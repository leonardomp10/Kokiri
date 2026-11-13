// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <memory>

#include "common/logging.h"
#include "common/settings.h"
#include "common/signal_chain.h"
#include "core/arm/nce/arm_nce.h"
#include "core/arm/nce/interpreter_visitor.h"
#include "core/arm/nce/patcher.h"
#include "core/core.h"
#include "core/memory.h"

#include "core/hle/kernel/k_process.h"

#include <dlfcn.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace Core {

namespace {

struct sigaction g_orig_bus_action;
struct sigaction g_orig_segv_action;

// A host fault that NCE does not own is passed to the handler that was installed before ours.
// When that handler is SIG_DFL (the usual case), calling sa_sigaction jumps to address 0 and the
// core dump only shows pc=0 inside the signal trampoline, losing the real fault. Instead, print the
// faulting context (async-signal-safe writes; dladdr is best effort, the process is dying anyway),
// restore the default action and return, so the instruction faults again and the kernel dumps the
// real pc.
void FaultWrite(const char* s) {
    size_t n = 0;
    while (s[n] != '\0') {
        ++n;
    }
    [[maybe_unused]] auto r = write(STDERR_FILENO, s, n);
}

void FaultWriteHex(const char* label, u64 v) {
    char buf[40];
    size_t i = 0;
    while (label[i] != '\0' && i < 16) {
        buf[i] = label[i];
        ++i;
    }
    buf[i++] = '0';
    buf[i++] = 'x';
    for (int sh = 60; sh >= 0; sh -= 4) {
        buf[i++] = "0123456789abcdef"[(v >> sh) & 0xf];
    }
    buf[i++] = ' ';
    buf[i] = '\0';
    FaultWrite(buf);
}

void FaultWriteWhere(const char* label, u64 addr) {
    FaultWriteHex(label, addr);
    Dl_info dl{};
    if (addr != 0 && dladdr(reinterpret_cast<void*>(addr), &dl) != 0 && dl.dli_fname != nullptr) {
        FaultWrite("(");
        FaultWrite(dl.dli_fname);
        FaultWriteHex("+", addr - reinterpret_cast<u64>(dl.dli_fbase));
        if (dl.dli_sname != nullptr) {
            FaultWrite(dl.dli_sname);
        }
        FaultWrite(")");
    } else {
        FaultWrite("(anon)");
    }
    FaultWrite("\n");
}

void ChainHostFault(const struct sigaction& orig, int sig, void* raw_info, void* raw_context) {
    if ((orig.sa_flags & SA_SIGINFO) != 0 && orig.sa_sigaction != nullptr) {
        return orig.sa_sigaction(sig, static_cast<siginfo_t*>(raw_info), raw_context);
    }
    if ((orig.sa_flags & SA_SIGINFO) == 0 && orig.sa_handler != SIG_DFL &&
        orig.sa_handler != SIG_IGN && orig.sa_handler != nullptr) {
        return orig.sa_handler(sig);
    }
    auto* info = static_cast<siginfo_t*>(raw_info);
    auto& mctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;
    char name[17]{};
    prctl(PR_GET_NAME, name, 0, 0, 0);
    FaultWrite("\n=== NX2 host fault nao tratado: sinal ");
    FaultWrite(sig == SIGSEGV ? "SIGSEGV" : (sig == SIGBUS ? "SIGBUS" : "?"));
    FaultWrite(" thread '");
    FaultWrite(name);
    FaultWrite("' ");
    FaultWriteHex("tid=", static_cast<u64>(syscall(SYS_gettid)));
    FaultWriteHex("code=", static_cast<u64>(info->si_code));
    FaultWriteHex("addr=", reinterpret_cast<u64>(info->si_addr));
    FaultWrite("\n");
    FaultWriteWhere("pc=", mctx.pc);
    FaultWriteWhere("lr=", mctx.regs[30]);
    for (int r = 0; r < 31; ++r) {
        char lbl[6] = {'x', static_cast<char>('0' + r / 10), static_cast<char>('0' + r % 10), '=',
                       '\0', '\0'};
        FaultWriteHex(lbl, mctx.regs[r]);
        if (r % 4 == 3) {
            FaultWrite("\n");
        }
    }
    FaultWriteHex("sp=", mctx.sp);
    FaultWrite("\n");
    // Frame-pointer walk; stops at the first pointer outside a sane stack window.
    u64 fp = mctx.regs[29];
    for (int depth = 0; depth < 24; ++depth) {
        if (fp == 0 || (fp & 0xf) != 0 || fp < mctx.sp || fp - mctx.sp > (64ULL << 20)) {
            break;
        }
        const u64* frame = reinterpret_cast<const u64*>(fp);
        FaultWriteWhere("  #", frame[1]);
        fp = frame[0];
    }
    FaultWrite("=== fim\n");
    // Back to the default action; returning re-executes the faulting instruction.
    struct sigaction dfl {};
    dfl.sa_handler = SIG_DFL;
    sigemptyset(&dfl.sa_mask);
    sigaction(sig, &dfl, nullptr);
}

// Verify assembly offsets.
using NativeExecutionParameters = Kernel::KThread::NativeExecutionParameters;
static_assert(offsetof(NativeExecutionParameters, native_context) == TpidrEl0NativeContext);
static_assert(offsetof(NativeExecutionParameters, lock) == TpidrEl0Lock);
static_assert(offsetof(NativeExecutionParameters, magic) == TpidrEl0TlsMagic);

fpsimd_context* GetFloatingPointState(mcontext_t& host_ctx) {
    _aarch64_ctx* header = reinterpret_cast<_aarch64_ctx*>(&host_ctx.__reserved);
    while (header->magic != FPSIMD_MAGIC) {
        header = reinterpret_cast<_aarch64_ctx*>(reinterpret_cast<char*>(header) + header->size);
    }
    return reinterpret_cast<fpsimd_context*>(header);
}

#ifndef ESR_MAGIC
#define ESR_MAGIC 0x45535201
struct esr_context {
    struct _aarch64_ctx head;
    __u64 esr;
};
#endif

// Whether the data abort that raised this signal was a write (ESR_EL1.WnR).
bool IsWriteFault(mcontext_t& host_ctx) {
    auto* header = reinterpret_cast<_aarch64_ctx*>(&host_ctx.__reserved);
    while (header->magic != 0 && header->size != 0) {
        if (header->magic == ESR_MAGIC) {
            const u64 esr = reinterpret_cast<esr_context*>(header)->esr;
            const u64 exception_class = (esr >> 26) & 0x3f;
            constexpr u64 DataAbortLowerEl = 0x24;
            constexpr u64 DataAbortSameEl = 0x25;
            return (exception_class == DataAbortLowerEl || exception_class == DataAbortSameEl) &&
                   (esr & (u64{1} << 6)) != 0;
        }
        header = reinterpret_cast<_aarch64_ctx*>(reinterpret_cast<char*>(header) + header->size);
    }
    return false;
}

// A store to a page that holds GPU-written buffer data. Handing the page back to the CPU
// downloads that data and waits for the host GPU, ~1 ms each time: Ocarina of Time writes small
// structures next to GPU data a few hundred times a second, which kept the GPU thread waiting
// ~70% of the time. Running the store here sends it to the buffer cache as a write of a desktop
// build does (collected for the GPU thread) and leaves the page with the GPU. A run of stores to
// one page, such as a copy loop, still hands the page back.
// Ocarina of Time only: Pokemon Pokopia stores to such pages ~20,000 times a second, and each
// store trapping (and taking the buffer cache lock the GPU thread holds) dropped it from ~40 to
// ~5 FPS for seconds at a time, where the old path hands a page over once.
bool RunStoreToGpuPage(Core::Memory::Memory& memory, mcontext_t& host_ctx, u64 page) {
    static constexpr u64 OcarinaOfTime = 0x0400486024DEC000ULL;
    // EDEN_NX2_NCE_GPU_STORES=1 or 0 turns this on or off for any game.
    static const int forced = [] {
        const char* value = std::getenv("EDEN_NX2_NCE_GPU_STORES");
        return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
    }();
    // Off for Ocarina of Time too: it stores command words next to the tile counts a compute
    // shader writes, and with this its lighting dispatches read garbage counts (71 million
    // groups), each running for seconds on a phone.
    (void)OcarinaOfTime;
    const bool enabled = forced == 1;
    if (!enabled || !IsWriteFault(host_ctx)) {
        return false;
    }
    struct Run {
        u64 page = ~u64{0};
        u32 stores = 0;
        std::chrono::steady_clock::time_point last;
    };
    thread_local Run run;
    const auto now = std::chrono::steady_clock::now();
    if (run.page != page || now - run.last > std::chrono::microseconds{200}) {
        run.page = page;
        run.stores = 0;
    }
    run.last = now;
    if (++run.stores > 32) {
        run.page = ~u64{0};
        return false;
    }
    if (!memory.HasGpuWrittenDataNCE(page, Memory::YUZU_PAGESIZE)) {
        return false;
    }
    const auto next_pc = MatchAndExecuteOneInstruction(memory, &host_ctx,
                                                       GetFloatingPointState(host_ctx));
    static std::atomic<u64> stores_run{};
    static std::atomic<u64> stores_unhandled{};
    if (!next_pc) {
        stores_unhandled.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (const u64 count = stores_run.fetch_add(1, std::memory_order_relaxed) + 1;
        (count & 0x3fff) == 0) {
        LOG_INFO(Core_ARM, "NCE: {} stores to GPU-written pages run in place, {} not decodable",
                 count, stores_unhandled.load(std::memory_order_relaxed));
    }
    host_ctx.pc = *next_pc;
    return true;
}

using namespace Common::Literals;
constexpr u32 StackSize = 128_KiB;

} // namespace

void* ArmNce::RestoreGuestContext(void* raw_context) {
    // Retrieve the host context.
    auto& host_ctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;

    // Thread-local parameters will be located in x9.
    auto* tpidr = reinterpret_cast<NativeExecutionParameters*>(host_ctx.regs[9]);
    auto* guest_ctx = static_cast<GuestContext*>(tpidr->native_context);

    // Retrieve the host floating point state.
    auto* fpctx = GetFloatingPointState(host_ctx);

    // Save host callee-saved registers.
    std::memcpy(guest_ctx->host_ctx.host_saved_vregs.data(), &fpctx->vregs[8],
                sizeof(guest_ctx->host_ctx.host_saved_vregs));
    std::memcpy(guest_ctx->host_ctx.host_saved_regs.data(), &host_ctx.regs[19],
                sizeof(guest_ctx->host_ctx.host_saved_regs));

    // Save stack pointer.
    guest_ctx->host_ctx.host_sp = host_ctx.sp;

    // Restore all guest state except tpidr_el0.
    host_ctx.sp = guest_ctx->sp;
    host_ctx.pc = guest_ctx->pc;
    host_ctx.pstate = guest_ctx->pstate;
    fpctx->fpcr = guest_ctx->fpcr;
    fpctx->fpsr = guest_ctx->fpsr;
    std::memcpy(host_ctx.regs, guest_ctx->cpu_registers.data(), sizeof(host_ctx.regs));
    std::memcpy(fpctx->vregs, guest_ctx->vector_registers.data(), sizeof(fpctx->vregs));

    // Return the new thread-local storage pointer.
    return tpidr;
}

void ArmNce::SaveGuestContext(GuestContext* guest_ctx, void* raw_context) {
    // Retrieve the host context.
    auto& host_ctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;

    // Retrieve the host floating point state.
    auto* fpctx = GetFloatingPointState(host_ctx);

    // Save all guest registers except tpidr_el0.
    std::memcpy(guest_ctx->cpu_registers.data(), host_ctx.regs, sizeof(host_ctx.regs));
    std::memcpy(guest_ctx->vector_registers.data(), fpctx->vregs, sizeof(fpctx->vregs));
    guest_ctx->fpsr = fpctx->fpsr;
    guest_ctx->fpcr = fpctx->fpcr;
    guest_ctx->pstate = static_cast<u32>(host_ctx.pstate);
    guest_ctx->pc = host_ctx.pc;
    guest_ctx->sp = host_ctx.sp;

    // Restore stack pointer.
    host_ctx.sp = guest_ctx->host_ctx.host_sp;

    // Restore host callee-saved registers.
    std::memcpy(&host_ctx.regs[19], guest_ctx->host_ctx.host_saved_regs.data(),
                sizeof(guest_ctx->host_ctx.host_saved_regs));
    std::memcpy(&fpctx->vregs[8], guest_ctx->host_ctx.host_saved_vregs.data(),
                sizeof(guest_ctx->host_ctx.host_saved_vregs));

    // Return from the call on exit by setting pc to x30.
    host_ctx.pc = guest_ctx->host_ctx.host_saved_regs[11];

    // Clear esr_el1 and return it.
    host_ctx.regs[0] = guest_ctx->esr_el1.exchange(0);
}

bool ArmNce::HandleFailedGuestFault(GuestContext* guest_ctx, void* raw_info, void* raw_context) {
    auto& host_ctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;
    auto* info = static_cast<siginfo_t*>(raw_info);

    // We can't handle the access, so determine why we crashed.
    const bool is_prefetch_abort = host_ctx.pc == reinterpret_cast<u64>(info->si_addr);

    // For data aborts, skip the instruction and return to guest code.
    // This will allow games to continue in many scenarios where they would otherwise crash.
    if (!is_prefetch_abort) {
        host_ctx.pc += 4;
        return true;
    }

    // This is a prefetch abort.
    guest_ctx->esr_el1.fetch_or(static_cast<u64>(HaltReason::PrefetchAbort));

    // Forcibly mark the context as locked. We are still running.
    // We may race with SignalInterrupt here:
    // - If we lose the race, then SignalInterrupt will send us a signal we are masking,
    //   and it will do nothing when it is unmasked, as we have already left guest code.
    // - If we win the race, then SignalInterrupt will wait for us to unlock first.
    auto& thread_params = guest_ctx->parent->m_running_thread->GetNativeExecutionParameters();
    thread_params.lock.store(SpinLockLocked);

    // Return to host.
    SaveGuestContext(guest_ctx, raw_context);
    return false;
}

bool ArmNce::HandleGuestAlignmentFault(GuestContext* guest_ctx, void* raw_info, void* raw_context) {
    auto& host_ctx = static_cast<ucontext_t*>(raw_context)->uc_mcontext;
    auto* fpctx = GetFloatingPointState(host_ctx);
    auto& memory = guest_ctx->parent->m_running_thread->GetOwnerProcess()->GetMemory();

    // Match and execute an instruction.
    auto next_pc = MatchAndExecuteOneInstruction(memory, &host_ctx, fpctx);
    if (next_pc) {
        host_ctx.pc = *next_pc;
        return true;
    }

    // We couldn't handle the access.
    return HandleFailedGuestFault(guest_ctx, raw_info, raw_context);
}

bool ArmNce::HandleGuestAccessFault(GuestContext* guest_ctx, void* raw_info, void* raw_context) {
    auto* info = static_cast<siginfo_t*>(raw_info);

    // Try to handle an invalid access.
    // TODO: handle accesses which split a page?
    const Common::ProcessAddress addr =
        (reinterpret_cast<u64>(info->si_addr) & ~Memory::YUZU_PAGEMASK);
    auto& memory = guest_ctx->parent->m_running_thread->GetOwnerProcess()->GetMemory();
    if (RunStoreToGpuPage(memory, static_cast<ucontext_t*>(raw_context)->uc_mcontext,
                          GetInteger(addr))) {
        return true;
    }
    if (memory.InvalidateNCE(addr, Memory::YUZU_PAGESIZE)) {
        // We handled the access successfully and are returning to guest code.
        return true;
    }

    // We couldn't handle the access.
    return HandleFailedGuestFault(guest_ctx, raw_info, raw_context);
}

void ArmNce::HandleHostAlignmentFault(int sig, void* raw_info, void* raw_context) {
    ChainHostFault(g_orig_bus_action, sig, raw_info, raw_context);
}

void ArmNce::HandleHostAccessFault(int sig, void* raw_info, void* raw_context) {
    ChainHostFault(g_orig_segv_action, sig, raw_info, raw_context);
}

void ArmNce::LockThread(Kernel::KThread* thread) {
    auto* thread_params = &thread->GetNativeExecutionParameters();
    LockThreadParameters(thread_params);
}

void ArmNce::UnlockThread(Kernel::KThread* thread) {
    auto* thread_params = &thread->GetNativeExecutionParameters();
    m_guest_ctx.tpidr_el0 = thread_params->tpidr_el0;
    m_guest_ctx.tpidrro_el0 = thread_params->tpidrro_el0;
    thread_params->native_context = nullptr;
    UnlockThreadParameters(thread_params);
}

HaltReason ArmNce::RunThread(Kernel::KThread* thread) {
    // Check if we're already interrupted.
    // If we are, we can just return immediately.
    HaltReason hr = static_cast<HaltReason>(m_guest_ctx.esr_el1.exchange(0));
    if (True(hr)) {
        return hr;
    }

    // Pre-fetch thread context data to improve cache locality
    auto* thread_params = &thread->GetNativeExecutionParameters();
    auto* process = thread->GetOwnerProcess();

    // Move non-critical operations outside the locked section
    const u64 tpidr_el0_cache = m_guest_ctx.tpidr_el0;
    const u64 tpidrro_el0_cache = m_guest_ctx.tpidrro_el0;

    // Critical section begins - minimize operations here
    m_running_thread = thread;
    m_guest_ctx.parent = this;
    thread_params->native_context = &m_guest_ctx;
    thread_params->tpidr_el0 = tpidr_el0_cache;
    thread_params->tpidrro_el0 = tpidrro_el0_cache;

    // Memory barrier to ensure visibility of changes
    std::atomic_thread_fence(std::memory_order_release);
    thread_params->is_running = true;

    // TODO: finding and creating the post handler needs to be locked
    // to deal with dynamic loading of NROs.
    const auto& post_handlers = process->GetPostHandlers();
    if (auto it = post_handlers.find(m_guest_ctx.pc); it != post_handlers.end()) {
        hr = ReturnToRunCodeByTrampoline(thread_params, &m_guest_ctx, it->second);
    } else {
        hr = ReturnToRunCodeByExceptionLevelChange(m_thread_id, thread_params);  // Android: Use "process handle SIGUSR2 -n true -p true -s false" (and SIGURG) in LLDB when debugging
    }

    // Critical section for thread cleanup
    std::atomic_thread_fence(std::memory_order_acquire);

    // Cache values before releasing thread
    const u64 final_tpidr_el0 = thread_params->tpidr_el0;

    // Minimize critical section
    thread_params->is_running = false;
    thread_params->native_context = nullptr;
    m_running_thread = nullptr;

    // Non-critical updates can happen after releasing the thread
    m_guest_ctx.tpidr_el0 = final_tpidr_el0;

    // Return the halt reason.
    return hr;
}

HaltReason ArmNce::StepThread(Kernel::KThread* thread) {
    return HaltReason::StepThread;
}

u32 ArmNce::GetSvcNumber() const {
    return m_guest_ctx.svc;
}

void ArmNce::GetSvcArguments(std::span<uint64_t, 8> args) const {
    for (size_t i = 0; i < 8; i++) {
        args[i] = m_guest_ctx.cpu_registers[i];
    }
}

void ArmNce::SetSvcArguments(std::span<const uint64_t, 8> args) {
    for (size_t i = 0; i < 8; i++) {
        m_guest_ctx.cpu_registers[i] = args[i];
    }
}

ArmNce::ArmNce(System& system, bool uses_wall_clock, std::size_t core_index)
    : ArmInterface{uses_wall_clock}, m_system{system}, m_core_index{core_index} {
    m_guest_ctx.system = &m_system;
}

ArmNce::~ArmNce() = default;

void ArmNce::Initialize() {
    if (m_thread_id == -1) {
        m_thread_id = gettid();
    }

    // Configure signal stack.
    if (!m_stack) {
        m_stack = std::make_unique<u8[]>(StackSize);

        stack_t ss{};
        ss.ss_sp = m_stack.get();
        ss.ss_size = StackSize;
        sigaltstack(&ss, nullptr);
    }

    // Set up signals.
    static std::once_flag flag;
    std::call_once(flag, [] {
        using HandlerType = decltype(sigaction::sa_sigaction);

        sigset_t signal_mask;
        sigemptyset(&signal_mask);
        sigaddset(&signal_mask, ReturnToRunCodeByExceptionLevelChangeSignal);
        sigaddset(&signal_mask, BreakFromRunCodeSignal);
        sigaddset(&signal_mask, GuestAlignmentFaultSignal);
        sigaddset(&signal_mask, GuestAccessFaultSignal);

        struct sigaction return_to_run_code_action {};
        return_to_run_code_action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        return_to_run_code_action.sa_sigaction = reinterpret_cast<HandlerType>(
            &ArmNce::ReturnToRunCodeByExceptionLevelChangeSignalHandler);
        return_to_run_code_action.sa_mask = signal_mask;
        Common::SigAction(ReturnToRunCodeByExceptionLevelChangeSignal, &return_to_run_code_action,
                          nullptr);

        struct sigaction break_from_run_code_action {};
        break_from_run_code_action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        break_from_run_code_action.sa_sigaction =
            reinterpret_cast<HandlerType>(&ArmNce::BreakFromRunCodeSignalHandler);
        break_from_run_code_action.sa_mask = signal_mask;
        Common::SigAction(BreakFromRunCodeSignal, &break_from_run_code_action, nullptr);

        struct sigaction alignment_fault_action {};
        alignment_fault_action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        alignment_fault_action.sa_sigaction =
            reinterpret_cast<HandlerType>(&ArmNce::GuestAlignmentFaultSignalHandler);
        alignment_fault_action.sa_mask = signal_mask;
        Common::SigAction(GuestAlignmentFaultSignal, &alignment_fault_action, nullptr);

        struct sigaction access_fault_action {};
        access_fault_action.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART;
        access_fault_action.sa_sigaction =
            reinterpret_cast<HandlerType>(&ArmNce::GuestAccessFaultSignalHandler);
        access_fault_action.sa_mask = signal_mask;
        Common::SigAction(GuestAccessFaultSignal, &access_fault_action, &g_orig_segv_action);
    });
}

void ArmNce::SetTpidrroEl0(u64 value) {
    m_guest_ctx.tpidrro_el0 = value;
}

void ArmNce::GetContext(Kernel::Svc::ThreadContext& ctx) const {
    for (size_t i = 0; i < 29; i++) {
        ctx.r[i] = m_guest_ctx.cpu_registers[i];
    }
    ctx.fp = m_guest_ctx.cpu_registers[29];
    ctx.lr = m_guest_ctx.cpu_registers[30];
    ctx.sp = m_guest_ctx.sp;
    ctx.pc = m_guest_ctx.pc;
    ctx.pstate = m_guest_ctx.pstate;
    ctx.v = m_guest_ctx.vector_registers;
    ctx.fpcr = m_guest_ctx.fpcr;
    ctx.fpsr = m_guest_ctx.fpsr;
    ctx.tpidr = m_guest_ctx.tpidr_el0;
}

void ArmNce::SetContext(const Kernel::Svc::ThreadContext& ctx) {
    for (size_t i = 0; i < 29; i++) {
        m_guest_ctx.cpu_registers[i] = ctx.r[i];
    }
    m_guest_ctx.cpu_registers[29] = ctx.fp;
    m_guest_ctx.cpu_registers[30] = ctx.lr;
    m_guest_ctx.sp = ctx.sp;
    m_guest_ctx.pc = ctx.pc;
    m_guest_ctx.pstate = ctx.pstate;
    m_guest_ctx.vector_registers = ctx.v;
    m_guest_ctx.fpcr = ctx.fpcr;
    m_guest_ctx.fpsr = ctx.fpsr;
    m_guest_ctx.tpidr_el0 = ctx.tpidr;
}

void ArmNce::SignalInterrupt(Kernel::KThread* thread) {
    constexpr u64 break_loop = static_cast<u64>(HaltReason::BreakLoop);
    if ((m_guest_ctx.esr_el1.fetch_or(break_loop) & break_loop) != 0) return;

    auto* params = &thread->GetNativeExecutionParameters();
    LockThreadParameters(params);

    // Ensure visibility of is_running after lock acquire
    std::atomic_thread_fence(std::memory_order_acquire);

    if (params->is_running) {
        // We should signal to the running thread.
        // The running thread will unlock the thread context.
        syscall(SYS_tkill, m_thread_id, BreakFromRunCodeSignal);
    } else {
        // If the thread is no longer running, we have nothing to do.
        UnlockThreadParameters(params);
    }
}

[[maybe_unused]] const std::size_t CACHE_PAGE_SIZE = 4096;

void ArmNce::ClearInstructionCache() {
#ifdef __aarch64__
    // Ensure all previous memory operations complete
    asm volatile("dsb ish\n"
                 "dsb ish\n"
                 "isb" ::: "memory");
#endif
}

void ArmNce::InvalidateCacheRange(u64 addr, std::size_t size) {
    this->ClearInstructionCache();
}

} // namespace Core

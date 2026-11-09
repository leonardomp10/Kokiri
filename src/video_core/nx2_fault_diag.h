// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

// EDEN_NX2_FAULT_DIAG=1 (diagnostics): what the guest CPU's writes into GPU-written pages make the
// GPU thread download (gpu.cpp InvalidateRegion with preserve_gpu_writes: a full host GPU wait).
// For each downloaded range: which binding path marked it GPU-written last (MarkWrittenBuffer's
// site) and over how many bytes, how many of the downloaded bytes differ from guest memory (the
// bytes the GPU really changed), and whether the page holds command lists the front fetched.
// Logged every 10 s.
namespace VideoCommon::Nx2FaultDiag {

[[nodiscard]] inline bool Enabled() {
    static const bool enabled = std::getenv("EDEN_NX2_FAULT_DIAG") != nullptr;
    return enabled;
}

/// Set by the GPU thread around the flush a guest write fault asks for, and the bytes it
/// downloaded.
inline thread_local bool t_in_fault = false;
inline thread_local u64 t_fault_bytes = 0;

inline constexpr u64 PageBits = 12;
inline std::array<std::atomic<u64>, (u64{1} << (36 - PageBits)) / 64> g_pushbuffer_pages{};

/// The front fetched command list words on [device, device + size).
inline void NotePushbuffer(u64 device, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 last = (std::min<u64>)((device + size - 1) >> PageBits, g_pushbuffer_pages.size() * 64 - 1);
    for (u64 page = device >> PageBits; page <= last; ++page) {
        auto& word = g_pushbuffer_pages[page / 64];
        const u64 bit = u64{1} << (page % 64);
        if ((word.load(std::memory_order_relaxed) & bit) == 0) {
            word.fetch_or(bit, std::memory_order_relaxed);
        }
    }
}

[[nodiscard]] inline bool IsPushbuffer(u64 device) {
    const u64 page = device >> PageBits;
    return page < g_pushbuffer_pages.size() * 64 &&
           ((g_pushbuffer_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) & 1);
}

struct Record {
    u64 addr;
    u32 size;
    u8 site;
};

struct State {
    std::mutex mutex;
    std::array<Record, 8192> records{};
    u64 record_count{};
    struct Stat {
        u64 ranges{};
        u64 bytes{};
        u64 diff_bytes{};
        u64 pushbuffer{};
    };
    /// By the last writer (site, binding address, binding size) of each downloaded range.
    std::map<std::tuple<u8, u64, u32>, Stat> stats;
    u64 faults{};
    u64 fault_us{};
    u64 fault_bytes{};
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};

inline State& Get() {
    static State state;
    return state;
}

inline void NoteWrite(u64 addr, u32 size, u8 site, [[maybe_unused]] u64 tick) {
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    state.records[state.record_count++ % state.records.size()] = {addr, size, site};
}

/// A range the fault's flush downloaded: `gpu` the host GPU's bytes, `guest` guest memory's.
inline void NoteDownload(u64 addr, u64 size, const u8* gpu, const u8* guest) {
    t_fault_bytes += size;
    u64 diff = 0;
    for (u64 i = 0; i < size; ++i) {
        diff += gpu[i] != guest[i] ? 1 : 0;
    }
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    Record writer{0, 0, 0};
    const u64 remembered = (std::min<u64>)(state.record_count, state.records.size());
    for (u64 i = 0; i < remembered; ++i) {
        const Record& record =
            state.records[(state.record_count - 1 - i) % state.records.size()];
        if (record.addr < addr + size && addr < record.addr + record.size) {
            writer = record;
            break;
        }
    }
    auto& stat = state.stats[{writer.site, writer.addr, writer.size}];
    ++stat.ranges;
    stat.bytes += size;
    stat.diff_bytes += diff;
    stat.pushbuffer += IsPushbuffer(addr) ? 1 : 0;
}

/// A guest write fault's flush ran: `bytes` downloaded in `us` microseconds.
inline void NoteFault(u64 us, u64 bytes) {
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.faults;
    state.fault_us += us;
    state.fault_bytes += bytes;
    const auto now = std::chrono::steady_clock::now();
    if (now - state.window < std::chrono::seconds{10}) {
        return;
    }
    LOG_INFO(HW_GPU, "Fault diag (10 s): {} write faults flushed, {:.1f} ms, {} KiB downloaded",
             state.faults, static_cast<double>(state.fault_us) / 1000.0, state.fault_bytes / 1024);
    std::vector<std::pair<u64, std::tuple<u8, u64, u32>>> order;
    for (const auto& [key, stat] : state.stats) {
        order.emplace_back(stat.ranges, key);
    }
    std::ranges::sort(order, std::greater{});
    for (size_t i = 0; i < order.size() && i < 10; ++i) {
        const auto& key = order[i].second;
        const auto& stat = state.stats[key];
        LOG_INFO(HW_GPU,
                 "Fault diag:   writer site {} binding {:#x}+{:#x}: {} ranges, {} bytes, {} "
                 "differ from guest memory, {} on fetched command list pages",
                 std::get<0>(key), std::get<1>(key), std::get<2>(key), stat.ranges, stat.bytes,
                 stat.diff_bytes, stat.pushbuffer);
    }
    state.stats.clear();
    state.faults = 0;
    state.fault_us = 0;
    state.fault_bytes = 0;
    state.window = now;
}

} // namespace VideoCommon::Nx2FaultDiag

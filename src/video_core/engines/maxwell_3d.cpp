// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <unordered_map>
#include <array>
#include <chrono>
#include <atomic>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_list_indirect.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/query_cache/frame_trace.h"
#include "video_core/query_dma.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

// EDEN_NX2_3D_METHODS: log the first write to each 3D register (finds methods Eden doesn't model).
static const bool g_trace_first_method_writes = std::getenv("EDEN_NX2_3D_METHODS") != nullptr;

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
    fast_maxwell3d = !record_write_history &&
                     !g_trace_first_method_writes && !VideoCommon::Nx2Profile::Enabled();
}

Maxwell3D::~Maxwell3D() = default;

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case RootTables::SelectMethod:
    case RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Nearly every call arrives whole in one method run (title 1 makes ~1,000,000 a second):
    // copy it to the stack instead of growing the parameter vector. Still a copy: the command
    // list is often guest memory read in place, which the guest may reuse while the macro runs
    // (and the macro memo stores the parameters after it ran). Only accurate GPU mode rewrites
    // the parameters in place (RefreshParameters), so it keeps the vector.
    // EDEN_NX2_MACRO_COPY=1: always use the vector (for comparisons).
    static const bool always_copy = std::getenv("EDEN_NX2_MACRO_COPY") != nullptr;
    if (is_last_call && macro_params.empty() && amount <= 64 && !always_copy &&
        !Settings::IsGPULevelHigh()) {
        std::array<u32, 64> parameters;
        std::copy_n(base_start, amount, parameters.begin());
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, std::span<const u32>{parameters.data(), amount});
        macro_segments.clear();
        current_macro_dirty = false;
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u64 stream_size = GetNativeMacroEngine().native_turing
                                    ? regs.NativeVertexStreamSize(index)
                                    : gpu_addr_end - gpu_addr_begin;
        const u32 address_size = static_cast<u32>((std::min<u64>)(stream_size, (std::numeric_limits<u32>::max)()));
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

u32 Maxwell3D::ProcessShadowRam(u32 method, u32 argument) {
    // Keep track of the register value in shadow_state when requested.
    auto const c = shadow_state.shadow_ram_control;
    if (c == Regs::ShadowRamControl::Track || c == Regs::ShadowRamControl::TrackWithFilter)
        return shadow_state.reg_array[method] = argument;
    else if (c == Regs::ShadowRamControl::Replay)
        return shadow_state.reg_array[method];
    return argument;
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    const auto control = shadow_state.shadow_ram_control;
    // Sunk methods are state writes only (not in execution_mask): skip unchanged values, as
    // CallMethod does.
    const bool skip_same = SkipSameValueWrites();
    const auto process = [&](u32 method, u32 value) {
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    };
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            process(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            process(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            process(method, value);
    }
    method_sink.clear();
}

void Maxwell3D::ProcessDirtyRegisters(u32 method, u32 argument) {
    // The instances after the first are drawn together once their loop ends, and they must see
    // the state they were begun with: NVN's draw macros reset the base instance (and its copy in
    // c[0][0x644]) right after the loop, before the macro returns.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    regs.reg_array[method] = argument;
    for (auto const& table : dirty.tables)
        dirty.flags[table[method]] = true;
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    if (GetNativeMacroEngine().native_turing && method == MAXWELL3D_REG_INDEX(clear_surface)) {
        LOG_DEBUG(HW_GPU, "Native GPU surface clear: value={:#x}", argument);
    }
    switch (method) {
    case RootTables::SelectMethod:
        root_tables.selected = argument & 7;
        root_tables.offset = (argument >> 8) & 0xff;
        return;
    case RootTables::LoadMethod:
        if (root_tables.offset < 256 && root_tables.offset % 4 == 0) {
            root_tables.tables[root_tables.selected][root_tables.offset / 4] = argument;
            root_tables.loaded = true;
        }
        root_tables.offset += 4;
        return;
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    // With macros on the split GPU thread's front (FrontWrite), the front loads and binds them.
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        if (front_macros) return;
        return GetNativeMacroEngine().ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        if (front_macros) return;
        return GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        if (front_macros) {
            ++regs.load_mme.start_address_ptr;
            return;
        }
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, std::span<const u32> parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(GetNativeMacroPositions().size());

    // Execute the current macro.
    GetNativeMacroEngine().Execute(system, *this, GetNativeMacroPositions()[entry], parameters);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::TraceFirstMethodWrite(u32 method, u32 argument) {
    static std::array<bool, Regs::NUM_REGS> seen{};
    if (method < seen.size() && !seen[method]) {
        seen[method] = true;
        LOG_INFO(HW_GPU, "3D method first write: {:#x} (byte {:#x}) = {:#x}", method, method * 4, argument);
    }
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (draw_manager.pending_vertex_arguments) [[unlikely]] {
        draw_manager.NoteVertexArgumentWrite(method);
    }
    if (!fast_maxwell3d || executing_macro != 0 || method >= MacroRegistersStart ||
        method == MAXWELL3D_REG_INDEX(shadow_ram_control)) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    if (VideoCommon::Nx2Perf::Verify()) {
        return VerifyCallMethod(system, method, method_argument, is_last_call);
    }
    CallMethodFast(method, method_argument, is_last_call);
}

namespace {

// Executable methods the draw manager does nothing for (ProcessDirtyRegisters is all they do).
bool IsStoreOnlyExecutable(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

// CallMethodSlow without the checks that cannot apply (CallMethod): no macro collects parameters,
// no write history, trace or profile, not a macro call or SET_SHADOW_RAM_CONTROL. The hot
// executable methods go straight to their handler.
bool Maxwell3D::TryFastDrawBegin(u32 argument) {
    // The conditions under which CallMethod reaches CallMethodFast for this method.
    if (draw_manager.pending_vertex_arguments || !fast_maxwell3d || executing_macro != 0 ||
        VideoCommon::Nx2Perf::Verify()) {
        return false;
    }
    constexpr u32 method = MAXWELL3D_REG_INDEX(draw.begin);
    ProcessDirtyRegisters(method, ProcessShadowRam(method, argument));
    draw_manager.DrawBegin(*this);
    return true;
}

void Maxwell3D::CallMethodFast(u32 method, u32 method_argument, bool is_last_call) {
    const u32 argument = ProcessShadowRam(method, method_argument);
    if (!execution_mask[method]) {
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method - cb_data < 16) {
        return ProcessCBData(argument);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        return draw_manager.DrawBegin(*this);
    case MAXWELL3D_REG_INDEX(draw.end):
        return draw_manager.DrawEnd(*this);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return;
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    default:
        return ProcessMethodCall(method, argument, method_argument, is_last_call);
    }
}

void Maxwell3D::VerifyCallMethod(Core::System& system, u32 method, u32 method_argument,
                                 bool is_last_call) {
    // Only register writes can run twice: handlers draw, upload and bind, and a deferred draw
    // ProcessDirtyRegisters may issue is a draw too.
    if ((execution_mask[method] && !IsStoreOnlyExecutable(method)) ||
        draw_manager.HasDeferredDraw()) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    const u32 reg = regs.reg_array[method];
    const u32 shadow = shadow_state.reg_array[method];
    const DirtyState::Flags flags = dirty.flags;
    CallMethodFast(method, method_argument, is_last_call);
    const u32 fast_reg = regs.reg_array[method];
    const u32 fast_shadow = shadow_state.reg_array[method];
    const DirtyState::Flags fast_flags = dirty.flags;
    regs.reg_array[method] = reg;
    shadow_state.reg_array[method] = shadow;
    dirty.flags = flags;
    CallMethodSlow(system, method, method_argument, is_last_call);
    if (fast_reg == regs.reg_array[method] && fast_shadow == shadow_state.reg_array[method] &&
        fast_flags == dirty.flags) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) >= 64) {
        return;
    }
    std::string flag_differences;
    const DirtyState::Flags differing = fast_flags ^ dirty.flags;
    for (size_t i = 0; i < differing.size(); ++i) {
        if (differing[i]) {
            flag_differences += fmt::format(" {}={}", i, fast_flags[i] ? 1 : 0);
        }
    }
    LOG_ERROR(HW_GPU,
              "Nx2Perf method: {:#x} = {:#x}: register {:#x} (old {:#x}), shadow {:#x} (old {:#x}), "
              "fast dirty flags differ:{}",
              method, method_argument, fast_reg, regs.reg_array[method], fast_shadow,
              shadow_state.reg_array[method], flag_differences);
}

void Maxwell3D::CallConstBufferWords(Core::System& system, const u32* methods,
                                     const u32* arguments, u32 count) {
    const u32 offset = regs.const_buffer.offset;
    const u64 bytes = u64{count} * sizeof(u32);
    const GPUVAddr address = regs.const_buffer.Address();
    // Keep per-word translation across device pages and per-word clipping at buffer limits.
    if (!fast_maxwell3d || executing_macro != 0 || address == 0 ||
        offset > regs.const_buffer.size || bytes > regs.const_buffer.size - offset ||
        bytes > Core::DEVICE_PAGESIZE - ((address + offset) & Core::DEVICE_PAGEMASK)) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        return;
    }
    std::array<u32, 64> values;
    const bool replay = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay;
    for (u32 i = 0; i < count; ++i) {
        values[i] = replay ? shadow_state.reg_array[methods[i]] : arguments[i];
    }
    if (VideoCommon::Nx2Perf::Verify()) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        std::array<u32, 64> written;
        const bool mapped = memory_manager.IsFullyMappedRange(address + offset, bytes);
        if (mapped) {
            memory_manager.ReadBlockUnsafe(address + offset, written.data(), bytes);
        }
        if (regs.const_buffer.offset != offset + bytes ||
            (mapped && std::memcmp(written.data(), values.data(), bytes) != 0)) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf method: {} constant buffer words at {:#x}+{:#x}: offset {:#x}, "
                          "expected {:#x}, memory {}",
                          count, address, offset, regs.const_buffer.offset, offset + bytes,
                          mapped && std::memcmp(written.data(), values.data(), bytes) != 0
                              ? "differs"
                              : "matches");
            }
        }
        return;
    }
    // As CallMethodFast for each word: the first may issue a deferred draw, before any upload.
    for (u32 i = 0; i < count; ++i) {
        ProcessDirtyRegisters(methods[i], ProcessShadowRam(methods[i], arguments[i]));
    }
    memory_manager.WriteWordsCached(address + offset, values.data(), count);
    regs.const_buffer.offset += static_cast<u32>(bytes);
}

void Maxwell3D::CallMethodSlow(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    // It is an error to write to a register other than the current macro's ARG register before
    // it has finished execution.
    if (executing_macro != 0 && method != executing_macro + 1) {
        DumpRecentWrites(method);
    }
    if (executing_macro != 0) {
        ASSERT(method == executing_macro + 1);
    }
    RecordWrite(method, method_argument, 1, is_last_call ? 1 : 2);

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }
    if (g_trace_first_method_writes) [[unlikely]] {
        TraceFirstMethodWrite(method, method_argument);
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::MethodWrites, 1);
        const bool same = regs.reg_array[method] == method_argument;
        if (same) {
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::SameValueWrites, 1);
        }
        VideoCommon::Nx2Profile::CountMethod(method, same);
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && front_macros) [[unlikely]] {
        back_control_writes[back_control_count++ % back_control_writes.size()] = {
            method_argument, static_cast<u32>(shadow_state.shadow_ram_control), is_last_call ? 1U : 0U};
    }
    const u32 argument = ProcessShadowRam(method, method_argument);
    // A state register written with the value it holds changes nothing: no dirty flags (they
    // would make the next draw re-check that state), no method processing. Title 1 makes ~6
    // million 3D register writes a second, ~60% of them same-value, most from macro replays.
    // Methods with side effects (execution_mask: draws, clears, constant buffer data and
    // binds, semaphores, ...) always run. EDEN_NX2_SKIP_SAME=0 turns it off.
    if (!execution_mask[method]) {
        // ProcessMethodCall (and the draw manager behind it) acts only on executable methods.
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::DumpRecentWrites(u32 method) {
    static u32 dumps{};
    if (dumps++ >= 4) {
        return;
    }
    LOG_ERROR(HW_GPU, "Write to {:#x} while macro {:#x} collects parameters ({} so far); last writes:",
              method, executing_macro, macro_params.size());
    if (!record_write_history) {
        LOG_ERROR(HW_GPU, "Register write history disabled; EDEN_NX2_3D_HISTORY=1 enables it");
        return;
    }
    for (size_t i = 0; i < recent_writes.size(); ++i) {
        const auto& w = recent_writes[(recent_write_index + i) % recent_writes.size()];
        if (w.amount != 0) {
            LOG_ERROR(HW_GPU, "  method {:#x} arg {:#x} amount {} pending {}", w.method, w.argument,
                      w.amount, w.pending);
        }
    }
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    // Non-incrementing: every word goes to `method`.
    if (draw_manager.pending_vertex_arguments && amount != 0) [[unlikely]] {
        draw_manager.NoteVertexArgumentWrite(method);
    }
    RecordWrite(method, amount ? base_start[0] : 0, amount, methods_pending);
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    // As in ProcessDirtyRegisters: the paths below write without it.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    const auto positions = GetNativeMacroPositions();
    const u32 index = regs.load_mme.start_address_ptr++;
    if (index < positions.size()) {
        positions[index] = data;
    }
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
    if (const u64 trace = VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::Write)) {
        LOG_INFO(HW_GPU,
                 "DKB_QUERY write n={} path=stamp as={} gpu={:#x} bytes={} value={:#x}",
                 trace, memory_manager.GetID(), sequence_address, long_query ? 8 : 4,
                 long_query ? payload : static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    if (rasterizer->AccelerateConditionalRendering()) {
        execute_on = true;
        return;
    }
    const GPUVAddr condition_address{regs.render_enable.Address()};
    Regs::ReportSemaphore::Compare cmp{};
    bool operands_read = false;
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
    if (const u64 trace =
            VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::CpuCondition)) {
        LOG_INFO(HW_GPU,
                 "DKB_CONDITION cpu n={} as={} gpu={:#x} mode={} override={} execute={} "
                 "operands_read={} initial={:#x}/{:#x} current={:#x}/{:#x}",
                 trace, memory_manager.GetID(), condition_address,
                 static_cast<u32>(regs.render_enable.mode),
                 static_cast<u32>(regs.render_enable_override), execute_on, operands_read,
                 cmp.initial_sequence, cmp.initial_mode, cmp.current_sequence, cmp.current_mode);
    }
}

void Maxwell3D::ProcessCounterReset() {
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    if (GetNativeMacroEngine().native_turing) {
        LOG_DEBUG(HW_GPU, "Native 3D syncpoint increment: id={}, raw={:#x}", regs.sync_info.sync_point.Value(), regs.reg_array[MAXWELL3D_REG_INDEX(sync_info)]);
    }
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    static std::array<std::array<bool, 32>, 5> seen{};
    if (!seen[stage_index][bind_data.shader_slot]) {
        seen[stage_index][bind_data.shader_slot] = true;
        LOG_INFO(HW_GPU, "Constant buffer bind: stage {} slot {} address {:#x} size {:#x}", stage_index,
                 bind_data.shader_slot.Value(), regs.const_buffer.Address(), regs.const_buffer.size);
    }
    if (bind_data.shader_slot >= Regs::MaxConstBuffers) {
        return; // slot field is 5 bits; the state array holds 18
    }
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    buffer.enabled = bind_data.valid.Value() != 0;
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    const size_t copy_size = amount * sizeof(u32);
    const size_t room = regs.const_buffer.offset < regs.const_buffer.size
                            ? regs.const_buffer.size - regs.const_buffer.offset
                            : 0;
    const size_t write_size = std::min(copy_size, room / sizeof(u32) * sizeof(u32));
    if (write_size != 0) {
        const GPUVAddr address{buffer_address + regs.const_buffer.offset};
        memory_manager.WriteBlockCached(address, start_base, write_size);
    }
    if (write_size != copy_size) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            LOG_ERROR(HW_GPU,
                      "Dropping constant buffer words past the end: offset={:#x}, size={:#x}, "
                      "bytes={:#x}, written={:#x}",
                      regs.const_buffer.offset, regs.const_buffer.size, copy_size, write_size);
        }
    }

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2). The front keeps
// what macros read: the registers as the back thread will hold them when it reaches the same
// point of the method stream. That is every write the front sends, with the shadow RAM applied,
// plus the few changes the back makes on its own (constant buffer offset, shadow_scratch[0] on
// falcon[4], the macro bind pointer); the front loads and binds the macro code itself. Macros
// that read memory run on the back instead (FrontRunOnBack), and the registers they can write
// are the back's until the front writes them or reads them back (FrontRegister).
struct Maxwell3D::FrontState {
    std::array<ControlWrite, 16> control_writes{};
    u32 control_count{};
    std::array<u32, Regs::NUM_REGS> regs{};
    std::array<u32, Regs::NUM_REGS> shadow{};
    Regs::ShadowRamControl control{};
    u32 executing_macro{};
    std::vector<u32> parameters;
    u64 calls{};
    /// A method the draw manager handles was sent since the back last ran DrawDeferred: only
    /// then can it have a deferred instanced draw to issue at the end of a macro.
    bool draw_pending{true};
    bool skip_same{}; ///< SkipSameValueWrites() && FrontSkipSame(), read once
    /// vertex_buffer.first (bit 0) and vertex_buffer.count (bit 1) came from GPU-written words
    /// the front didn't read (FrontDrawFromGpuArguments): regs holds older values until each is
    /// written again, so a write of the same value isn't dropped.
    u32 vertex_arguments_unknown{};
    /// Macro DMA read snapshots (FrontDmaRead): device pages a drained read flushed whole, and the
    /// split position right after. Cleared past 8192 pages.
    std::unordered_map<u64, u64> dma_snapshots;
    /// Registers a macro the back ran may have written (FrontRunOnBack), and their shadow RAM
    /// entries: the ring position after the last such macro, 0 once the front knows the value.
    std::array<u64, Regs::NUM_REGS> back_regs{};
    std::array<u64, Regs::NUM_REGS> back_shadow{};
    std::bitset<Regs::NUM_REGS> back_owned; ///< back_regs or back_shadow set
    struct BackMacro {
        u32 start{};
        bool back{};                           ///< Runs on the back (else on the front)
        std::vector<u32> writes;               ///< The registers its code can write
        std::bitset<Regs::NUM_REGS> write_set; ///< The same, for the back's check
        std::atomic<bool> escaped{};           ///< The back saw it write another register
    };
    /// By start in the macro code of back_generation, once called.
    std::vector<std::unique_ptr<BackMacro>> back_macros;
    u64 back_generation{~u64{0}};
    std::vector<u32> back_demoted; ///< Starts of macros that wrote outside their write set
    /// The back's macro engine, its code a copy of the front's (when back_code_copied).
    std::unique_ptr<MacroEngine> back_engine;
    bool back_code_copied{};
    u64 back_last{};                 ///< The position after the last macro sent to the back
    std::atomic<u32> back_escapes{}; ///< Back macros that wrote outside their write set
    u32 back_escapes_seen{};
};

namespace {

// Notes the guest memory the back writes for a report or an inline upload sent from the front,
// for its macro DMA read snapshots (FrontDmaRead). `value` is the method's argument.
void NoteFrontMemoryWrite(VideoCommon::GpuSplit& split, MemoryManager& memory,
                          const std::array<u32, Maxwell3D::Regs::NUM_REGS>& regs, u32 method,
                          u32 value) {
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        constexpr u32 base = MAXWELL3D_REG_INDEX(report_semaphore);
        const GPUVAddr address = (GPUVAddr{regs[base]} << 32) | regs[base + 1];
        if (const auto device = memory.GpuToCpuAddress(address)) {
            split.NoteMemoryWrite(*device, 16);
        }
        return;
    }
    if (method != MAXWELL3D_REG_INDEX(launch_dma)) {
        return;
    }
    constexpr u32 base = MAXWELL3D_REG_INDEX(upload);
    const u64 line_length = regs[base];
    const u64 lines = (std::max)(regs[base + 1], 1U);
    const u64 pitch = (std::max<u64>)(regs[base + 4], line_length);
    const GPUVAddr address = (GPUVAddr{regs[base + 2]} << 32) | regs[base + 3];
    const u64 size = pitch * (lines - 1) + line_length;
    const auto device = memory.GpuToCpuAddress(address);
    // LAUNCH_DMA: bit 0 pitch (linear) layout; completion 2 also releases a semaphore.
    const bool linear = (value & 1) != 0;
    if (!linear || ((value >> 4) & 3) == 2 || !device || size == 0 || size > (u64{64} << 20) ||
        memory.GpuToCpuAddress(address + size - 1) != *device + size - 1) {
        split.NoteUnrangedWriter();
        return;
    }
    split.NoteMemoryWrite(*device, size);
}

// EDEN_NX2_SPLIT_VERIFY=1: every 4096 front macro calls, wait for the back and compare the
// front's registers with the back's (logs and resynchronizes differences).
bool FrontVerifyEnabled() {
    static const bool verify = std::getenv("EDEN_NX2_SPLIT_VERIFY") != nullptr;
    return verify;
}

// Executable methods ProcessMethodCall handles itself; the rest go to the draw manager.
bool FrontHandledOutsideDrawManager(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return true;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

// EDEN_NX2_SPLIT_SKIP=0: send state writes that keep the register's value too.
bool FrontSkipSame() {
    static const bool skip = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT_SKIP");
        return value == nullptr || value[0] != '0';
    }();
    return skip;
}

// Whether a macro the split's back runs may write `method` (FrontRunOnBack): the front then only
// loses track of the register's value. Not the methods with effects the front tracks (macro code
// and bindings, the shadow RAM mode, draws, constant buffer data, reports, uploads, syncpoints),
// nor the registers it reads to note their memory writes, nor registers the back changes itself,
// nor MME_DMA_READ, whose words land in the front's data RAM.
bool BackWritable(const Maxwell3D& maxwell3d, u32 method) {
    constexpr u32 const_buffer = MAXWELL3D_REG_INDEX(const_buffer);
    constexpr u32 report = MAXWELL3D_REG_INDEX(report_semaphore);
    constexpr u32 upload = MAXWELL3D_REG_INDEX(upload);
    constexpr u32 upload_end = MAXWELL3D_REG_INDEX(launch_dma);
    if (method - const_buffer < 4 || method - report < 4 || method - upload < upload_end - upload) {
        return false;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(load_mme.start_address_ptr): // Advanced by load_mme.start_address
    case MAXWELL3D_REG_INDEX(shadow_scratch):             // Set by falcon[4]
    case MAXWELL3D_REG_INDEX(logic_op.enable):            // Cleared by draws on AMD
    case 0x157:
        return false;
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true; // Executable, with effects on the back only
    default:
        return method < MacroRegistersStart && !maxwell3d.execution_mask[method];
    }
}

} // Anonymous namespace

Maxwell3D::FrontState& Maxwell3D::Front(Core::System& system, VideoCommon::GpuSplit& split) {
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle (state writes it
        // still holds in its method sink first).
        split.Drain(VideoCommon::DrainSite::FrontStart);
        ConsumeSink(system);
        front = std::make_unique<FrontState>();
        std::memcpy(front->regs.data(), regs.reg_array.data(), sizeof(front->regs));
        std::memcpy(front->shadow.data(), shadow_state.reg_array.data(), sizeof(front->shadow));
        front->control = shadow_state.shadow_ram_control;
        front->skip_same = SkipSameValueWrites() && FrontSkipSame();
        front->executing_macro = executing_macro;
        front->parameters = std::move(macro_params);
        macro_params.clear();
        executing_macro = 0;
        front_macros = true;
    }
    return *front;
}

void Maxwell3D::FrontWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                           u32 argument, bool is_last_call, bool from_macro) {
    namespace Profile = VideoCommon::Nx2Profile;
    // As ProcessShadowRam does on the back. A register a macro the back ran may have written is
    // the front's again, and its write is sent even if the front's copies hold the value.
    u32 value = argument;
    bool shadow_kept = !(fs.back_owned[method] && FrontTakeOver(split, fs, method));
    if (fs.control == Regs::ShadowRamControl::Track ||
        fs.control == Regs::ShadowRamControl::TrackWithFilter) {
        shadow_kept = shadow_kept && fs.shadow[method] == argument;
        fs.shadow[method] = argument;
    } else if (fs.control == Regs::ShadowRamControl::Replay) {
        value = fs.shadow[method];
    }
    if (fs.vertex_arguments_unknown != 0) [[unlikely]] {
        const u32 bit = method == MAXWELL3D_REG_INDEX(vertex_buffer.first)   ? 1U
                        : method == MAXWELL3D_REG_INDEX(vertex_buffer.count) ? 2U
                                                                             : 0U;
        if ((fs.vertex_arguments_unknown & bit) != 0) {
            fs.vertex_arguments_unknown &= ~bit;
            shadow_kept = false; // Sent even if regs holds the value: the back's may differ.
        }
    }
    if (Profile::Enabled()) [[unlikely]] {
        Profile::Add(Profile::Counter::MethodWrites, 1);
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
        fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
            argument, static_cast<u32>(fs.control), from_macro ? 2U : 1U};
    }
    if (!execution_mask[method]) {
        // Nothing would change on the back: CallMethod and ConsumeSinkImpl skip same-value state
        // writes, and the shadow RAM (tracked by default) holds the value already.
        if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            return;
        }
        fs.regs[method] = value;
    } else {
        fs.regs[method] = value;
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        switch (method) {
        case MAXWELL3D_REG_INDEX(shadow_ram_control):
            // The back's mode is shadow_state.shadow_ram_control: the shadow copy of this very
            // register, which a later Replay write reads back.
            fs.control = static_cast<Regs::ShadowRamControl>(argument);
            fs.shadow[method] = argument;
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
            GetNativeMacroEngine().ClearCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)]);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction):
            GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.start_address): {
            const auto positions = GetNativeMacroPositions();
            u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
            if (pointer < positions.size()) {
                positions[pointer] = value;
            }
            ++pointer;
            break;
        }
        case MAXWELL3D_REG_INDEX(falcon[4]):
            fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
            break;
        default:
            if (method >= cb_data && method < cb_data + 16) {
                fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += sizeof(u32);
            } else if (!FrontHandledOutsideDrawManager(method)) {
                fs.draw_pending = true;
            }
            break;
        }
    }
    if (from_macro) {
        split.MacroWrite(this, method, argument);
    } else {
        split.Method(this, method, argument, is_last_call, nullptr);
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) ||
        method == MAXWELL3D_REG_INDEX(launch_dma)) {
        NoteFrontMemoryWrite(split, memory_manager, fs.regs, method, value);
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        FrontNoteReport(split, fs);
    }
}

void Maxwell3D::FrontNoteReport(VideoCommon::GpuSplit& split, const FrontState& fs) {
    // What ProcessQueryGet will write on the back: a payload report writes the payload, a counter
    // what the GPU counted, a reduction what it makes of the word (port of Lichii #805 fb4d4e47).
    Regs::ReportSemaphore report;
    std::memcpy(&report, &fs.regs[MAXWELL3D_REG_INDEX(report_semaphore)], sizeof(report));
    const auto operation = report.query.operation.Value();
    if (operation != Regs::ReportSemaphore::Operation::Release &&
        operation != Regs::ReportSemaphore::Operation::ReportOnly) {
        return;
    }
    const auto device = memory_manager.GpuToCpuAddress(report.Address());
    if (!device) {
        return;
    }
    const bool payload = report.query.report == Regs::ReportSemaphore::Report::Payload &&
                         report.query.reduction_enable == 0;
    split.NoteSemaphoreWrite(*device, report.payload, payload);
    if (report.query.short_query == 0) {
        // 64 bits (a timestamp follows).
        split.NoteSemaphoreWrite(*device + 4, 0, payload);
    }
}

void Maxwell3D::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                u32 argument, bool is_last_call) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    FrontWrite(split, fs, method, argument, is_last_call, false);
}

void Maxwell3D::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, const u32* base_start, u32 amount,
                                     u32 methods_pending) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.insert(fs.parameters.end(), base_start, base_start + amount);
        if (amount == methods_pending) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    // What CallMultiMethod does to the registers on the back, then one entry for all the words
    // (sent whatever the front's copies hold).
    if (fs.back_owned[method]) {
        FrontTakeOver(split, fs, method);
    }
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto tracked = [&fs] {
        return fs.control == Regs::ShadowRamControl::Track ||
               fs.control == Regs::ShadowRamControl::TrackWithFilter;
    };
    if (method >= cb_data && method < cb_data + 16) {
        fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += amount * sizeof(u32);
    } else if (method == MAXWELL3D_REG_INDEX(inline_data)) {
        // Upload data only.
    } else if ((method == MAXWELL3D_REG_INDEX(draw_inline_index) ||
                method == MAXWELL3D_REG_INDEX(inline_index_2x16.even) ||
                method == MAXWELL3D_REG_INDEX(inline_index_4x8.index0)) &&
               fs.control != Regs::ShadowRamControl::Replay) {
        if (amount != 0) {
            const u32 last = base_start[amount - 1];
            if (tracked()) {
                fs.shadow[method] = last;
            }
            fs.regs[method] = last;
        }
        fs.draw_pending = true;
    } else {
        // CallMethod for each word, macro uploads among them (load_mme.instruction). A word may
        // change the shadow RAM mode for the next (a run of SET_SHADOW_RAM_CONTROL writes).
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = base_start[i];
            u32 value = argument;
            if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
                fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
                    argument, static_cast<u32>(fs.control), 3U};
            }
            if (tracked()) {
                fs.shadow[method] = argument;
            } else if (fs.control == Regs::ShadowRamControl::Replay) {
                value = fs.shadow[method];
            }
            fs.regs[method] = value;
            if (!execution_mask[method]) {
                continue;
            }
            switch (method) {
            case MAXWELL3D_REG_INDEX(shadow_ram_control):
                fs.control = static_cast<Regs::ShadowRamControl>(argument);
                fs.shadow[method] = argument; // The same word on the back (see FrontWrite)
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
                GetNativeMacroEngine().ClearCode(value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction):
                GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.start_address): {
                const auto positions = GetNativeMacroPositions();
                u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
                if (pointer < positions.size()) {
                    positions[pointer] = value;
                }
                ++pointer;
                break;
            }
            case MAXWELL3D_REG_INDEX(falcon[4]):
                fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
                break;
            default:
                if (!FrontHandledOutsideDrawManager(method)) {
                    fs.draw_pending = true;
                }
                break;
            }
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr);
    if (amount != 0 && (method == MAXWELL3D_REG_INDEX(report_semaphore.query) ||
                        method == MAXWELL3D_REG_INDEX(launch_dma))) {
        NoteFrontMemoryWrite(split, memory_manager, fs.regs, method, fs.regs[method]);
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) && amount != 0) {
        FrontNoteReport(split, fs);
    }
}

namespace {

// GPU-written list words the channel's compute macros left in its data RAM (Nx2ListIndirect):
// the front runs 3D macros on the words themselves, so a macro at `start` that may read one
// gets them first, read back from their slots.
void ConcretizeListWords(VideoCommon::GpuSplit& split, VideoCore::RasterizerInterface* rasterizer,
                         Nx2ListIndirect::DataRamWords& channel, std::span<u32> data_ram,
                         const MacroEngine& engine, u32 start) {
    const auto& reads =
        channel.reads_3d.Get(engine.native_code, engine.native_code_generation, start);
    if (!Nx2ListIndirect::MayRead(reads.data_ram_known, reads.data_ram, channel.words)) {
        return;
    }
    const auto release = [&](u32 slot) { Nx2ListIndirect::ReleaseSlot(split, rasterizer, slot); };
    Nx2ListIndirect::Prune(channel.words, [data_ram](u32 word) { return data_ram[word]; }, release);
    if (!Nx2ListIndirect::MayRead(reads.data_ram_known, reads.data_ram, channel.words)) {
        return;
    }
    static const bool stats = std::getenv("EDEN_NX2_LIST_WAIT_STATS") != nullptr;
    if (stats) {
        static u64 count{};
        if ((++count & 255) == 1) {
            LOG_INFO(HW_GPU, "List indirect: 3D macro at {:#x} reads list words ({} so far)", start,
                     count);
        }
    }
    Nx2ListIndirect::ConcretizeDataRam(
        channel.words, data_ram,
        [&](std::span<const u32> slots, std::span<u32> words) {
            Nx2ListIndirect::ReadSlots(split, rasterizer, slots, words);
        },
        release);
}

} // Anonymous namespace

void Maxwell3D::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                              FrontState& fs) {
    // EDEN_NX2_FRONT_DELAY_NS (experiment): spin this long per front macro call, to measure how
    // the frame rate follows the front's macro cost (what a faster interpreter or a JIT could win).
    static const s64 front_delay_ns = [] {
        const char* value = std::getenv("EDEN_NX2_FRONT_DELAY_NS");
        return value != nullptr ? std::strtoll(value, nullptr, 10) : 0;
    }();
    if (front_delay_ns > 0) [[unlikely]] {
        const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds{front_delay_ns};
        while (std::chrono::steady_clock::now() < until) {
        }
    }
    const u32 method = fs.executing_macro;
    fs.executing_macro = 0;
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    VideoCommon::Nx2Profile::CountMethod(method, false);
    if (!FrontRunOnBack(system, split, fs, engine, start)) {
        VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
        if (auto* channel = Nx2ListIndirect::FindChannel(&native_data_ram);
            channel != nullptr && !channel->words.empty()) {
            ConcretizeListWords(split, rasterizer, *channel, native_data_ram, engine, start);
        }
        engine.native_interpreter.active_dram = &native_data_ram;
        const auto parameter_count = fs.parameters.size();
        const auto read = [&](u32 address) {
            if (address >= Regs::NUM_REGS) {
                return 0U;
            }
            return fs.back_owned[address] ? FrontRegister(split, fs, address) : fs.regs[address];
        };
        const auto emit = [&](u32 address, u32 value) {
            if (address >= MacroRegistersStart) {
                return false;
            }
            FrontWrite(split, fs, address, value, true, true);
            // MME_DMA_READ (into data RAM) and MME_DMA_READ_FIFOED (into the load FIFO).
            if (address == 0x157 || address == 0x158) {
                FrontDmaRead(split, fs, method, address, value);
            }
            return true;
        };
#ifdef ARCHITECTURE_x86_64
        if (!engine.native_jit) {
            engine.native_jit = std::make_shared<TuringMacroJit>();
        }
        engine.native_jit->Prepare(engine.native_code, engine.native_code_generation);
        const auto result = engine.native_jit->Execute(
            start, fs.parameters, std::span<u32>{native_data_ram}, read, emit);
#else
        const auto result =
            engine.native_interpreter.Execute(engine.native_code, start, fs.parameters, read, emit);
#endif
        if (!result.success) {
            engine.native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Native MME (split front) failed: start={:#x}, consumed={}/{}, error={}",
                      start, result.consumed, parameter_count, result.error);
        }
    }
    fs.parameters.clear();
    if (fs.draw_pending) {
        fs.draw_pending = false;
        split.Defer([this, system_ptr = &system] { FinishFrontMacro(*system_ptr); });
    }
    if (FrontVerifyEnabled() && (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}

// A macro whose code reads memory into its FIFO (MME_DMA_READ_FIFOED) and writes only registers
// BackWritable allows runs on the back, at its point of the method stream: it reads the words as
// a drained front would, and the front goes on without waiting for the work queued before it.
// The registers it can write (and, in a tracked shadow RAM mode, their entries) are the back's
// until the front writes them (FrontTakeOver) or needs them (FrontRegister). Returns whether the
// call went to the back.
bool Maxwell3D::FrontRunOnBack(Core::System& system, VideoCommon::GpuSplit& split, FrontState& fs,
                               const MacroEngine& engine, u32 start) {
    using DrainSite = VideoCommon::DrainSite;
    using BackMacro = FrontState::BackMacro;
    if (fs.parameters.size() > 0x1000) {
        return false;
    }
    // A macro the back ran wrote another register: the front's copies may be stale anywhere. It
    // runs on the front from now on, and the front takes the back's registers (FrontVerify).
    const auto take_escapes = [&] {
        fs.back_escapes_seen = fs.back_escapes.load(std::memory_order_acquire);
        for (const auto& macro : fs.back_macros) {
            if (macro && macro->back && macro->escaped.load(std::memory_order_relaxed)) {
                macro->back = false;
                fs.back_demoted.push_back(macro->start);
                LOG_ERROR(HW_GPU,
                          "Split back macro at {:#x} wrote outside the registers its code "
                          "can write; it runs on the front from now on",
                          macro->start);
            }
        }
        FrontVerify(split, fs);
    };
    if (fs.back_generation != engine.native_code_generation) {
        // The macros sent before run the code they were sent with, and check their records.
        split.WaitUntil(fs.back_last, DrainSite::BackCode);
        if (fs.back_escapes.load(std::memory_order_relaxed) != fs.back_escapes_seen) {
            take_escapes();
        }
        fs.back_macros.clear();
        fs.back_macros.resize(engine.native_code.size() / 3);
        fs.back_generation = engine.native_code_generation;
        fs.back_code_copied = false;
    } else if (fs.back_escapes.load(std::memory_order_relaxed) != fs.back_escapes_seen) {
        take_escapes();
    }
    if (start >= fs.back_macros.size()) {
        return false;
    }
    auto& known = fs.back_macros[start];
    if (!known) {
        known = std::make_unique<BackMacro>();
        known->start = start;
        const auto effects = TuringMacroInterpreter::AnalyzeEffects(engine.native_code, start);
        const auto& methods = effects.methods;
        const bool fifo_read = std::ranges::binary_search(methods, 0x158U);
        if (fifo_read || std::ranges::binary_search(methods, 0x157U)) {
            std::string reason;
            if (std::ranges::find(fs.back_demoted, start) != fs.back_demoted.end()) {
                reason = "it wrote another register on the back before";
            } else if (!effects.methods_known) {
                reason = "it computes the method of a write";
            } else if (effects.data_ram) {
                reason = "it uses the data RAM";
            } else if (const auto it = std::ranges::find_if(
                           methods, [this](u32 m) { return !BackWritable(*this, m); });
                       it != methods.end()) {
                reason = fmt::format("it writes {:#x}", *it);
            }
            known->back = reason.empty() && fifo_read;
            if (known->back) {
                known->writes = methods;
                for (const u32 written : methods) {
                    known->write_set.set(written);
                }
            }
            static u32 logged{};
            if (logged < 32) {
                ++logged;
                LOG_INFO(HW_GPU, "Split macro at {:#x} reads memory and runs on the {}: {}", start,
                         known->back ? "back" : "front",
                         known->back ? fmt::format("writes {:#x}", fmt::join(methods, ","))
                                     : reason);
            }
        }
    }
    if (!known->back) {
        return false;
    }
    if (!fs.back_engine) {
        fs.back_engine = std::make_unique<MacroEngine>(true);
        fs.back_engine->native_turing = true;
    }
    MacroEngine& back = *fs.back_engine;
    if (!fs.back_code_copied) {
        // Nothing sent with the old code is left to run (waited for above).
        back.native_code = engine.native_code;
        back.native_code_generation = engine.native_code_generation;
        fs.back_code_copied = true;
    }
    split.DeferWords(
        [this, system_ptr = &system, engine_ptr = &back, macro = known.get(),
         escapes = &fs.back_escapes](std::span<const u32> parameters) {
            // As ProcessMacro calls a macro, with the back's engine.
            ConsumeSink(*system_ptr);
            engine_ptr->split_writes = &macro->write_set;
            const u64 unexpected = engine_ptr->split_unexpected_writes;
            engine_ptr->Execute(*system_ptr, *this, macro->start, parameters);
            if (engine_ptr->split_unexpected_writes != unexpected) {
                macro->escaped.store(true, std::memory_order_relaxed);
                escapes->fetch_add(1, std::memory_order_release);
            }
        },
        fs.parameters);
    const u64 position = split.Position();
    fs.back_last = position;
    const bool tracked = fs.control == Regs::ShadowRamControl::Track ||
                         fs.control == Regs::ShadowRamControl::TrackWithFilter;
    for (const u32 written : known->writes) {
        fs.back_regs[written] = position;
        if (tracked) {
            fs.back_shadow[written] = position;
        }
        fs.back_owned[written] = true;
    }
    return true;
}

// A register as the back holds it once it ran what the front sent: the front's copy, or the
// back's once it ran the last macro there that can write it. No entry sent after that macro
// writes the register (a front write takes it over first, a later macro there would be the one
// waited for), so the back's value stays until the front writes the register again.
u32 Maxwell3D::FrontRegister(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    if (fs.back_regs[method] != 0) {
        split.WaitUntil(fs.back_regs[method], VideoCommon::DrainSite::BackRegister);
        fs.regs[method] = regs.reg_array[method];
        fs.back_regs[method] = 0;
        fs.back_owned[method] = fs.back_shadow[method] != 0;
    }
    return fs.regs[method];
}

// Before the front writes `method`: the register, and its shadow RAM entry unless the mode
// leaves that alone, are the front's again; a replay takes the entry from the back first (as
// FrontRegister). Returns whether the back may hold another value or entry than the front's
// copies, so that the write must be sent even if it seems to change nothing.
bool Maxwell3D::FrontTakeOver(VideoCommon::GpuSplit& split, FrontState& fs, u32 method) {
    using Control = Regs::ShadowRamControl;
    const bool tracked = fs.control == Control::Track || fs.control == Control::TrackWithFilter;
    if (fs.back_shadow[method] != 0 && fs.control == Control::Replay) {
        split.WaitUntil(fs.back_shadow[method], VideoCommon::DrainSite::BackRegister);
        fs.shadow[method] = shadow_state.reg_array[method];
        fs.back_shadow[method] = 0;
    }
    const bool differs = fs.back_regs[method] != 0 || (tracked && fs.back_shadow[method] != 0);
    fs.back_regs[method] = 0;
    if (tracked) {
        fs.back_shadow[method] = 0;
    }
    fs.back_owned[method] = fs.back_shadow[method] != 0;
    return differs;
}

std::optional<Maxwell3D::GpuArgumentDraw> Maxwell3D::FrontGpuArgumentDraw(u32 start) {
    auto& engine = GetNativeMacroEngine();
    if (front_gpu_argument_generation != engine.native_code_generation) {
        front_gpu_argument_generation = engine.native_code_generation;
        front_gpu_argument_draws.clear();
    }
    if (const auto it = front_gpu_argument_draws.find(start); it != front_gpu_argument_draws.end()) {
        return it->second;
    }
    // Run the macro on two pairs of marker parameters in a sandbox (its own data RAM, register
    // reads answered with 0). It qualifies only if both runs consume exactly the two parameters
    // and emit exactly draw.begin, vertex_buffer.first = the second, vertex_buffer.count = the
    // first and draw.end, with the same constant begin and end values, reading nothing else.
    const auto run = [&](u32 count, u32 first) -> std::optional<GpuArgumentDraw> {
        TuringMacroInterpreter sandbox;
        sandbox.dram = native_data_ram;
        sandbox.active_dram = &sandbox.dram;
        std::vector<u32> parameters{count, first};
        std::vector<std::pair<u32, u32>> writes;
        bool read_register = false;
        const auto result = sandbox.Execute(
            engine.native_code, start, parameters,
            [&read_register](u32) {
                read_register = true;
                return 0U;
            },
            [&writes](u32 address, u32 value) {
                writes.emplace_back(address, value);
                return writes.size() <= 8;
            });
        if (!result.success || result.consumed != 2 || read_register || sandbox.read_registers ||
            !sandbox.data_ram_reads.empty() || !sandbox.data_ram_writes.empty() ||
            writes.size() != 4 || writes[0].first != MAXWELL3D_REG_INDEX(draw.begin) ||
            writes[1] != std::pair{u32{MAXWELL3D_REG_INDEX(vertex_buffer.first)}, first} ||
            writes[2] != std::pair{u32{MAXWELL3D_REG_INDEX(vertex_buffer.count)}, count} ||
            writes[3].first != MAXWELL3D_REG_INDEX(draw.end)) {
            return std::nullopt;
        }
        return GpuArgumentDraw{.begin = writes[0].second, .end = writes[3].second};
    };
    std::optional<GpuArgumentDraw> draw = run(0x13579, 0x2468a);
    if (const auto again = run(0x5a5a5, 0xa5a5a);
        !draw || !again || draw->begin != again->begin || draw->end != again->end) {
        draw.reset();
    }
    if (draw) {
        LOG_INFO(HW_GPU, "Native macro at {:#x} draws with its two parameters as vertex count "
                         "and first vertex: GPU-written ones stay in GPU memory",
                 start);
    }
    front_gpu_argument_draws.emplace(start, draw);
    return draw;
}

bool Maxwell3D::FrontDrawFromGpuArguments(Core::System& system, VideoCommon::GpuSplit& split,
                                          u32 method, GPUVAddr arguments, u32 words) {
    // Port of Lichii #980: in Ocarina of Time a culling dispatch writes the task group count of
    // each mesh draw into a command list entry that calls such a macro, which the front had to
    // wait for the dispatch to read. EDEN_NX2_GPU_ARGUMENT_DRAWS=0 reads them as before.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_ARGUMENT_DRAWS");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled || words != 2 || method < MacroRegistersStart || (method & 1) != 0) {
        return false;
    }
    auto& fs = Front(system, split);
    // In Replay mode the macro's writes would take the shadow RAM's values instead.
    if (fs.executing_macro != 0 || !fs.parameters.empty() ||
        fs.control == Regs::ShadowRamControl::Replay) {
        return false;
    }
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    const auto draw = FrontGpuArgumentDraw(start);
    if (!draw) {
        return false;
    }
    // Both words on one device page, contiguous.
    const auto device = memory_manager.GpuToCpuAddress(arguments);
    if (!device || (*device & 3) != 0 ||
        memory_manager.GpuToCpuAddress(arguments + sizeof(u32)) != *device + sizeof(u32)) {
        return false;
    }
    // What the macro does, but the two registers' values stay with the back (and in GPU
    // memory): draw.begin as the macro's write, then the rest on the back.
    VideoCommon::Nx2Profile::CountMethod(method, false);
    FrontWrite(split, fs, MAXWELL3D_REG_INDEX(draw.begin), draw->begin, false, true);
    // draw.end as FrontWrite would note it (the back runs it in DrawFromGpuArguments).
    constexpr u32 end_method = MAXWELL3D_REG_INDEX(draw.end);
    if (fs.control == Regs::ShadowRamControl::Track ||
        fs.control == Regs::ShadowRamControl::TrackWithFilter) {
        fs.shadow[end_method] = draw->end;
    }
    fs.regs[end_method] = draw->end;
    fs.vertex_arguments_unknown = 3;
    split.Defer([this, system_ptr = &system, arguments, device = *device, end = draw->end] {
        DrawFromGpuArguments(*system_ptr, arguments, device, end);
    });
    // The end of the macro, as FrontRunMacro ends it (its writes include draw manager methods).
    fs.draw_pending = false;
    split.Defer([this, system_ptr = &system] { FinishFrontMacro(*system_ptr); });
    return true;
}

void Maxwell3D::DrawFromGpuArguments(Core::System& system, GPUVAddr arguments, DAddr device,
                                     u32 end_value) {
    // As the macro's direct writes of vertex_buffer.first, vertex_buffer.count and draw.end,
    // with the two registers left pending (DrawManager::ResolveVertexArguments).
    ConsumeSink(system);
    draw_manager.pending_vertex_arguments =
        DrawManager::GpuVertexArguments{.address = arguments, .device = device, .unknown = 3};
    draw_manager.gpu_draw_arguments = draw_manager.pending_vertex_arguments;
    CallMethod(system, MAXWELL3D_REG_INDEX(draw.end), end_value, true);
    draw_manager.gpu_draw_arguments.reset();
}

void Maxwell3D::FrontDmaRead(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                             u32 method, u32 count) {
    // The words may be written by methods still queued for the back (indirect draw arguments
    // from a culling dispatch), and the caches that may hold newer ones belong to it.
    const GPUVAddr source = (GPUVAddr{FrontRegister(split, fs, 0x154) & 0x1ffffff} << 32) |
                            FrontRegister(split, fs, 0x155);
    const u32 ram = FrontRegister(split, fs, 0x156);
    const u32 words = std::min<u32>(count, 0x1000);
    const size_t read_bytes = size_t{words} * sizeof(u32);
    std::vector<u32> diag_before;
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        diag_before.resize(words);
        memory_manager.ReadBlockUnsafe(source, diag_before.data(), read_bytes);
    }
    VideoCommon::Nx2Profile::CountDmaRead(macro);
    // Snapshots (EvilOOT's dma-snapshot fix, as ported by Frizti): Ocarina of Time reads an
    // occlusion report for each object (macro 0xe2c, tens of thousands of times a second) from
    // GPU-written pages, and each read drained the back, so the front and the back took turns.
    // A drained read flushes its whole device pages and remembers the split position; later
    // reads of those pages skip the drain while nothing sent since may write them
    // (GpuSplit::MayBeWrittenSince) and the query cache lists no report there.
    // EDEN_NX2_DMA_SNAPSHOT=0 drains every read.
    static const bool snapshots = [] {
        const char* value = std::getenv("EDEN_NX2_DMA_SNAPSHOT");
        return value == nullptr || value[0] != '0';
    }();
    if (snapshots && method == 0x158 && words != 0 && !Tegra::Nx2CmdPageDiag::g_enabled) {
        const auto device = memory_manager.GpuToCpuAddress(source);
        if (device &&
            memory_manager.GpuToCpuAddress(source + read_bytes - 1) == *device + read_bytes - 1) {
            const u64 first_page = *device >> 12;
            const u64 last_page = (*device + read_bytes - 1) >> 12;
            bool hit = true;
            for (u64 page = first_page; page <= last_page && hit; ++page) {
                const auto it = fs.dma_snapshots.find(page);
                hit = it != fs.dma_snapshots.end() &&
                      !split.MayBeWrittenSince(it->second, *device, read_bytes);
            }
            // A report's value is up to 8 bytes, listed by its first 4: look 12 bytes back. A
            // listed report may not be in guest memory yet: take the drained path below.
            if (!hit || !rasterizer->HasCachedReports(*device - 12, u64{read_bytes} + 12)) {
                thread_local std::vector<u32> snap_words;
                snap_words.resize(words);
                if (!hit) {
                    // The GPU data of the whole pages (a flush of only the bytes read leaves stale
                    // neighbours for later hits), and the pending reports of the bytes read.
                    const GPUVAddr flush_begin = source & ~GPUVAddr{0xfff};
                    const GPUVAddr flush_end = (source + read_bytes + 0xfff) & ~GPUVAddr{0xfff};
                    const size_t flush_bytes = static_cast<size_t>(flush_end - flush_begin);
                    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
                    if (memory_manager.IsMemoryDirty(flush_begin, flush_bytes,
                                                     VideoCommon::CacheType::BufferCache)) {
                        split.Run(
                            [memory = &memory_manager, flush_begin, flush_bytes] {
                                memory->FlushRegion(flush_begin, flush_bytes,
                                                    VideoCommon::CacheType::BufferCache);
                            },
                            VideoCommon::DrainSite::MacroDmaFlush);
                    }
                    split.Run(
                        [memory = &memory_manager, source, read_bytes] {
                            memory->FlushRegion(source, read_bytes,
                                                VideoCommon::CacheType::QueryCache);
                        },
                        VideoCommon::DrainSite::MacroDmaFlush);
                    if (fs.dma_snapshots.size() > 8192) {
                        fs.dma_snapshots.clear();
                    }
                    for (u64 page = first_page; page <= last_page; ++page) {
                        fs.dma_snapshots[page] = split.SentPosition();
                    }
                }
                memory_manager.ReadBlockUnsafe(source, snap_words.data(), read_bytes);
                fs.parameters.insert(fs.parameters.end(), snap_words.begin(), snap_words.end());
                return;
            }
        }
    }
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    if (VideoCommon::QueryDma::TraceEnabled()) {
        split.Run(
            [this, macro, method, source, read_bytes] {
                if (VideoCommon::QueryDma::HasPendingQueries(memory_manager, *rasterizer, source,
                                                             read_bytes)) {
                    VideoCommon::QueryDma::Trace(memory_manager, *rasterizer, macro, method,
                                                 source, read_bytes);
                }
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    }
    if (VideoCommon::Nx2Profile::Enabled()) {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DmaReadBytes, read_bytes);
    }
    if (memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache)) {
        split.Run(
            [memory = &memory_manager, source, read_bytes] {
                memory->FlushRegion(source, read_bytes, VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    }
    u32 changed{};
    for (u32 i = 0; i < words; ++i) {
        const u32 word = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
        if (method == 0x158) {
            fs.parameters.push_back(word);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = word;
        }
        if (!diag_before.empty() && diag_before[i] != word) {
            ++changed;
        }
    }
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        bool gpu_written = false;
        for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + read_bytes; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            gpu_written |= device && Tegra::Nx2IsGpuWritten(*device);
        }
        Tegra::Nx2CmdPageDiag::RecordDmaRead(source, words, changed, gpu_written);
    }
}

void Maxwell3D::FrontVerify(VideoCommon::GpuSplit& split, FrontState& fs) {
    split.Drain(VideoCommon::DrainSite::FrontVerify);
    static u64 checks{};
    static u64 differences{};
    ++checks;
    // State writes the back still holds in its method sink count as done.
    std::array<u32, Regs::NUM_REGS> expected;
    std::memcpy(expected.data(), regs.reg_array.data(), sizeof(expected));
    for (const auto& [method, value] : method_sink) {
        expected[method] = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay
                               ? shadow_state.reg_array[method]
                               : value;
    }
    // What macros the back ran may have written, the front takes over (FrontRunOnBack).
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.back_regs[i] != 0) {
            fs.regs[i] = expected[i];
        }
        if (fs.back_shadow[i] != 0) {
            fs.shadow[i] = shadow_state.reg_array[i];
        }
    }
    fs.back_regs.fill(0);
    fs.back_shadow.fill(0);
    fs.back_owned.reset();
    static u64 control_differences{};
    if ((fs.control != shadow_state.shadow_ram_control ||
         fs.shadow[MAXWELL3D_REG_INDEX(shadow_ram_control)] != shadow_state.reg_array[MAXWELL3D_REG_INDEX(shadow_ram_control)] ||
         fs.regs[MAXWELL3D_REG_INDEX(shadow_ram_control)] != expected[MAXWELL3D_REG_INDEX(shadow_ram_control)]) &&
        control_differences < 4) {
        LOG_WARNING(HW_GPU, "Split shadow RAM control writes, front ({} total) / back ({} total):",
                    fs.control_count, back_control_count);
        for (u32 i = 0; i < 16; ++i) {
            const auto& f = fs.control_writes[(fs.control_count + i) % 16];
            const auto& b = back_control_writes[(back_control_count + i) % 16];
            LOG_WARNING(HW_GPU, "  front {:#x} (mode {}, path {}) | back {:#x} (mode {}, last {})",
                        f.argument, f.before, f.path, b.argument, b.before, b.path);
        }
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.regs[i] != expected[i]) {
            if (differences++ < 64) {
                LOG_WARNING(HW_GPU, "Split front register {:#x}: front {:#x}, back {:#x}", i,
                            fs.regs[i], expected[i]);
            }
            fs.regs[i] = expected[i];
        }
    }
    if (fs.control != shadow_state.shadow_ram_control) {
        if (control_differences++ < 16) {
            LOG_WARNING(HW_GPU, "Split front shadow RAM control {} vs back {}",
                        static_cast<u32>(fs.control),
                        static_cast<u32>(shadow_state.shadow_ram_control));
        }
        fs.control = shadow_state.shadow_ram_control;
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.shadow[i] != shadow_state.reg_array[i]) {
            if (control_differences++ < 16) {
                LOG_WARNING(HW_GPU, "Split front shadow register {:#x}: front {:#x}, back {:#x}", i,
                            fs.shadow[i], shadow_state.reg_array[i]);
            }
            fs.shadow[i] = shadow_state.reg_array[i];
        }
    }
    if ((checks & 63) == 1) {
        LOG_INFO(HW_GPU, "Split front register checks: {} done, {} differences", checks, differences);
    }
}

void Maxwell3D::FinishFrontMacro(Core::System& system) {
    // What ProcessMacro and CallMacroMethod do around the macro.
    ConsumeSink(system);
    draw_manager.DrawDeferred(*this);
}

} // namespace Tegra::Engines

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "common/common_types.h"

namespace VideoCommon {

/// EDEN_NX2_TEXTURE_FILL=1 (off by default until measured): texture uploads fill their staging
/// memory on a thread of their own ("TexFill") instead of on the GPU back thread.
///
/// The back still decides everything an upload does and records its copy (or compute decode) in
/// order; only the CPU work that writes the staging memory (the unswizzle, or the copy of the
/// guest bytes an accelerated decode reads) moves here. The GPU reads that memory only once the
/// command buffer is submitted, so Scheduler::SubmitExecution waits, on the Vulkan worker thread,
/// for the fills queued before it (WaitDone): the back never waits for them.
class TextureFill {
public:
    [[nodiscard]] static bool Enabled() noexcept;
    [[nodiscard]] static TextureFill& Get();

    /// Queues `job`; returns its sequence number (1, 2, ...).
    u64 Push(std::function<void()>&& job);

    /// The sequence number of the last job queued (0: none yet).
    [[nodiscard]] u64 Queued() const noexcept {
        return queued.load(std::memory_order_acquire);
    }

    /// Blocks until every job up to `sequence` ran.
    void WaitDone(u64 sequence);

    TextureFill(const TextureFill&) = delete;
    TextureFill& operator=(const TextureFill&) = delete;

private:
    TextureFill();
    ~TextureFill();

    void Run(std::stop_token stop_token);

    std::mutex mutex;
    std::condition_variable_any work_cv;
    std::condition_variable done_cv;
    std::deque<std::function<void()>> jobs;
    std::atomic<u64> queued{};
    std::atomic<u64> done{};
    std::atomic<bool> waiting{};
    std::jthread thread;
};

} // namespace VideoCommon

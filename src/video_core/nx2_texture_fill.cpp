// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>

#include "common/logging.h"
#include "common/thread.h"
#include "video_core/nx2_texture_fill.h"

namespace VideoCommon {

bool TextureFill::Enabled() noexcept {
    // Read on first use (a texture upload), after main() applied nx2_switches.txt.
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_TEXTURE_FILL");
        const bool on = value == nullptr || value[0] != '0'; // Kokiri: on by default
        if (on) {
            LOG_INFO(HW_GPU, "Texture uploads fill their staging memory "
                             "on the TexFill thread");
        }
        return on;
    }();
    return enabled;
}

TextureFill& TextureFill::Get() {
    static TextureFill fill;
    return fill;
}

TextureFill::TextureFill()
    : thread{[this](std::stop_token stop_token) { Run(stop_token); }} {}

TextureFill::~TextureFill() {
    thread.request_stop();
    work_cv.notify_all();
}

u64 TextureFill::Push(std::function<void()>&& job) {
    u64 sequence;
    {
        std::scoped_lock lock{mutex};
        jobs.push_back(std::move(job));
        sequence = queued.load(std::memory_order_relaxed) + 1;
        queued.store(sequence, std::memory_order_release);
    }
    work_cv.notify_one();
    return sequence;
}

void TextureFill::WaitDone(u64 sequence) {
    if (done.load(std::memory_order_acquire) >= sequence) {
        return;
    }
    std::unique_lock lock{mutex};
    waiting.store(true, std::memory_order_relaxed);
    done_cv.wait(lock, [&] { return done.load(std::memory_order_acquire) >= sequence; });
    waiting.store(false, std::memory_order_relaxed);
}

void TextureFill::Run(std::stop_token stop_token) {
    Common::SetCurrentThreadName("TexFill");
    // Off the cores the GPU threads are pinned to (Linux pinning), like the background builds.
    if (const u64 mask = Common::BackgroundWorkCpuMask(); mask != 0) {
        Common::SetCurrentThreadCpuMask(mask);
    }
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock lock{mutex};
            work_cv.wait(lock, stop_token, [&] { return !jobs.empty(); });
            if (jobs.empty()) {
                return; // Stop requested
            }
            job = std::move(jobs.front());
            jobs.pop_front();
        }
        job();
        {
            // Under the lock, so a WaitDone between its check and its wait can't miss this.
            std::scoped_lock lock{mutex};
            done.fetch_add(1, std::memory_order_acq_rel);
        }
        done_cv.notify_all();
    }
}

} // namespace VideoCommon

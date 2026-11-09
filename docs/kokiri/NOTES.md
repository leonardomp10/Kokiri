# Kokiri: performance notes from the AYN Thor Pro

Measurements of The Legend of Zelda: Ocarina of Time (Switch 2) on an AYN Thor Pro (Snapdragon 8 Gen 2, Adreno 740 with Mesa Turnip, 12 GB RAM, Linux) at 0.5x resolution, GPU accuracy Low, anti-aliasing *No sharpening* with Max Resolution, NCE.

## Method

- **Gameplay benchmark:** a save in Kokiri Forest is loaded and Link walks around with the camera kept at the horizon and turned horizontally, for 10 minutes. Frame rate is counted per 10 s window from presented frames.
- **Warm shader cache:** windows where pipelines are being compiled are reported separately. The steady state after the cache is built is what matters for play.
- **Profiling:** per-thread profile zones (`EDEN_NX2_GPU_PROFILE=1`), `perf` from a container on the GPU threads, devfreq residency of the GPU, and per-thread page faults (`/proc`).

## History on this device

| Step | fps |
|---|---|
| Upstream Lichii, software HMMA at warp 32 not available on a7xx | ~1.5 |
| Wave64 software HMMA (one host wave = two guest warps) | 6.6 |
| Shared-B HMMA, lower settings | 7.2 |
| Bypass of the game's sharpening network + Max Resolution | 11.5–12 (intro, cold cache) |
| Same, gameplay in Kokiri with a warm shader cache | ~17 (12–22) |

## What limits the frame rate (Thor, old base)

- **Not the GPU:** the Adreno spends ~88% of the time at its lowest clock (348 MHz). Pinning it at 680 MHz changed nothing.
- **A serial chain of round trips between the split GPU thread's front, its back, and the host GPU.** Removing one wait moved the time to the next one: the front's blocked time stayed ~800 ms/s across five different fixes. `EDEN_NX2_SPLIT=0` (one thread) gave the same frame rate as the split. What helps is less work on that chain, not fewer waits.
- **Front CPU time:** the 3D macro interpreter is 26–39% of the front thread (no MME JIT on ARM).
- **Memory:** ~1,800 major page faults/s on the back thread from zram swap. This is why the memory savings matter on 12 GB devices.
- **Guest write faults on GPU-written pages:** compute shaders bind storage buffers several MB wide. A guest write anywhere in them used to force a download after a full host GPU wait, ~55 times a second. Only ~3% of the downloaded bytes differed from guest memory. Upstream now marks only as far as the shader's stores reach (#816).
- **Render pass breaks:** re-uploading 64 KiB native uniform banks on every bind ended ~6,700 render passes a second. That reupload is not in this base.

## Tried and not kept

- Merging guest write faults with a snapshot diff, without the download. It removed the waits, but whole frames flashed white: the game reads GPU-written words (eye adaptation) that then reach guest memory a fence late.
- Emulating each guest store to GPU pages in place (~44,000 signals a second), which was slower.
- Fence flush batching off, GPU clock pinned, uploads reordered per submission: no measurable change.

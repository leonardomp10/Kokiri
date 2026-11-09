<!--
# SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
-->

<p align="center">
  <img src="dist/branding/kokiri_icon_with_logo.png" alt="Kokiri" width="600">
</p>

# Kokiri

**Kokiri** is a Nintendo Switch / Switch 2 emulator fork for **ARM64 handhelds with limited memory and older mobile GPUs**: Linux gaming handhelds and similar devices built on Snapdragon SoCs with Adreno GPUs (Mesa Turnip), with 12 GB of RAM or less.

It builds on [Lichii for Android](https://github.com/pavelgajdo/Lichii-Android) by pavelgajdo, itself a fork of [Lichii](https://github.com/zortos293/Lichii) (Eglantine / Eden / yuzu lineage), and adds the changes those devices need. Like the Kokiri of the forest, it is meant for the small ones.

This is an independent, experimental fork. It is not affiliated with Nintendo or with the upstream projects it builds on. Games, keys and firmware are not included: only use games and system files you have legally obtained.

## Tested on

| | |
|---|---|
| Device | AYN Thor Pro (256 GB storage + 12 GB RAM) |
| SoC | Snapdragon 8 Gen 2 |
| GPU | Adreno 740 (a7xx), Mesa Turnip (Vulkan 1.4) |
| RAM | 12 GB (about 11 GiB usable), zram swap |
| OS | Linux (ArmadaOS, Fedora based), kernel 7.2 |
| CPU backend | NCE |

Other Adreno a6xx/a7xx devices running Linux with Turnip are the intended targets, but only the device above is tested so far.

## What Kokiri changes

On top of the latest Lichii for Android:

- **Wave64 software tensor math on Adreno a7xx.** Turnip on a7xx cannot run compute at the guest's 32-lane warp size; Kokiri requests 64-lane subgroups and lets one host wave carry two guest warps in the software HMMA emulation (matrix B broadcast per guest warp). On the Thor this took Ocarina of Time from ~1.5 fps to playable. Optional shared-B path: `EDEN_NX2_HMMA_SHARED_B=1`.
- **Cheaper software matrix B gathers** on guarded Turnip pipelines (from NickWalters' Turnip work).
- **Memory savings for 12 GB devices on Linux aarch64** (previously Android-only upstream): 128 MiB NUP decompression cache, and render targets drawn below 1x keep only their scaled image (~0.9 GB saved in Ocarina of Time at 0.5x).

Investigation notes from the Thor (what limits the frame rate, what was tried and measured) are kept in [docs/kokiri/NOTES.md](docs/kokiri/NOTES.md).

## Game status (AYN Thor Pro)

| Game | Status |
|---|---|
| The Legend of Zelda: Ocarina of Time (Switch 2) | Playable at 0.5x: about 17 fps average in Kokiri Forest gameplay (12–22 fps) with a warm shader cache. Movies and the intro run full speed. |

## Recommended settings (Ocarina of Time on Adreno)

- **Graphics:** resolution 0.5x, GPU accuracy Low, asynchronous shader building on, disk shader cache on.
- **Ocarina of Time anti-aliasing:** *No sharpening* (skips the game's tensor-core sharpening network), with the built-in *Max Resolution* patch enabled.
- **Environment** (launch script): `EDEN_NX2_HMMA_SHARED_B=1`, `EDEN_NX2_ASTC_MAX_DIM=1024`, and on 12 GB devices `EDEN_NX2_VRAM_LIMIT_MIB=3200`.
- Keep the shader cache between sessions: the first visit to an area compiles pipelines and stutters; after that the area runs at its normal frame rate.

## Building (Linux aarch64)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DYUZU_TESTS=OFF -DYUZU_CMD=OFF -DYUZU_ROOM=OFF -DYUZU_ROOM_STANDALONE=OFF \
  -DYUZU_USE_BUNDLED_QT=OFF -DYUZU_USE_BUNDLED_SDL3=ON -DENABLE_UPDATE_CHECKER=OFF \
  -DUSE_DISCORD_PRESENCE=OFF -DCMAKE_PREFIX_PATH=/path/to/Qt/6.8.3/gcc_arm64
cmake --build build --target yuzu --parallel 6
```

Builds were made in an Ubuntu 24.04 arm64 container with Qt 6.8.3 (aqtinstall). The binary is `build/bin/lichii`. On a fanless or thermally limited handheld, pause the build when the SoC gets hot.

The Android build from upstream still works as described in the [Lichii for Android README](https://github.com/pavelgajdo/Lichii-Android#building).

## Reporting problems

[Open an issue](https://github.com/leonardomp10/Kokiri/issues) with your device, OS, GPU driver (Mesa version), the game, what happened, and the log (`~/.local/share/eden/log/eden_log.txt` or your profile's `log` folder). Never upload games, keys, firmware or account data.

## Credits and license

Kokiri builds on Lichii for Android by pavelgajdo, Lichii by Zortos, the split GPU thread and Switch 2 work by jpkto and the Lichii contributors, NickWalters' Turnip software-matrix work, and Eglantine, Eden, yuzu and Sudachi and their contributors, including Eglantine's original developer Camille LaVey. Original copyright notices and third-party licenses are retained.

See [LICENSE.txt](LICENSE.txt) and the source files for the applicable license terms.

---

### Resumo em português

Kokiri é um fork do emulador de Switch / Switch 2 para portáteis ARM64 com pouca memória e GPUs Adreno mais antigas (Linux + Mesa Turnip). Testado em um AYN Thor Pro 256 GB + 12 GB (Snapdragon 8 Gen 2, Adreno 740). Traz HMMA por software em wave64 para Adreno a7xx, economia de memória para aparelhos de 12 GB e notas de investigação de desempenho. Ocarina of Time (Switch 2) é jogável a 0.5x, cerca de 17 fps de média em Kokiri com o cache de shaders já compilado.

# Hokkaido 0.6.1

The first release under the new name: Lichii for Android is now **Hokkaido - Switch 2 Emulator for Android**. Your games, saves and settings carry over (the app ID is unchanged).

Tested on a Samsung Galaxy S26 (Exynos 2600, Xclipse 960, 12 GB) and a Galaxy S24 (Exynos 2400, Xclipse 940, 8 GB), both on Android 17.

## Highlights

- **New look.** Hokkaido name, melon icon and colours, a new title on the games screen and orange progress bars.
- **Fast Fusion** shows its menu again instead of a black screen. An upstream texture-cache shortcut for render targets that a clear overwrites now runs only for Ocarina of Time, where it was measured. This may also fix black screens in other games.
- **Pokémon Pokopia**
  - No longer closed for low memory on 8 GB phones.
  - Runs smoother: it stopped copying every GPU-written buffer back at each frame, since only the parts the game reads are needed.
- **Loose Switch 2 game folders** (an `exefs` folder with `main`, like Pokopia's dumps): if a game folder holds one and the app lacks "All files access", Hokkaido now offers to open that permission, then lists the game.
- **Docked mode** reports a 1080p maximum output to games on Android, as handheld does.

## Known issues

- Ocarina of Time on a 12 GB phone can still be closed for low memory, most often on the first start while shaders compile. Use **0.5x** resolution.
- Pokopia's sky is black on Xclipse phones.
- Microphone input is not supported on Android.
- The phone heats up during long sessions and slows down until it cools.

## For testers

- Install the APK, add your games folder, and set up keys and firmware as usual. None are included.
- Recommended: Handheld mode, NCE CPU backend, 0.5x resolution, and a custom RADV driver on Xclipse GPUs.
- Diagnostic switches can be set in `Android/data/dev.lichii.android/files/env.txt` (see the README).
- Report problems at https://github.com/pavelgajdo/Hokkaido/issues with your phone, driver and `eden_log.txt`.

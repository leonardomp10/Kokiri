// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

package org.yuzu.yuzu_emu.utils

import android.content.SharedPreferences
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.DocumentsContract
import androidx.preference.PreferenceManager
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.yuzu.yuzu_emu.NativeLibrary
import org.yuzu.yuzu_emu.YuzuApplication
import org.yuzu.yuzu_emu.model.DirectoryType
import org.yuzu.yuzu_emu.model.Game
import org.yuzu.yuzu_emu.model.GameDir
import org.yuzu.yuzu_emu.model.MinimalDocumentFile
import androidx.core.content.edit
import androidx.core.net.toUri

object GameHelper {
    private const val KEY_OLD_GAME_PATH = "game_path"
    const val KEY_GAMES = "Games"

    var cachedGameList = mutableListOf<Game>()

    // Set by getGames: a game folder holds loose ExeFS games that only all files access can list.
    var looseExeFSNeedAllFilesAccess = false
        private set

    const val PREF_EXEFS_ACCESS_PROMPT_DISMISSED = "ExeFSAccessPromptDismissed"

    private lateinit var preferences: SharedPreferences

    fun getGames(): List<Game> {
        val games = mutableListOf<Game>()
        val gamesByProgramId = mutableMapOf<String, Game>()
        val context = YuzuApplication.appContext
        preferences = PreferenceManager.getDefaultSharedPreferences(context)

        val gameDirs = mutableListOf<GameDir>()
        val oldGamesDir = preferences.getString(KEY_OLD_GAME_PATH, "") ?: ""
        if (oldGamesDir.isNotEmpty()) {
            gameDirs.add(GameDir(oldGamesDir, true))
            preferences.edit() { remove(KEY_OLD_GAME_PATH) }
        }
        gameDirs.addAll(NativeConfig.getGameDirs())

        // Ensure keys are loaded so that ROM metadata can be decrypted.
        NativeLibrary.reloadKeys()

        // Reset metadata so we don't use stale information
        GameMetadata.resetMetadata()

        // Remove previous filesystem provider information so we can get up to date version info
        NativeLibrary.clearFilesystemProvider()

        val mountedContainerUris = mutableSetOf<String>()
        mountExternalContentDirectories(mountedContainerUris)

        var lookForExeFS = !hasAllFilesAccess() &&
            !preferences.getBoolean(PREF_EXEFS_ACCESS_PROMPT_DISMISSED, false)
        looseExeFSNeedAllFilesAccess = false
        val badDirs = mutableListOf<Int>()
        gameDirs.forEachIndexed { index: Int, gameDir: GameDir ->
            val gameDirUri = gameDir.uriString.toUri()
            val isValid = FileUtil.isTreeUriValid(gameDirUri)
            if (isValid) {
                val scanDepth = if (gameDir.deepScan) 3 else 1
                val files = FileUtil.listFiles(gameDirUri)

                addGamesRecursive(
                    games,
                    gamesByProgramId,
                    files,
                    scanDepth,
                    mountedContainerUris
                )
                addLibraryExeFSGames(games, gameDir)
                if (lookForExeFS && gameDir.type == DirectoryType.GAME) {
                    val exeFSDepth =
                        if (gameDir.deepScan) EXEFS_SEARCH_DEPTH + 2 else EXEFS_SEARCH_DEPTH
                    if (containsExeFSFolder(files, exeFSDepth)) {
                        looseExeFSNeedAllFilesAccess = true
                        lookForExeFS = false
                    }
                }
            } else {
                badDirs.add(index)
            }
        }

        // Remove all game dirs with insufficient permissions from config
        if (badDirs.isNotEmpty()) {
            var offset = 0
            badDirs.forEach {
                gameDirs.removeAt(it - offset)
                offset++
            }
        }
        NativeConfig.setGameDirs(gameDirs.toTypedArray())

        // Cache list of games found on disk
        val serializedGames = mutableSetOf<String>()
        games.forEach {
            serializedGames.add(Json.encodeToString(it))
        }
        preferences.edit() {
            remove(KEY_GAMES)
                .putStringSet(KEY_GAMES, serializedGames)
        }

        cachedGameList = games.toMutableList()
        return games.toList()
    }

    private fun addLibraryExeFSGames(games: MutableList<Game>, gameDir: GameDir) {
        findLibraryExeFSGames(gameDir).forEach { game ->
            val duplicate = games.any {
                it.path == game.path ||
                    (game.programId.isNotEmpty() && it.programId == game.programId)
            }
            if (duplicate) {
                return@forEach
            }
            games.add(game)
            if (preferences.getLong(game.keyAddedToLibraryTime, 0L) == 0L) {
                preferences.edit {
                    putLong(game.keyAddedToLibraryTime, System.currentTimeMillis())
                }
            }
        }
    }

    fun restoreContentForGame(game: Game) {
        NativeLibrary.reloadKeys()

        val mountedContainerUris = mutableSetOf<String>()
        mountExternalContentDirectories(mountedContainerUris)
        mountGameFolderContent(Uri.parse(game.path), mountedContainerUris)
        NativeLibrary.addFileToFilesystemProvider(game.path)
    }

    // File extensions considered as external content, buuut should
    // be done better imo.
    private val externalContentExtensions = setOf("nsp", "xci", "nsz", "xcz")

    private fun scanContentContainersRecursive(
        files: Array<MinimalDocumentFile>,
        depth: Int,
        onContainerFound: (MinimalDocumentFile) -> Unit
    ) {
        if (depth <= 0) {
            return
        }

        files.forEach {
            if (it.isDirectory) {
                scanContentContainersRecursive(
                    FileUtil.listFiles(it.uri),
                    depth - 1,
                    onContainerFound
                )
            } else {
                val extension = FileUtil.getExtension(it.uri).lowercase()
                if (externalContentExtensions.contains(extension)) {
                    onContainerFound(it)
                }
            }
        }
    }

    private fun addGamesRecursive(
        games: MutableList<Game>,
        gamesByProgramId: MutableMap<String, Game>,
        files: Array<MinimalDocumentFile>,
        depth: Int,
        mountedContainerUris: MutableSet<String>
    ) {
        if (depth <= 0) {
            return
        }

        files.forEach {
            if (it.isDirectory) {
                addGamesRecursive(
                    games,
                    gamesByProgramId,
                    FileUtil.listFiles(it.uri),
                    depth - 1,
                    mountedContainerUris
                )
            } else {
                val extension = FileUtil.getExtension(it.uri).lowercase()
                val filePath = it.uri.toString()

                val mountedContainer = externalContentExtensions.contains(extension) &&
                    mountedContainerUris.add(filePath)
                if (mountedContainer) {
                    NativeLibrary.addGameFolderFileToFilesystemProvider(filePath)
                }

                if (Game.extensions.contains(extension)) {
                    val game = getGame(it.uri, true, false)
                    if (game != null) {
                        games.add(game)
                        if (game.programId != "0") {
                            gamesByProgramId[game.programId] = game
                        }
                    } else if (mountedContainer) {
                        GameMetadata.getProgramId(filePath).toLongOrNull()?.let { programId ->
                            gamesByProgramId[(programId and 0x800L.inv()).toString()]
                        }?.let { existingGame ->
                            NativeLibrary.getPatchesForFile(existingGame.path, existingGame.programId)
                            existingGame.version = GameMetadata.getVersion(
                                existingGame.path,
                                true
                            )
                            GameIconUtils.refreshGameIcon(existingGame)
                        }
                    }
                }
            }
        }
    }

    private fun mountExternalContentDirectories(mountedContainerUris: MutableSet<String>) {
        val uniqueExternalContentDirs = linkedSetOf<String>()
        NativeConfig.getExternalContentDirs().forEach { externalDir ->
            if (externalDir.isNotEmpty()) {
                uniqueExternalContentDirs.add(externalDir)
            }
        }

        for (externalDir in uniqueExternalContentDirs) {
            val externalDirUri = externalDir.toUri()
            if (FileUtil.isTreeUriValid(externalDirUri)) {
                scanContentContainersRecursive(FileUtil.listFiles(externalDirUri), 3) {
                    val containerUri = it.uri.toString()
                    if (mountedContainerUris.add(containerUri)) {
                        NativeLibrary.addFileToFilesystemProvider(containerUri)
                    }
                }
            }
        }
    }

    private fun mountGameFolderContent(gameUri: Uri, mountedContainerUris: MutableSet<String>) {
        if (gameUri.scheme == "content") {
            val parentUri = getParentDocumentUri(gameUri) ?: return
            scanContentContainersRecursive(FileUtil.listFiles(parentUri), 1) {
                val containerUri = it.uri.toString()
                if (mountedContainerUris.add(containerUri)) {
                    NativeLibrary.addGameFolderFileToFilesystemProvider(containerUri)
                }
            }
            return
        }

        val gameFile = File(gameUri.path ?: gameUri.toString())
        val parentDir = gameFile.parentFile ?: return
        parentDir.listFiles()?.forEach { sibling ->
            if (!sibling.isFile) {
                return@forEach
            }

            val extension = sibling.extension.lowercase()
            if (externalContentExtensions.contains(extension)) {
                val containerUri = Uri.fromFile(sibling).toString()
                if (mountedContainerUris.add(containerUri)) {
                    NativeLibrary.addGameFolderFileToFilesystemProvider(containerUri)
                }
            }
        }
    }

    private fun getParentDocumentUri(uri: Uri): Uri? {
        return try {
            val documentId = DocumentsContract.getDocumentId(uri)
            val separatorIndex = documentId.lastIndexOf('/')
            if (separatorIndex == -1) {
                null
            } else {
                val parentDocumentId = documentId.substring(0, separatorIndex)
                DocumentsContract.buildDocumentUriUsingTree(uri, parentDocumentId)
            }
        } catch (_: Exception) {
            null
        }
    }

    private const val EXEFS_SEARCH_DEPTH = 4
    private const val NPDM_MAX_SIZE = 1024 * 1024
    private const val NPDM_MAGIC = 0x4154454D // "META"
    private const val ACI0_MAGIC = 0x30494341 // "ACI0"
    private val exeFSSkippedDirs = setOf("romfs", "dlc")
    private val exeFSNonTitleDirs = setOf("exefs", "base", "updates", "update")
    private val exeFSStorageLevels = setOf("base", "update", "updates")

    // Loose ExeFS titles (Switch 2) boot from exefs/main by real path. Accepts the game folder
    // (containing exefs/) or the exefs folder itself.
    fun getExeFSGame(folder: File): Game? = findExeFSGames(folder).firstOrNull()

    // Searches a picked folder for exefs/main candidates, newest numbered update first.
    fun findExeFSGames(folder: File, maxDepth: Int = EXEFS_SEARCH_DEPTH): List<Game> {
        val mains = mutableListOf<File>()

        // Check if folder itself is an exefs folder
        File(folder, "main").takeIf { it.isFile }?.let { mains.add(it) }

        // Check direct child folders
        val exefsFolder = File(folder, "exefs")
        if (exefsFolder.isDirectory) {
            File(exefsFolder, "main").takeIf { it.isFile }?.let { mains.add(it) }
        }

        // Check base/exefs/main layout
        val baseExefsFolder = File(folder, "base/exefs")
        if (baseExefsFolder.isDirectory) {
            File(baseExefsFolder, "main").takeIf { it.isFile }?.let { mains.add(it) }
        }

        // Recursively search subdirectories
        collectExeFSMains(folder, 0, maxDepth, mains)

        return mains.distinctBy { it.absolutePath }
            .sortedWith(
                compareByDescending<File> { exeFSVersion(it) }
                    .thenBy { it.parentFile?.relativeTo(folder)?.path ?: "" }
            )
            .map {
                Game(
                    title = exeFSTitle(it, folder),
                    path = it.absolutePath,
                    programId = exeFSProgramId(it)
                )
            }
    }

    // Loose ExeFS games need real paths, which only all files access can read.
    fun hasAllFilesAccess(): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.R || Environment.isExternalStorageManager()

    // One library entry per loose ExeFS game in a game folder: its newest update, else base.
    fun findLibraryExeFSGames(gameDir: GameDir): List<Game> {
        if (gameDir.type != DirectoryType.GAME || !hasAllFilesAccess()) {
            return emptyList()
        }
        val path = PathUtil.getPathFromUri(gameDir.uriString.toUri()) ?: return emptyList()
        val folder = File(path)
        if (!folder.exists() || !folder.isDirectory) {
            return emptyList()
        }
        val maxDepth = if (gameDir.deepScan) EXEFS_SEARCH_DEPTH + 2 else EXEFS_SEARCH_DEPTH
        return findExeFSGames(folder, maxDepth).distinctBy {
            if (it.programId.isEmpty()) it.title else it.programId
        }
    }

    // Looks for an exefs folder through content URIs, which need no all files access.
    private fun containsExeFSFolder(files: Array<MinimalDocumentFile>, depth: Int): Boolean {
        if (depth <= 0) {
            return false
        }
        val dirs = files.filter { it.isDirectory }
        return dirs.any { it.filename.equals("exefs", true) } ||
            dirs.any {
                it.filename.lowercase() !in exeFSSkippedDirs &&
                    containsExeFSFolder(FileUtil.listFiles(it.uri), depth - 1)
            }
    }

    private fun collectExeFSMains(dir: File, depth: Int, maxDepth: Int, mains: MutableList<File>) {
        if (depth >= maxDepth) {
            return
        }
        val children = dir.listFiles() ?: return
        for (child in children) {
            if (!child.isDirectory || isSymlink(child)) {
                continue
            }
            val name = child.name.lowercase()
            if (name in exeFSSkippedDirs && depth > 0) {
                continue
            }
            if (name == "exefs") {
                File(child, "main").takeIf { it.isFile }?.let { mains.add(it) }
                // Also scan inside exefs for update folders
                collectExeFSMains(child, depth + 1, maxDepth, mains)
                continue
            }
            collectExeFSMains(child, depth + 1, maxDepth, mains)
        }
    }

    private fun isSymlink(file: File): Boolean {
        return try {
            val parent = file.parentFile?.canonicalFile ?: return false
            val resolved = File(parent, file.name)
            resolved.canonicalFile != resolved.absoluteFile
        } catch (_: Exception) {
            true
        }
    }

    // Title ID from the ACI0 section of exefs/main.npdm, as a decimal string like GameMetadata.
    private fun exeFSProgramId(main: File): String {
        val npdm = File(main.parentFile, "main.npdm")
        return try {
            if (!npdm.isFile || npdm.length() > NPDM_MAX_SIZE) {
                return ""
            }
            val buffer = ByteBuffer.wrap(npdm.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            if (buffer.limit() < 0x80 || buffer.getInt(0) != NPDM_MAGIC) {
                return ""
            }
            val aciOffset = buffer.getInt(0x70)
            if (aciOffset < 0 || aciOffset + 0x18 > buffer.limit() ||
                buffer.getInt(aciOffset) != ACI0_MAGIC
            ) {
                return ""
            }
            val programId = buffer.getLong(aciOffset + 0x10)
            if (programId == 0L) "" else java.lang.Long.toUnsignedString(programId)
        } catch (_: Exception) {
            ""
        }
    }

    private fun exeFSVersion(main: File): Long {
        var dir = main.parentFile ?: return -1
        if (dir.name.equals("exefs", true)) {
            dir = dir.parentFile ?: return -1
        }
        return dir.name.toLongOrNull() ?: -1
    }

    // Find RomFS for a loose ExeFS dump: the first .romfs file next to main,
    // else a romfs/ directory next to the exefs/ directory
    private fun findLooseRomFS(mainFile: File): File? {
        val exefsDir = mainFile.parentFile ?: return null

        // Check for .romfs files in exefs directory
        exefsDir.listFiles()?.forEach { file ->
            if (file.isFile && file.extension.equals("romfs", ignoreCase = true)) {
                return file
            }
        }

        // Check for romfs/ directory next to exefs/
        val parentDir = exefsDir.parentFile ?: return null
        val romfsDir = File(parentDir, "romfs")
        if (romfsDir.isDirectory) {
            return romfsDir
        }

        return null
    }

    // Calculate total size of a loose dump (exefs + romfs)
    private fun calculateLooseDumpSize(mainFile: File): Long {
        val exefsDir = mainFile.parentFile ?: return mainFile.length()
        var totalSize = 0L

        // Sum exefs directory
        exefsDir.walkTopDown().forEach { file ->
            if (file.isFile) {
                totalSize += file.length()
            }
        }

        // Add romfs if present
        findLooseRomFS(mainFile)?.let { romfs ->
            if (romfs.isDirectory) {
                romfs.walkTopDown().forEach { file ->
                    if (file.isFile) {
                        totalSize += file.length()
                    }
                }
            } else {
                totalSize += romfs.length()
            }
        }

        return totalSize
    }

    private fun exeFSTitle(main: File, folder: File): String {
        var dir = main.parentFile
        var isUpdate = false

        // Skip storage levels (base, update, updates, hex IDs)
        while (dir != null && dir != folder) {
            val name = dir.name
            val nameLower = name.lowercase()

            // Check if this is an update indicator
            if (nameLower in setOf("update", "updates")) {
                isUpdate = true
            }

            // Check if it's a storage level (base, update, or hex ID)
            val isStorageLevel = nameLower in exeFSStorageLevels ||
                name.all { it.isDigit() || it in 'a'..'f' || it in 'A'..'F' }

            if (!isStorageLevel && nameLower !in exeFSNonTitleDirs) {
                val title = if (isUpdate && !name.contains("update", ignoreCase = true)) {
                    "$name (update)"
                } else {
                    name
                }
                return title
            }

            dir = dir.parentFile
        }

        return folder.name
    }

    fun getGame(
        uri: Uri,
        addedToLibrary: Boolean,
        registerFilesystemProvider: Boolean = true
    ): Game? {
        val filePath = uri.toString()
        if (!GameMetadata.getIsValid(filePath)) {
            return null
        }

        if (registerFilesystemProvider) {
            // Needed to update installed content information
            NativeLibrary.addFileToFilesystemProvider(filePath)
        }

        var name = GameMetadata.getTitle(filePath)

        // If the game's title field is empty, use the filename.
        if (name.isEmpty()) {
            name = FileUtil.getFilename(uri)
        }
        var programId = GameMetadata.getProgramId(filePath)

        // If the game's ID field is empty, use the filename without extension.
        if (programId.isEmpty()) {
            programId = name.substring(0, name.lastIndexOf("."))
        }

        val newGame = Game(
            name,
            filePath,
            programId,
            GameMetadata.getDeveloper(filePath),
            GameMetadata.getVersion(filePath, false),
            GameMetadata.getIsHomebrew(filePath)
        )

        if (addedToLibrary) {
            val addedTime = preferences.getLong(newGame.keyAddedToLibraryTime, 0L)
            if (addedTime == 0L) {
                preferences.edit()
                    .putLong(newGame.keyAddedToLibraryTime, System.currentTimeMillis())
                    .apply()
            }
        }

        return newGame
    }
}

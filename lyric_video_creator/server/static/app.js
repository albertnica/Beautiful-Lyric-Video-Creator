/**
 * Beautiful Lyric Video Creator • Web Studio Frontend Controller
 * Manages folder launching, batch render telemetry polling, Windows Explorer
 * video filtering/sorting/deletion, and integrated HTML5 video playback.
 */

let allVideos = [];
let selectedVideoSet = new Set();
let activeVideoFilename = null;
let explorerViewMode = "details"; // 'details' | 'grid'
let searchQuery = "";
let sortMode = "date_desc";
let pollTimer = null;

/**
 * Display a brief floating notification toast in the bottom-right corner.
 * @param {string} message - Notification text to display.
 * @param {boolean} [isError=false] - Whether to style the toast as an error alert.
 */
function showToast(message, isError = false) {
  const toast = document.getElementById("studio-toast");
  if (!toast) return;
  toast.textContent = message;
  toast.style.borderColor = isError ? "#ff4d5e" : "rgba(0, 255, 135, 0.55)";
  toast.classList.remove("hidden");
  clearTimeout(window._toastTimeout);
  window._toastTimeout = setTimeout(() => {
    toast.classList.add("hidden");
  }, 3400);
}

let currentStudioTab = "main";
let configLoadedOnce = false;

/**
 * Fetch the latest studio status, folder counts, render progress, and output videos.
 * @param {boolean} [manual=false] - True when triggered explicitly by the user's Refresh button.
 */
async function refreshStudioStatus(manual = false) {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error(`Status HTTP ${res.status}`);
    const data = await res.json();

    document.getElementById("count-songs").textContent = data.songs_count;
    document.getElementById("count-lyrics").textContent = data.lyrics_count;
    document.getElementById("count-videos").textContent = data.videos_count;

    document.getElementById("lbl-hw-backend").textContent = `⚡ ${data.active_backend}`;
    document.getElementById("lbl-hw-encoder").textContent = `🎞 ${data.active_encoder}`;
    document.getElementById("lbl-explorer-path").textContent = data.output_dir;

    const songsDisp = document.getElementById("display-songs-path");
    const lyricsDisp = document.getElementById("display-lyrics-path");
    const outputDisp = document.getElementById("display-output-path");
    if (songsDisp && data.songs_dir) songsDisp.textContent = data.songs_dir;
    if (lyricsDisp && data.lyrics_dir) lyricsDisp.textContent = data.lyrics_dir;
    if (outputDisp && data.output_dir) outputDisp.textContent = data.output_dir;

    if (data.config) {
      updateProfilePill(data.config);
      if (!configLoadedOnce) {
        populateConfigForm(data.config);
        configLoadedOnce = true;
      }
    }

    updateRenderBanner(data.render_progress);

    // Preserve selection only for files that still exist
    const currentNames = new Set(data.videos.map((v) => v.filename));
    for (const name of Array.from(selectedVideoSet)) {
      if (!currentNames.has(name)) {
        selectedVideoSet.delete(name);
      }
    }

    const prevCount = allVideos.length;
    allVideos = data.videos || [];

    // Re-render explorer if video list changed or manual refresh requested
    if (manual || prevCount !== allVideos.length || document.getElementById("explorer-body").children.length === 0) {
      renderExplorerItems();
    } else {
      updateSelectionToolbar();
    }

    if (manual) {
      showToast("Output folder refreshed.");
    }
  } catch (err) {
    console.error("Failed to fetch studio status:", err);
  }
}

/**
 * Update the live batch rendering banner and Launch button state.
 * @param {object} progress - RenderProgressModel payload from `/api/status`.
 */
function updateRenderBanner(progress) {
  const banner = document.getElementById("render-banner");
  const badge = document.getElementById("badge-render-state");
  const launchTitle = document.getElementById("lbl-launch-title");
  const launchSub = document.getElementById("lbl-launch-sub");
  const stopBtn = document.getElementById("btn-stop-render");

  if (!progress) return;

  if (progress.is_rendering) {
    banner.classList.remove("hidden");
    stopBtn.classList.remove("hidden");
    badge.textContent = `${progress.percent.toFixed(0)}%`;
    badge.style.background = "#ffbd2e";
    launchTitle.textContent = "Rendering in Progress...";
    launchSub.textContent = progress.current_track || "Encoding frames...";

    document.getElementById("banner-msg").textContent = progress.message;
    document.getElementById("banner-counts").textContent =
      `${progress.current_index} / ${progress.total_files} tracks • ` +
      `${progress.processed} rendered • ${progress.skipped} skipped • ${progress.failed} failed`;
    document.getElementById("banner-progress-fill").style.width = `${progress.percent}%`;
  } else {
    badge.textContent = "READY";
    badge.style.background = "var(--spotify-green)";
    launchTitle.textContent = "Launch Video Creation";
    launchSub.textContent = "Batch render all tracks into output/";
    stopBtn.classList.add("hidden");

    if (activeRenderingTrack) {
      showToast(`Rendering finished: ${activeRenderingTrack}`);
      activeRenderingTrack = null;
      renderTracksItems();
    }

    if (progress.total_files > 0 && progress.message !== "Idle") {
      banner.classList.remove("hidden");
      document.getElementById("banner-msg").textContent = progress.message;
      document.getElementById("banner-counts").textContent =
        `${progress.processed} rendered • ${progress.skipped} skipped • ${progress.failed} failed`;
      document.getElementById("banner-progress-fill").style.width = `${progress.percent}%`;
    }
  }
}

/**
 * Open a project folder ('songs', 'lyrics', 'output', 'logs') in native Windows Explorer.
 * @param {string} target - Folder identifier.
 */
async function openFolder(target) {
  try {
    const res = await fetch("/api/open-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Could not open folder");
    showToast(`Opened ${target}/ folder in Windows Explorer.`);
  } catch (err) {
    showToast(`Error opening folder: ${err.message}`, true);
  }
}

/**
 * Trigger batch video creation on the backend server using the active Studio configuration.
 */
async function launchVideoCreation() {
  const backend = document.getElementById("sel-backend").value;
  const encoder = document.getElementById("sel-encoder").value;

  try {
    const res = await fetch("/api/render/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend, encoder }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Failed to start render");

    if (data.status === "empty") {
      showToast(data.message, true);
    } else {
      showToast(data.message);
    }
    await refreshStudioStatus(true);
  } catch (err) {
    showToast(`Launch error: ${err.message}`, true);
  }
}

/**
 * Request graceful stop of an active batch render after the current song finishes.
 */
async function stopVideoCreation() {
  try {
    const res = await fetch("/api/render/stop", { method: "POST" });
    const data = await res.json();
    showToast(data.message || "Stopping render...");
    await refreshStudioStatus();
  } catch (err) {
    showToast(`Stop error: ${err.message}`, true);
  }
}

/**
 * Return the filtered and sorted array of VideoItemModel objects.
 * @returns {Array<object>} Filtered and sorted video items.
 */
function getFilteredSortedVideos() {
  const q = searchQuery.trim().toLowerCase();
  const filtered = allVideos.filter((v) => !q || v.filename.toLowerCase().includes(q));

  filtered.sort((a, b) => {
    switch (sortMode) {
      case "date_asc":
        return a.modified_ts - b.modified_ts;
      case "name_asc":
        return a.filename.localeCompare(b.filename, undefined, { numeric: true });
      case "name_desc":
        return b.filename.localeCompare(a.filename, undefined, { numeric: true });
      case "size_desc":
        return b.size_bytes - a.size_bytes;
      case "date_desc":
      default:
        return b.modified_ts - a.modified_ts;
    }
  });
  return filtered;
}

/**
 * Render the Windows Explorer file list or grid inside `#explorer-body`.
 */
function renderExplorerItems() {
  const container = document.getElementById("explorer-body");
  const header = document.getElementById("explorer-details-header");
  const videos = getFilteredSortedVideos();

  document.getElementById("lbl-item-count").textContent = `${videos.length} item(s)`;
  document.getElementById("lbl-footer-summary").textContent =
    `${allVideos.length} rendered video(s) in output/ (${selectedVideoSet.size} selected)`;

  if (explorerViewMode === "grid") {
    container.classList.add("grid-mode");
    header.classList.add("hidden");
  } else {
    container.classList.remove("grid-mode");
    header.classList.remove("hidden");
  }

  container.innerHTML = "";

  if (videos.length === 0) {
    const empty = document.createElement("div");
    empty.className = "explorer-empty";
    empty.innerHTML = `
      <div style="font-size: 36px;">📂</div>
      <strong>No rendered videos found in <code>output/</code></strong>
      <span>Place your tracks in <code>songs/</code> and click <b>Launch Video Creation</b> above.</span>
    `;
    container.appendChild(empty);
    updateSelectionToolbar();
    return;
  }

  for (const video of videos) {
    const isSelected = selectedVideoSet.has(video.filename);
    const isActive = activeVideoFilename === video.filename;

    if (explorerViewMode === "grid") {
      const tile = document.createElement("div");
      tile.className = `file-tile${isActive ? " active" : ""}`;
      tile.onclick = () => playVideoInStudio(video.filename);

      tile.innerHTML = `
        <div class="tile-top">
          <input type="checkbox" class="file-chk" ${isSelected ? "checked" : ""}>
          <div style="display:flex;gap:6px;">
            <button class="btn-row-action play" title="Play video">▶ Play</button>
            <button class="btn-row-action del" title="Delete video">🗑</button>
          </div>
        </div>
        <div class="tile-preview">🎬</div>
        <div class="tile-title" title="${escapeHtml(video.filename)}">${escapeHtml(video.title)}</div>
        <div class="tile-meta">
          <span>${escapeHtml(video.size_human)}</span>
          <span>${escapeHtml(video.modified_iso.slice(0, 10))}</span>
        </div>
      `;

      const chk = tile.querySelector(".file-chk");
      chk.onclick = (e) => {
        e.stopPropagation();
        toggleVideoSelection(video.filename, chk.checked);
      };
      tile.querySelector(".btn-row-action.play").onclick = (e) => {
        e.stopPropagation();
        playVideoInStudio(video.filename);
      };
      tile.querySelector(".btn-row-action.del").onclick = (e) => {
        e.stopPropagation();
        deleteSingleVideo(video.filename);
      };
      container.appendChild(tile);
    } else {
      const row = document.createElement("div");
      row.className = `file-row${isActive ? " active" : ""}`;
      row.onclick = () => playVideoInStudio(video.filename);

      row.innerHTML = `
        <div class="col-check">
          <input type="checkbox" class="file-chk" ${isSelected ? "checked" : ""}>
        </div>
        <div class="file-name-cell">
          <div class="file-icon">🎬</div>
          <span class="file-title-text" title="${escapeHtml(video.filename)}">${escapeHtml(video.filename)}</span>
        </div>
        <div class="file-date-cell">${escapeHtml(video.modified_iso)}</div>
        <div class="file-size-cell">${escapeHtml(video.size_human)}</div>
        <div class="file-actions-cell">
          <button class="btn-row-action play" title="Play in Studio Player">▶ Play</button>
          <button class="btn-row-action del" title="Delete from output/">🗑 Delete</button>
        </div>
      `;

      const chk = row.querySelector(".file-chk");
      chk.onclick = (e) => {
        e.stopPropagation();
        toggleVideoSelection(video.filename, chk.checked);
      };
      row.querySelector(".btn-row-action.play").onclick = (e) => {
        e.stopPropagation();
        playVideoInStudio(video.filename);
      };
      row.querySelector(".btn-row-action.del").onclick = (e) => {
        e.stopPropagation();
        deleteSingleVideo(video.filename);
      };
      container.appendChild(row);
    }
  }

  updateSelectionToolbar();
}

/**
 * Load and play the specified video inside the integrated right-hand HTML5 video player.
 * @param {string} filename - Basename of the `.mp4` file in `output/`.
 */
function playVideoInStudio(filename) {
  const videoObj = allVideos.find((v) => v.filename === filename);
  if (!videoObj) return;

  activeVideoFilename = filename;
  const player = document.getElementById("main-video-player");
  const emptyState = document.getElementById("empty-player-state");

  emptyState.classList.add("hidden");
  player.classList.remove("hidden");
  player.src = `${videoObj.url}?t=${videoObj.modified_ts}`;
  player.play().catch(() => {});

  document.getElementById("lbl-player-filename").textContent = videoObj.filename;
  document.getElementById("lbl-inspector-title").textContent = videoObj.title;
  document.getElementById("lbl-inspector-sub").textContent =
    `${videoObj.size_human} • Modified ${videoObj.modified_iso}`;

  document.getElementById("btn-play-os").disabled = false;
  document.getElementById("btn-delete-current").disabled = false;

  renderExplorerItems();
}

/**
 * Open the currently active video in the native OS media player.
 */
async function playCurrentInOSPlayer() {
  if (!activeVideoFilename) return;
  try {
    const res = await fetch("/api/videos/play-external", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: activeVideoFilename }),
    });
    if (!res.ok) throw new Error("Failed to launch external player");
    showToast(`Opened '${activeVideoFilename}' in external player.`);
  } catch (err) {
    showToast(err.message, true);
  }
}

/**
 * Toggle selection state for a single video filename.
 * @param {string} filename - Video filename.
 * @param {boolean} checked - Whether the checkbox is checked.
 */
function toggleVideoSelection(filename, checked) {
  if (checked) {
    selectedVideoSet.add(filename);
  } else {
    selectedVideoSet.delete(filename);
  }
  updateSelectionToolbar();
}

/**
 * Select or deselect all currently filtered videos.
 * @param {boolean} checked - Master checkbox state.
 */
function toggleSelectAll(checked) {
  const visible = getFilteredSortedVideos();
  for (const v of visible) {
    if (checked) {
      selectedVideoSet.add(v.filename);
    } else {
      selectedVideoSet.delete(v.filename);
    }
  }
  renderExplorerItems();
}

/**
 * Synchronize the selection count badge and Delete Selected button state.
 */
function updateSelectionToolbar() {
  const count = selectedVideoSet.size;
  document.getElementById("lbl-selected-count").textContent = count;
  document.getElementById("btn-delete-selected").disabled = count === 0;
  document.getElementById("lbl-footer-summary").textContent =
    `${allVideos.length} rendered video(s) in output/ (${count} selected)`;

  const visible = getFilteredSortedVideos();
  const selectAllChk = document.getElementById("chk-select-all");
  selectAllChk.checked = visible.length > 0 && visible.every((v) => selectedVideoSet.has(v.filename));
}

/**
 * Delete a single video file after user confirmation.
 * @param {string} filename - Basename of the `.mp4` file to delete.
 */
async function deleteSingleVideo(filename) {
  if (!confirm(`Delete '${filename}' permanently from the output folder?`)) {
    return;
  }
  await performDeleteVideos([filename]);
}

/**
 * Delete the currently playing video.
 */
async function deleteCurrentVideo() {
  if (!activeVideoFilename) return;
  await deleteSingleVideo(activeVideoFilename);
}

/**
 * Delete all currently checked videos in bulk.
 */
async function deleteSelectedVideos() {
  const targets = Array.from(selectedVideoSet);
  if (targets.length === 0) return;
  if (!confirm(`Delete ${targets.length} selected video(s) permanently from the output folder?`)) {
    return;
  }
  await performDeleteVideos(targets);
}

/**
 * Execute the `/api/videos/delete` call and clean up active player state if deleted.
 * @param {Array<string>} filenames - Array of `.mp4` basenames to delete.
 */
async function performDeleteVideos(filenames) {
  // If the currently playing video is among those being deleted, unload the video element first
  // so Windows releases the file handle immediately.
  if (activeVideoFilename && filenames.includes(activeVideoFilename)) {
    const player = document.getElementById("main-video-player");
    player.pause();
    player.removeAttribute("src");
    player.load();
    player.classList.add("hidden");
    document.getElementById("empty-player-state").classList.remove("hidden");
    document.getElementById("lbl-player-filename").textContent = "No video selected";
    document.getElementById("lbl-inspector-title").textContent = "Select a track from the Explorer";
    document.getElementById("lbl-inspector-sub").textContent = "Video unloaded.";
    document.getElementById("btn-play-os").disabled = true;
    document.getElementById("btn-delete-current").disabled = true;
    activeVideoFilename = null;
  }

  try {
    const res = await fetch("/api/videos/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filenames }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Failed to delete videos");

    for (const name of data.deleted || []) {
      selectedVideoSet.delete(name);
    }
    showToast(`Deleted ${(data.deleted || []).length} video(s).`);
    await refreshStudioStatus(true);
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, true);
  }
}

/**
 * Switch Explorer view mode between 'details' and 'grid'.
 * @param {string} mode - Target view mode ('details' or 'grid').
 */
function setExplorerView(mode) {
  explorerViewMode = mode;
  document.getElementById("btn-view-details").classList.toggle("active", mode === "details");
  document.getElementById("btn-view-grid").classList.toggle("active", mode === "grid");
  renderExplorerItems();
}

/**
 * Filter Explorer items as the user types in the search box.
 * @param {string} value - Search substring.
 */
function onSearchInput(value) {
  searchQuery = value || "";
  renderExplorerItems();
}

/**
 * Change Explorer sorting criterion.
 * @param {string} value - Sort key ('date_desc', 'date_asc', 'name_asc', 'name_desc', 'size_desc').
 */
function onSortChange(value) {
  sortMode = value || "date_desc";
  renderExplorerItems();
}

/**
 * Gracefully stop the background FastAPI Web Studio server.
 */
async function shutdownStudioServer() {
  if (!confirm("Stop the Beautiful Lyric Video Creator Web Studio server?")) return;
  try {
    await fetch("/api/shutdown", { method: "POST" });
    document.body.innerHTML = `
      <div style="height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#070a08;color:#fff;font-family:'Outfit',sans-serif;gap:12px;">
        <h2 style="color:#00ff87;">Web Studio Stopped</h2>
        <p style="color:#93a897;">You can safely close this browser tab.</p>
      </div>
    `;
  } catch (_) {}
}

/**
 * Escape HTML special characters to prevent XSS when rendering filenames.
 * @param {string} str - Input string.
 * @returns {string} Escaped HTML string.
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

let allTracks = [];
let availableLrcs = [];
let tracksSearchQuery = "";
let allPresets = [];
let activePresetId = null;
let activeRenderingTrack = null;

/**
 * Switch between Studio & Output Explorer ('main'), Tracks & Single Render ('tracks'),
 * and Configuration & Presets ('config').
 * @param {string} tab - Target tab identifier ('main', 'tracks', or 'config').
 */
function switchStudioTab(tab) {
  currentStudioTab = tab || "main";
  const mainView = document.getElementById("tab-view-main");
  const tracksView = document.getElementById("tab-view-tracks");
  const configView = document.getElementById("tab-view-config");

  const navMain = document.getElementById("nav-tab-main");
  const navTracks = document.getElementById("nav-tab-tracks");
  const navConfig = document.getElementById("nav-tab-config");

  // Reset visibility
  mainView.classList.add("hidden");
  if (tracksView) tracksView.classList.add("hidden");
  configView.classList.add("hidden");

  navMain.classList.remove("active");
  navMain.setAttribute("aria-selected", "false");
  if (navTracks) {
    navTracks.classList.remove("active");
    navTracks.setAttribute("aria-selected", "false");
  }
  navConfig.classList.remove("active");
  navConfig.setAttribute("aria-selected", "false");

  if (currentStudioTab === "tracks") {
    if (tracksView) tracksView.classList.remove("hidden");
    if (navTracks) {
      navTracks.classList.add("active");
      navTracks.setAttribute("aria-selected", "true");
    }
    loadTracksList();
    loadPresets();
  } else if (currentStudioTab === "config") {
    configView.classList.remove("hidden");
    navConfig.classList.add("active");
    navConfig.setAttribute("aria-selected", "true");
    loadStudioConfig();
    loadPresets();
  } else {
    mainView.classList.remove("hidden");
    navMain.classList.add("active");
    navMain.setAttribute("aria-selected", "true");
  }
}

/**
 * Fetch and render discovered tracks from `/api/tracks`.
 * @param {boolean} [manual=false] - True when triggered by manual user click.
 */
async function loadTracksList(manual = false) {
  try {
    const res = await fetch("/api/tracks");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    allTracks = data.tracks || [];
    availableLrcs = data.available_lrcs || [];

    const summary = document.getElementById("lbl-tracks-summary");
    if (summary) {
      summary.textContent = `${allTracks.length} track(s) discovered`;
    }

    renderTracksItems();
    if (manual) showToast("Tracks catalog refreshed.");
  } catch (err) {
    console.error("Failed to load tracks list:", err);
    showToast(`Failed to load tracks: ${err.message}`, true);
  }
}

/**
 * Filter tracks by user search input.
 * @param {string} value - Substring search text.
 */
function onTracksSearchInput(value) {
  tracksSearchQuery = (value || "").toLowerCase().trim();
  renderTracksItems();
}

/**
 * Render discovered track rows into the Tracks list table.
 */
function renderTracksItems() {
  const container = document.getElementById("tracks-body");
  if (!container) return;

  const filtered = allTracks.filter((t) => {
    if (!tracksSearchQuery) return true;
    return (
      t.filename.toLowerCase().includes(tracksSearchQuery) ||
      t.title.toLowerCase().includes(tracksSearchQuery) ||
      t.artist.toLowerCase().includes(tracksSearchQuery)
    );
  });

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="padding: 40px; text-align: center; color: var(--text-muted); font-size: 13px;">
        ${allTracks.length === 0 ? "No audio tracks found in active songs folder. Add songs (.flac, .mp3, .m4a, .ogg, .wav)." : "No tracks matching search query."}
      </div>
    `;
    return;
  }

  container.innerHTML = "";
  filtered.forEach((track, idx) => {
    const artClass = track.has_cover ? "track-art-indicator has-art" : "track-art-indicator";
    const artIcon = track.has_cover ? "🎨" : "🎵";

    const autoLabel = track.matched_lrc
      ? `[Auto] ${escapeHtml(track.matched_lrc)}`
      : "[Auto] No LRC auto-matched";

    let lrcOptions = `<option value="">${autoLabel}</option>`;
    lrcOptions += `<option value="__none__">[Cover Only] No Lyrics</option>`;

    for (const lrcName of availableLrcs) {
      if (lrcName !== track.matched_lrc) {
        lrcOptions += `<option value="${escapeHtml(lrcName)}">${escapeHtml(lrcName)}</option>`;
      }
    }

    let presetOptions = `<option value="">Active Configuration</option>`;
    for (const p of allPresets) {
      presetOptions += `<option value="${p.id}">${escapeHtml(p.name)}</option>`;
    }

    const row = document.createElement("div");
    row.className = "track-row";
    row.innerHTML = `
      <div class="${artClass}" title="${track.has_cover ? 'Cover artwork available' : 'No embedded or sidecar artwork'}">${artIcon}</div>
      <div class="track-title-col">
        <span class="track-name-text" title="${escapeHtml(track.title)}">${escapeHtml(track.title)}</span>
        <span class="track-artist-text">${escapeHtml(track.artist)} • ${escapeHtml(track.album)}</span>
      </div>
      <div class="track-meta-chips">
        <span>⏱ ${track.duration_human}</span>
        <span>📦 ${track.format.toUpperCase()} (${track.size_human})</span>
      </div>
      <div class="track-lrc-picker">
        <select id="track-lrc-${idx}">
          ${lrcOptions}
        </select>
      </div>
      <div class="track-preset-picker">
        <select id="track-preset-${idx}">
          ${presetOptions}
        </select>
      </div>
      <div>
        <button class="btn-render-single" id="btn-render-track-${idx}">
          🚀 Render Track
        </button>
      </div>
    `;

    const renderBtn = row.querySelector(`#btn-render-track-${idx}`);
    if (renderBtn) {
      renderBtn.onclick = () => renderSingleTrack(track.filename, idx);
    }

    container.appendChild(row);
  });
}

/**
 * Launch single-track video creation with optional LRC and preset overrides.
 * @param {string} audioFilename - Basename of audio track.
 * @param {number} idx - Index in DOM.
 */
async function renderSingleTrack(audioFilename, idx) {
  const btn = document.getElementById(`btn-render-track-${idx}`);
  const lrcSel = document.getElementById(`track-lrc-${idx}`);
  const presetSel = document.getElementById(`track-preset-${idx}`);

  const lrcVal = lrcSel ? lrcSel.value : null;
  const presetVal = presetSel ? presetSel.value : null;

  const payload = {
    audio_filename: audioFilename,
    lrc_filename: lrcVal || null,
    preset: presetVal || null,
  };

  const originalHtml = btn ? btn.innerHTML : "🚀 Render Track";
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `⏳ Encoding...`;
    btn.style.opacity = "0.75";
  }

  activeRenderingTrack = audioFilename;

  try {
    const res = await fetch("/api/render/single", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.status === "error" || data.status === "busy") {
      throw new Error(data.message || "Failed to start render");
    }

    showToast(data.message || `Rendering started for ${audioFilename}`);
    await refreshStudioStatus(false);
  } catch (err) {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = originalHtml;
      btn.style.opacity = "1";
    }
    activeRenderingTrack = null;
    showToast(`Render failed: ${err.message}`, true);
  }
}

/**
 * Fetch and render standard configuration presets.
 */
async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    allPresets = data.presets || [];
    activePresetId = data.active_preset;

    renderPresetCards();
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

/**
 * Render standard preset cards into `#presets-grid`.
 */
function renderPresetCards() {
  const container = document.getElementById("presets-grid");
  if (!container) return;

  container.innerHTML = allPresets
    .map((p) => {
      const isActive = p.id === activePresetId;
      const cardClass = isActive ? "preset-card active" : "preset-card";
      const activeBadge = isActive ? `<span class="preset-badge">ACTIVE</span>` : "";

      return `
        <div class="${cardClass}" onclick="applyPreset('${p.id}')">
          ${activeBadge}
          <div class="preset-title">${escapeHtml(p.name)}</div>
          <div class="preset-desc">${escapeHtml(p.description)}</div>
          <div class="preset-specs">
            <span class="spec-chip">📐 ${p.width}×${p.height}</span>
            <span class="spec-chip">⚡ ${p.fps} FPS</span>
            <span class="spec-chip">🎞 ${p.video_bitrate}</span>
            <span class="spec-chip">✨ Blur: ${p.blur_strength}px</span>
          </div>
        </div>
      `;
    })
    .join("");
}

/**
 * Apply a standard preset via `POST /api/presets/apply`.
 * @param {string} presetId - Preset identifier.
 */
async function applyPreset(presetId) {
  try {
    const res = await fetch("/api/presets/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset_id: presetId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Failed to apply preset");

    activePresetId = presetId;
    populateConfigForm(data.config);
    updateProfilePill(data.config);
    renderPresetCards();
    showToast(data.message || "Preset applied.");
    await refreshStudioStatus(false);
  } catch (err) {
    showToast(`Error applying preset: ${err.message}`, true);
  }
}

/**
 * Prompt user for a directory path and apply it via `POST /api/folders/set`.
 * @param {string} target - 'songs', 'lyrics', or 'output'.
 */
async function promptChangeFolder(target) {
  const currentPath = document.getElementById(`display-${target}-path`)?.textContent || "";
  const newPath = prompt(`Enter absolute folder path for ${target}:`, currentPath.trim());
  if (!newPath || !newPath.trim() || newPath.trim() === currentPath.trim()) return;

  await setFolder(target, newPath.trim());
}

/**
 * Reset folder back to project default via `POST /api/folders/set`.
 * @param {string} target - 'songs', 'lyrics', or 'output'.
 */
async function resetFolderDefault(target) {
  if (!confirm(`Restore default project folder for ${target}?`)) return;
  await setFolder(target, "__default__");
}

/**
 * Update media directory path via `POST /api/folders/set`.
 * @param {string} target - 'songs', 'lyrics', or 'output'.
 * @param {string} path - Target path string or '__default__'.
 */
async function setFolder(target, path) {
  try {
    const res = await fetch("/api/folders/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target, path }),
    });
    const data = await res.json();
    if (!res.ok || data.status === "error") {
      throw new Error(data.message || "Failed to set folder");
    }

    showToast(data.message);
    await refreshStudioStatus(false);
    await loadTracksList(false);
  } catch (err) {
    showToast(`Folder update error: ${err.message}`, true);
  }
}

/**
 * Update the bottom bar active profile summary pill.
 * @param {object} cfg - RenderConfig dictionary.
 */
function updateProfilePill(cfg) {
  const pill = document.getElementById("lbl-active-profile");
  if (!pill || !cfg) return;
  pill.textContent =
    `Active Profile: ${cfg.width}×${cfg.height} • ${cfg.fps} FPS • ` +
    `${cfg.video_bitrate} video • ${cfg.audio_bitrate} audio • ` +
    `${cfg.backend}/${cfg.encoder}`;
}

/**
 * Fetch the active and default configurations from `/api/config` and populate the form.
 */
async function loadStudioConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.config) {
      populateConfigForm(data.config);
      updateProfilePill(data.config);
      configLoadedOnce = true;
    }
  } catch (err) {
    console.error("Failed to load studio configuration:", err);
  }
}

const NUMERIC_CONFIG_FIELDS = [
  "width",
  "height",
  "fps",
  "max_lyrics_length",
  "default_last_line_duration",
  "left_margin",
  "glow_blur_ksize",
  "blur_strength",
  "glow_intensity_base",
  "fixed_text_glow_intensity",
  "assets_glow_intensity",
  "glow_pulse_amp",
  "glow_pulse_freq",
  "glow_edge_px",
  "glow_boost",
  "alpha_min",
  "background_size",
  "rotation_speed",
  "min_glow_sat",
  "min_glow_val",
  "block_size",
  "ping_pong_buffers",
  "queue_max",
];

const STRING_CONFIG_FIELDS = [
  "video_bitrate",
  "audio_bitrate",
  "backend",
  "encoder",
];

/**
 * Populate all inputs inside the Configuration tab and quick selectors from a RenderConfig object.
 * @param {object} cfg - RenderConfig dictionary.
 */
function populateConfigForm(cfg) {
  if (!cfg) return;

  for (const key of [...NUMERIC_CONFIG_FIELDS, ...STRING_CONFIG_FIELDS]) {
    const el = document.getElementById(`cfg-${key}`);
    if (el && cfg[key] !== undefined) {
      el.value = cfg[key];
    }
  }

  const debugChk = document.getElementById("cfg-debug");
  if (debugChk) {
    debugChk.checked = Boolean(cfg.debug);
  }

  // Sync quick hardware selectors in the Main Tab footer
  const quickBackend = document.getElementById("sel-backend");
  const quickEncoder = document.getElementById("sel-encoder");
  if (quickBackend && cfg.backend) quickBackend.value = cfg.backend;
  if (quickEncoder && cfg.encoder) quickEncoder.value = cfg.encoder;

  syncPresetDropdown();
}

/**
 * Update width and height inputs when the user selects a standard resolution preset.
 * @param {string} preset - Preset value (e.g., '1920x1080', '1280x720', 'custom').
 */
function onResolutionPresetChange(preset) {
  if (!preset || preset === "custom") return;
  const parts = preset.split("x");
  if (parts.length === 2) {
    document.getElementById("cfg-width").value = parseInt(parts[0], 10);
    document.getElementById("cfg-height").value = parseInt(parts[1], 10);
  }
}

/**
 * Synchronize the Resolution Preset dropdown to match the current Width and Height inputs.
 */
function syncPresetDropdown() {
  const w = document.getElementById("cfg-width")?.value;
  const h = document.getElementById("cfg-height")?.value;
  const sel = document.getElementById("sel-res-preset");
  if (!sel || !w || !h) return;

  const key = `${w}x${h}`;
  const matchingOption = Array.from(sel.options).some((opt) => opt.value === key);
  sel.value = matchingOption ? key : "custom";
}

/**
 * Sync changes from the Main Tab quick hardware selectors (`#sel-backend`, `#sel-encoder`)
 * into the Configuration form and persist them to `/api/config`.
 */
async function syncQuickHardwareToConfig() {
  const backend = document.getElementById("sel-backend").value;
  const encoder = document.getElementById("sel-encoder").value;
  const cfgBackend = document.getElementById("cfg-backend");
  const cfgEncoder = document.getElementById("cfg-encoder");
  if (cfgBackend) cfgBackend.value = backend;
  if (cfgEncoder) cfgEncoder.value = encoder;

  try {
    const res = await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend, encoder }),
    });
    if (res.ok) {
      const data = await res.json();
      updateProfilePill(data.config);
      await refreshStudioStatus(false);
    }
  } catch (err) {
    console.error("Failed to sync quick hardware settings:", err);
  }
}

/**
 * Collect all form fields from the Configuration tab and persist via `PUT /api/config`.
 */
async function saveStudioConfig() {
  const payload = {};

  const intFields = new Set([
    "width",
    "height",
    "fps",
    "max_lyrics_length",
    "left_margin",
    "glow_blur_ksize",
    "blur_strength",
    "glow_edge_px",
    "alpha_min",
    "background_size",
    "block_size",
    "ping_pong_buffers",
    "queue_max",
  ]);

  for (const key of NUMERIC_CONFIG_FIELDS) {
    const el = document.getElementById(`cfg-${key}`);
    if (!el) continue;
    const raw = el.value;
    const num = intFields.has(key) ? parseInt(raw, 10) : parseFloat(raw);
    if (!Number.isNaN(num)) {
      payload[key] = num;
    }
  }

  // Ensure blur kernels are odd integers >= 3 so OpenCV/CUDA separable filters remain valid
  for (const oddKey of ["glow_blur_ksize", "blur_strength"]) {
    if (payload[oddKey] !== undefined) {
      if (payload[oddKey] < 3) payload[oddKey] = 3;
      if (payload[oddKey] % 2 === 0) payload[oddKey] += 1;
    }
  }

  // Ensure width/height are even integers for YUV420p compatibility
  for (const evenKey of ["width", "height"]) {
    if (payload[evenKey] !== undefined && payload[evenKey] % 2 !== 0) {
      payload[evenKey] += 1;
    }
  }

  for (const key of STRING_CONFIG_FIELDS) {
    const el = document.getElementById(`cfg-${key}`);
    if (el && el.value.trim()) {
      payload[key] = el.value.trim();
    }
  }

  const debugChk = document.getElementById("cfg-debug");
  if (debugChk) {
    payload.debug = Boolean(debugChk.checked);
  }

  try {
    const res = await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Failed to save configuration");

    populateConfigForm(data.config);
    updateProfilePill(data.config);
    showToast(data.message || "Configuration saved.");
    await refreshStudioStatus(false);
  } catch (err) {
    showToast(`Save error: ${err.message}`, true);
  }
}

/**
 * Reset all configuration parameters back to the user's original defaults via `POST /api/config/reset`.
 */
async function resetStudioConfig() {
  if (!confirm("Reset all resolution, framerate, bitrate, and visual parameters to your default profile (1920×1080 @ 20 FPS)?")) {
    return;
  }
  try {
    const res = await fetch("/api/config/reset", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Failed to reset configuration");

    populateConfigForm(data.config);
    updateProfilePill(data.config);
    showToast(data.message || "Reset to default values.");
    await refreshStudioStatus(false);
  } catch (err) {
    showToast(`Reset error: ${err.message}`, true);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  refreshStudioStatus(true);
  loadStudioConfig();
  loadPresets();
  loadTracksList();
  pollTimer = setInterval(() => refreshStudioStatus(false), 2000);
});

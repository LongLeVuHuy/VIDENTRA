"""Embedded standalone UI for Search and Playback Inspector."""

UI_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VIDENTRA — Multimodal Video Moment Search</title>
  <style>
    :root {
      --bg: #0f172a;
      --bg-elevated: #0b1120;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --border: #334155;
      --badge-supported: #22c55e;
      --badge-partial: #eab308;
      --badge-contradicted: #ef4444;
      --badge-unknown: #64748b;
      --warn-bg: rgba(245, 158, 11, 0.14);
      --warn-border: rgba(245, 158, 11, 0.45);
      --err-bg: rgba(239, 68, 68, 0.14);
      --err-border: rgba(239, 68, 68, 0.45);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 20px 24px; min-height: 100vh; }
    .container { max-width: 1400px; margin: 0 auto; }

    .header { margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 14px; display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
    .header h1 { font-size: 24px; font-weight: 700; color: var(--primary); letter-spacing: -0.5px; }
    .header .subtitle { font-size: 13px; color: var(--text-muted); margin-top: 2px; }
    .header-actions { display: flex; gap: 8px; flex-wrap: wrap; }

    .search-panel { background: var(--card-bg); padding: 18px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 16px; }
    .query-box { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
    .query-input { flex: 1; min-width: 240px; padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg-elevated); color: var(--text); font-size: 15px; outline: none; transition: border-color 0.15s; }
    .query-input:focus { border-color: var(--primary); }
    .btn { padding: 10px 18px; border-radius: 8px; background: var(--primary); color: #0b1120; font-weight: 600; cursor: pointer; border: none; font-size: 14px; transition: background 0.15s, opacity 0.15s; display: inline-flex; align-items: center; gap: 6px; }
    .btn:hover:not(:disabled) { background: var(--primary-hover); color: #fff; }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .btn-secondary { background: var(--border); color: var(--text); }
    .btn-secondary:hover:not(:disabled) { background: #475569; }
    .btn-sm { padding: 6px 11px; font-size: 12px; border-radius: 6px; }

    .controls { display: flex; gap: 14px; align-items: center; font-size: 13px; color: var(--text-muted); flex-wrap: wrap; }
    .control-item { display: inline-flex; align-items: center; gap: 6px; }
    .select-ctl, .input-ctl { padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg-elevated); color: var(--text); font-size: 13px; outline: none; }
    .select-ctl:focus, .input-ctl:focus { border-color: var(--primary); }
    .checkbox-ctl { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; color: var(--text); }
    .checkbox-ctl input { accent-color: var(--primary); width: 15px; height: 15px; cursor: pointer; }

    /* Status Banners */
    .banner { display: none; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; line-height: 1.5; border: 1px solid transparent; }
    .banner-degraded { background: var(--warn-bg); border-color: var(--warn-border); color: #fde68a; }
    .banner-error { background: var(--err-bg); border-color: var(--err-border); color: #fecaca; }
    .banner-loading { background: rgba(56, 189, 248, 0.1); border-color: rgba(56, 189, 248, 0.35); color: #bae6fd; align-items: center; gap: 10px; }
    .spinner { width: 16px; height: 16px; border: 2px solid rgba(56, 189, 248, 0.3); border-top-color: var(--primary); border-radius: 50%; animation: spin 0.7s linear infinite; display: inline-block; }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* QueryGraph Inspector Panel */
    .graph-panel { display: none; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; }
    .graph-title { font-size: 13px; font-weight: 600; color: var(--primary); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .chip { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin: 2px 4px 2px 0; background: var(--card-bg); border: 1px solid var(--border); }
    .chip-entity { border-color: #38bdf8; color: #38bdf8; }
    .chip-pred { border-color: #a855f7; color: #c084fc; }
    .chip-logic { border-color: #f59e0b; color: #fbbf24; }
    .chip-btn { cursor: pointer; transition: background 0.15s, border-color 0.15s; }
    .chip-btn:hover { background: #1e293b; border-color: var(--primary); color: #fff; }

    .meta-bar { font-size: 13px; color: var(--text-muted); margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 18px; }
    .card { background: var(--card-bg); border-radius: 10px; border: 1px solid var(--border); overflow: hidden; display: flex; flex-direction: column; transition: transform 0.15s, border-color 0.15s; }
    .card:hover { transform: translateY(-2px); border-color: var(--primary); }
    .thumb-wrap { position: relative; background: #020617; cursor: pointer; }
    .thumb { width: 100%; height: 170px; display: flex; align-items: center; justify-content: center; color: var(--text-muted); font-size: 13px; object-fit: cover; }
    .thumb-time-pill { position: absolute; bottom: 8px; right: 8px; background: rgba(2, 6, 23, 0.85); color: #f8fafc; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-family: monospace; border: 1px solid rgba(148, 163, 184, 0.3); }
    .card-body { padding: 14px 16px; flex: 1; display: flex; flex-direction: column; gap: 8px; font-size: 13px; }
    .vid-title { font-weight: 600; color: var(--primary); word-break: break-all; font-family: monospace; font-size: 12px; }
    .time-info { color: var(--text); font-weight: 500; }
    .frame-info { color: var(--text-muted); font-size: 12px; }
    .score-badge { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-top: 2px; flex-wrap: wrap; }
    .evidence-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 2px; }
    .badge { padding: 2px 8px; border-radius: 4px; font-weight: 600; text-transform: uppercase; font-size: 10px; letter-spacing: 0.3px; }
    .badge-supported { background: rgba(34, 197, 94, 0.2); color: var(--badge-supported); }
    .badge-partial { background: rgba(234, 179, 8, 0.2); color: var(--badge-partial); }
    .badge-contradicted { background: rgba(239, 68, 68, 0.2); color: var(--badge-contradicted); }
    .badge-unknown { background: rgba(100, 116, 139, 0.25); color: #cbd5e1; }
    .badge-visual { background: rgba(56, 189, 248, 0.16); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-ocr { background: rgba(168, 85, 247, 0.18); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.35); }
    .badge-asr { background: rgba(16, 185, 129, 0.18); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.35); }
    .card-footer { padding: 10px 16px; border-top: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; gap: 8px; }

    /* Modal Inspector */
    .modal-overlay { position: fixed; inset: 0; background: rgba(2, 6, 23, 0.88); display: none; align-items: center; justify-content: center; z-index: 100; backdrop-filter: blur(4px); padding: 12px; }
    .modal { background: var(--card-bg); width: 100%; max-width: min(1480px, 96vw); border-radius: 12px; border: 1px solid var(--border); overflow: hidden; display: flex; flex-direction: column; max-height: 95vh; transition: max-width 0.18s ease, height 0.18s ease; }
    .modal.theater { max-width: 98vw; width: 98vw; height: 96vh; max-height: 96vh; }
    .modal-header { padding: 12px 18px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-shrink: 0; }
    .modal-header-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
    .modal-body { padding: 16px 18px; overflow-y: auto; flex: 1; min-height: 0; }
    .inspector-layout { display: grid; grid-template-columns: 1fr; gap: 16px; align-items: start; }
    .inspector-layout.split-view { grid-template-columns: minmax(0, 2.2fr) minmax(320px, 1fr); }
    .modal.theater .inspector-layout,
    .modal.theater .inspector-layout.split-view { grid-template-columns: 1fr; }
    .inspector-primary { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
    .inspector-sidebar { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; min-width: 0; }
    .inspector-layout.split-view .inspector-sidebar { display: flex; flex-direction: column; gap: 12px; }
    .video-container { position: relative; width: 100%; aspect-ratio: 16 / 9; min-height: 360px; max-height: 74vh; flex-shrink: 0; background: #000; border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
    .modal.theater .video-container { min-height: 440px; max-height: 82vh; }
    .video-container video, video { position: absolute; inset: 0; width: 100%; height: 100%; max-height: 100%; object-fit: contain; display: block; outline: none; background: #000; }
    .player-status { position: absolute; top: 10px; right: 10px; background: rgba(2, 6, 23, 0.82); color: #e2e8f0; font-size: 11px; padding: 3px 8px; border-radius: 4px; border: 1px solid var(--border); display: none; pointer-events: none; }

    .timeline-bar-wrap { display: flex; flex-direction: column; gap: 6px; background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; flex-shrink: 0; }
    .timeline-meta { display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: var(--text-muted); font-family: monospace; gap: 8px; flex-wrap: wrap; }
    .timeline-track { position: relative; height: 10px; background: #1e293b; border-radius: 999px; cursor: pointer; overflow: hidden; border: 1px solid var(--border); }
    .timeline-moment-window { position: absolute; top: 0; bottom: 0; background: rgba(56, 189, 248, 0.35); border-left: 2px solid var(--primary); border-right: 2px solid var(--primary); pointer-events: none; }
    .timeline-playhead { position: absolute; top: 0; bottom: 0; left: 0; width: 3px; background: #f8fafc; box-shadow: 0 0 6px rgba(248, 250, 252, 0.8); pointer-events: none; }

    .jump-pills { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .jump-pill { padding: 4px 10px; border-radius: 999px; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text); font-size: 12px; cursor: pointer; transition: border-color 0.15s, background 0.15s; }
    .jump-pill:hover { border-color: var(--primary); background: #172554; }
    .jump-pill-primary { border-color: rgba(56, 189, 248, 0.5); color: var(--primary); }

    .inspector-controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .kbd { background: var(--bg-elevated); border: 1px solid var(--border); padding: 1px 5px; border-radius: 4px; font-size: 11px; font-family: monospace; color: var(--text); }
    .shortcuts-hint { font-size: 12px; color: var(--text-muted); margin-left: auto; }
    .inspector-card { background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 12px; }
    .inspector-card-header { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
    .evidence-tabs { display: flex; gap: 4px; flex-wrap: wrap; }
    .evidence-tab { padding: 3px 8px; border-radius: 999px; border: 1px solid var(--border); background: var(--card-bg); color: var(--text-muted); font-size: 11px; cursor: pointer; }
    .evidence-tab.active { border-color: var(--primary); color: var(--primary); background: rgba(56, 189, 248, 0.12); }
    .frame-preview { border-radius: 8px; border: 1px solid var(--border); width: 100%; max-height: 220px; object-fit: contain; background: #000; }
    .filmstrip-row { display: flex; gap: 8px; overflow-x: auto; padding: 4px 2px 6px; }
    .filmstrip-item { flex: 0 0 112px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 4px; cursor: pointer; text-align: center; transition: border-color 0.15s; }
    .filmstrip-item:hover { border-color: var(--primary); }
    .filmstrip-item.active { border-color: var(--primary); box-shadow: 0 0 0 1px var(--primary); }
    .filmstrip-item img { width: 100%; height: 64px; object-fit: cover; border-radius: 4px; background: #020617; display: block; margin-bottom: 4px; }
    .filmstrip-item span { font-size: 11px; color: var(--text-muted); font-family: monospace; }

    .toast { position: fixed; bottom: 20px; right: 20px; background: #0f172a; color: #f8fafc; border: 1px solid var(--primary); padding: 10px 14px; border-radius: 8px; font-size: 13px; z-index: 200; display: none; box-shadow: 0 10px 25px rgba(0,0,0,0.4); }

    @media (max-width: 960px) {
      .inspector-layout,
      .inspector-layout.split-view { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      body { padding: 14px; }
      .modal-overlay { padding: 8px; }
      .modal-body { padding: 12px; }
      .video-container { min-height: 220px; max-height: 55vh; }
      .inspector-sidebar { grid-template-columns: 1fr; }
      .shortcuts-hint { margin-left: 0; width: 100%; }
      .frame-verify-row { flex-direction: column; align-items: flex-start !important; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>VIDENTRA</h1>
        <div class="subtitle">Multi-Modal Video Retrieval Engine • Snapshot: current</div>
      </div>
      <div class="header-actions">
        <button class="btn btn-secondary btn-sm" onclick="exportData('aic_kis')">Export AIC KIS</button>
        <button class="btn btn-secondary btn-sm" onclick="exportData('vbs_csv')">Export VBS CSV</button>
        <button class="btn btn-secondary btn-sm" onclick="exportData('csv')">Export CSV</button>
        <button class="btn btn-secondary btn-sm" onclick="exportData('json')">Export JSON</button>
      </div>
    </div>

    <div class="search-panel">
      <div class="query-box">
        <input type="text" id="queryInput" class="query-input" placeholder="Tìm người mặc áo đỏ, xe máy rẽ trái, chữ SALE, tiếng nói..." autofocus>
        <button id="searchBtn" class="btn" onclick="performSearch()">Tìm kiếm</button>
      </div>
      <div class="controls">
        <label class="control-item">Top-K:
          <select id="topKSelect" class="select-ctl">
            <option value="5">5</option>
            <option value="10" selected>10</option>
            <option value="20">20</option>
            <option value="50">50</option>
          </select>
        </label>
        <label class="control-item">Chế độ:
          <select id="modeSelect" class="select-ctl">
            <option value="fast">Fast (Visual Vector)</option>
            <option value="accurate" selected>Accurate (QueryGraph + Multimodal)</option>
          </select>
        </label>
        <label class="control-item">Xếp hạng:
          <select id="rankModeSelect" class="select-ctl">
            <option value="moment" selected>Moment Ranking</option>
            <option value="diverse_video">Diverse Video</option>
          </select>
        </label>
        <label class="control-item">Dataset:
          <input id="datasetInput" list="datasetPresets" class="input-ctl" type="text" placeholder="Tất cả (tuỳ chọn)" style="width:130px;">
          <datalist id="datasetPresets">
            <option value="msrvtt"></option>
            <option value="avc"></option>
            <option value="qvhighlights"></option>
          </datalist>
        </label>
        <label class="control-item">Backend:
          <select id="backendSelect" class="select-ctl">
            <option value="auto" selected>Auto</option>
            <option value="siglip2">SigLIP2</option>
            <option value="clip">CLIP</option>
          </select>
        </label>
        <label class="checkbox-ctl">
          <input type="checkbox" id="verifyRelationsChk">
          <span>Verify relations</span>
        </label>
      </div>
    </div>

    <div id="loadingBar" class="banner banner-loading">
      <span class="spinner"></span>
      <span>Đang truy vấn đa phương thức và tổng hợp khoảnh khắc...</span>
    </div>
    <div id="errorBanner" class="banner banner-error"></div>
    <div id="degradedBanner" class="banner banner-degraded"></div>

    <!-- QueryGraph Inspector -->
    <div class="graph-panel" id="graphPanel">
      <div class="graph-title">
        <span>Phân tích Truy vấn (QueryGraph v1)</span>
        <span id="graphStatusBadge" class="badge badge-supported">validated</span>
      </div>
      <div id="graphEntities" style="margin-bottom: 6px;"></div>
      <div id="graphPredicates"></div>
    </div>

    <div class="meta-bar">
      <span id="resultsMeta">Chưa có tìm kiếm. Nhập câu hỏi và nhấn Tìm kiếm (hoặc Enter).</span>
      <span id="timingsMeta"></span>
    </div>

    <div class="grid" id="resultsGrid"></div>
  </div>

  <!-- Modal Inspector -->
  <div class="modal-overlay" id="modalOverlay" onclick="closeModalOnBg(event)">
    <div class="modal" id="inspectorModal" role="dialog" aria-modal="true" aria-labelledby="modalVideoId">
      <div class="modal-header">
        <div>
          <strong id="modalVideoId" style="color:var(--primary); font-family:monospace; font-size:13px;"></strong>
          <div id="modalTiming" style="font-size:12px; color:var(--text-muted); margin-top:2px;"></div>
        </div>
        <div class="modal-header-actions">
          <button id="layoutToggleBtn" class="btn btn-secondary btn-sm" onclick="toggleSplitLayout()">Chia 2 cột</button>
          <button id="theaterToggleBtn" class="btn btn-secondary btn-sm" onclick="toggleTheaterMode()">Phóng to (F)</button>
          <button class="btn btn-secondary btn-sm" onclick="closeModal()">Đóng (Esc)</button>
        </div>
      </div>
      <div class="modal-body">
        <div id="inspectorLayout" class="inspector-layout">
          <div class="inspector-primary">
            <div class="video-container">
              <video id="player" controls preload="metadata" playsinline></video>
              <div id="playerStatus" class="player-status">Đang tải...</div>
            </div>

            <div class="timeline-bar-wrap">
              <div class="timeline-meta">
                <span id="playerTimeDisplay">00:00.00 / 00:00.00</span>
                <span id="momentWindowDisplay">Khoảnh khắc: --</span>
              </div>
              <div id="timelineTrack" class="timeline-track" title="Nhấn để tua đến vị trí trên timeline" onclick="seekFromTimeline(event)">
                <div id="timelineMomentWindow" class="timeline-moment-window" style="left:0%; width:0%;"></div>
                <div id="timelinePlayhead" class="timeline-playhead" style="left:0%;"></div>
              </div>
            </div>

            <div id="momentJumpPills" class="jump-pills"></div>

            <div class="inspector-card">
              <div class="inspector-card-header">
                <strong style="color:var(--primary);">Duyệt Keyframes Lân Cận (Neighbor Filmstrip)</strong>
                <span id="neighborMeta" style="color:var(--text-muted); font-size:11px;"></span>
              </div>
              <div id="neighborFilmstrip" class="filmstrip-row">
                <span style="color:var(--text-muted); font-size:12px;">Đang tải các frame lân cận...</span>
              </div>
            </div>

            <div class="inspector-controls">
              <button class="btn btn-sm" onclick="replayMoment()">Replay Moment (<span class="kbd">R</span>)</button>
              <button class="btn btn-secondary btn-sm" onclick="seekRelative(-5)">−5s (<span class="kbd">←</span>)</button>
              <button class="btn btn-secondary btn-sm" onclick="seekRelative(5)">+5s (<span class="kbd">→</span>)</button>
              <label class="control-item" style="font-size:12px; color:var(--text-muted);">Tốc độ:
                <select id="playbackSpeedSelect" class="select-ctl" style="padding:4px 8px; font-size:12px;" onchange="setPlaybackSpeed(this.value)">
                  <option value="0.75">0.75x</option>
                  <option value="1" selected>1x</option>
                  <option value="1.25">1.25x</option>
                  <option value="1.5">1.5x</option>
                </select>
              </label>
              <button class="btn btn-secondary btn-sm" onclick="stepFrame(-1)">−1 Frame (<span class="kbd">,</span>)</button>
              <button class="btn btn-secondary btn-sm" onclick="stepFrame(1)">+1 Frame (<span class="kbd">.</span>)</button>
              <button class="btn btn-secondary btn-sm" onclick="copyInfo()">Copy Video + Frame</button>
              <span class="shortcuts-hint">
                <span class="kbd">Space</span> Play/Pause • <span class="kbd">←</span>/<span class="kbd">→</span> ±5s • <span class="kbd">F</span> Phóng to • <span class="kbd">Esc</span> Đóng
              </span>
            </div>
          </div>

          <div class="inspector-sidebar">
            <div id="modalEvidenceBox" class="inspector-card">
              <div class="inspector-card-header">
                <strong style="color:var(--primary);">Bằng chứng đa phương thức (Evidence Items)</strong>
                <div id="evidenceFilterTabs" class="evidence-tabs">
                  <button class="evidence-tab active" data-filter="all" onclick="setEvidenceFilter('all')">Tất cả</button>
                  <button class="evidence-tab" data-filter="visual" onclick="setEvidenceFilter('visual')">Visual</button>
                  <button class="evidence-tab" data-filter="ocr" onclick="setEvidenceFilter('ocr')">OCR</button>
                  <button class="evidence-tab" data-filter="asr" onclick="setEvidenceFilter('asr')">ASR</button>
                </div>
              </div>
              <div id="modalEvidenceList" style="display:flex; flex-wrap:wrap; gap:6px;"></div>
            </div>

            <div class="inspector-card">
              <div class="inspector-card-header">
                <strong style="color:var(--primary);">Exact Bit-Level Decoded Frame (Ledger Verification)</strong>
              </div>
              <div class="frame-verify-row" style="display:flex; flex-direction:column; gap:10px;">
                <img id="exactFrameImg" class="frame-preview" alt="Exact Frame">
                <div id="frameDetails" style="font-size:12px; color:var(--text-muted); line-height:1.6;"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div id="toast" class="toast"></div>

  <script>
    let currentResults = [];
    let activeMoment = null;
    let currentOrdinal = 0;
    let inspectorToken = 0;
    let seekAbortController = null;
    let currentFrameObjectUrl = null;
    let toastTimer = null;

    function escapeHtml(str) {
      return String(str ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function showToast(msg) {
      const toast = document.getElementById("toast");
      toast.textContent = msg;
      toast.style.display = "block";
      if (toastTimer) clearTimeout(toastTimer);
      toastTimer = setTimeout(() => { toast.style.display = "none"; }, 2200);
    }

    function setLoading(isLoading) {
      const loadingBar = document.getElementById("loadingBar");
      const searchBtn = document.getElementById("searchBtn");
      loadingBar.style.display = isLoading ? "flex" : "none";
      searchBtn.disabled = isLoading;
      searchBtn.textContent = isLoading ? "Đang tìm..." : "Tìm kiếm";
    }

    function showErrorBanner(msg) {
      const el = document.getElementById("errorBanner");
      if (!msg) {
        el.style.display = "none";
        el.textContent = "";
        return;
      }
      el.textContent = msg;
      el.style.display = "block";
    }

    function renderDegradedBanner(data) {
      const el = document.getElementById("degradedBanner");
      const branchStatuses = data.branch_statuses || {};
      const degradedChannels = Object.entries(branchStatuses)
        .filter(([, st]) => ["degraded", "error", "missing_index", "unavailable", "failed"].includes(String(st).toLowerCase()))
        .map(([ch, st]) => `${ch}: ${st}`);
      const isDegraded = data.status === "degraded" || degradedChannels.length > 0;
      const warnings = Array.isArray(data.warnings) ? data.warnings : [];

      if (!isDegraded && warnings.length === 0) {
        el.style.display = "none";
        el.innerHTML = "";
        return;
      }

      const parts = [];
      if (isDegraded) {
        const detail = degradedChannels.length > 0 ? degradedChannels.join(", ") : "một số kênh bị suy giảm";
        parts.push(`<strong>Chế độ suy giảm (Degraded):</strong> ${escapeHtml(detail)}`);
      }
      if (warnings.length > 0) {
        parts.push(`<strong>Lưu ý:</strong> ${warnings.map(escapeHtml).join(" • ")}`);
      }
      el.innerHTML = parts.join("<br>");
      el.style.display = "block";
    }

    document.getElementById("queryInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter") performSearch();
    });

    window.addEventListener("keydown", (e) => {
      const overlay = document.getElementById("modalOverlay");
      if (overlay.style.display !== "flex") return;

      const tag = (document.activeElement && document.activeElement.tagName || "").toLowerCase();
      const inFormControl = tag === "input" || tag === "textarea" || tag === "select";

      if (e.key === "Escape") {
        closeModal();
        return;
      }
      if (inFormControl) return;

      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        togglePlayPause();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        seekRelative(-5);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        seekRelative(5);
      } else if (e.key === "r" || e.key === "R") {
        replayMoment();
      } else if (e.key === "f" || e.key === "F") {
        toggleTheaterMode();
      } else if (e.key === ",") {
        stepFrame(-1);
      } else if (e.key === ".") {
        stepFrame(1);
      }
    });

    function formatTimeSec(sec) {
      if (!Number.isFinite(sec) || sec < 0) return "00:00.00";
      const mins = Math.floor(sec / 60);
      const rem = sec - mins * 60;
      const minStr = String(mins).padStart(2, "0");
      const secStr = rem.toFixed(2).padStart(5, "0");
      return `${minStr}:${secStr}`;
    }

    function updateTimelineUI() {
      const player = document.getElementById("player");
      const timeDisplay = document.getElementById("playerTimeDisplay");
      const momentDisplay = document.getElementById("momentWindowDisplay");
      const playhead = document.getElementById("timelinePlayhead");
      const winEl = document.getElementById("timelineMomentWindow");

      const dur = Number.isFinite(player.duration) && player.duration > 0 ? player.duration : 0;
      const cur = Number.isFinite(player.currentTime) ? player.currentTime : 0;
      timeDisplay.textContent = `${formatTimeSec(cur)} / ${formatTimeSec(dur)}`;

      if (dur > 0) {
        const pct = Math.min(100, Math.max(0, (cur / dur) * 100));
        playhead.style.left = `${pct}%`;
      } else {
        playhead.style.left = "0%";
      }

      if (activeMoment) {
        const startSec = Number(activeMoment.moment_start_seconds || 0);
        const endSec = Number(activeMoment.moment_end_seconds || startSec);
        momentDisplay.textContent = `Khoảnh khắc: ${startSec.toFixed(2)}s – ${endSec.toFixed(2)}s`;
        if (dur > 0) {
          const leftPct = Math.min(100, Math.max(0, (startSec / dur) * 100));
          const rightPct = Math.min(100, Math.max(leftPct, (endSec / dur) * 100));
          winEl.style.left = `${leftPct}%`;
          winEl.style.width = `${Math.max(1.5, rightPct - leftPct)}%`;
        } else {
          winEl.style.left = "0%";
          winEl.style.width = "0%";
        }
      } else {
        momentDisplay.textContent = "Khoảnh khắc: --";
        winEl.style.left = "0%";
        winEl.style.width = "0%";
      }
    }

    function seekFromTimeline(event) {
      const player = document.getElementById("player");
      const track = document.getElementById("timelineTrack");
      const dur = Number.isFinite(player.duration) && player.duration > 0 ? player.duration : 0;
      if (dur <= 0) return;
      const rect = track.getBoundingClientRect();
      if (!rect || rect.width <= 0) return;
      const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
      seekPlayerSafely(ratio * dur, false);
    }

    function toggleTheaterMode() {
      const modal = document.getElementById("inspectorModal");
      const btn = document.getElementById("theaterToggleBtn");
      if (!modal || !btn) return;
      const isTheater = modal.classList.toggle("theater");
      btn.textContent = isTheater ? "Thu gọn (F)" : "Phóng to (F)";
    }

    function toggleSplitLayout() {
      const layout = document.getElementById("inspectorLayout");
      const btn = document.getElementById("layoutToggleBtn");
      if (!layout || !btn) return;
      const isSplit = layout.classList.toggle("split-view");
      btn.textContent = isSplit ? "1 cột lớn" : "Chia 2 cột";
    }

    // Player buffering & timeline feedback
    (function setupPlayerEvents() {
      const player = document.getElementById("player");
      const statusEl = document.getElementById("playerStatus");
      player.addEventListener("waiting", () => {
        statusEl.textContent = "Đang đệm video...";
        statusEl.style.display = "block";
      });
      player.addEventListener("seeking", () => {
        statusEl.textContent = "Đang tua...";
        statusEl.style.display = "block";
      });
      ["canplay", "seeked", "playing", "pause"].forEach((evt) => {
        player.addEventListener(evt, () => {
          statusEl.style.display = "none";
          updateTimelineUI();
        });
      });
      ["loadedmetadata", "timeupdate", "durationchange"].forEach((evt) => {
        player.addEventListener(evt, updateTimelineUI);
      });
      player.addEventListener("error", () => {
        if (player.getAttribute("src")) {
          statusEl.textContent = "Không thể phát luồng video.";
          statusEl.style.display = "block";
        }
      });
    })();

    async function performSearch() {
      const query = document.getElementById("queryInput").value.trim();
      if (!query) return;

      const topK = parseInt(document.getElementById("topKSelect").value, 10);
      const mode = document.getElementById("modeSelect").value;
      const rankMode = document.getElementById("rankModeSelect").value;
      const dataset = document.getElementById("datasetInput").value.trim();
      const backend = document.getElementById("backendSelect").value;
      const verifyRelations = document.getElementById("verifyRelationsChk").checked;

      const filters = {};
      if (dataset) filters.dataset = dataset;
      if (backend && backend !== "auto") filters.backend = backend;

      showErrorBanner("");
      setLoading(true);
      document.getElementById("resultsMeta").textContent = "Đang tìm kiếm...";

      try {
        const resp = await fetch("/v1/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            top_k: topK,
            mode,
            ranking_mode: rankMode,
            filters,
            verify_relations: verifyRelations,
          }),
        });
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status}`);
        }
        const data = await resp.json();
        currentResults = data.results || [];

        renderDegradedBanner(data);
        document.getElementById("resultsMeta").textContent = `${currentResults.length} kết quả cho "${query}" • Status: ${data.status}`;
        document.getElementById("timingsMeta").textContent = data.timings && data.timings.elapsed_ms != null
          ? `${data.timings.elapsed_ms} ms`
          : "";

        renderQueryGraph(data.query_graph);
        renderGrid(currentResults);
      } catch (err) {
        showErrorBanner(`Lỗi tìm kiếm: ${err.message}`);
        document.getElementById("resultsMeta").textContent = "Tìm kiếm thất bại.";
      } finally {
        setLoading(false);
      }
    }

    function renderQueryGraph(qg) {
      const panel = document.getElementById("graphPanel");
      if (!qg) {
        panel.style.display = "none";
        return;
      }
      panel.style.display = "block";
      const statusBadge = document.getElementById("graphStatusBadge");
      const parserStatus = qg.parser?.status || "unknown";
      statusBadge.textContent = parserStatus;
      statusBadge.className = `badge ${parserStatus === "validated" ? "badge-supported" : "badge-partial"}`;

      const entDiv = document.getElementById("graphEntities");
      entDiv.innerHTML = "<strong>Thực thể:</strong> ";
      if (qg.entities && qg.entities.length > 0) {
        qg.entities.forEach((e) => {
          entDiv.innerHTML += `<span class="chip chip-entity">${escapeHtml(e.mention)} (${escapeHtml(e.type)})</span>`;
        });
      } else {
        entDiv.innerHTML += "<span style='color:var(--text-muted); font-size:12px;'>Không có thực thể riêng lẻ</span>";
      }

      const predDiv = document.getElementById("graphPredicates");
      predDiv.innerHTML = "<strong>Vị từ:</strong> ";
      if (qg.predicates && qg.predicates.length > 0) {
        qg.predicates.forEach((p) => {
          const val = p.value ? `: "${escapeHtml(p.value)}"` : "";
          const pol = p.polarity === "negative" ? " [NOT]" : "";
          predDiv.innerHTML += `<span class="chip chip-pred">${escapeHtml(p.kind)}${pol}${val}</span>`;
        });
      } else {
        predDiv.innerHTML += "<span style='color:var(--text-muted); font-size:12px;'>Không có ràng buộc vị từ</span>";
      }
    }

    function summarizeEvidenceBadges(item) {
      const evs = Array.isArray(item.evidence) ? item.evidence : [];
      const counts = { visual: 0, ocr: 0, asr: 0, other: 0 };
      evs.forEach((ev) => {
        const k = String(ev.kind || ev.modality || "").toLowerCase();
        if (k.includes("ocr")) counts.ocr += 1;
        else if (k.includes("asr") || k.includes("audio") || k.includes("speech")) counts.asr += 1;
        else if (k.includes("visual") || k.includes("frame")) counts.visual += 1;
        else counts.other += 1;
      });
      const badges = [];
      if (counts.visual > 0) badges.push(`<span class="badge badge-visual">Visual ×${counts.visual}</span>`);
      if (counts.ocr > 0) badges.push(`<span class="badge badge-ocr">OCR ×${counts.ocr}</span>`);
      if (counts.asr > 0) badges.push(`<span class="badge badge-asr">ASR ×${counts.asr}</span>`);
      if (counts.other > 0) badges.push(`<span class="badge badge-unknown">Evidence ×${counts.other}</span>`);
      return badges.join("");
    }

    function renderGrid(results) {
      const grid = document.getElementById("resultsGrid");
      grid.innerHTML = "";

      if (!results || results.length === 0) {
        grid.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 48px; color:var(--text-muted);">Không tìm thấy khoảnh khắc phù hợp.</div>`;
        return;
      }

      results.forEach((item, idx) => {
        const card = document.createElement("div");
        card.className = "card";
        const repOrd = item.representative_original_frame_index || 0;
        const thumbUrl = `/v1/frames/${encodeURIComponent(item.video_id)}/${repOrd}`;
        const startVal = Number(item.moment_start_seconds || 0);
        const endVal = Number(item.moment_end_seconds || startVal);
        const startSec = startVal.toFixed(2);
        const endSec = endVal.toFixed(2);
        const durSec = Math.max(0, endVal - startVal).toFixed(2);
        const score = Number(item.ranking_score || 0).toFixed(4);

        let badgeClass = "badge-partial";
        if (item.match_status === "supported") badgeClass = "badge-supported";
        else if (item.match_status === "contradicted") badgeClass = "badge-contradicted";
        else if (item.match_status === "unknown") badgeClass = "badge-unknown";

        const unresolvedWarn = (item.unresolved_predicates && item.unresolved_predicates.length > 0)
          ? `<div style="font-size:11px; color:#f87171;">Chưa giải quyết: ${item.unresolved_predicates.map(escapeHtml).join(", ")}</div>`
          : "";

        const evidenceBadges = summarizeEvidenceBadges(item);

        card.innerHTML = `
          <div class="thumb-wrap" onclick="openInspector(${idx})">
            <img src="${thumbUrl}" class="thumb" alt="Thumbnail" loading="lazy" onerror="this.removeAttribute('src'); this.alt='Không có thumbnail';">
            <span class="thumb-time-pill">${startSec}s – ${endSec}s</span>
          </div>
          <div class="card-body">
            <div class="vid-title" title="${escapeHtml(item.video_id)}">${escapeHtml(item.video_id)}</div>
            <div class="time-info">${startSec}s → ${endSec}s (${durSec}s)</div>
            <div class="frame-info">Frame ordinal: ${item.start_original_frame_index ?? 0} → ${item.end_original_frame_index ?? 0} (rep: ${repOrd})</div>
            <div class="score-badge">
              <span class="badge ${badgeClass}">${escapeHtml(item.match_status || "unknown")}</span>
              <span style="color:var(--text-muted);">Score: <strong>${score}</strong></span>
            </div>
            ${evidenceBadges ? `<div class="evidence-badges">${evidenceBadges}</div>` : ""}
            ${unresolvedWarn}
          </div>
          <div class="card-footer">
            <button class="btn btn-sm" onclick="openInspector(${idx})">Mở Inspector</button>
            <button class="btn btn-secondary btn-sm" onclick='searchSimilar(${JSON.stringify(item.video_id)}, ${startVal}, ${repOrd})'>Tương tự</button>
            <button class="btn btn-secondary btn-sm" onclick='copySingle(${JSON.stringify(item.video_id)}, ${repOrd})'>Copy ID</button>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    function seekPlayerSafely(targetSeconds, autoPlay = false) {
      const player = document.getElementById("player");
      const token = inspectorToken;
      const safeTarget = Number.isFinite(targetSeconds) ? Math.max(0, targetSeconds) : 0;

      if (seekAbortController) {
        seekAbortController.abort();
        seekAbortController = null;
      }

      const applySeek = () => {
        if (token !== inspectorToken) return;
        const maxDur = Number.isFinite(player.duration) && player.duration > 0 ? player.duration : null;
        const clamped = maxDur !== null ? Math.min(safeTarget, Math.max(0, maxDur - 0.05)) : safeTarget;
        try {
          player.currentTime = clamped;
        } catch (_) {
          // Ignore seek errors if media is not yet seekable
        }
        if (autoPlay) {
          player.play().catch(() => {});
        }
      };

      if (player.readyState >= 1) {
        applySeek();
      } else {
        seekAbortController = new AbortController();
        player.addEventListener("loadedmetadata", applySeek, {
          once: true,
          signal: seekAbortController.signal,
        });
      }
    }

    function renderJumpPills(item) {
      const container = document.getElementById("momentJumpPills");
      container.innerHTML = "";
      const startSec = Number(item.moment_start_seconds || 0);
      const endSec = Number(item.moment_end_seconds || startSec);
      const midSec = (startSec + endSec) / 2;

      const pills = [
        { label: `Bắt đầu (${startSec.toFixed(2)}s)`, sec: startSec, primary: true },
        { label: `Giữa (${midSec.toFixed(2)}s)`, sec: midSec, primary: false },
        { label: `Kết thúc (${endSec.toFixed(2)}s)`, sec: endSec, primary: false },
      ];

      pills.forEach((p) => {
        const btn = document.createElement("button");
        btn.className = `jump-pill ${p.primary ? "jump-pill-primary" : ""}`;
        btn.textContent = p.label;
        btn.onclick = () => seekPlayerSafely(p.sec, true);
        container.appendChild(btn);
      });
    }

    let activeEvidenceFilter = "all";

    function classifyEvidenceModality(ev) {
      const k = String(ev.kind || ev.modality || "").toLowerCase();
      if (k.includes("ocr")) return "ocr";
      if (k.includes("asr") || k.includes("audio") || k.includes("speech")) return "asr";
      if (k.includes("visual") || k.includes("frame")) return "visual";
      return "other";
    }

    function setEvidenceFilter(filter) {
      activeEvidenceFilter = filter || "all";
      document.querySelectorAll("#evidenceFilterTabs .evidence-tab").forEach((btn) => {
        btn.classList.toggle("active", btn.getAttribute("data-filter") === activeEvidenceFilter);
      });
      renderEvidenceList(activeMoment);
    }

    function renderEvidenceList(item) {
      const evList = document.getElementById("modalEvidenceList");
      if (!evList) return;
      evList.innerHTML = "";
      const allEv = item && Array.isArray(item.evidence) ? item.evidence : [];
      const filtered = activeEvidenceFilter === "all"
        ? allEv
        : allEv.filter((ev) => classifyEvidenceModality(ev) === activeEvidenceFilter);

      if (filtered.length > 0) {
        filtered.forEach((ev) => {
          const chip = document.createElement("span");
          const hasTs = Number.isFinite(ev.timestamp_us);
          chip.className = `chip chip-logic ${hasTs ? "chip-btn" : ""}`;
          const label = ev.evidence_id || (hasTs ? `${(ev.timestamp_us / 1e6).toFixed(2)}s` : "item");
          chip.textContent = `${ev.kind || "evidence"}: ${label}`;
          if (hasTs) {
            chip.title = "Nhấn để nhảy tới thời điểm bằng chứng";
            chip.onclick = () => seekPlayerSafely(ev.timestamp_us / 1e6, true);
          }
          evList.appendChild(chip);
        });
      } else if (allEv.length > 0) {
        evList.innerHTML = "<span style='color:var(--text-muted);'>Không có bằng chứng thuộc kênh đã lọc.</span>";
      } else {
        evList.innerHTML = "<span style='color:var(--text-muted);'>Chỉ có bằng chứng khung hình đại diện.</span>";
      }
    }

    function openInspector(idx) {
      const item = currentResults[idx];
      if (!item) return;
      inspectorToken += 1;
      activeMoment = item;
      currentOrdinal = item.representative_original_frame_index || item.start_original_frame_index || 0;

      document.getElementById("modalVideoId").textContent = item.video_id;
      const startSec = Number(item.moment_start_seconds || 0);
      const endSec = Number(item.moment_end_seconds || startSec);
      document.getElementById("modalTiming").textContent =
        `Khoảnh khắc: ${startSec.toFixed(2)}s → ${endSec.toFixed(2)}s | Score: ${Number(item.ranking_score || 0).toFixed(4)} | Match: ${item.match_status || "unknown"}`;

      renderJumpPills(item);
      setEvidenceFilter("all");

      const player = document.getElementById("player");
      const speedSel = document.getElementById("playbackSpeedSelect");
      player.playbackRate = parseFloat(speedSel.value || "1");
      document.getElementById("playerStatus").style.display = "none";

      const targetStart = Number.isFinite(item.start_ms)
        ? item.start_ms / 1000.0
        : startSec;
      player.src = `/v1/media/${encodeURIComponent(item.video_id)}`;
      seekPlayerSafely(targetStart, false);
      updateTimelineUI();

      loadExactFrame(item.video_id, currentOrdinal);
      loadNeighborKeyframes(item.video_id, targetStart);
      document.getElementById("modalOverlay").style.display = "flex";
    }

    function closeModal() {
      inspectorToken += 1;
      activeMoment = null;
      if (seekAbortController) {
        seekAbortController.abort();
        seekAbortController = null;
      }
      const player = document.getElementById("player");
      player.pause();
      player.removeAttribute("src");
      player.load();
      updateTimelineUI();
      document.getElementById("playerStatus").style.display = "none";
      document.getElementById("modalOverlay").style.display = "none";
    }

    function closeModalOnBg(e) {
      if (e.target.id === "modalOverlay") closeModal();
    }

    function togglePlayPause() {
      const player = document.getElementById("player");
      if (player.paused) {
        player.play().catch(() => {});
      } else {
        player.pause();
      }
    }

    function seekRelative(deltaSeconds) {
      const player = document.getElementById("player");
      const next = Math.max(0, (player.currentTime || 0) + deltaSeconds);
      seekPlayerSafely(next, false);
    }

    function setPlaybackSpeed(rate) {
      const player = document.getElementById("player");
      const parsed = parseFloat(rate);
      if (Number.isFinite(parsed) && parsed > 0) {
        player.playbackRate = parsed;
      }
    }

    function replayMoment() {
      if (!activeMoment) return;
      const targetStart = Number.isFinite(activeMoment.start_ms)
        ? activeMoment.start_ms / 1000.0
        : Number(activeMoment.moment_start_seconds || 0);
      seekPlayerSafely(targetStart, true);
    }

    function stepFrame(delta) {
      if (!activeMoment) return;
      currentOrdinal = Math.max(0, currentOrdinal + delta);
      loadExactFrame(activeMoment.video_id, currentOrdinal);
    }

    async function loadExactFrame(videoId, ordinal) {
      const img = document.getElementById("exactFrameImg");
      const details = document.getElementById("frameDetails");
      details.innerHTML = "Đang nạp frame từ server...";

      try {
        const url = `/v1/frames/${encodeURIComponent(videoId)}/${ordinal}`;
        const resp = await fetch(url);
        if (!resp.ok) {
          details.innerHTML = `<span style="color:#f87171;">Lỗi nạp frame: HTTP ${resp.status}</span>`;
          return;
        }

        const pts = resp.headers.get("X-PTS") || "N/A";
        const tsUs = resp.headers.get("X-Timestamp-Us") || "N/A";
        const rev = resp.headers.get("X-Timeline-Revision") || "N/A";

        const blob = await resp.blob();
        if (currentFrameObjectUrl) {
          URL.revokeObjectURL(currentFrameObjectUrl);
        }
        currentFrameObjectUrl = URL.createObjectURL(blob);
        img.src = currentFrameObjectUrl;

        const parsedTs = parseInt(tsUs, 10);
        const tsDisplay = Number.isFinite(parsedTs)
          ? `${tsUs} µs (${(parsedTs / 1e6).toFixed(4)}s)`
          : "N/A";
        details.innerHTML = `
          <div><strong>Ordinal:</strong> ${ordinal}</div>
          <div><strong>PTS:</strong> ${escapeHtml(pts)}</div>
          <div><strong>Timestamp:</strong> ${escapeHtml(tsDisplay)}</div>
          <div><strong>Revision:</strong> ${escapeHtml(rev)}</div>
        `;
      } catch (err) {
        details.innerHTML = `<span style="color:#f87171;">Lỗi: ${escapeHtml(err.message)}</span>`;
      }
    }

    function copyInfo() {
      if (!activeMoment) return;
      const text = `${activeMoment.video_id}:${currentOrdinal} [${activeMoment.moment_start_seconds}-${activeMoment.moment_end_seconds}s]`;
      navigator.clipboard.writeText(text).then(() => showToast("Đã sao chép: " + text)).catch(() => {});
    }

    function copySingle(vid, ord) {
      const text = `${vid}:${ord}`;
      navigator.clipboard.writeText(text).then(() => showToast("Đã sao chép: " + text)).catch(() => {});
    }

    async function loadNeighborKeyframes(videoId, centerSec) {
      const strip = document.getElementById("neighborFilmstrip");
      const meta = document.getElementById("neighborMeta");
      if (!strip) return;
      strip.innerHTML = "<span style='color:var(--text-muted); font-size:12px;'>Đang tải keyframe lân cận...</span>";
      try {
        const url = `/v1/videos/${encodeURIComponent(videoId)}/neighbors?timestamp=${encodeURIComponent(centerSec.toFixed(2))}&window=25&limit=15`;
        const resp = await fetch(url);
        if (!resp.ok) {
          strip.innerHTML = "<span style='color:var(--text-muted); font-size:12px;'>Không có keyframe lân cận.</span>";
          if (meta) meta.textContent = "";
          return;
        }
        const data = await resp.json();
        const list = data.neighbors || [];
        if (meta) meta.textContent = `${list.length} keyframes (±25s)`;
        strip.innerHTML = "";
        if (list.length === 0) {
          strip.innerHTML = "<span style='color:var(--text-muted); font-size:12px;'>Không có keyframe lân cận.</span>";
          return;
        }
        list.forEach((kf) => {
          const item = document.createElement("div");
          item.className = `filmstrip-item ${kf.is_anchor ? "active" : ""}`;
          item.innerHTML = `<img src="${kf.thumb_url}" alt="${kf.pts_time}s" loading="lazy"><span>${Number(kf.pts_time).toFixed(1)}s (#${kf.source_frame_index})</span>`;
          item.onclick = () => {
            currentOrdinal = kf.source_frame_index;
            seekPlayerSafely(kf.pts_time, false);
            loadExactFrame(videoId, kf.source_frame_index);
            strip.querySelectorAll(".filmstrip-item").forEach((el) => el.classList.remove("active"));
            item.classList.add("active");
          };
          strip.appendChild(item);
        });
      } catch (err) {
        strip.innerHTML = `<span style='color:var(--text-muted); font-size:12px;'>Không thể tải filmstrip: ${escapeHtml(err.message)}</span>`;
      }
    }

    async function searchSimilar(videoId, tsSec, frameOrd) {
      const topK = parseInt(document.getElementById("topKSelect").value, 10);
      const rankMode = document.getElementById("rankModeSelect").value;
      const dataset = document.getElementById("datasetInput").value.trim();
      const filters = dataset ? { dataset } : {};
      showErrorBanner("");
      setLoading(true);
      document.getElementById("resultsMeta").textContent = `Đang tìm cảnh tương tự ${videoId} @ ${Number(tsSec).toFixed(1)}s...`;
      try {
        const resp = await fetch("/v1/search/similar", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            video_id: videoId,
            timestamp_seconds: Number(tsSec),
            frame_index: Number(frameOrd),
            top_k: topK,
            ranking_mode: rankMode,
            filters,
          }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        currentResults = data.results || [];
        renderDegradedBanner(data);
        document.getElementById("resultsMeta").textContent = `${currentResults.length} cảnh tương tự với ${videoId} • Status: ${data.status}`;
        document.getElementById("timingsMeta").textContent = data.timings && data.timings.elapsed_ms != null ? `${data.timings.elapsed_ms} ms` : "";
        renderQueryGraph(null);
        renderGrid(currentResults);
      } catch (err) {
        showErrorBanner(`Lỗi tìm cảnh tương tự: ${err.message}`);
      } finally {
        setLoading(false);
      }
    }

    async function exportData(format) {
      if (!currentResults || currentResults.length === 0) {
        showToast("Chưa có kết quả để xuất.");
        return;
      }
      try {
        const resp = await fetch("/v1/exports", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            results: currentResults,
            format: format,
          }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const blob = await resp.blob();
        const a = document.createElement("a");
        const objUrl = URL.createObjectURL(blob);
        a.href = objUrl;
        const ext = format === "json" ? "json" : "csv";
        a.download = `videntra_${format}.${ext}`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(objUrl), 1000);
        showToast(`Đã xuất file ${format.toUpperCase()}`);
      } catch (err) {
        showToast("Lỗi xuất file: " + err.message);
      }
    }
  </script>
</body>
</html>
"""


def render_ui_html() -> str:
    """Return the embedded standalone Search & Playback Inspector HTML."""
    return UI_HTML

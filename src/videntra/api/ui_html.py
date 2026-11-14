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
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
    .header { margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
    .header h1 { font-size: 24px; font-weight: 700; color: var(--primary); letter-spacing: -0.5px; }
    .header .subtitle { font-size: 13px; color: var(--text-muted); }

    .search-panel { background: var(--card-bg); padding: 20px; border-radius: 12px; border: 1px solid var(--border); margin-bottom: 20px; }
    .query-box { display: flex; gap: 12px; margin-bottom: 16px; }
    .query-input { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border); background: #0b1120; color: var(--text); font-size: 15px; outline: none; }
    .query-input:focus { border-color: var(--primary); }
    .btn { padding: 12px 24px; border-radius: 8px; background: var(--primary); color: #0b1120; font-weight: 600; cursor: pointer; border: none; font-size: 14px; transition: background 0.2s; }
    .btn:hover { background: var(--primary-hover); }
    .btn-secondary { background: var(--border); color: var(--text); margin-left: 8px; }
    .btn-secondary:hover { background: #475569; }

    .controls { display: flex; gap: 16px; align-items: center; font-size: 13px; color: var(--text-muted); flex-wrap: wrap; }
    .select-ctl { padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border); background: #0b1120; color: var(--text); font-size: 13px; }

    /* QueryGraph Inspector Panel */
    .graph-panel { display: none; background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 20px; }
    .graph-title { font-size: 13px; font-weight: 600; color: var(--primary); margin-bottom: 10px; display: flex; justify-content: space-between; }
    .chip { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin: 2px 4px 2px 0; background: var(--card-bg); border: 1px solid var(--border); }
    .chip-entity { border-color: #38bdf8; color: #38bdf8; }
    .chip-pred { border-color: #a855f7; color: #c084fc; }
    .chip-logic { border-color: #f59e0b; color: #fbbf24; }

    .meta-bar { font-size: 13px; color: var(--text-muted); margin-bottom: 16px; display: flex; justify-content: space-between; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
    .card { background: var(--card-bg); border-radius: 10px; border: 1px solid var(--border); overflow: hidden; display: flex; flex-direction: column; transition: transform 0.15s, border-color 0.15s; }
    .card:hover { transform: translateY(-2px); border-color: var(--primary); }
    .thumb { width: 100%; height: 160px; background: #020617; display: flex; align-items: center; justify-content: center; color: var(--text-muted); font-size: 13px; object-fit: contain; }
    .card-body { padding: 16px; flex: 1; display: flex; flex-direction: column; gap: 8px; font-size: 13px; }
    .vid-title { font-weight: 600; color: var(--primary); word-break: break-all; font-family: monospace; font-size: 12px; }
    .time-info { color: var(--text); font-weight: 500; }
    .frame-info { color: var(--text-muted); font-size: 12px; }
    .score-badge { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-top: 4px; flex-wrap: wrap; }
    .badge { padding: 2px 8px; border-radius: 4px; font-weight: 600; text-transform: uppercase; font-size: 10px; }
    .badge-supported { background: rgba(34, 197, 94, 0.2); color: var(--badge-supported); }
    .badge-partial { background: rgba(234, 179, 8, 0.2); color: var(--badge-partial); }
    .badge-contradicted { background: rgba(239, 68, 68, 0.2); color: var(--badge-contradicted); }
    .card-footer { padding: 12px 16px; border-top: 1px solid var(--border); display: flex; justify-content: space-between; }

    /* Modal Inspector */
    .modal-overlay { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.85); display: none; align-items: center; justify-content: center; z-index: 100; backdrop-filter: blur(4px); }
    .modal { background: var(--card-bg); width: 90%; max-width: 920px; border-radius: 12px; border: 1px solid var(--border); overflow: hidden; display: flex; flex-direction: column; max-height: 90vh; }
    .modal-header { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
    .modal-body { padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
    .video-container { background: #000; border-radius: 8px; overflow: hidden; display: flex; justify-content: center; }
    video { max-width: 100%; max-height: 400px; }
    .frame-preview { border-radius: 8px; border: 1px solid var(--border); max-width: 100%; max-height: 250px; object-fit: contain; background: #000; }
    .inspector-controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .kbd { background: #0b1120; border: 1px solid var(--border); padding: 2px 6px; border-radius: 4px; font-size: 11px; font-family: monospace; }
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>VIDENTRA</h1>
      <div class="subtitle">Multi-Modal Video Retrieval Engine • Snapshot: current</div>
    </div>
    <div>
      <button class="btn btn-secondary" onclick="exportData('csv')">Export CSV</button>
      <button class="btn btn-secondary" onclick="exportData('json')">Export JSON</button>
    </div>
  </div>

  <div class="search-panel">
    <div class="query-box">
      <input type="text" id="queryInput" class="query-input" placeholder="Tìm người mặc áo đỏ, xe máy rẽ trái, chữ SALE, tiếng nói..." autofocus>
      <button class="btn" onclick="performSearch()">Tìm kiếm</button>
    </div>
    <div class="controls">
      <label>Top-K:
        <select id="topKSelect" class="select-ctl">
          <option value="5">5</option>
          <option value="10" selected>10</option>
          <option value="20">20</option>
          <option value="50">50</option>
        </select>
      </label>
      <label>Chế độ:
        <select id="modeSelect" class="select-ctl">
          <option value="fast">Fast (Visual Vector)</option>
          <option value="accurate" selected>Accurate (QueryGraph + Multimodal)</option>
        </select>
      </label>
      <label>Xếp hạng:
        <select id="rankModeSelect" class="select-ctl">
          <option value="moment" selected>Moment Ranking</option>
          <option value="diverse_video">Diverse Video</option>
        </select>
      </label>
    </div>
  </div>

  <!-- QueryGraph Inspector -->
  <div class="graph-panel" id="graphPanel">
    <div class="graph-title">
      <span>📐 Phân tích Truy vấn (QueryGraph v1)</span>
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

  <!-- Modal Inspector -->
  <div class="modal-overlay" id="modalOverlay" onclick="closeModalOnBg(event)">
    <div class="modal">
      <div class="modal-header">
        <div>
          <strong id="modalVideoId" style="color:var(--primary); font-family:monospace; font-size:13px;"></strong>
          <div id="modalTiming" style="font-size:12px; color:var(--text-muted); margin-top:2px;"></div>
        </div>
        <button class="btn btn-secondary" style="padding:6px 12px;" onclick="closeModal()">Đóng (Esc)</button>
      </div>
      <div class="modal-body">
        <div class="video-container">
          <video id="player" controls preload="metadata"></video>
        </div>

        <div class="inspector-controls">
          <button class="btn" style="padding:6px 14px;" onclick="replayMoment()">Replay Moment (<span class="kbd">R</span>)</button>
          <button class="btn btn-secondary" style="padding:6px 14px;" onclick="stepFrame(-1)">−1 Frame (<span class="kbd">,</span>)</button>
          <button class="btn btn-secondary" style="padding:6px 14px;" onclick="stepFrame(1)">+1 Frame (<span class="kbd">.</span>)</button>
          <button class="btn btn-secondary" style="padding:6px 14px;" onclick="copyInfo()">Copy Video + Frame</button>
          <span style="font-size:12px; color:var(--text-muted); margin-left:auto;">
            Phím tắt: <span class="kbd">Space</span> Play/Pause • <span class="kbd">Esc</span> Đóng
          </span>
        </div>

        <div id="modalEvidenceBox" style="background:#0b1120; border:1px solid var(--border); border-radius:8px; padding:12px; font-size:12px;">
          <strong style="color:var(--primary);">Bằng chứng đa phương thức (Evidence Items):</strong>
          <div id="modalEvidenceList" style="margin-top:6px; display:flex; flex-wrap:wrap; gap:6px;"></div>
        </div>

        <div style="border-top:1px solid var(--border); padding-top:12px;">
          <h4 style="font-size:13px; color:var(--primary); margin-bottom:8px;">Exact Bit-Level Decoded Frame (Ledger Verification)</h4>
          <div style="display:flex; gap:16px; align-items:center;">
            <img id="exactFrameImg" class="frame-preview" alt="Exact Frame">
            <div id="frameDetails" style="font-size:12px; color:var(--text-muted); line-height:1.6;"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentResults = [];
    let activeMoment = null;
    let currentOrdinal = 0;

    document.getElementById("queryInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter") performSearch();
    });

    window.addEventListener("keydown", (e) => {
      if (document.getElementById("modalOverlay").style.display === "flex") {
        if (e.key === "Escape") closeModal();
        if (e.key === "r" || e.key === "R") replayMoment();
        if (e.key === ",") stepFrame(-1);
        if (e.key === ".") stepFrame(1);
      }
    });

    async function performSearch() {
      const query = document.getElementById("queryInput").value.trim();
      if (!query) return;

      const topK = parseInt(document.getElementById("topKSelect").value);
      const mode = document.getElementById("modeSelect").value;
      const rankMode = document.getElementById("rankModeSelect").value;

      document.getElementById("resultsMeta").textContent = "Đang tìm kiếm...";

      try {
        const resp = await fetch("/v1/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: topK, mode, ranking_mode: rankMode })
        });
        const data = await resp.json();
        currentResults = data.results || [];

        document.getElementById("resultsMeta").textContent = `${currentResults.length} kết quả cho "${query}" • Status: ${data.status}`;
        document.getElementById("timingsMeta").textContent = `${data.timings.elapsed_ms} ms`;

        // Render QueryGraph panel if available
        renderQueryGraph(data.query_graph);

        renderGrid(currentResults);
      } catch (err) {
        document.getElementById("resultsMeta").textContent = `Lỗi tìm kiếm: ${err.message}`;
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
      statusBadge.textContent = qg.parser?.status || "unknown";
      statusBadge.className = `badge ${qg.parser?.status === "validated" ? "badge-supported" : "badge-partial"}`;

      const entDiv = document.getElementById("graphEntities");
      entDiv.innerHTML = "<strong>Thực thể:</strong> ";
      if (qg.entities && qg.entities.length > 0) {
        qg.entities.forEach(e => {
          entDiv.innerHTML += `<span class="chip chip-entity">${e.mention} (${e.type})</span>`;
        });
      } else {
        entDiv.innerHTML += "<span style='color:var(--text-muted); font-size:12px;'>Không có thực thể riêng lẻ</span>";
      }

      const predDiv = document.getElementById("graphPredicates");
      predDiv.innerHTML = "<strong>Vị từ:</strong> ";
      if (qg.predicates && qg.predicates.length > 0) {
        qg.predicates.forEach(p => {
          const val = p.value ? `: "${p.value}"` : "";
          const pol = p.polarity === "negative" ? " [NOT]" : "";
          predDiv.innerHTML += `<span class="chip chip-pred">${p.kind}${pol}${val}</span>`;
        });
      } else {
        predDiv.innerHTML += "<span style='color:var(--text-muted); font-size:12px;'>Không có ràng buộc vị từ</span>";
      }
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
        const thumbUrl = `/v1/frames/${encodeURIComponent(item.video_id)}/${item.representative_original_frame_index || 0}`;
        const startSec = item.moment_start_seconds.toFixed(2);
        const endSec = item.moment_end_seconds.toFixed(2);
        const score = item.ranking_score.toFixed(4);

        let badgeClass = "badge-partial";
        if (item.match_status === "supported") badgeClass = "badge-supported";
        else if (item.match_status === "contradicted") badgeClass = "badge-contradicted";

        const unresolvedWarn = (item.unresolved_predicates && item.unresolved_predicates.length > 0)
          ? `<div style="font-size:11px; color:#f87171;">⚠️ Chưa giải quyết: ${item.unresolved_predicates.join(", ")}</div>`
          : "";

        card.innerHTML = `
          <img src="${thumbUrl}" class="thumb" alt="Thumbnail" loading="lazy" onerror="this.src=''; this.alt='Không có thumbnail';">
          <div class="card-body">
            <div class="vid-title" title="${item.video_id}">${item.video_id}</div>
            <div class="time-info">⏱ ${startSec}s → ${endSec}s (${(endSec - startSec).toFixed(2)}s)</div>
            <div class="frame-info">Frame ordinal: ${item.start_original_frame_index} → ${item.end_original_frame_index} (rep: ${item.representative_original_frame_index})</div>
            <div class="score-badge">
              <span class="badge ${badgeClass}">${item.match_status}</span>
              <span style="color:var(--text-muted);">Rank score: <strong>${score}</strong></span>
            </div>
            ${unresolvedWarn}
          </div>
          <div class="card-footer">
            <button class="btn btn-secondary" style="padding:4px 10px; font-size:12px;" onclick='openInspector(${idx})'>Mở Inspector</button>
            <button class="btn btn-secondary" style="padding:4px 10px; font-size:12px;" onclick='copySingle("${item.video_id}", ${item.representative_original_frame_index})'>Copy ID</button>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    function openInspector(idx) {
      const item = currentResults[idx];
      activeMoment = item;
      currentOrdinal = item.representative_original_frame_index || item.start_original_frame_index || 0;

      document.getElementById("modalVideoId").textContent = item.video_id;
      document.getElementById("modalTiming").textContent = `Khoảnh khắc: ${item.moment_start_seconds}s → ${item.moment_end_seconds}s | Rank Score: ${item.ranking_score} | Match: ${item.match_status}`;

      // Populate evidence items
      const evList = document.getElementById("modalEvidenceList");
      evList.innerHTML = "";
      if (item.evidence && item.evidence.length > 0) {
        item.evidence.forEach(ev => {
          evList.innerHTML += `<span class="chip chip-logic">${ev.kind}: ${ev.evidence_id || ev.timestamp_us + 'µs'}</span>`;
        });
      } else {
        evList.innerHTML = "<span style='color:var(--text-muted);'>Chỉ có bằng chứng khung hình đại diện.</span>";
      }

      const player = document.getElementById("player");
      player.src = `/v1/media/${encodeURIComponent(item.video_id)}`;
      player.currentTime = item.moment_start_seconds;

      loadExactFrame(item.video_id, currentOrdinal);

      document.getElementById("modalOverlay").style.display = "flex";
    }

    function closeModal() {
      const player = document.getElementById("player");
      player.pause();
      player.src = "";
      document.getElementById("modalOverlay").style.display = "none";
    }

    function closeModalOnBg(e) {
      if (e.target.id === "modalOverlay") closeModal();
    }

    function replayMoment() {
      if (!activeMoment) return;
      const player = document.getElementById("player");
      player.currentTime = activeMoment.moment_start_seconds;
      player.play();
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
        img.src = URL.createObjectURL(blob);

        details.innerHTML = `
          <div><strong>Ordinal:</strong> ${ordinal}</div>
          <div><strong>PTS:</strong> ${pts}</div>
          <div><strong>Timestamp:</strong> ${tsUs} µs (${(parseInt(tsUs)/1e6).toFixed(4)}s)</div>
          <div><strong>Revision:</strong> ${rev}</div>
        `;
      } catch (err) {
        details.innerHTML = `<span style="color:#f87171;">Lỗi: ${err.message}</span>`;
      }
    }

    function copyInfo() {
      if (!activeMoment) return;
      const text = `${activeMoment.video_id}:${currentOrdinal} [${activeMoment.moment_start_seconds}-${activeMoment.moment_end_seconds}s]`;
      navigator.clipboard.writeText(text);
      alert("Đã sao chép: " + text);
    }

    function copySingle(vid, ord) {
      const text = `${vid}:${ord}`;
      navigator.clipboard.writeText(text);
      alert("Đã sao chép: " + text);
    }

    async function exportData(format) {
      if (!currentResults || currentResults.length === 0) {
        alert("Chưa có kết quả để xuất.");
        return;
      }
      try {
        const resp = await fetch(`/v1/exports?format=${format}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(currentResults)
        });
        const blob = await resp.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `videntra_export.${format}`;
        a.click();
      } catch (err) {
        alert("Lỗi xuất file: " + err.message);
      }
    }
  </script>
</body>
</html>
"""

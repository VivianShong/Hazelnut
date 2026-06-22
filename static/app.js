"use strict";

const $ = (sel) => document.querySelector(sel);

const STAGE_LABELS = {
  propose: "Propose",
  patch: "Patch",
  train: "Train",
  evaluate: "Evaluate",
  compare: "Compare",
  commit: "Commit",
};

// ---- API helpers ---------------------------------------------------------
async function api(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  return res.json();
}

// ---- Controls ------------------------------------------------------------
$("#btn-start").addEventListener("click", async () => {
  const raw = $("#max-runs").value.trim();
  const max_runs = raw === "" ? null : parseInt(raw, 10);
  await api("/api/start", "POST", { max_runs });
  refresh();
});

$("#btn-stop").addEventListener("click", async () => {
  await api("/api/stop", "POST");
  refresh();
});

$("#btn-restore").addEventListener("click", async () => {
  const r = await api("/api/restore_best", "POST");
  if (!r.ok) alert("No best model yet to switch to.");
});

// ---- Renderers -----------------------------------------------------------
function renderBadge(s) {
  const badge = $("#status-badge");
  badge.textContent = s.running ? "running" : "idle";
  badge.className = "badge " + (s.running ? "running" : "idle");
  $("#btn-start").disabled = s.running;
  $("#btn-stop").disabled = !s.running;
}

function renderPipeline(s) {
  const activeIdx = s.stages.indexOf(s.stage);
  const html = s.stages
    .map((stage, i) => {
      let cls = "stage";
      if (s.running && i === activeIdx) cls += " active";
      else if (activeIdx >= 0 && i < activeIdx) cls += " done";
      return `<div class="${cls}"><span class="num">${i + 1}</span>${STAGE_LABELS[stage] || stage}</div>`;
    })
    .join("");
  $("#pipeline").innerHTML = html;
}

function renderCurrent(s) {
  const meta = $("#current-meta");
  const bar = $("#progress-bar");
  const label = $("#progress-label");
  const stats = $("#live-stats");

  if (!s.current) {
    meta.textContent = "No experiment running.";
    bar.style.width = "0%";
    label.textContent = "0%";
    stats.innerHTML = "";
    return;
  }

  const c = s.current;
  const cfg = Object.entries(c.config || {})
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
  meta.innerHTML = `Run <b>#${c.run_id}</b> · <span class="variant">${c.variant}</span><br/><small>${cfg}</small>`;

  const live = s.live || {};
  const pct = live.pct != null ? live.pct : 0;
  bar.style.width = pct + "%";
  label.textContent = pct.toFixed(1) + "%";

  const cards = [
    ["step", live.step ?? "—"],
    ["loss", live.loss != null ? live.loss.toFixed(4) : "—"],
    ["mfu %", live.mfu != null ? live.mfu.toFixed(1) : "—"],
    ["tok/sec", live.tok_per_sec != null ? live.tok_per_sec.toLocaleString() : "—"],
    ["epoch", live.epoch ?? "—"],
    ["remaining", live.remaining != null ? live.remaining + "s" : "—"],
  ];
  stats.innerHTML = cards
    .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");
}

function renderBest(s) {
  const el = $("#best-panel");
  if (!s.best) {
    el.textContent = "No results yet.";
    return;
  }
  const b = s.best;
  const cfg = Object.entries(b.config || {})
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
  el.innerHTML = `
    <div class="bpb">${b.val_bpb.toFixed(6)}</div>
    <div class="row">Run <b>#${b.run_id}</b> · <b>${b.variant}</b></div>
    <div class="row">VRAM: <b>${b.peak_vram_mb != null ? b.peak_vram_mb.toFixed(0) : "—"} MB</b> ·
      Params: <b>${b.num_params_M != null ? b.num_params_M.toFixed(1) : "—"}M</b></div>
    <div class="row"><small>${cfg}</small></div>`;
}

function renderHistory(s) {
  const tbody = $("#history tbody");
  if (!s.history.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="color:var(--muted)">No runs yet.</td></tr>`;
    return;
  }
  const bestId = s.best ? s.best.run_id : -1;
  tbody.innerHTML = s.history
    .slice()
    .reverse()
    .map((r) => {
      const isBest = r.run_id === bestId ? " best-row" : "";
      const bpb = r.val_bpb != null ? r.val_bpb.toFixed(6) : "—";
      const vram = r.peak_vram_mb != null ? r.peak_vram_mb.toFixed(0) : "—";
      return `<tr class="${isBest.trim()}">
        <td class="num">${r.run_id}</td>
        <td>${r.variant}</td>
        <td class="num">${bpb}</td>
        <td class="num">${vram}</td>
        <td><span class="tag ${r.status}">${r.status}</span></td>
        <td>${r.note || ""}</td>
      </tr>`;
    })
    .join("");
}

// ---- Chart (no dependencies) --------------------------------------------
function renderChart(s) {
  const canvas = $("#chart");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 800;
  const cssH = 220;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const pts = s.history.filter((r) => r.val_bpb != null && r.val_bpb > 0);
  if (pts.length === 0) {
    ctx.fillStyle = "#8b949e";
    ctx.font = "13px sans-serif";
    ctx.fillText("Waiting for results…", 12, 24);
    return;
  }

  const pad = { l: 50, r: 16, t: 14, b: 26 };
  const w = cssW - pad.l - pad.r;
  const h = cssH - pad.t - pad.b;
  const vals = pts.map((p) => p.val_bpb);
  let min = Math.min(...vals);
  let max = Math.max(...vals);
  if (min === max) { min -= 0.01; max += 0.01; }
  const range = max - min;
  min -= range * 0.1;
  max += range * 0.1;

  const x = (i) => pad.l + (pts.length === 1 ? w / 2 : (i / (pts.length - 1)) * w);
  const y = (v) => pad.t + (1 - (v - min) / (max - min)) * h;

  // grid + y labels
  ctx.strokeStyle = "#2a313c";
  ctx.fillStyle = "#8b949e";
  ctx.font = "11px monospace";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const yy = pad.t + (g / 4) * h;
    const val = max - (g / 4) * (max - min);
    ctx.beginPath();
    ctx.moveTo(pad.l, yy);
    ctx.lineTo(cssW - pad.r, yy);
    ctx.stroke();
    ctx.fillText(val.toFixed(3), 6, yy + 3);
  }

  // best line
  const bestV = Math.min(...vals);
  ctx.strokeStyle = "rgba(63,185,80,0.5)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, y(bestV));
  ctx.lineTo(cssW - pad.r, y(bestV));
  ctx.stroke();
  ctx.setLineDash([]);

  // line
  ctx.strokeStyle = "#4f9dff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach((p, i) => {
    const px = x(i), py = y(p.val_bpb);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();

  // points
  pts.forEach((p, i) => {
    ctx.fillStyle = p.val_bpb === bestV ? "#3fb950" : "#4f9dff";
    ctx.beginPath();
    ctx.arc(x(i), y(p.val_bpb), p.val_bpb === bestV ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#8b949e";
    ctx.fillText("#" + p.run_id, x(i) - 8, cssH - 8);
  });
}

function renderLog(s) {
  const log = $("#log");
  log.textContent = (s.log_tail || []).join("\n");
  log.scrollTop = log.scrollHeight;
}

// ---- Poll loop -----------------------------------------------------------
async function refresh() {
  try {
    const s = await api("/api/status");
    $("#message").textContent = s.message || "";
    renderBadge(s);
    renderPipeline(s);
    renderCurrent(s);
    renderBest(s);
    renderHistory(s);
    renderChart(s);
    renderLog(s);
  } catch (e) {
    $("#message").textContent = "Connection lost — is dashboard.py running?";
  }
}

refresh();
setInterval(refresh, 1000);

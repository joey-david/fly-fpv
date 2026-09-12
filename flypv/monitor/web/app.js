/* flypv training monitor.
 *
 * Three live views share one websocket:
 *   - the connectome, drawn at real soma coordinates and rotatable
 *   - the compound eye, one hexagon per ommatidium, split into ON and OFF
 *   - the course, with the fly's trail through it
 *
 * Neuron activity arrives as base64 uint8 rather than JSON numbers: 14,000
 * floats twenty times a second is megabytes of parsing, and one byte is finer
 * than a 2-pixel dot can show.
 */
const ROLE = [
  ["Intrinsic",  "#465671"],
  ["Retina",     "#4fd2e8"],
  ["Sensory",    "#7c6bf5"],
  ["Motor",      "#ff5e7a"],
  ["Descending", "#ffb03a"],
  ["Ascending",  "#43c08a"],
];

const S = {
  graph: null, eye: null, info: null,
  act: null, actHi: 1, eyeFrame: null, flight: null, pools: null,
  live: {}, metrics: {}, hist: {},
};

const el = (id) => document.getElementById(id);
const b64 = (s) => {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
};
const fmt = (v, d = 2) =>
  v == null || Number.isNaN(v) ? "—" :
  Math.abs(v) >= 10000 ? v.toLocaleString(undefined, { maximumFractionDigits: 0 })
                       : v.toFixed(d);

/* ------------------------------------------------------------------ socket */
async function seedHistory() {
  // The server keeps the full series, so a page opened or reloaded part-way
  // through a run shows the whole curve instead of starting blank.
  try {
    const r = await fetch("/api/series");
    const d = await r.json();
    for (const k of ["ep_return", "ep_gates", "crash_rate", "entropy", "difficulty"]) {
      if (d[k]) S.hist[k] = d[k].slice(-600);
    }
    if (d.step && d.step.length) lastStep = d.step[d.step.length - 1];
  } catch (_) { /* trainer not up yet; frames will fill it in */ }
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { el("conn").classList.add("live"); seedHistory(); };
  ws.onclose = () => {
    el("conn").classList.remove("live");
    el("conn").lastChild.textContent = "reconnecting";
    setTimeout(connect, 1200);
  };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.kind === "static") { takeStatic(m); return; }
    if (m.act)       { S.act = b64(m.act.data); S.actHi = m.act.hi; }
    if (m.eye_frame) S.eyeFrame = m.eye_frame;
    if (m.flight)    S.flight = m.flight;
    if (m.pools)     S.pools = m.pools;
    if (m.live)      S.live = m.live;
    if (m.metrics)   { S.metrics = m.metrics; pushHistory(m.metrics); }
    el("conn").lastChild.textContent = "live";
  };
}

let lastStep = -1;
function pushHistory(m) {
  // one point per PPO update, not per websocket frame — otherwise the same
  // value is repeated ~300 times between updates and the curve is a staircase
  if (m.step == null || m.step === lastStep) return;
  lastStep = m.step;
  for (const k of ["ep_return", "ep_gates", "crash_rate", "entropy", "difficulty"]) {
    if (m[k] == null) continue;
    (S.hist[k] ||= []).push(m[k]);
    if (S.hist[k].length > 600) S.hist[k].shift();
  }
}

function takeStatic(m) {
  if (m.graph) { S.graph = m.graph; prepareGraph(); }
  if (m.eye)   { S.eye = m.eye; prepareEye(); }
  if (m.info)  { S.info = m.info; renderHeader(); renderKey(); }
}

/* ------------------------------------------------------------------ header */
function renderHeader() {
  const i = S.info;
  el("dataset").textContent =
    `${i.dataset} — ${i.neurons.toLocaleString()} neurons, ` +
    `${i.connections.toLocaleString()} connections, ${i.synapses.toLocaleString()} synapses`;
  el("graph-note").textContent =
    `Drag to rotate. ${i.displayed.toLocaleString()} of ${i.neurons.toLocaleString()} neurons shown, ` +
    `coloured by role; edge colour is the transmitter's sign.`;
}

function renderCounters() {
  const m = S.metrics, l = S.live, i = S.info || {};
  const rows = [
    ["Env steps", m.step ? Math.round(m.step).toLocaleString() : "—"],
    ["Return", fmt(m.ep_return)],
    [`Gates of ${i.n_gates ?? "—"}`, fmt(m.ep_gates, 2)],
    ["Crashes", m.crash_rate == null ? "—" : `${Math.round(m.crash_rate * 100)}%`],
    ["Course difficulty", fmt(m.difficulty, 2)],
    ["Airspeed", l.speed == null ? "—" : `${fmt(l.speed * 100, 1)} cm/s`],
    ["Power vs hover", l.power_rel == null ? "—" : `${fmt(l.power_rel, 2)}×`],
    ["Steps/s", fmt(m.fps, 0)],
  ];
  el("counters").innerHTML = rows
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

function renderKey() {
  el("key").innerHTML = ROLE.map(([n, c], i) => {
    const counts = S.graph ? S.graph.role.filter((r) => r === i).length : 0;
    return `<li><b style="background:${c}"></b>${n}<span>${counts.toLocaleString()}</span></li>`;
  }).join("") +
    `<li><b style="background:var(--exc)"></b>Excitatory synapse</li>` +
    `<li><b style="background:var(--inh)"></b>Inhibitory synapse</li>`;
}

/* ------------------------------------------------------ connectome drawing */
const G = { yaw: -0.35, pitch: 0.12, drag: null, sub: null, dirty: true, proj: null };

function fitCanvas(c) {
  const r = c.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) {
    c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
    return true;
  }
  return false;
}

function prepareGraph() {
  const g = S.graph, n = g.n;
  // centre and scale the anatomy once
  const cx = avg(g.x), cy = avg(g.y), cz = avg(g.z);
  g.px = new Float32Array(n); g.py = new Float32Array(n); g.pz = new Float32Array(n);
  for (let i = 0; i < n; i++) { g.px[i] = g.x[i] - cx; g.py[i] = g.y[i] - cy; g.pz[i] = g.z[i] - cz; }
  let ext = 1;
  for (let i = 0; i < n; i++) ext = Math.max(ext, Math.abs(g.px[i]), Math.abs(g.py[i]), Math.abs(g.pz[i]));
  g.ext = ext;
  G.proj = { sx: new Float32Array(n), sy: new Float32Array(n), depth: new Float32Array(n) };
  G.sub = document.createElement("canvas");
  G.dirty = true;
}
const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;

function project(w, h) {
  const g = S.graph, p = G.proj;
  const cy = Math.cos(G.yaw), sy = Math.sin(G.yaw);
  const cp = Math.cos(G.pitch), sp = Math.sin(G.pitch);
  // the fly's long axis (z in dataset coordinates) runs down the screen, so the
  // brain sits above the nerve cord the way it does in the animal
  const k = Math.min(w, h) * 0.46 / g.ext;
  for (let i = 0; i < g.n; i++) {
    const x = g.px[i], y = g.py[i], z = g.pz[i];
    const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
    const y2 = y1 * cp - z * sp, z2 = y1 * sp + z * cp;
    p.sx[i] = w / 2 + x1 * k;
    p.sy[i] = h / 2 + z2 * k;
    p.depth[i] = y2;
  }
}

function drawEdges(w, h, dpr) {
  const g = S.graph, p = G.proj, c = G.sub;
  c.width = w; c.height = h;
  const x = c.getContext("2d");
  x.clearRect(0, 0, w, h);
  x.lineWidth = Math.max(0.5, 0.55 * dpr);
  const E = g.edges.length / 2;
  for (const wantExc of [0, 1]) {
    x.beginPath();
    x.strokeStyle = wantExc ? "rgba(232,196,106,0.075)" : "rgba(91,140,255,0.085)";
    for (let e = 0; e < E; e++) {
      if (g.edge_sign[e] !== wantExc) continue;
      const a = g.edges[2 * e], b = g.edges[2 * e + 1];
      x.moveTo(p.sx[a], p.sy[a]); x.lineTo(p.sx[b], p.sy[b]);
    }
    x.stroke();
  }
}

function drawGraph() {
  const c = el("graph"); if (!S.graph) return;
  const resized = fitCanvas(c);
  const w = c.width, h = c.height, dpr = Math.min(devicePixelRatio || 1, 2);
  const ctx = c.getContext("2d");

  if (resized || G.dirty) { project(w, h); drawEdges(w, h, dpr); G.dirty = false; }

  ctx.fillStyle = "#05070b";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(G.sub, 0, 0);

  const g = S.graph, p = G.proj, act = S.act;
  const r = 1.5 * dpr;
  // draw quiet neurons first so active ones sit on top
  for (const pass of [0, 1]) {
    for (let i = 0; i < g.n; i++) {
      const a = act ? act[i] / 255 : 0;
      const hot = a > 0.28;
      if ((pass === 1) !== hot) continue;
      const role = g.role[i];
      const col = ROLE[role][1];
      if (pass === 0) {
        ctx.globalAlpha = role === 0 ? 0.30 + a * 0.5 : 0.55 + a * 0.4;
        ctx.fillStyle = col;
        ctx.fillRect(p.sx[i] - r / 2, p.sy[i] - r / 2, r, r);
      } else {
        ctx.globalAlpha = Math.min(1, 0.45 + a);
        ctx.fillStyle = col;
        const rr = r + a * 2.4 * dpr;
        ctx.beginPath(); ctx.arc(p.sx[i], p.sy[i], rr / 2, 0, 6.283); ctx.fill();
      }
    }
  }
  ctx.globalAlpha = 1;
}

(function graphDrag() {
  const c = el("graph");
  const down = (e) => { G.drag = { x: e.clientX, y: e.clientY, yaw: G.yaw, pitch: G.pitch }; };
  const move = (e) => {
    if (!G.drag) return;
    G.yaw = G.drag.yaw + (e.clientX - G.drag.x) * 0.006;
    G.pitch = Math.max(-1.3, Math.min(1.3, G.drag.pitch + (e.clientY - G.drag.y) * 0.006));
    G.dirty = true;
  };
  const up = () => { G.drag = null; };
  c.addEventListener("pointerdown", down);
  addEventListener("pointermove", move);
  addEventListener("pointerup", up);

  // hover readout: which neuron is this, and what is it doing right now
  c.addEventListener("pointermove", (e) => {
    if (G.drag || !S.graph || !G.proj) return;
    const r = c.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    const mx = (e.clientX - r.left) * dpr, my = (e.clientY - r.top) * dpr;
    const p = G.proj, g = S.graph;
    let best = -1, bd = 100 * dpr * dpr;
    for (let i = 0; i < g.n; i++) {
      const dx = p.sx[i] - mx, dy = p.sy[i] - my, d2 = dx * dx + dy * dy;
      if (d2 < bd) { bd = d2; best = i; }
    }
    const tip = el("tip");
    if (best < 0) { tip.hidden = true; return; }
    const a = S.act ? (S.act[best] / 255 * S.actHi) : 0;
    tip.hidden = false;
    tip.style.left = `${e.clientX - r.left + 12}px`;
    tip.style.top = `${e.clientY - r.top + 12}px`;
    tip.innerHTML = `<b style="background:${ROLE[g.role[best]][1]}"></b>` +
      `${g.types[best] || "unnamed"} <span>${ROLE[g.role[best]][0].toLowerCase()}</span>` +
      `<span>rate ${a.toFixed(3)}</span>`;
  });
  c.addEventListener("pointerleave", () => { el("tip").hidden = true; });
})();

/* ------------------------------------------------------------- compound eye */
const EY = { pts: null };
function prepareEye() {
  const e = S.eye;
  EY.pts = { R: [], L: [] };
  for (let i = 0; i < e.n; i++) (e.side[i] ? EY.pts.R : EY.pts.L).push(i);
  // The ommatidial field is a diagonal band in hex coordinates (h1 and h2
  // correlate at ~0.57), so drawn raw it is a thin slash across the panel.
  // Rotate each eye onto its own principal axis — a change of viewing angle
  // only; neighbours stay neighbours and the lattice is untouched.
  EY.xy = { };
  for (const side of ["L", "R"]) {
    const ids = EY.pts[side];
    const mx = ids.reduce((s2, i) => s2 + e.x[i], 0) / ids.length;
    const my = ids.reduce((s2, i) => s2 + e.y[i], 0) / ids.length;
    let sxx = 0, syy = 0, sxy = 0;
    for (const i of ids) {
      const dx = e.x[i] - mx, dy = e.y[i] - my;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const ct = Math.cos(-th), st = Math.sin(-th);
    const X = new Float32Array(e.n), Y = new Float32Array(e.n);
    for (const i of ids) {
      const dx = e.x[i] - mx, dy = e.y[i] - my;
      X[i] = dx * ct - dy * st; Y[i] = dx * st + dy * ct;
    }
    EY.xy[side] = { X, Y };
  }
  const norm = (side) => {
    const { X, Y } = EY.xy[side], ids = EY.pts[side];
    const xs = ids.map((i) => X[i]), ys = ids.map((i) => Y[i]);
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  };
  EY.bounds = { R: norm("R"), L: norm("L") };
}

function drawEye() {
  const c = el("eye"); fitCanvas(c);
  const ctx = c.getContext("2d"), w = c.width, h = c.height;
  ctx.fillStyle = "#0a0e15"; ctx.fillRect(0, 0, w, h);
  if (!S.eye || !EY.pts) return;

  const f = S.eyeFrame;
  const data = f ? { lum: b64(f.lum), on: b64(f.on), off: b64(f.off) } : null;
  // one row per channel, left eye then right eye — the way a fly's visual field
  // is actually split, with almost no binocular overlap
  const rows = [
    { key: "lum", label: "Luminance", tint: null },
    { key: "on",  label: "ON  ·  L1, L5, Mi1", tint: [79, 210, 232] },
    { key: "off", label: "OFF  ·  L2, L3, Tm1, Tm2, Tm9", tint: [255, 94, 122] },
  ];
  const rh = h / rows.length, pad = 5 * (devicePixelRatio > 1 ? 2 : 1);

  rows.forEach((row, ri) => {
    const y0 = ri * rh;
    ["L", "R"].forEach((side, si) => {
      const ids = EY.pts[side], b = EY.bounds[side];
      const cw = w / 2;
      const availW = cw - pad * 2, availH = rh - pad * 2;
      const s = Math.min(availW / Math.max(b.x1 - b.x0, 1e-6),
                         availH / Math.max(b.y1 - b.y0, 1e-6));
      const ox = si * cw + (cw - (b.x1 - b.x0) * s) / 2;
      const oy = y0 + rh - pad - (availH - (b.y1 - b.y0) * s) / 2;
      const rad = Math.max(s * 0.60, 0.8);
      for (const i of ids) {
        const v = data ? data[row.key][i] / 255 : 0;
        if (row.tint) {
          const t = row.tint;
          ctx.fillStyle = `rgba(${t[0]},${t[1]},${t[2]},${0.06 + v * 0.94})`;
        } else {
          const g = Math.round(16 + v * 222);
          ctx.fillStyle = `rgb(${g},${g},${Math.min(255, g + 14)})`;
        }
        const XY = EY.xy[side];
        hexAt(ctx, ox + (XY.X[i] - b.x0) * s, oy - (XY.Y[i] - b.y0) * s, rad);
      }
    });
    ctx.strokeStyle = "#161e2b"; ctx.lineWidth = 1;
    if (ri) { ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(w, y0); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(w / 2, y0); ctx.lineTo(w / 2, y0 + rh); ctx.stroke();
  });

  if (!data) {
    ctx.fillStyle = "#3d4a5e";
    ctx.font = `${12 * Math.min(devicePixelRatio || 1, 2)}px "Helvetica Neue", sans-serif`;
    ctx.fillText("waiting for the first flight", 10, h / 2);
  }
}

function hexAt(ctx, x, y, r) {
  ctx.beginPath();
  for (let k = 0; k < 6; k++) {
    const a = (Math.PI / 3) * k + Math.PI / 6;
    const px = x + r * Math.cos(a), py = y + r * Math.sin(a);
    k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  }
  ctx.closePath(); ctx.fill();
}

/* ------------------------------------------------------------------- course */
function drawCourse() {
  const c = el("course"); fitCanvas(c);
  const ctx = c.getContext("2d"), w = c.width, h = c.height;
  ctx.fillStyle = "#0a0e15"; ctx.fillRect(0, 0, w, h);
  const fl = S.flight; if (!fl) return;
  const co = fl.course;

  const pts = co.center.concat([fl.pos], co.center);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]), zs = pts.map((p) => p[2]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 0.4);
  const k = Math.min(w, h * 1.7) * 0.40 / span;
  // plan view with altitude as a vertical offset, so climbs stay readable
  const P = (p) => [w / 2 + (p[0] - cx) * k, h * 0.68 - (p[1] - cy) * k * 0.62 - (p[2] - 0.3) * k * 0.7];

  ctx.lineWidth = 1; ctx.strokeStyle = "#1b2331";
  ctx.beginPath();
  co.center.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(...q) : ctx.moveTo(...q); });
  ctx.stroke();

  co.center.forEach((p, i) => {
    const q = P(p), next = i === fl.next_gate;
    ctx.strokeStyle = next ? "#ffb03a" : (i < fl.next_gate ? "#2b3848" : "#31405a");
    ctx.lineWidth = next ? 2 : 1.2;
    ctx.beginPath();
    ctx.ellipse(q[0], q[1], co.r_in[i] * k, co.r_in[i] * k * 0.45, 0, 0, 6.283);
    ctx.stroke();
  });

  if (fl.trail && fl.trail.length > 1) {
    ctx.strokeStyle = "#4fd2e8"; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.85;
    ctx.beginPath();
    fl.trail.forEach((p, i) => { const q = P(p); i ? ctx.lineTo(...q) : ctx.moveTo(...q); });
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  const q = P(fl.pos);
  ctx.fillStyle = "#ff5e7a";
  ctx.beginPath(); ctx.arc(q[0], q[1], 3.2, 0, 6.283); ctx.fill();

  el("course-note").textContent =
    `gate ${fl.next_gate + 1} of ${co.center.length}, ${fmt(fl.speed * 100, 1)} cm/s`;
}

/* ----------------------------------------------------------------- wingbeat */
function pushWing() { /* the env supplies the waveform; nothing to accumulate */ }

function drawWing() {
  const c = el("wing"); fitCanvas(c);
  const ctx = c.getContext("2d"), w = c.width, h = c.height;
  ctx.fillStyle = "#0a0e15"; ctx.fillRect(0, 0, w, h);
  const fl = S.flight;
  if (!fl || !fl.scope) return;
  const [phiR, phiL, alR, alL] = fl.scope;

  ctx.strokeStyle = "#141c28"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

  const series = [
    [phiR, "#ff5e7a", 1.6], [phiL, "#ffb03a", 1.6],
    [alR, "#4fd2e8", 1.6], [alL, "#7c6bf5", 1.6],
  ];
  for (const [arr, col, rng] of series) {
    if (!arr || arr.length < 2) continue;
    ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.beginPath();
    arr.forEach((v, i) => {
      const x = (i / (arr.length - 1)) * w;
      const y = h / 2 - (v / rng) * h * 0.44;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  el("wing-note").textContent =
    `${fmt(fl.wing.freq, 0)} Hz, about two strokes — stroke angle (red, amber) ` +
    `and wing pitch (cyan, violet), right and left`;
}

/* --------------------------------------------------------------- motor bars */
function drawMotor() {
  if (!S.pools) return;
  const m = S.pools.motor;
  const keys = Object.keys(m);
  const host = el("motor");
  if (host.children.length !== keys.length) {
    host.innerHTML = keys.map((k) =>
      `<div class="row"><span class="name">${k}</span>` +
      `<span class="track"><i class="fill" data-k="${k}"></i></span>` +
      `<span class="val" data-v="${k}">0.00</span></div>`).join("");
  }
  const hi = Math.max(...keys.map((k) => m[k]), 1e-3);
  for (const k of keys) {
    host.querySelector(`[data-k="${k}"]`).style.width = `${(m[k] / hi) * 100}%`;
    host.querySelector(`[data-v="${k}"]`).textContent = fmt(m[k], 3);
  }
}

/* ------------------------------------------------------------------- curves */
const CURVES = [
  ["ep_return", "#4fd2e8", "Return"],
  ["ep_gates", "#43c08a", "Gates cleared"],
  ["crash_rate", "#ff5e7a", "Crash rate"],
  ["entropy", "#7c6bf5", "Policy entropy"],
  ["difficulty", "#ffb03a", "Difficulty"],
];
function drawCurves() {
  const c = el("curves"); fitCanvas(c);
  const ctx = c.getContext("2d"), w = c.width, h = c.height;
  ctx.fillStyle = "#0a0e15"; ctx.fillRect(0, 0, w, h);
  for (const [k, col] of CURVES) {
    const a = S.hist[k]; if (!a || a.length < 2) continue;
    const lo = Math.min(...a), hi = Math.max(...a), r = hi - lo || 1;
    ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.beginPath();
    a.forEach((v, i) => {
      const x = (i / (a.length - 1)) * w, y = h - 6 - ((v - lo) / r) * (h - 12);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  if (!el("curve-key").children.length) {
    el("curve-key").innerHTML = CURVES.map(([k, c2, n]) =>
      `<li><b style="background:${c2}"></b>${n}</li>`).join("");
  }
  el("curve-note").textContent = S.metrics.update
    ? `update ${Math.round(S.metrics.update)}, each series on its own scale` : "";
}

/* -------------------------------------------------------------------- loop */
function frame() {
  drawGraph(); drawEye(); drawCourse(); pushWing(); drawWing();
  drawMotor(); drawCurves(); renderCounters();
  requestAnimationFrame(frame);
}
connect();
requestAnimationFrame(frame);

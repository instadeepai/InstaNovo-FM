"use strict";
/* Arrays, CAT/NUM, numVal/catStr, ensureCols and N come from
   static/js/data-loader.js, which fetches them as binary shards. */

/* MODE3D is runtime state here: one page serves the published 2-D layout
   and the separate 3-component fit, instead of shipping the payload twice. */
let MODE3D = false;


const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const SLOTS = () => [1,2,3,4,5,6,7,8].map(i => css('--s' + i));
const SEQ   = () => [0,1,2,3,4].map(i => css('--seq' + i));


const S = {
  colorField:'analyser', kind:'cat',
  space:'plane',   /* 'fit3d' = umap3d_1/2/3 */
  zField:'collision_energy',
  hi:[], hidden:new Set(),
  filters:[], showBg:true,
  size:2.4, hsize:2.4, opac:0.5, npts:100000,
  renderer:'gl',            /* 'gl' = scattergl, 'svg' = plain scatter */
  pep:'', selIdx:null, saved:[], pinned:null, box:null, showBox:false, camera:null, resetCam:false
};

/* --------------------------------------------------------------- formatting */
const fmtInt = n => n.toLocaleString('en-US');
function fmt(v, d){
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  if (d !== undefined) return v.toFixed(d);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}
const pct = f => (100 * f).toFixed(f >= 0.1 ? 0 : 1) + '%';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/* ------------------------------------------------------------- third axis */
const is3dFit = () => S.space === 'fit3d';
const zVal = i => is3dFit() ? Z3[i] : numVal(S.zField, i);
const zLabel = () => is3dFit() ? 'UMAP3D 3' : NUM[S.zField].label;
const axLabels = () => is3dFit() ? ['UMAP3D 1', 'UMAP3D 2', 'UMAP3D 3']
                                 : ['UMAP 1', 'UMAP 2', NUM[S.zField].label];
function setSpace(v){
  S.space = v;
  applySpace();
  for (const k of Object.keys(ZCLIP)) delete ZCLIP[k];
}
function zRange(){
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < N; i++){ const v = zVal(i); if (Number.isFinite(v)){ if (v < lo) lo = v; if (v > hi) hi = v; } }
  return [lo, hi];
}
/* A measured variable usually has a long tail that would flatten the cloud
   against one wall, so the drawn axis is clipped to the 1st-99th percentile. */
const ZCLIP = {};
function zAxisRange(){
  if (is3dFit()) return undefined;            /* a fitted axis is shown in full */
  if (ZCLIP[S.zField]) return ZCLIP[S.zField];
  const v = [];
  for (let i = 0; i < N; i++){ const t = zVal(i); if (Number.isFinite(t)) v.push(t); }
  v.sort((a, b) => a - b);
  if (!v.length) return ZCLIP[S.zField] = [0, 1];
  const q = p => v[Math.floor(p * (v.length - 1))];
  let lo = q(0.01), hi = q(0.99);
  if (hi <= lo){ lo = v[0]; hi = v[v.length - 1]; }
  const pad = (hi - lo) * 0.04 || 0.5;
  return ZCLIP[S.zField] = [lo - pad, hi + pad];
}

/* ---------------------------------------------------------------- filtering */
function passFilters(i){
  for (const f of S.filters){
    if (!f.values.size) continue;
    if (!f.values.has(CAT[f.field].codes[i])) return false;
  }
  return true;
}
function computeSets(){
  const inn = [], out = [];
  const cf = S.kind === 'cat' ? CAT[S.colorField] : null;
  for (let i = 0; i < N; i++){
    let ok = passFilters(i);
    if (ok && cf && S.hidden.size && S.hidden.has(cf.codes[i])) ok = false;
    if (ok && MODE3D && !Number.isFinite(zVal(i))) ok = false;
    (ok ? inn : out).push(i);
  }
  return {inn:Uint32Array.from(inn), out:Uint32Array.from(out)};
}
function stride(idx, cap){
  if (idx.length <= cap) return idx;
  const step = idx.length / cap, o = new Uint32Array(cap);
  for (let j = 0; j < cap; j++) o[j] = idx[Math.floor(j * step)];
  return o;
}
const gather = (arr, idx) => { const o = new Float32Array(idx.length); for (let j = 0; j < idx.length; j++) o[j] = arr[idx[j]]; return o; };
const gatherZ = idx => { const o = new Float32Array(idx.length); for (let j = 0; j < idx.length; j++) o[j] = zVal(idx[j]); return o; };
const filterKey = () => S.filters.map(f => f.field + ':' + [...f.values].sort((a, b) => a - b).join(',')).join('|') +
                        '#' + (S.kind === 'cat' ? S.colorField + ':' + [...S.hidden].join(',') : '') +
                        '@' + S.space + (S.space === 'plane' ? ':' + S.zField : '');

/* ------------------------------------------------------------------ plotting
   Highlight model: selected classes are drawn in palette colour at an enlarged
   marker on top; every other spectrum keeps the base grey marker at the base
   size, so adding or dropping a highlight never disturbs the background. */
const PLOT = document.getElementById('plot');
const SVG_MAX_PTS = 15000;   /* SVG cannot carry 100k markers */
function webglAvailable(){
  try {
    const c = document.createElement('canvas');
    const g = c.getContext('webgl2') || c.getContext('webgl');
    if (!g) return false;
    /* a context alone is not proof it paints; clear and read one pixel */
    const c2 = document.createElement('canvas'); c2.width = c2.height = 2;
    const g2 = c2.getContext('webgl2', {preserveDrawingBuffer:true})
            || c2.getContext('webgl',  {preserveDrawingBuffer:true});
    if (!g2) return false;
    g2.clearColor(0, 1, 0, 1); g2.clear(g2.COLOR_BUFFER_BIT);
    const px = new Uint8Array(4);
    g2.readPixels(0, 0, 1, 1, g2.RGBA, g2.UNSIGNED_BYTE, px);
    return px[1] > 200;
  } catch (e) { return false; }
}
const TT = () => MODE3D ? 'scatter3d'
                        : (S.renderer === 'svg' ? 'scatter' : 'scattergl');
let SETS = null, TRACEIDX = [], HLIDX = null, SCALE = null;

function axStyle(){
  const grid = css('--surface-2'), zl = css('--line');
  return {gridcolor:grid, zerolinecolor:zl, linecolor:zl, tickfont:{size:10, color:css('--text-3')},
          showline:true, ticks:'outside', ticklen:3, tickcolor:zl};
}
function liveCamera(){
  const sc = PLOT._fullLayout && PLOT._fullLayout.scene;
  return sc && sc.camera ? JSON.parse(JSON.stringify(sc.camera)) : null;
}
function layout(){
  const base = {
    margin:MODE3D ? {l:0, r:0, t:0, b:0} : {l:44, r:12, t:8, b:34},
    paper_bgcolor:css('--surface-0'), plot_bgcolor:css('--surface-1'),
    font:{color:css('--text-2'), family:'ui-sans-serif,-apple-system,sans-serif'},
    showlegend:false, uirevision:'keep', hovermode:'closest'
  };
  if (MODE3D){
    const a = axStyle(), wall = css('--surface-1');
    base.scene = {
      aspectmode:is3dFit() ? 'data' : 'cube',
      bgcolor:css('--surface-1'),
      xaxis:Object.assign({title:{text:axLabels()[0]}, backgroundcolor:wall, showbackground:true}, a),
      yaxis:Object.assign({title:{text:axLabels()[1]}, backgroundcolor:wall, showbackground:true}, a),
      zaxis:Object.assign({title:{text:axLabels()[2]}, backgroundcolor:wall, showbackground:true,
                           range:zAxisRange()}, a)
    };
    if (S.camera) base.scene.camera = S.camera;
  } else {
    base.dragmode = 'pan';
    base.xaxis = Object.assign({title:{text:'UMAP 1', font:{size:11}}, mirror:true, constrain:'domain'}, axStyle());
    base.yaxis = Object.assign({title:{text:'UMAP 2', font:{size:11}}, mirror:true, constrain:'domain',
                                scaleanchor:'x', scaleratio:1}, axStyle());
  }
  return base;
}
const CONF = () => ({responsive:true, scrollZoom:true, displaylogo:false,
  modeBarButtonsToRemove:MODE3D ? [] : ['toggleSpikelines','hoverClosestGl2d','autoScale2d'],
  modeBarButtonsToAdd:MODE3D ? [] : ['select2d','lasso2d'],
  toImageButtonOptions:{format:'png', scale:2, filename:'umap'}});

const MAXHI = 8;
/* One colour per class, keyed on the class itself and ranked by how common
   it is, so a class keeps the same colour whether it is highlighted, shown
   as grey context, or sitting beside a different selection. Selection order
   and hidden classes deliberately do not enter into it. Cached per colour
   field and per palette, so switching theme recomputes it. */
let CCM = null, CCM_KEY = null;
function classColors(){
  const slots = SLOTS(), key = S.colorField + '|' + slots[0];
  if (CCM_KEY === key) return CCM;
  const count = CAT[S.colorField].count, m = new Map();
  count.map((_, l) => l)
       .sort((a, b) => count[b] - count[a])
       .slice(0, MAXHI)
       .forEach((l, k) => m.set(l, slots[k]));
  CCM_KEY = key; CCM = m;
  return m;
}
function hiColorMap(){
  const slots = SLOTS(), all = classColors(), m = new Map();
  for (const l of S.hi.slice(0, MAXHI))
    /* A class outside the leading few still needs a colour once picked;
       derive it from the level so it too is stable between selections. */
    m.set(l, all.get(l) || slots[l % slots.length]);
  return m;
}
function baseColorMap(){
  const m = new Map();
  for (const [l, col] of classColors())
    if (!S.hidden.has(l) && m.size < 3) m.set(l, col);
  return m;
}
function mk(rows, marker){
  const t = {type:TT(), mode:'markers', x:gather(X, rows), y:gather(Y, rows), marker,
             hoverinfo:'none', showlegend:false};
  if (MODE3D) t.z = gatherZ(rows);
  return t;
}
function boxTrace(b){
  const [x0, x1] = b.x, [y0, y1] = b.y, [z0, z1] = b.z;
  const P = [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]];
  const E = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const seg = [];
  for (const [a, c] of E) seg.push(P[a], P[c], [null, null, null]);
  return {type:'scatter3d', mode:'lines',
          x:seg.map(p => p[0]), y:seg.map(p => p[1]), z:seg.map(p => p[2]),
          line:{color:css('--hl'), width:2}, hoverinfo:'none', showlegend:false};
}

function neededCols(){
  const cols = [[S.kind, S.colorField]];
  for (const f of S.filters) if (f.field) cols.push(['cat', f.field]);
  if (S.pep) cols.push(['cat', 'sequence']);
  if (MODE3D){
    if (is3dFit()) cols.push(['array','x3'], ['array','y3'], ['array','z3']);
    else cols.push(['num', S.zField]);
  }
  return cols;
}
function applySpace(){
  if (is3dFit() && X3 && Y3){ X = X3; Y = Y3; } else { X = X2; Y = Y2; }
}
/** Every redraw funnels through here, so this is the one place that has to
    wait for a column the reader just asked to see. */
async function draw(){
  await ensureCols(neededCols());
  applySpace();
  drawSync();
}
function drawSync(){
  if (MODE3D){
    if (S.resetCam){ S.camera = null; S.resetCam = false; }
    else { const c = liveCamera(); if (c) S.camera = c; }
  }
  SETS = computeSets();
  const idx = stride(SETS.inn, S.npts);
  const traces = []; TRACEIDX = [];
  const push = (t, rows) => { traces.push(t); TRACEIDX.push(rows); };

  if (S.showBg && SETS.out.length){
    const b = stride(SETS.out, Math.min(MODE3D ? 20000 : 60000, S.npts));
    push(mk(b, {size:S.size, color:css('--bg-pt'), opacity:Math.min(0.45, S.opac)}), b);
  }

  if (S.kind === 'cat' && S.hi.length){
    const c = CAT[S.colorField], hiSet = new Set(S.hi.slice(0, MAXHI)), cmap = hiColorMap();
    const rest = [], byLevel = new Map();
    for (const i of idx){
      const l = c.codes[i];
      if (hiSet.has(l)){ if (!byLevel.has(l)) byLevel.set(l, []); byLevel.get(l).push(i); }
      else rest.push(i);
    }
    const r = Uint32Array.from(rest);
    push(mk(r, {size:S.size, color:css('--other'), opacity:S.opac}), r);
    for (const l of S.hi.slice(0, MAXHI)){
      const rows = Uint32Array.from(byLevel.get(l) || []);
      if (!rows.length) continue;
      push(mk(rows, {size:S.size * S.hsize, color:cmap.get(l), opacity:Math.min(1, S.opac * 1.7 + 0.2)}), rows);
    }
  } else {
    const m = {size:S.size, opacity:S.opac};
    if (S.kind === 'cat'){
      const map = baseColorMap(), other = css('--other'), c = CAT[S.colorField];
      const col = new Array(idx.length);
      for (let j = 0; j < idx.length; j++) col[j] = map.get(c.codes[idx[j]]) || other;
      m.color = col;
    } else {
      const v = new Float32Array(idx.length);
      for (let j = 0; j < idx.length; j++) v[j] = numVal(S.colorField, idx[j]);
      const st = SEQ();
      m.color = v; m.colorscale = st.map((cc, i) => [i / (st.length - 1), cc]);
      const lo = quantile(v, 0.02), hi = quantile(v, 0.98);
      if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo){ m.cmin = lo; m.cmax = hi; }
      SCALE = {lo:m.cmin, hi:m.cmax, field:S.colorField};
      if (MODE3D) m.showscale = false;    /* drawn in the side panel instead */
      else {
        m.showscale = true;
        m.colorbar = {title:{text:NUM[S.colorField].label, side:'right', font:{size:10}},
                      thickness:9, len:0.42, y:0.5, x:1.005, tickfont:{size:9},
                      outlinewidth:0, bgcolor:'rgba(0,0,0,0)'};
      }
    }
    push(mk(idx, m), idx);
  }

  const hl = highlightIdx();
  if (hl.length) push(mk(hl, {size:Math.max(MODE3D ? 4 : 6, S.size * 2.8), color:css('--hl'), opacity:0.95}), hl);
  HLIDX = hl.length ? hl : null;

  if (MODE3D && S.showBox && S.box && Number.isFinite(S.box.x[0])){ traces.push(boxTrace(S.box)); TRACEIDX.push(null); }

  const lay = layout();
  if (!MODE3D){
    const live = PLOT._fullLayout && PLOT._fullLayout.xaxis ? currentRanges() : null;
    if (live && !PLOT._fullLayout.xaxis.autorange){
      lay.xaxis.range = live.x.slice(); lay.yaxis.range = live.y.slice();
      lay.xaxis.autorange = false; lay.yaxis.autorange = false;
    }
  }
  Plotly.react(PLOT, traces, lay, CONF());
  document.getElementById('stN').textContent =
    `${fmtInt(SETS.inn.length)} of ${fmtInt(N)} pass filters · ${fmtInt(idx.length)} drawn`;
  renderLegend();
  updateViewLabel();
}

function quantile(arr, q){
  const v = Array.from(arr).filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  return v[Math.min(v.length - 1, Math.max(0, Math.floor(q * (v.length - 1))))];
}
function highlightIdx(){
  if (!S.pep) return [];
  const q = S.pep.toUpperCase(), lv = CAT.sequence.levels, codes = CAT.sequence.codes;
  const hit = new Set();
  for (let l = 0; l < lv.length; l++) if (lv[l].toUpperCase().includes(q)) hit.add(l);
  if (!hit.size) return [];
  const o = [];
  for (let i = 0; i < N; i++) if (hit.has(codes[i]) && (!MODE3D || Number.isFinite(zVal(i)))) o.push(i);
  return Uint32Array.from(o);
}

/* -------------------------------------------------------------------- legend */
function renderLegend(){
  const el = document.getElementById('legend'), st = document.getElementById('hiState');
  if (S.kind !== 'cat'){
    st.textContent = '';
    if (MODE3D && SCALE && SCALE.field === S.colorField && Number.isFinite(SCALE.lo)){
      const stops = SEQ().map((c, i) => `${c} ${100 * i / (SEQ().length - 1)}%`).join(', ');
      el.innerHTML = `<div style="padding:7px">
        <div style="font-size:11px;color:var(--text-2);margin-bottom:4px">${esc(NUM[S.colorField].label)}</div>
        <div style="height:9px;border-radius:3px;background:linear-gradient(90deg,${stops})"></div>
        <div style="display:flex;justify-content:space-between;font:10.5px ui-monospace,Menlo,monospace;
                    color:var(--text-3);margin-top:3px">
          <span>${esc(fmt(SCALE.lo))}</span><span>${esc(fmt(SCALE.hi))}</span></div>
        <div class="hint">2nd–98th percentile</div></div>`;
    } else el.innerHTML = '';
    return;
  }
  const c = CAT[S.colorField], cmap = hiColorMap(), bmap = S.hi.length ? new Map() : baseColorMap();
  const rows = c.levels.map((nm, l) => ({l, nm, n:c.count[l]})).sort((a, b) => b.n - a.n).slice(0, 400);
  el.innerHTML = rows.map(r => {
    const col = cmap.get(r.l) || bmap.get(r.l) || css('--other');
    return `<div class="lg" data-l="${r.l}" data-off="${S.hidden.has(r.l)}" data-hi="${cmap.has(r.l)}" title="${esc(r.nm)}">
       <span class="sw" style="background:${col}"></span>
       <span class="nm">${esc(r.nm)}</span><span class="ct">${fmtInt(r.n)}</span></div>`;
  }).join('') +
    (c.nlevels > 400 ? `<div style="padding:4px 7px;color:var(--text-3);font-size:11px">…${fmtInt(c.nlevels - 400)} rarer classes not listed</div>` : '');
  el.querySelectorAll('.lg').forEach(d => d.onclick = ev => {
    const l = +d.dataset.l;
    if (ev.shiftKey){ S.hidden.has(l) ? S.hidden.delete(l) : S.hidden.add(l); }
    else {
      const k = S.hi.indexOf(l);
      if (k >= 0) S.hi.splice(k, 1);
      else if (S.hi.length < MAXHI) S.hi.push(l);
    }
    draw();
  });
  st.textContent = S.hi.length
    ? `${S.hi.length} highlighted${S.hi.length >= MAXHI ? ' (max)' : ''}${S.hidden.size ? ` · ${S.hidden.size} hidden` : ''}`
    : (S.hidden.size ? `${S.hidden.size} hidden` : '');
}

/* ------------------------------------------------------- region composition */
const GS = {};
function globalStats(k){
  if (GS[k]) return GS[k];
  const v = []; for (let i = 0; i < N; i++){ const t = numVal(k, i); if (Number.isFinite(t)) v.push(t); }
  v.sort((a, b) => a - b);
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length);
  return GS[k] = {med:v[v.length >> 1], sd, mean};
}
function boundsOf(idx){
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (const i of idx){
    if (X[i] < x0) x0 = X[i]; if (X[i] > x1) x1 = X[i];
    if (Y[i] < y0) y0 = Y[i]; if (Y[i] > y1) y1 = Y[i];
    if (MODE3D){ const v = zVal(i); if (v < z0) z0 = v; if (v > z1) z1 = v; }
  }
  const b = {x:[x0, x1], y:[y0, y1]};
  if (MODE3D) b.z = [z0, z1];
  return b;
}

function report(idx, title, bounds){
  const n = idx.length;
  if (!n) return '<div class="head">No spectra in this region.</div>';
  const bx = bounds || boundsOf(idx);

  const marks = [], blocks = [];
  for (const k of REPORT_CATS){
    const c = CAT[k], cnt = new Map(), projOf = new Map();
    for (const i of idx){
      const l = c.codes[i];
      cnt.set(l, (cnt.get(l) || 0) + 1);
      if (!projOf.has(l)) projOf.set(l, new Set());
      projOf.get(l).add(CAT.search_project.codes[i]);
    }
    const rows = [...cnt.entries()].map(([l, v]) => {
      const share = v / n, glob = c.count[l] / N;
      return {nm:c.levels[l], v, share, lift:share / glob, nproj:projOf.get(l).size};
    }).sort((a, b) => b.v - a.v);
    for (const r of rows.slice(0, 4)) if (r.share >= 0.12 && r.lift >= 1.25)
      marks.push({field:c.label, ...r, score:r.share * Math.log2(r.lift)});
    blocks.push(`<tr><th colspan="5" style="padding-top:7px">${esc(c.label)} <span style="text-transform:none;font-weight:400">· ${cnt.size} classes</span></th></tr>` +
      rows.slice(0, 5).map(r =>
        `<tr><td>${esc(r.nm)}</td><td class="n">${fmtInt(r.v)}</td><td class="n">${pct(r.share)}</td>
          <td class="n" style="color:${r.lift >= 2 ? 'var(--s3)' : (r.lift <= 0.5 ? 'var(--text-3)' : 'var(--text-2)')}">${r.lift >= 100 ? '&gt;99' : fmt(r.lift, 2)}×</td>
          <td class="n">${fmtInt(r.nproj)}</td></tr>`).join(''));
  }
  marks.sort((a, b) => b.score - a.score);

  const numRows = REPORT_NUMS.map(k => {
    const v = []; for (const i of idx){ const t = numVal(k, i); if (Number.isFinite(t)) v.push(t); }
    if (!v.length) return null;
    v.sort((a, b) => a - b);
    const g = globalStats(k), med = v[v.length >> 1];
    return {label:NUM[k].label, med, p10:v[Math.floor(0.1 * (v.length - 1))], p90:v[Math.floor(0.9 * (v.length - 1))],
            z:g.sd ? (med - g.med) / g.sd : 0};
  }).filter(Boolean).sort((a, b) => Math.abs(b.z) - Math.abs(a.z));

  const nd = k => { const s = new Set(); for (const i of idx) s.add(CAT[k].codes[i]); return s.size; };
  const pepRows = (() => {
    const c = CAT.sequence, m = new Map();
    for (const i of idx) m.set(c.codes[i], (m.get(c.codes[i]) || 0) + 1);
    return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([l, v]) => ({nm:c.levels[l], v}));
  })();

  return `
  <div class="head"><b>${esc(title)}</b> — ${fmtInt(n)} spectra (${pct(n / N)} of the pool)</div>
  <div class="mono" style="color:var(--text-2);font-size:11px">
    x ∈ [${fmt(bx.x[0], 2)}, ${fmt(bx.x[1], 2)}] &nbsp; y ∈ [${fmt(bx.y[0], 2)}, ${fmt(bx.y[1], 2)}]
    ${bx.z ? `<br>${esc(zLabel())} ∈ [${fmt(bx.z[0], 2)}, ${fmt(bx.z[1], 2)}]` : ''}</div>
  <div class="row wrap" style="margin-top:6px">
    <button data-copy="ax">Copy axis limits</button><button data-copy="py">Copy pandas query</button></div>

  <h2>Enriched labels</h2>
  ${marks.length ? marks.slice(0, 9).map(m =>
      `<div class="chip" title="${esc(m.field)}"><b>${esc(m.nm)}</b> ${pct(m.share)} · ${fmt(m.lift, 1)}× · ${fmtInt(m.nproj)} proj</div>`).join('')
    : '<div class="hint">nothing above 1.25×</div>'}

  <h2>Distinct entities</h2>
  <div class="row wrap">
    <span class="chip">${fmtInt(nd('search_project'))} projects</span>
    <span class="chip">${fmtInt(nd('experiment_name'))} runs</span>
    <span class="chip">${fmtInt(nd('sequence'))} peptidoforms</span>
    <span class="chip">${fmtInt(nd('search_instrument'))} instruments</span>
    <span class="chip">${fmtInt(nd('protein'))} proteins</span>
  </div>
  <table style="margin-top:6px"><tr><th>Repeated peptidoform</th><th class="n">spectra</th></tr>
    ${pepRows.map(p => `<tr class="clk" data-pep="${esc(p.nm)}"><td class="mono">${esc(p.nm)}</td><td class="n">${p.v}</td></tr>`).join('')}</table>

  <h2>Numeric shifts vs pool</h2>
  <table><tr><th>Field</th><th class="n">median</th><th class="n">p10–p90</th><th class="n" title="shift of the median in pool SD units">Δ SD</th></tr>
    ${numRows.slice(0, 10).map(r => `<tr><td>${esc(r.label)}</td><td class="n">${fmt(r.med)}</td>
      <td class="n">${fmt(r.p10)}–${fmt(r.p90)}</td>
      <td class="n" style="color:${Math.abs(r.z) >= 0.5 ? 'var(--s2)' : 'var(--text-3)'}">${r.z >= 0 ? '+' : ''}${fmt(r.z, 2)}</td></tr>`).join('')}
  </table>

  <h2>Full composition</h2>
  <table><tr><th>Class</th><th class="n">n</th><th class="n">share</th><th class="n" title="share here ÷ share in the pool">enr.</th><th class="n" title="distinct projects contributing this class here">proj</th></tr>${blocks.join('')}</table>`;
}

let LASTREG = null;
async function showRegion(idx, title, bounds){
  LASTREG = {idx, title, bounds};
  await ensureCols([...REPORT_CATS.map(k => ['cat', k]), ...REPORT_NUMS.map(k => ['num', k])]);
  document.getElementById('regionOut').innerHTML = report(idx, title, bounds);
  tab('Region');
  document.querySelectorAll('#regionOut [data-copy]').forEach(b => b.onclick = () => {
    const r = LASTREG.bounds || boundsOf(LASTREG.idx);
    let t;
    if (b.dataset.copy === 'ax')
      t = `ax.set_xlim(${fmt(r.x[0], 2)}, ${fmt(r.x[1], 2)}); ax.set_ylim(${fmt(r.y[0], 2)}, ${fmt(r.y[1], 2)})`;
    else {
      t = `sub = df[df.umap_x.between(${fmt(r.x[0], 3)}, ${fmt(r.x[1], 3)}) & df.umap_y.between(${fmt(r.y[0], 3)}, ${fmt(r.y[1], 3)})]`;
      if (r.z && S.zField !== '__coord__')
        t += `  # and ${S.zField} in [${fmt(r.z[0], 3)}, ${fmt(r.z[1], 3)}]`;
    }
    navigator.clipboard.writeText(t);
    const was = b.textContent; b.textContent = 'copied'; setTimeout(() => b.textContent = was, 1200);
  });
  document.querySelectorAll('#regionOut [data-pep]').forEach(tr => tr.onclick = () => {
    document.getElementById('pepSearch').value = tr.dataset.pep; S.pep = tr.dataset.pep; draw(); updatePepHint();
  });
}


/* -------------------------------------------------------- candidate windows */
/* --------------------------------------------------------------- point panel */
function usiFor(i){
  const run = catStr('experiment_name', i).replace(/\.(mzML|mzXML|raw|RAW|mgf|d)$/, '');
  const ch = catStr('charge_cat', i).replace('+', '');
  return `mzspec:${catStr('search_project', i)}:${run}:scan:${SCAN[i]}:${catStr('sequence', i)}/${ch === 'not reported (0)' ? 0 : ch}`;
}
function pointHtml(i){
  const nv = (k, d) => k in NUM ? fmt(numVal(k, i), d) : null;
  const kv = [
    ['peptidoform', `<span class="mono">${esc(catStr('sequence', i))}</span>`],
    ['protein', esc(catStr('protein', i))],
    ['charge', esc(catStr('charge_cat', i))],
    ['precursor m/z', nv('precursor_mz', 4)],
    ['length', nv('sequence_length', 0) ? nv('sequence_length', 0) + ' aa' : null],
    ['GRAVY', nv('hydrophobicity', 2)],
    ['analyser', esc(catStr('analyser', i))],
    ['activation', esc(catStr('activation', i))],
    ['NCE', nv('collision_energy')],
    ['instrument', esc(catStr('search_instrument', i))],
    ['label chem.', esc(catStr('label_chem', i))],
    ['enrichment', esc(catStr('enrichment', i))],
    ['modification', esc(catStr('modification_types', i))],
    ['organism', esc(catStr('search_organism', i))],
    ['project', esc(catStr('search_project', i))],
    ['run', `<span class="mono" style="font-size:10px">${esc(catStr('experiment_name', i))}</span>`],
    ['RT (s)', nv('retention_time')],
    ['peaks', nv('n_peaks', 0)],
    ['annot. frac', nv('annotation_ratio', 3)],
    ['backbone cov.', nv('backbone_coverage', 3)],
    ['confidence', nv('spectrum_confidence', 3)],
    ['hyperscore', nv('hyperscore', 2)],
    ['position', MODE3D ? `${fmt(X[i], 3)}, ${fmt(Y[i], 3)}, ${fmt(zVal(i), 3)}` : `${fmt(X[i], 3)}, ${fmt(Y[i], 3)}`],
    ['space', MODE3D ? (is3dFit() ? '3-D UMAP fit' : 'published 2-D layout + ' + NUM[S.zField].label) : 'published 2-D layout'],
    ['sample_idx', fmtInt(SAMPLE[i])]
  ];
  return `<dl class="kv">${kv.filter(r => r[1] !== null && r[1] !== undefined && r[1] !== '—')
      .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>
    <div class="sec"><div class="mono" style="overflow-wrap:anywhere">${esc(usiFor(i))}</div>
    <div class="row"><button id="copyUsi">Copy USI</button>
    <button id="pepFromPoint">Highlight peptidoform</button></div></div>`;
}
function setPinned(i){
  S.pinned = i;
  document.getElementById('pinbar').dataset.on = i !== null;
  if (i !== null) showPoint(i);
}
function showPoint(i){
  if (!detailReady()){
    document.getElementById('pointOut').innerHTML =
      '<div class="hint">Loading the per-spectrum record…</div>';
    onDetailReady(() => { if (S.pinned === i || S.pinned === null) showPoint(i); });
    return;
  }
  document.getElementById('pointOut').innerHTML = pointHtml(i);
  document.getElementById('copyUsi').onclick = e => {
    navigator.clipboard.writeText(usiFor(i)); e.target.textContent = 'copied'; };
  document.getElementById('pepFromPoint').onclick = () => {
    const seq = catStr('sequence', i);
    document.getElementById('pepSearch').value = seq; S.pep = seq; draw(); updatePepHint(); };
}

/* ------------------------------------------------------------ saved regions */
/* ------------------------------------------------------------------ filters */
function fieldOptions(sel){
  const groups = {};
  for (const k of D.catOrder) (groups[CAT[k].group] = groups[CAT[k].group] || []).push([k, CAT[k].label]);
  return Object.entries(groups).map(([g, arr]) =>
    `<optgroup label="${esc(g)}">${arr.map(([k, l]) => `<option value="${k}"${k === sel ? ' selected' : ''}>${esc(l)}</option>`).join('')}</optgroup>`).join('');
}
async function renderFilters(){
  const host = document.getElementById('filters');
  await ensureCols(S.filters.filter(f => f.field).map(f => ['cat', f.field]));
  host.innerHTML = S.filters.map((f, k) => {
    const c = CAT[f.field];
    const rows = c.levels.map((nm, l) => ({l, nm, n:c.count[l]})).sort((a, b) => b.n - a.n).slice(0, 300);
    return `<div class="filter" data-k="${k}">
      <div class="fh"><select data-role="field">${fieldOptions(f.field)}</select>
        <button class="fx" data-role="del">×</button></div>
      <div class="vals">${rows.map(r =>
        `<div data-l="${r.l}" aria-selected="${f.values.has(r.l)}">${esc(r.nm)}<span class="c">${fmtInt(r.n)}</span></div>`).join('')}</div>
      <div class="hint">${f.values.size ? f.values.size + ' selected' : 'unrestricted'}</div></div>`;
  }).join('');
  host.querySelectorAll('.filter').forEach(fd => {
    const k = +fd.dataset.k;
    fd.querySelector('[data-role=field]').onchange = e => { S.filters[k] = {field:e.target.value, values:new Set()}; renderFilters(); draw(); };
    fd.querySelector('[data-role=del]').onclick = () => { S.filters.splice(k, 1); renderFilters(); draw(); };
    fd.querySelectorAll('.vals div').forEach(d => d.onclick = () => {
      const l = +d.dataset.l, v = S.filters[k].values;
      v.has(l) ? v.delete(l) : v.add(l);
      renderFilters(); draw();
    });
  });
}

/* ------------------------------------------------------------------- export
   PNG is a straight raster at 2× . SVG keeps axes, ticks and text as vectors;
   for the 2D map the point layer is converted to true vector marks when the
   drawn count is manageable, otherwise — and always in 3D, where the scene is
   a WebGL canvas — the point layer arrives as an embedded raster. PDF goes
   through the browser's own print-to-PDF on that same SVG. */
const SVG_VECTOR_LIMIT = 40000;
function exportName(ext){
  const bits = ['umap', MODE3D ? '3d' : '2d', S.colorField];
  if (S.hi.length) bits.push(S.hi.slice(0, 3).map(l => CAT[S.colorField].levels[l].replace(/[^A-Za-z0-9]+/g, '-')).join('_'));
  return bits.join('_').slice(0, 90) + '.' + ext;
}
const drawnCount = () => TRACEIDX.reduce((a, m) => a + (m ? m.length : 0), 0);
function setExHint(t){ document.getElementById('exHint').textContent = t; }

function exportPng(){
  Plotly.downloadImage(PLOT, {format:'png', scale:2, width:PLOT.clientWidth, height:PLOT.clientHeight,
                              filename:exportName('png').replace(/\.png$/, '')});
  setExHint('PNG written at 2× the on-screen size.');
}
async function withVectorTraces(fn){
  if (MODE3D || drawnCount() > SVG_VECTOR_LIMIT) return fn(false);
  const types = PLOT.data.map(t => t.type);
  await Plotly.restyle(PLOT, {type:'scatter'});
  try { return await fn(true); }
  finally { await Plotly.restyle(PLOT, {type:types}); }
}
function svgString(){
  return withVectorTraces(async vector => {
    const url = await Plotly.toImage(PLOT, {format:'svg', width:PLOT.clientWidth, height:PLOT.clientHeight});
    return {svg:decodeURIComponent(url.replace(/^data:image\/svg\+xml,/, '')), vector};
  });
}
async function exportSvg(){
  const {svg, vector} = await svgString();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([svg], {type:'image/svg+xml'}));
  a.download = exportName('svg'); a.click();
  setExHint(vector ? 'SVG written with vector point marks.'
                   : MODE3D
                     ? 'SVG written; the 3D scene is an embedded raster. Use PNG 2× for a crisp 3D image.'
                     : `SVG written; point layer is raster above ${fmtInt(SVG_VECTOR_LIMIT)} drawn points — lower “Spectra drawn” or untick the grey context for vector marks.`);
}
async function exportPdf(){
  const {svg, vector} = await svgString();
  const w = window.open('', '_blank');
  if (!w){ setExHint('Pop-up blocked — allow pop-ups for this file to export PDF.'); return; }
  w.document.write(`<!doctype html><title>${exportName('pdf')}</title>
    <style>@page{size:auto;margin:8mm}html,body{margin:0}svg{width:100%;height:auto}</style>${svg}`);
  w.document.close(); w.focus();
  setTimeout(() => w.print(), 400);
  setExHint(`Print dialog opened — choose “Save as PDF”. ${vector ? 'Vector marks.' : 'Point layer is raster.'}`);
}

/* ---------------------------------------------------------------- UI wiring */
const TABS = {Region:['tbRegion','pnRegion'], Point:['tbPoint','pnPoint']};
function tab(name){
  for (const [n, [b, p]] of Object.entries(TABS)){
    document.getElementById(b).setAttribute('aria-pressed', n === name);
    document.getElementById(p).setAttribute('data-active', n === name);
  }
}
async function updatePepHint(){
  const el = document.getElementById('pepHint');
  await ensureCols([['cat','sequence'], ['cat','experiment_name'], ['cat','search_project']]);
  if (!S.pep){ el.textContent = ''; return; }
  const idx = HLIDX || [];
  if (!idx.length){ el.textContent = 'no match'; return; }
  const runs = new Set(), projs = new Set(), forms = new Set();
  for (const i of idx){ runs.add(CAT.experiment_name.codes[i]); projs.add(CAT.search_project.codes[i]); forms.add(CAT.sequence.codes[i]); }
  const b = boundsOf(idx);
  el.innerHTML = `<b>${fmtInt(idx.length)}</b> spectra · ${forms.size} peptidoform(s) · ${runs.size} runs · ${projs.size} projects<br>
    <span class="mono">x ${fmt(b.x[0], 1)}–${fmt(b.x[1], 1)} · y ${fmt(b.y[0], 1)}–${fmt(b.y[1], 1)}</span>
    <button style="margin-top:4px" id="pepProfile">Profile these</button>`;
  document.getElementById('pepProfile').onclick = () => showRegion(idx, 'peptidoform “' + S.pep + '”', null);
}
function setTheme(t){
  document.documentElement.dataset.theme = t;
  document.getElementById('thDark').setAttribute('aria-pressed', t === 'dark');
  document.getElementById('thLight').setAttribute('aria-pressed', t === 'light');
  draw();
}
function currentRanges(){
  if (!PLOT._fullLayout || !PLOT._fullLayout.xaxis)
    return {x:[-Infinity, Infinity], y:[-Infinity, Infinity]};
  if (MODE3D) return S.box || {x:[-Infinity, Infinity], y:[-Infinity, Infinity], z:[-Infinity, Infinity]};
  const fl = PLOT._fullLayout;
  return {x:fl.xaxis.range.slice(), y:fl.yaxis.range.slice()};
}
function inRegion(i, r){
  if (X[i] < r.x[0] || X[i] > r.x[1] || Y[i] < r.y[0] || Y[i] > r.y[1]) return false;
  if (MODE3D && r.z){ const v = zVal(i); if (v < r.z[0] || v > r.z[1]) return false; }
  return true;
}
function regionIdx(){
  const r = currentRanges(), out = [];
  for (const i of SETS.inn) if (inRegion(i, r)) out.push(i);
  return {idx:Uint32Array.from(out), r};
}
function updateViewLabel(){
  const el = document.getElementById('stView');
  if (MODE3D){
    el.textContent = S.box
      ? `box x [${fmt(S.box.x[0], 2)}, ${fmt(S.box.x[1], 2)}]  y [${fmt(S.box.y[0], 2)}, ${fmt(S.box.y[1], 2)}]  z [${fmt(S.box.z[0], 2)}, ${fmt(S.box.z[1], 2)}]`
      : 'box: all data';
  } else {
    const r = currentRanges();
    el.textContent = `view x [${fmt(r.x[0], 2)}, ${fmt(r.x[1], 2)}]  y [${fmt(r.y[0], 2)}, ${fmt(r.y[1], 2)}]`;
  }
}
function setBox(b){
  S.box = b;
  const put = (id, v, d) => document.getElementById(id).value = Number.isFinite(v) ? fmt(v, d) : '';
  put('bx0', b.x[0], 2); put('bx1', b.x[1], 2);
  put('by0', b.y[0], 2); put('by1', b.y[1], 2);
  put('bz0', b.z[0], 3); put('bz1', b.z[1], 3);
  updateViewLabel();
}
function readBox(){
  const g = id => parseFloat(document.getElementById(id).value);
  const b = {x:[g('bx0'), g('bx1')], y:[g('by0'), g('by1')], z:[g('bz0'), g('bz1')]};
  for (const k of ['x', 'y', 'z']) if (!Number.isFinite(b[k][0]) || !Number.isFinite(b[k][1])) return null;
  return b;
}

async function init(){
  /* Fall back to SVG when WebGL cannot paint, and honour ?renderer=svg so a
     blank map can be worked around without waiting for a fix. */
  const forced = new URLSearchParams(location.search).get('renderer');
  if (forced === 'svg' || (forced !== 'gl' && !webglAvailable())){
    S.renderer = 'svg';
    S.npts = Math.min(S.npts, SVG_MAX_PTS);
  }
  const sel = document.getElementById('colorField');
  const gcat = {}, gnum = {};
  for (const k of D.catOrder) (gcat[CAT[k].group] = gcat[CAT[k].group] || []).push([k, CAT[k].label, CAT[k].nlevels]);
  for (const k of D.numOrder) (gnum[NUM[k].group] = gnum[NUM[k].group] || []).push([k, NUM[k].label]);
  let html = '';
  let zHtml = HAS3D ? '<option value="fit3d">3-D UMAP fit — umap3d_1/2/3</option>' : '';
  for (const [g, arr] of Object.entries(gcat)){
    html += `<optgroup label="Labels · ${esc(g)}">` + arr.map(([k, l, nl]) =>
      `<option value="cat:${k}">${esc(l)} (${nl})</option>`).join('') + '</optgroup>';
  }
  for (const [g, arr] of Object.entries(gnum)){
    html += `<optgroup label="Continuous · ${esc(g)}">` + arr.map(([k, l]) =>
      `<option value="num:${k}">${esc(l)}</option>`).join('') + '</optgroup>';
    zHtml += `<optgroup label="2-D layout + ${esc(g)}">` + arr.map(([k, l]) =>
      `<option value="plane:${k}">${esc(l)}</option>`).join('') + '</optgroup>';
  }
  sel.innerHTML = html; sel.value = 'cat:analyser';
  sel.onchange = () => {
    const [kind, k] = sel.value.split(':');
    S.kind = kind; S.colorField = k; S.hi = []; S.hidden = new Set(); draw();
  };

  {
    setSpace(S.space);          /* point X/Y at the active space before anything draws */
    const zsel = document.getElementById('zField');
    zsel.innerHTML = zHtml;
    zsel.value = is3dFit() ? 'fit3d' : 'plane:' + S.zField;
    document.getElementById('zBlock').hidden = !MODE3D;
    document.getElementById('boxBlock').hidden = !MODE3D;
    const zhint = () => document.getElementById('zHint').textContent = is3dFit()
      ? 'Its own optimisation — these axes are not the published umap_x / umap_y.'
      : 'Published 2-D layout, with the measured variable above as the vertical axis.';
    const fitAll = () => {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < N; i++){
        if (X[i] < x0) x0 = X[i]; if (X[i] > x1) x1 = X[i];
        if (Y[i] < y0) y0 = Y[i]; if (Y[i] > y1) y1 = Y[i];
      }
      setBox({x:[x0, x1], y:[y0, y1], z:zRange()});
    };
    zhint3d = zhint; fitAll3d = fitAll;
    zhint();
    if (MODE3D) fitAll();   /* reads the plot layout, which 2-D has not drawn yet */
    zsel.onchange = () => {
      const v = zsel.value;
      if (v === 'fit3d') setSpace('fit3d');
      else { S.zField = v.slice(6); setSpace('plane'); }
      document.getElementById('bzlab').textContent = is3dFit() ? 'u3' : 'z';
      S.resetCam = true;                   /* a new space deserves a fresh framing */
      zhint(); fitAll(); draw();
    };
    document.getElementById('boxAll').onclick = () => { fitAll(); S.showBox = false; draw(); };
    document.getElementById('boxShow').onclick = () => {
      const b = readBox(); if (!b) return;
      S.box = b; S.showBox = !S.showBox; draw();
    };
    ['bx0','bx1','by0','by1','bz0','bz1'].forEach(id =>
      document.getElementById(id).onchange = () => {
        const b = readBox(); if (b){ S.box = b; updateViewLabel(); if (S.showBox) draw(); } });
  }

  const dl = CAT.replicate_peptide;
  const dupOpts = dl.levels.map((nm, l) => ({nm, n:dl.count[l]}))
    .filter(o => !o.nm.startsWith('(')).sort((a, b) => b.n - a.n);
  document.getElementById('dupSel').innerHTML =
    '<option value="">— most-replicated peptidoforms —</option>' +
    dupOpts.map(o => `<option value="${esc(o.nm)}">${esc(o.nm.length > 30 ? o.nm.slice(0, 29) + '…' : o.nm)} · ${o.n}</option>`).join('');
  document.getElementById('dupSel').onchange = e => {
    S.pep = e.target.value; document.getElementById('pepSearch').value = e.target.value; draw(); updatePepHint(); };
  const pep = document.getElementById('pepSearch');
  let t = null;
  pep.oninput = () => { clearTimeout(t); t = setTimeout(() => { S.pep = pep.value.trim(); draw(); updatePepHint(); }, 220); };

  document.getElementById('lgReset').onclick = () => { S.hi = []; S.hidden = new Set(); draw(); };
  document.getElementById('addFilter').onclick = () => { S.filters.push({field:'search_project', values:new Set()}); renderFilters(); };
  document.getElementById('clearFilters').onclick = () => { S.filters = []; renderFilters(); draw(); };
  document.getElementById('showBg').onchange = e => { S.showBg = e.target.checked; draw(); };
  document.getElementById('exPng').onclick = exportPng;
  document.getElementById('exSvg').onclick = exportSvg;
  document.getElementById('exPdf').onclick = exportPdf;

  const bind = (id, out, key, f) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = S[key];
    document.getElementById(out).textContent = f ? f(S[key]) : S[key];
    el.oninput = () => { S[key] = +el.value; document.getElementById(out).textContent = f ? f(el.value) : el.value; draw(); };
  };
  bind('size', 'szv', 'size'); bind('hsize', 'hzv', 'hsize'); bind('opac', 'opv', 'opac');
  bind('npts', 'npv', 'npts', v => fmtInt(+v));

  document.getElementById('md2d').onclick = () => setMode3d(false);
  document.getElementById('md3d').onclick = () => setMode3d(true);
  document.getElementById('rdGl').onclick  = () => setRenderer('gl');
  document.getElementById('rdSvg').onclick = () => setRenderer('svg');
  document.getElementById('rdGl').setAttribute('aria-pressed', S.renderer === 'gl');
  document.getElementById('rdSvg').setAttribute('aria-pressed', S.renderer === 'svg');
  if (S.renderer === 'svg'){
    const np = document.getElementById('npts');
    np.max = SVG_MAX_PTS; np.value = S.npts;
    setExHint('SVG renderer: at most ' + fmtInt(SVG_MAX_PTS) + ' spectra drawn.');
  }
  document.getElementById('thDark').onclick = () => setTheme('dark');
  document.getElementById('thLight').onclick = () => setTheme('light');
  for (const [n, [b]] of Object.entries(TABS)) document.getElementById(b).onclick = () => tab(n);

  document.getElementById('btnReset').onclick = () => {
    if (MODE3D){ S.resetCam = true; document.getElementById('boxAll').click(); }
    else Plotly.relayout(PLOT, {'xaxis.autorange':true, 'yaxis.autorange':true});
  };
  document.getElementById('btnProfileView').onclick = () => {
    const {idx, r} = regionIdx();
    showRegion(idx, MODE3D ? 'region box' : 'current view', r);
  };

  renderFilters(); updatePepHint(); setPinned(null);
  document.getElementById('pointOut').innerHTML =
    `<div class="hint">Hover the map for a spectrum; click to pin the full record.</div>`;
  await draw();

  wireEvents();
  document.getElementById('overlay').remove();
  const fit = () => Plotly.Plots.resize(PLOT);
  requestAnimationFrame(fit); setTimeout(fit, 250);
  new ResizeObserver(fit).observe(document.getElementById('center'));
  document.getElementById('hdrsub').textContent =
    `${D.dataset} \u00b7 ${fmtInt(N)} spectra \u00b7 ${D.catOrder.length} label fields \u00b7 ${D.numOrder.length} measured fields`;
}
let zhint3d = () => {}, fitAll3d = () => {};
/** Re-applied after every Plotly.purge, i.e. on each 2-D/3-D switch. */
function wireEvents(){
  const rowOf = ev => {
    const p = ev.points && ev.points[0]; if (!p) return undefined;
    const map = TRACEIDX[p.curveNumber];
    return map ? map[p.pointNumber] : undefined;
  };
  if (!MODE3D){
    document.getElementById('stSel').textContent = 'no selection';
    PLOT.on('plotly_relayout', updateViewLabel);
    PLOT.on('plotly_selected', ev => {
      if (!ev || !ev.points || !ev.points.length){ S.selIdx = null; document.getElementById('stSel').textContent = 'no selection'; return; }
      const rows = new Set();
      for (const p of ev.points){
        const map = TRACEIDX[p.curveNumber];
        if (map && map[p.pointNumber] !== undefined) rows.add(map[p.pointNumber]);
      }
      S.selIdx = Uint32Array.from(rows);
      document.getElementById('stSel').textContent = `${fmtInt(S.selIdx.length)} selected`;
      const b = boundsOf(S.selIdx), full = [];
      for (const i of SETS.inn) if (inRegion(i, b)) full.push(i);
      showRegion(Uint32Array.from(full), 'selection (bounding box)', b);
    });
    PLOT.on('plotly_deselect', () => { S.selIdx = null; document.getElementById('stSel').textContent = 'no selection'; });
  } else {
    document.getElementById('stSel').textContent = 'drag to rotate · hover to inspect';
  }
  let hovT = 0;
  PLOT.on('plotly_hover', ev => {
    const now = performance.now(); if (now - hovT < 45) return; hovT = now;
    const i = rowOf(ev);
    if (i === undefined) return;
    /* the side panel follows the cursor, unless a point is pinned */
    if (S.pinned === null) showPoint(i);
  });
  PLOT.on('plotly_click', ev => {
    const i = rowOf(ev);
    if (i === undefined) return;
    setPinned(S.pinned === i ? null : i);   /* clicking the pinned point releases it */
    if (S.pinned !== null) tab('Point');
  });
  document.getElementById('unpin').onclick = () => setPinned(null);
  window.addEventListener('keydown', e => { if (e.key === 'Escape') setPinned(null); });
}

/* The 2-D/3-D switch rebuilds the plot: Plotly cannot morph a scattergl trace
   into a scatter3d one in place. */
async function setRenderer(kind){
  if (S.renderer === kind) return;
  S.renderer = kind;
  S.npts = kind === 'svg' ? Math.min(S.npts, SVG_MAX_PTS) : 100000;
  const el = document.getElementById('npts');
  if (el){
    el.max = kind === 'svg' ? SVG_MAX_PTS : 100000;
    el.value = S.npts; el.dispatchEvent(new Event('input'));
  }
  for (const [id, on] of [['rdGl', kind === 'gl'], ['rdSvg', kind === 'svg']]){
    const btn = document.getElementById(id);
    if (btn) btn.setAttribute('aria-pressed', on);
  }
  setExHint(kind === 'svg'
    ? 'SVG renderer: at most ' + fmtInt(SVG_MAX_PTS) + ' spectra drawn.'
    : '');
  Plotly.purge(PLOT);        /* scatter and scattergl are different trace types */
  await draw();
  wireEvents();
}

async function setMode3d(on){
  if (MODE3D === on) return;
  MODE3D = on;
  S.space = (on && HAS3D) ? 'fit3d' : 'plane';
  S.size = on ? 1.8 : 2.4;
  S.npts = on ? 40000 : 100000;
  S.selIdx = null; S.camera = null; S.showBox = false;
  for (const [id, val] of [['size', S.size], ['npts', S.npts]]){
    const el = document.getElementById(id);
    if (el){ el.value = val; el.dispatchEvent(new Event('input')); }
  }
  document.getElementById('md2d').setAttribute('aria-pressed', !on);
  document.getElementById('md3d').setAttribute('aria-pressed', on);
  document.documentElement.dataset.mode = on ? '3d' : '2d';
  document.getElementById('zBlock').hidden = !on;
  document.getElementById('boxBlock').hidden = !on;
  const zsel = document.getElementById('zField');
  zsel.value = is3dFit() ? 'fit3d' : 'plane:' + S.zField;
  await ensureCols(neededCols());
  applySpace();
  zhint3d(); fitAll3d();
  S.resetCam = true;
  Plotly.purge(PLOT);              /* scattergl cannot morph into scatter3d */
  drawSync();
  wireEvents();
}

bootData().then(init).catch(e => {
  console.error(e);
  const ov = document.getElementById('overlay');
  if (!ov) return;
  const detail = '<div class="boot-detail mono">' + String(e.message || e) + '</div>';
  if (e && e.kind === 'auth'){
    /* The page loaded but its data requests did not: the private-site sign-in is
       not reaching them. Almost always a browser withholding the cookie from
       subresource requests rather than anything wrong with the site. */
    ov.innerHTML =
      '<div class="boot-err"><b>Signed out of the data requests.</b>' +
      '<p>The page loaded, but its data files were redirected to the GitHub ' +
      'sign-in. That happens when the browser withholds the sign-in cookie from ' +
      'background requests.</p><ul>' +
      '<li><b>Reload the page</b> &mdash; this clears it most of the time.</li>' +
      '<li><b>Brave:</b> click the shields icon in the address bar and turn ' +
      'shields <b>down</b> for this site, then reload.</li>' +
      '<li>Otherwise allow cross-site cookies for <span class="mono">github.com</span> ' +
      'and <span class="mono">pages.github.io</span>.</li>' +
      '</ul>' + detail + '</div>';
  } else if (e && e.kind === 'network'){
    ov.innerHTML =
      '<div class="boot-err"><b>Could not reach the data files.</b>' +
      '<p>The request did not complete. Check the connection, then reload. ' +
      'An ad-blocker or content blocker can also stop these requests.</p>' +
      detail + '</div>';
  } else {
    ov.innerHTML =
      '<div class="boot-err"><b>Could not load the UMAP data.</b>' + detail +
      '<p><a href="selftest.html">Run the renderer self-test</a></p></div>';
  }
});


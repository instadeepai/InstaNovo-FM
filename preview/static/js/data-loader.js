"use strict";
/* ---------------------------------------------------------------------------
   Tiered loader for the explorer's format-2 payload.

   Four waves:

     boot    a row prefix of the coordinates and three cheap labels, awaited
             before first paint. Skipped when the payload declares no boot
             shards, which it does when the row count is already small enough
             that a prefix would be the whole column.
     core    the rest of those columns plus the acquisition-physics labels.
     detail  the point panel and the filter fields.
     lazy    fetched on demand: the alternate layouts, the seldom-used numerics
             and the three high-cardinality level tables.

   Two things differ from the published loader beyond the tiering.

   Coordinates arrive as uint16 and are dequantised once, here, into Float32Array.
   Every downstream read of X[i] is then unchanged, and the saving is on the wire
   rather than in memory -- which is the right way round, since the wire is the
   constraint and 8 MB of resident floats is not.

   Level tables for the high-cardinality fields are a NUL-separated blob plus an
   offset array rather than JSON. At ~400,000 distinct peptides the JSON form is
   ~10 MB and about a second of blocking parse before anything renders, to
   materialise several hundred thousand strings of which the UI ever shows a few
   hundred. Here a level is decoded on demand and memoised.
   --------------------------------------------------------------------------- */

const TA = {uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array, float32: Float32Array};
const NAN_CODE = 65535, QUANT_MAX = 65534;
const FORMAT = 2;

let D = null, N = 0, NL = 0;                   /* catalog, rows, rows loaded so far */
let X2 = null, Y2 = null, X3 = null, Y3 = null, Z3 = null;
let X = null, Y = null;                        /* active horizontal plane */
let SAMPLE = null, SCAN = null;
let LAYOUT = null, HAS3D = false;
const CAT = {}, NUM = {}, ARR = {};
let REPORT_CATS = [], REPORT_NUMS = [];

const dequant = (code, lo, hi) =>
  code === NAN_CODE ? NaN : lo + code * (hi - lo) / QUANT_MAX;

const numVal = (k, i) => {
  const c = NUM[k];
  if (!c || !c.q) return NaN;
  return dequant(c.q[i], c.lo, c.hi);
};

/* Decode one level, memoised. The blob is only present for the high-cardinality
   fields; the rest carry their levels inline in the catalog. */
function levelAt(key, l){
  const c = CAT[key];
  if (!c || l == null || l < 0) return '—';
  if (c.levels) return c.levels[l] ?? '—';
  if (!c.blob || !c.offsets || l + 1 >= c.offsets.length) return '—';
  const hit = c.memo.get(l);
  if (hit !== undefined) return hit;
  const text = DECODER.decode(c.blob.subarray(c.offsets[l], c.offsets[l + 1] - 1));
  c.memo.set(l, text);
  return text;
}
const DECODER = new TextDecoder('utf-8');

const catStr = (k, i) => {
  const c = CAT[k];
  if (!c || !c.codes) return '—';
  return levelAt(k, c.codes[i]);
};

/* Per-level counts over the whole pool. The live counts the legend shows are
   computed by the viewer over the passing rows; these are the denominator. */
const catCount = (k, l) => {
  const c = CAT[k];
  return c && c.count ? (c.count[l] || 0) : 0;
};

/* ------------------------------------------------------------------ fetching */
let BASE = 'data/';
const inflight = new Map();

/* A privately published GitHub Pages site answers unauthenticated requests with a
   302 to github.com/pages/auth. A top-level navigation follows that happily, but
   fetch() cannot follow a cross-origin redirect, so it rejects with a bare
   "Failed to fetch". Detect it and say so, because the raw message sends people
   looking for a bug in the data. */
class LoadError extends Error {
  constructor(kind, message){ super(message); this.kind = kind; }
}

async function fetchChecked(path){
  const url = BASE + path;
  let r;
  try {
    r = await fetch(url);
  } catch (e) {
    try {
      const probe = await fetch(url, {redirect: 'manual'});
      if (probe.type === 'opaqueredirect' || (probe.status >= 300 && probe.status < 400))
        throw new LoadError('auth', 'the sign-in redirected the data request');
    } catch (inner) {
      if (inner instanceof LoadError) throw inner;
    }
    throw new LoadError('network', e.message || 'request failed');
  }
  if (r.redirected && new URL(r.url).origin !== location.origin)
    throw new LoadError('auth', 'the data request was redirected off-site');
  if (r.status === 401 || r.status === 403)
    throw new LoadError('auth', `HTTP ${r.status} on ${path}`);
  if (!r.ok)
    throw new LoadError('http', `${path}: HTTP ${r.status}`);
  return r;
}

async function fetchTyped(path, dtype){
  const buf = await (await fetchChecked(path)).arrayBuffer();
  return new TA[dtype](buf);
}
async function fetchBytes(path){
  return new Uint8Array(await (await fetchChecked(path)).arrayBuffer());
}
async function fetchJson(path){
  return (await fetchChecked(path)).json();
}

/* --------------------------------------------------------------- coordinates */

/** Dequantise a uint16 coordinate column into the Float32Array the viewer reads.
 *
 *  `into` lets the boot prefix and the full column share one allocation, so the
 *  prefix is never a separate array the viewer would have to know about.
 */
function toFloats(codes, lo, hi, into, rows){
  const out = into || new Float32Array(rows);
  for (let i = 0; i < codes.length; i++){
    const v = codes[i];
    out[i] = v === NAN_CODE ? NaN : lo + v * (hi - lo) / QUANT_MAX;
  }
  return out;
}

const axisKey = (layout, axis) => `${layout}.${axis}`;

/** Load one axis of one layout. Layouts differ only in coordinates, so switching
 *  between them refetches at most five small columns and nothing else. */
function loadAxis(layout, axis){
  const spec = D.layouts[layout];
  if (!spec || !spec.axes[axis]) return Promise.resolve();
  const key = axisKey(layout, axis);
  if (ARR[key]) return Promise.resolve();
  const id = 'axis:' + key;
  if (inflight.has(id)) return inflight.get(id);

  const m = spec.axes[axis];
  const p = fetchTyped(m.path, 'uint16').then(codes => {
    ARR[key] = toFloats(codes, m.lo, m.hi, null, N);
  });
  inflight.set(id, p);
  return p;
}

/** Point the active plane at a layout, fetching its axes first. */
async function applyLayout(layout, mode){
  const spec = D.layouts[layout];
  if (!spec) throw new Error(`unknown layout ${layout}`);
  const axes = mode === 'fit3d' && spec.has3d ? ['x3', 'y3', 'z3'] : ['x', 'y'];
  await Promise.all(axes.map(a => loadAxis(layout, a)));
  LAYOUT = layout;
  HAS3D = !!spec.has3d;
  if (mode === 'fit3d' && spec.has3d){
    X = X3 = ARR[axisKey(layout, 'x3')];
    Y = Y3 = ARR[axisKey(layout, 'y3')];
    Z3 = ARR[axisKey(layout, 'z3')];
  } else {
    X = X2 = ARR[axisKey(layout, 'x')];
    Y = Y2 = ARR[axisKey(layout, 'y')];
  }
  return spec;
}

/* ------------------------------------------------------------------- columns */

/** Load one column exactly once; concurrent callers share the promise. */
function loadOne(kind, key){
  const id = kind + ':' + key;
  if (inflight.has(id)) return inflight.get(id);

  let p;
  if (kind === 'array'){
    const m = D.arrays[key];
    if (!m || ARR[key]) return Promise.resolve();
    p = fetchTyped(m.path, m.dtype).then(a => {
      ARR[key] = a;
      if (key === 'sample_idx') SAMPLE = a;
      if (key === 'scan') SCAN = a;
    });
  } else if (kind === 'cat'){
    const c = CAT[key];
    if (!c || c.codes) return Promise.resolve();
    const m = D.cats[key];
    const jobs = [fetchTyped(m.path, m.dtype)];
    /* Sidecar level table and counts, for the fields too large to inline. */
    jobs.push(m.levelsPath ? fetchBytes(m.levelsPath) : Promise.resolve(null));
    jobs.push(m.levelsOffsetPath ? fetchTyped(m.levelsOffsetPath, 'uint32') : Promise.resolve(null));
    jobs.push(m.countPath ? fetchTyped(m.countPath, 'uint32') : Promise.resolve(null));
    p = Promise.all(jobs).then(([codes, blob, offsets, count]) => {
      c.codes = codes;
      if (blob) c.blob = blob;
      if (offsets) c.offsets = offsets;
      if (count) c.count = count;
    });
  } else {
    const c = NUM[key];
    if (!c || c.q) return Promise.resolve();
    p = fetchTyped(D.nums[key].path, 'uint16').then(q => { c.q = q; });
  }

  inflight.set(id, p);
  return p;
}

/** Ensure a list of [kind, key] pairs is resident. */
function ensureCols(cols){
  return Promise.all((cols || []).map(([kind, key]) => loadOne(kind, key)));
}

function tierCols(tier){
  const out = [];
  for (const [k, m] of Object.entries(D.arrays)) if (m.tier === tier) out.push(['array', k]);
  for (const [k, m] of Object.entries(D.cats))   if (m.tier === tier) out.push(['cat', k]);
  for (const [k, m] of Object.entries(D.nums))   if (m.tier === tier) out.push(['num', k]);
  return out;
}

/** True once every column the point panel reads is resident. */
function detailReady(){
  return tierCols('detail').every(([kind, key]) =>
    kind === 'array' ? !!ARR[key] : kind === 'cat' ? !!CAT[key].codes : !!NUM[key].q);
}

let detailPromise = null;
const detailWaiters = [];
function onDetailReady(fn){ detailReady() ? fn() : detailWaiters.push(fn); }

/* ------------------------------------------------------------------ the boot */

/** Fetch the boot prefix into full-length arrays, so a partial load is just a
 *  smaller NL rather than a different data shape the viewer must branch on. */
async function loadBootShards(layout){
  const boot = D.boot || {};
  if (!Object.keys(boot).length) return 0;

  const jobs = [];
  for (const axis of ['x', 'y']){
    const path = boot[axisKey(layout, axis)];
    if (!path) return 0;                       /* boot covers another layout; skip it */
    const m = D.layouts[layout].axes[axis];
    jobs.push(fetchTyped(path, 'uint16').then(codes => {
      const full = new Float32Array(N);
      full.fill(NaN);
      toFloats(codes, m.lo, m.hi, full, N);
      ARR[axisKey(layout, axis)] = full;
      return codes.length;
    }));
  }
  for (const key of Object.keys(boot)){
    if (key.includes('.')) continue;           /* an axis, handled above */
    const m = D.cats[key];
    if (!m) continue;
    jobs.push(fetchTyped(boot[key], m.dtype).then(codes => {
      const full = new TA[m.dtype](N);
      full.set(codes);
      CAT[key].codes = full;
      CAT[key].partial = codes.length;
      return codes.length;
    }));
  }
  const counts = await Promise.all(jobs);
  return Math.min(...counts);
}

/* ---------------------------------------------------------------------- boot */

async function bootData(base){
  if (base) BASE = base.endsWith('/') ? base : base + '/';
  D = await fetchJson('catalog.json');
  if (D.format !== FORMAT)
    throw new LoadError('format',
      `payload is format ${D.format}, this viewer reads format ${FORMAT}`);
  N = D.n;

  /* Declare every column up front so the menus, legend and level counts are all
     addressable before any array has been fetched. */
  for (const k of D.catOrder){
    const m = D.cats[k];
    CAT[k] = {label: m.label, group: m.group, nlevels: m.nlevels,
              levels: m.levels || null, count: m.count || null,
              codes: null, blob: null, offsets: null, memo: new Map(), partial: 0};
  }
  for (const k of D.numOrder){
    const m = D.nums[k];
    NUM[k] = {label: m.label, group: m.group, lo: m.lo, hi: m.hi, q: null};
  }
  REPORT_CATS = D.catOrder.slice();
  REPORT_NUMS = D.numOrder.slice();

  const layout = D.defaultLayout || Object.keys(D.layouts)[0];

  /* First paint on the boot prefix if there is one, otherwise straight to the
     full core tier -- which is what a small payload does, having no boot shards. */
  const booted = await loadBootShards(layout);
  NL = booted || 0;
  if (!booted){
    await applyLayout(layout, 'plane');
    await ensureCols(tierCols('core'));
    NL = N;
  } else {
    /* The prefix is already in the full-length arrays, so point the plane at them
       without refetching, then let the remainder land in the background. */
    LAYOUT = layout;
    X = X2 = ARR[axisKey(layout, 'x')];
    Y = Y2 = ARR[axisKey(layout, 'y')];
    HAS3D = !!D.layouts[layout].has3d;
  }

  const rest = (async () => {
    if (booted){
      /* Refetch the full columns over the prefix. The prefix rows are identical,
         so a viewer reading them mid-flight sees no discontinuity, only more rows. */
      for (const k of Object.keys(CAT)) if (CAT[k].partial){ CAT[k].codes = null; CAT[k].partial = 0; }
      inflight.clear();
      await applyLayout(layout, 'plane');
      await ensureCols(tierCols('core'));
      NL = N;
    }
    detailPromise = ensureCols(tierCols('detail')).then(() => {
      while (detailWaiters.length) detailWaiters.shift()();
    }).catch(e => console.error('detail tier failed', e));
    return detailPromise;
  })();

  return {layout, booted, rest};
}

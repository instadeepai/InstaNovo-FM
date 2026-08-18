"use strict";
/* ---------------------------------------------------------------------------
   Tiered loader for the Figure-3 UMAP point set.

   The reviewed prototype carried all 8.2 MB of arrays inline as base64 inside a
   single `const D = {...}`. Here the same arrays live as raw binary shards and
   arrive in three waves:

     core    blocked on before first paint  (2-D coords + acquisition-physics labels)
     detail  fetched in the background right after first paint (point panel, filters)
     lazy    fetched on demand              (the 3-D fit, seldom-used numerics)

   The globals published below are exactly the shapes the viewer expects, so the
   plotting code is unchanged from the prototype apart from the load guards.
   --------------------------------------------------------------------------- */

const TA = {uint8:Uint8Array, uint16:Uint16Array, uint32:Uint32Array, float32:Float32Array};

let D = null, N = 0;
let X2 = null, Y2 = null, X3 = null, Y3 = null, Z3 = null;
let X = null, Y = null;                        /* active horizontal plane */
let SAMPLE = null, SCAN = null;
let HAS3D = false;
const CAT = {}, NUM = {}, ARR = {};
let REPORT_CATS = [], REPORT_NUMS = [];

const numVal = (k, i) => {
  const c = NUM[k];
  if (!c || !c.q) return NaN;
  const v = c.q[i];
  return v === 65535 ? NaN : c.lo + v * (c.hi - c.lo) / 65534;
};
const catStr = (k, i) => {
  const c = CAT[k];
  if (!c || !c.codes || !c.levels) return '—';
  return c.levels[c.codes[i]];
};

/* ------------------------------------------------------------------ fetching */
const BASE = 'data/';
const inflight = new Map();

async function fetchTyped(path, dtype){
  const r = await fetch(BASE + path.replace(/^data\//, ''));
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  return new TA[dtype](buf);
}
async function fetchJson(path){
  const r = await fetch(BASE + path.replace(/^data\//, ''));
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

/** Load one column/array exactly once; concurrent callers share the promise. */
function loadOne(kind, key){
  const id = kind + ':' + key;
  if (inflight.has(id)) return inflight.get(id);

  let p;
  if (kind === 'array'){
    const m = D.arrays[key];
    if (!m) return Promise.resolve();
    if (ARR[key]) return Promise.resolve();
    p = fetchTyped(m.path, m.dtype).then(a => {
      ARR[key] = a;
      if (key === 'x'){ X2 = a; if (!X) X = a; }
      if (key === 'y'){ Y2 = a; if (!Y) Y = a; }
      if (key === 'x3') X3 = a;
      if (key === 'y3') Y3 = a;
      if (key === 'z3') Z3 = a;
      if (key === 'sample_idx') SAMPLE = a;
      if (key === 'scan') SCAN = a;
    });
  } else if (kind === 'cat'){
    const c = CAT[key];
    if (!c || c.codes) return Promise.resolve();
    const m = D.cats[key];
    p = Promise.all([
      fetchTyped(m.path, m.dtype),
      m.levelsPath ? fetchJson(m.levelsPath) : Promise.resolve(m.levels)
    ]).then(([codes, levels]) => { c.codes = codes; c.levels = levels; });
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

/* ---------------------------------------------------------------------- boot */
async function bootData(){
  D = await fetchJson('manifest.json');
  N = D.n;
  HAS3D = !!(D.arrays.x3 && D.arrays.y3 && D.arrays.z3);

  /* Declare every column up front so `k in CAT` / `k in NUM` and the label,
     group, level-count and per-level counts are all available for the menus
     and legend before any array has been fetched. */
  for (const k of D.catOrder){
    const m = D.cats[k];
    CAT[k] = {label:m.label, group:m.group, levels:m.levels || null,
              nlevels:m.nlevels, count:m.count, codes:null};
  }
  for (const k of D.numOrder){
    const m = D.nums[k];
    NUM[k] = {label:m.label, group:m.group, lo:m.lo, hi:m.hi, q:null};
  }
  REPORT_CATS = D.reportCats.filter(k => k in CAT);
  REPORT_NUMS = D.reportNums.filter(k => k in NUM);

  await ensureCols(tierCols('core'));

  /* Non-blocking: the rest of the record streams in while the map is already up. */
  detailPromise = ensureCols(tierCols('detail')).then(() => {
    while (detailWaiters.length) detailWaiters.shift()();
  }).catch(e => console.error('detail tier failed', e));
}

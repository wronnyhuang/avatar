/* percept_worker.js — ONE MediaPipe model per worker (face | hand | pose), driven by app/perception.js.
 *
 * init  {type:'init', model, assets:{bundle, loader, wasm, model}}   (absolute or blob: URLs)
 *   → {type:'ready', model, edges?:Uint16Array}   face: FACE_LANDMARKS_TESSELATION pairs (sent once);
 *                                                  hand: HAND_CONNECTIONS pairs; pose: POSE_CONNECTIONS pairs
 *   → {type:'error', model, fatal:true, message}
 * frame {type:'frame', bmp:ImageBitmap (transferred), t:ms (performance timeline), id}
 *   → {type:'result', model, t, id, infer, w, h, ...compact payload...}
 *        face: present, lm:Float32Array(478*3) RAW x,y,z (unmirrored image coords 0..1), bs:Float32Array(52)
 *              (order = `names` sent once with 'ready'), mat:Float32Array(16) column-major or null
 *        hand: hands:[{lm:Float32Array(63) RAW x,y,z, label:'Left'|'Right' (MediaPipe's raw handedness label),
 *              lscore, gesture, gscore}]
 *        pose: present, lm:Float32Array(33*4) RAW x,y,z,visibility
 *        (unified mode: init {world:true} adds pose wlm:Float32Array(33*4) world x,y,z (metres),visibility and
 *         per hand wlm:Float32Array(63) world x,y,z (metres); without it the output is unchanged)
 *   → {type:'error', model, fatal:false, message}   (one bad frame; the worker stays alive)
 * All geometry (mirroring, angles, filtering) happens on the main thread — this file only runs the model.
 */
'use strict';
let task = null, model = '', lastTs = 0, names = null, world = false;

function fetchBuf(url, ms) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  return fetch(url, {signal: ctl.signal, cache: 'force-cache'}).then(r => {
    if (!r.ok) throw Error('asset HTTP ' + r.status);
    return r.arrayBuffer();
  }).catch(e => { throw Error((e && e.name === 'AbortError') ? 'asset timeout after ' + (ms / 1000) + ' s' : String(e && e.message || e)); })
    .finally(() => clearTimeout(timer));
}
function pairs(list) {
  const out = new Uint16Array((list || []).length * 2);
  (list || []).forEach((c, i) => { out[2 * i] = c.start; out[2 * i + 1] = c.end; });
  return out;
}

async function init(m) {
  model = m.model;
  world = !!m.world;
  importScripts(m.assets.bundle);
  const buf = await fetchBuf(m.assets.model, 120000);
  const wasm = {wasmLoaderPath: m.assets.loader, wasmBinaryPath: m.assets.wasm};
  const base = {baseOptions: {modelAssetBuffer: new Uint8Array(buf), delegate: 'CPU'}, runningMode: 'VIDEO'};
  let edges = null;
  if (model === 'face') {
    task = await Vision.FaceLandmarker.createFromOptions(wasm, Object.assign(base, {numFaces: 1,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true}));
    edges = pairs(Vision.FaceLandmarker.FACE_LANDMARKS_TESSELATION);
  } else if (model === 'hand') {
    task = await Vision.GestureRecognizer.createFromOptions(wasm, Object.assign(base, {numHands: 2}));
    edges = pairs(Vision.GestureRecognizer.HAND_CONNECTIONS);
  } else if (model === 'pose') {
    task = await Vision.PoseLandmarker.createFromOptions(wasm, Object.assign(base, {numPoses: 1}));
    edges = pairs(Vision.PoseLandmarker.POSE_CONNECTIONS);
  } else throw Error('unknown model ' + model);
  const msg = {type: 'ready', model, edges};
  self.postMessage(msg, edges ? [edges.buffer] : []);
}

function flat(lms, stride) {
  const out = new Float32Array(lms.length * stride);
  for (let i = 0; i < lms.length; i++) {
    const p = lms[i], o = i * stride;
    out[o] = p.x; out[o + 1] = p.y; out[o + 2] = p.z || 0;
    if (stride === 4) out[o + 3] = p.visibility == null ? 1 : p.visibility;
  }
  return out;
}

function runFrame(m) {
  const bmp = m.bmp, w = bmp.width, h = bmp.height;
  let ts = Math.round(m.t);
  if (ts <= lastTs) ts = lastTs + 1;   // VIDEO mode needs strictly increasing timestamps
  lastTs = ts;
  const t0 = performance.now();
  let r;
  try { r = model === 'hand' ? task.recognizeForVideo(bmp, ts) : task.detectForVideo(bmp, ts); }
  finally { bmp.close(); }
  const infer = performance.now() - t0;
  const out = {type: 'result', model, t: m.t, id: m.id, infer, w, h};
  const transfer = [];
  if (model === 'face') {
    const lm = r.faceLandmarks && r.faceLandmarks[0];
    out.present = !!(lm && lm.length);
    if (out.present) {
      out.lm = flat(lm, 3); transfer.push(out.lm.buffer);
      const cats = (r.faceBlendshapes && r.faceBlendshapes[0] && r.faceBlendshapes[0].categories) || [];
      if (!names && cats.length) { names = cats.map(c => c.categoryName); out.names = names; }
      out.bs = new Float32Array(cats.length);
      for (let i = 0; i < cats.length; i++) out.bs[i] = cats[i].score;
      transfer.push(out.bs.buffer);
      const mat = r.facialTransformationMatrixes && r.facialTransformationMatrixes[0];
      out.mat = mat && mat.data ? Float32Array.from(mat.data) : null;
      if (out.mat) transfer.push(out.mat.buffer);
    }
  } else if (model === 'hand') {
    out.hands = (r.landmarks || []).map((lm, i) => {
      const hd = (r.handedness && r.handedness[i] && r.handedness[i][0]) || (r.handednesses && r.handednesses[i] && r.handednesses[i][0]) || {};
      const g = (r.gestures && r.gestures[i] && r.gestures[i][0]) || {};
      const f = flat(lm, 3); transfer.push(f.buffer);
      const o = {lm: f, label: hd.categoryName || '', lscore: hd.score || 0, gesture: g.categoryName || 'None', gscore: g.score || 0};
      if (world) { const wl = r.worldLandmarks && r.worldLandmarks[i]; if (wl && wl.length) { o.wlm = flat(wl, 3); transfer.push(o.wlm.buffer); } }
      return o;
    });
  } else {
    const lm = r.landmarks && r.landmarks[0];
    out.present = !!(lm && lm.length);
    if (out.present) { out.lm = flat(lm, 4); transfer.push(out.lm.buffer); }
    if (out.present && world) { const wl = r.worldLandmarks && r.worldLandmarks[0]; if (wl && wl.length) { out.wlm = flat(wl, 4); transfer.push(out.wlm.buffer); } }
  }
  self.postMessage(out, transfer);
}

self.onmessage = async e => {
  const m = e.data;
  if (m.type === 'init') {
    try { await init(m); }
    catch (err) { self.postMessage({type: 'error', model: m.model, fatal: true, message: String(err && err.message || err).slice(0, 240)}); }
    return;
  }
  if (m.type === 'frame') {
    if (!task) { try { m.bmp.close(); } catch (_) {} self.postMessage({type: 'result', model, t: m.t, id: m.id, skipped: true}); return; }
    try { runFrame(m); }
    catch (err) { self.postMessage({type: 'error', model, fatal: false, t: m.t, id: m.id, message: 'inference: ' + String(err && err.message || err).slice(0, 200)}); }
  }
};

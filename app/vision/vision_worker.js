/* Camera inference stays in this worker. Only compact landmarks/gesture labels go to the page. */
'use strict';
let face, hands, ready = false, busy = false, mode = 'both', lastFaceY = null, lastFaceT = 0;
let palmTrail = [], previousWaveSign = 0, waveTurns = 0;
const clip = (v, lo=0, hi=1) => Math.max(lo, Math.min(hi, v));
function withTimeout(url, ms) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), ms);
  return fetch(url, {signal:ctl.signal, cache:'force-cache'}).then(r => {
    if (!r.ok) throw Error('asset HTTP ' + r.status);
    return r.arrayBuffer();
  }).finally(() => clearTimeout(t));
}
function shapeMap(rows) {
  const out = {};
  const values=Array.isArray(rows)?rows:(rows && Array.isArray(rows.categories)?rows.categories:[]);
  for (const r of values) out[r.categoryName] = r.score;
  return out;
}
function signalFromFace(r, t) {
  if (!r.faceLandmarks || !r.faceLandmarks.length) return {present:false};
  const lm = r.faceLandmarks[0], b = shapeMap((r.faceBlendshapes || [])[0]);
  const eyeL = lm[33], eyeR = lm[263], nose = lm[1], mouth = lm[13];
  const eyeMidX = (eyeL.x + eyeR.x) / 2, eyeMidY = (eyeL.y + eyeR.y) / 2;
  const eyeSpan = Math.max(.08, Math.abs(eyeL.x - eyeR.x));
  const faceY = (nose.y + eyeMidY) / 2;
  const dy = lastFaceY === null ? 0 : (faceY - lastFaceY) / Math.max(.033, (t-lastFaceT)/1000);
  lastFaceY = faceY; lastFaceT = t;
  const smile = ((b.mouthSmileLeft||0)+(b.mouthSmileRight||0))/2;
  const frown = ((b.mouthFrownLeft||0)+(b.mouthFrownRight||0))/2;
  const surprise = clip(((b.jawOpen||0)-.22)*1.7 + ((b.eyeWideLeft||0)+(b.eyeWideRight||0))*.35);
  const yaw = clip((nose.x-eyeMidX)/eyeSpan*2.2,-1,1);
  const pitch = clip(((nose.y-eyeMidY)/eyeSpan-.58)*1.7,-1,1);
  return {present:true,x:clip(nose.x),y:clip(nose.y),yaw,pitch,smile,frown,surprise,
          mesh:lm.map(p=>[p.x,p.y]),
          blink:((b.eyeBlinkLeft||0)+(b.eyeBlinkRight||0))/2,
          mouth: b.jawOpen||0,nod: Math.abs(dy)>.28 && Math.abs(dy)<3 ? Math.sign(dy):0,
          capture_ms:t, detector:'mediapipe-face'};
}
function signalFromHands(r,t) {
  const lm = (r.landmarks||[])[0];
  if (!lm) { palmTrail=[];waveTurns=0;previousWaveSign=0;return {present:false}; }
  const raw = ((r.gestures||[])[0]||[])[0];
  const label = raw && raw.score>.45 ? raw.categoryName : 'None';
  const wrist = lm[0], palm = lm[9], index = lm[8], thumb = lm[4];
  palmTrail.push({t,x:wrist.x});
  while(palmTrail.length && palmTrail[0].t<t-750) palmTrail.shift();
  if(palmTrail.length>=2) {
    const a=palmTrail[palmTrail.length-2], dx=(wrist.x-a.x)/Math.max(.02,(t-a.t)/1000);
    const sign=Math.abs(dx)>.33?Math.sign(dx):0;
    if(sign && previousWaveSign && sign!==previousWaveSign) waveTurns++;
    if(sign) previousWaveSign=sign;
  }
  const wave=(label==='Open_Palm' && waveTurns>=1);
  const pinch=Math.hypot(index.x-thumb.x,index.y-thumb.y)<.075;
  const held=pinch && wrist.y<.67;
  return {present:true,x:clip(wrist.x),y:clip(wrist.y),label,wave,thumbs_up:label==='Thumb_Up',
          skeleton:(r.landmarks||[]).slice(0,2).map(hand=>hand.map(p=>[p.x,p.y])),
          point:label==='Pointing_Up',hold_up:held,score:raw?raw.score:0,
          index:{x:clip(index.x),y:clip(index.y)},capture_ms:t,detector:'mediapipe-gesture'};
}
self.onmessage=async e=>{
  const m=e.data;
  if(m.type==='init') {
    try {
      mode=m.mode||'both';
      importScripts(m.assets.bundle);
      const wasm={wasmLoaderPath:m.assets.loader,wasmBinaryPath:m.assets.wasm};
      if(mode==='face'||mode==='both'){
        const fb=await withTimeout(m.assets.face,120000);
        face=await Vision.FaceLandmarker.createFromOptions(wasm,{baseOptions:{modelAssetBuffer:new Uint8Array(fb),delegate:'CPU'},
          runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true});
      }
      if(mode==='hand'||mode==='both'){
        const hb=await withTimeout(m.assets.gesture,120000);
        hands=await Vision.GestureRecognizer.createFromOptions(wasm,{baseOptions:{modelAssetBuffer:new Uint8Array(hb),delegate:'CPU'},
          runningMode:'VIDEO',numHands:2});
      }
      ready=true;self.postMessage({type:'ready',mode});
    }catch(err){self.postMessage({type:'error',message:String(err).slice(0,180)});}
    return;
  }
  if(m.type==='frame') {
    if(!ready||busy){m.bmp.close();return;}
    busy=true;
    try {
      const t=Math.max(1,Math.floor(m.t));
      const t0=performance.now(),fr=face?face.detectForVideo(m.bmp,t):null,
            t1=performance.now(),hr=hands?hands.recognizeForVideo(m.bmp,t):null,t2=performance.now();
      const result={type:'perception',mode,face:fr?signalFromFace(fr,t):null,hand:hr?signalFromHands(hr,t):null,
                    capture_ms:m.t,done_ms:performance.now(),infer_ms:t2-t0,face_ms:t1-t0,hand_ms:t2-t1};
      self.postMessage(result);
    }catch(err){self.postMessage({type:'error',message:'camera inference: '+String(err).slice(0,140)});}
    finally{m.bmp.close();busy=false;}
  }
};

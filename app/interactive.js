/* Phone-side perception and 3-D pose controller. MediaPipe assets are first-party. */
(() => {
  'use strict';
  const stage = document.getElementById('stage');
  const opt = document.querySelector('.opts');
  const css = document.createElement('style');
  css.textContent = `#selfcam,#track{position:absolute;right:8px;bottom:8px;width:23%;aspect-ratio:4/3;object-fit:cover;
      transform:scaleX(-1);border:2px solid #fff9;border-radius:9px;display:none;background:#222}
    #track{border:0;background:transparent;pointer-events:none;opacity:.67}
    #percept{font-size:12px;color:#93a0b0;margin:3px 2px}
    #cameraLabel{color:#e8ecf1}`;
  document.head.appendChild(css);
  const cam=document.createElement('video');cam.id='selfcam';cam.autoplay=true;cam.muted=true;cam.playsInline=true;stage.appendChild(cam);
  const track=document.createElement('canvas');track.id='track';stage.appendChild(track);
  const tx=track.getContext('2d',{alpha:true});
  const lbl=document.createElement('label');lbl.id='cameraLabel';lbl.innerHTML='<input type="checkbox" id="camon" checked> See me';
  opt.prepend(lbl);
  const tlabel=document.createElement('label');tlabel.innerHTML='<input type="checkbox" id="trackon" checked> Show tracking';
  opt.insertBefore(tlabel,lbl.nextSibling);
  const status=document.createElement('div');status.id='percept';status.textContent='Camera is optional. If enabled, it reads expressions and gestures on your phone; one small still per second may go to GPT Live while talking.';
  opt.insertAdjacentElement('afterend',status);
  const cbox=document.getElementById('camon');
  const trackOn=document.getElementById('trackon');
  const R={on:false,ready:false,readyModes:{face:false,hand:false},workers:{},stream:null,
    busy:{face:false,hand:false},perception:null,emotion:'calm',
    action:null,actionUntil:0,actionAt:0,seenAt:0,from:'idle',lastShot:0,lastPose:0,
    lastGesture:'',lastGestureAt:0,lastExpr:'calm',lastExprAt:0,expressionState:{accepted:'calm',pending:'calm',pendingAt:0},commandSeen:{},
    channels:{},smoothPose:{},poseAt:0,transcriptBuffer:'',commandLog:[],lastLocalParts:[],lastLocalAt:0,
    measurements:{face:[],gesture:[],commands:[],inference:[],frames:0,stills:0,errors:[]}};
  const A=(name)=>url('vision/'+name);
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const avg=(xs)=>xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:0;
  // BEGIN EMOTION FILTER: keep subtle blendshape noise from making the face chatter.
  function emotionFromFace(f,state,t){
    const scores={surprised:Number(f?.surprise)||0,happy:Number(f?.smile)||0,concerned:Number(f?.frown)||0};
    const enter={surprised:.48,happy:.37,concerned:.35};
    const exit={surprised:.32,happy:.25,concerned:.23};
    let target=state.accepted;
    if(!f?.present||!exit[target]||scores[target]<exit[target])target='calm';
    for(const name of ['surprised','happy','concerned']){
      if(scores[name]>enter[name] && (target==='calm'||scores[name]>scores[target]+.12)){
        target=name;break;
      }
    }
    if(target!==state.pending){state.pending=target;state.pendingAt=t;}
    if(target!==state.accepted){
      const strong=target!=='calm'&&scores[target]>enter[target]+.20;
      const hold=target==='calm'?180:100;
      if(strong||t-state.pendingAt>=hold)state.accepted=target;
    }
    return state.accepted;
  }
  // END EMOTION FILTER
  function note(s, bad=false){status.textContent=s;status.style.color=bad?'#e5534b':'#93a0b0';}
  function record(kind,ms){const q=R.measurements[kind];if(q){q.push(Math.round(ms));if(q.length>200)q.shift();}}
  const PARTS=new Set(['head_yaw','head_pitch','head_tilt','head_nod','head_shake','left_ear_up','right_ear_up',
    'left_ear_down','right_ear_down','left_ear_flop','right_ear_flop','left_ear_wiggle','right_ear_wiggle',
    'ears_wiggle','nose_wiggle','nose_scrunch','nose_sniff','eyes_x','eyes_y',
    'eyes_blink','left_eye_wink','right_eye_wink','eyes_widen','eyes_squint','left_brow_raise','right_brow_raise',
    'left_brow_furrow','right_brow_furrow','mouth_open','mouth_smile','mouth_frown','mouth_pout','tongue_out',
    'cheeks_puff','cheeks_blush','left_arm_raise','right_arm_raise','left_arm_wave','right_arm_wave',
    'left_paw_point','right_paw_point','left_paw_open','right_paw_open','clap','thumbs_up','tail_wag',
    'tail_raise','body_lean','body_bounce']);
  function animate(motions,source='model'){
    const now=performance.now();let count=0;
    for(const m of motions||[]){if(!m||!PARTS.has(m.part))continue;
      const from=v(m.part,now);
      R.channels[m.part]={amount:clamp(Number(m.amount)||0,-1,1),from,at:now,
        until:now+clamp(Number(m.duration_ms)||2500,100,30000),repeat:!!m.repeat};count++;}
    if(count){R.commandPending={action:'animate',source,t:now,heardAt:R.commandHeardAt||null};
      note((R.on?'Camera on · ':'')+'Avatar: '+motions.slice(0,3).map(m=>m.part.replace(/_/g,' ')).join(', '));}
  }
  function v(part,now){const m=R.channels[part];if(!m)return 0;
    const ease=x=>{x=clamp(x,0,1);return x*x*(3-2*x);};
    const reached=m.from+(m.amount-m.from)*ease((now-m.at)/360);
    if(m.repeat||now<=m.until)return reached;
    const fade=1-ease((now-m.until)/520);
    if(fade<=0){delete R.channels[part];return 0;}
    return reached*fade;
  }
  function setAction(name,source='model',args={}){
    if(name==='animate'){animate(args.motions,source);return;}
    const valid=['look_up','look_down','look_left','look_right','turn_head','raise_hand','wave','point','open_mouth',
      'smile','frown','surprised','nod','shake_head','thumbs_up','set_emotion'];
    if(!valid.includes(name))return;
    if(name==='set_emotion'){R.emotion=String(args.emotion||'calm');R.actionUntil=performance.now()+2300;return;}
    R.action=name;R.actionAt=performance.now();R.actionUntil=R.actionAt+(name==='wave'?3500:2500);R.from=source;
    const legacy={look_up:[['eyes_y',-1],['head_pitch',-0.6]],look_down:[['eyes_y',1],['head_pitch',0.6]],
      look_left:[['eyes_x',-1]],look_right:[['eyes_x',1]],turn_head:[['head_yaw',1]],
      raise_hand:[['right_arm_raise',1]],wave:[['right_arm_wave',1]],point:[['right_paw_point',1],['right_arm_raise',.5]],
      open_mouth:[['mouth_open',1]],smile:[['mouth_smile',1]],frown:[['mouth_frown',1]],
      surprised:[['eyes_widen',1],['mouth_open',.8]],nod:[['head_nod',1]],
      shake_head:[['head_shake',1]],thumbs_up:[['thumbs_up',1],['right_arm_raise',.7]]};
    if(legacy[name])animate(legacy[name].map(([part,amount])=>({part,amount,duration_ms:name==='wave'?3500:2500,repeat:false})),source);
    R.commandPending={action:name,source,t:R.actionAt,heardAt:R.commandHeardAt||null};
    note((R.on?'Camera on · ':'')+'Avatar: '+name.replace(/_/g,' '));
  }
  function applyPerception(p){
    const now=performance.now();
    const prev=R.perception||{};
    R.perception={...prev,...p,face:p.face||prev.face||null,hand:p.hand||prev.hand||null};
    R.seenAt=now;R.busy[p.mode||'face']=false;
    drawTracking();
    R.measurements.frames++;record('inference',now-p.capture_ms);
    const f=R.perception.face||{},h=R.perception.hand||{};
    const emotion=p.face?emotionFromFace(f,R.expressionState,p.capture_ms):R.lastExpr;
    if(p.face&&emotion!==R.lastExpr){R.lastExpr=emotion;R.lastExprAt=p.capture_ms;R.pendingExpr={value:emotion,t:p.capture_ms};}
    const gesture=h.wave?'wave':h.thumbs_up?'thumbs_up':h.point?'point':h.hold_up?'hold_up':'';
    if(p.hand&&gesture && gesture!==R.lastGesture){R.lastGesture=gesture;R.lastGestureAt=p.capture_ms;R.pendingGesture={value:gesture,t:p.capture_ms};}
    if(p.hand&&!gesture && now-R.lastGestureAt>500)R.lastGesture='';
    note('Camera on · '+(f.present?emotion:'face not in view')+(gesture?' · '+gesture.replace(/_/g,' '):'')+
      ' · '+Math.round(avg(R.measurements.inference.slice(-12)))+' ms capture→perception');
  }
  const FACE_LINES=[[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10],
    [33,160,158,133,153,144,33],[362,385,387,263,373,380,362],
    [61,40,37,0,267,270,291,321,314,17,84,91,61],[168,6,1,4,5,195]];
  const HAND_LINES=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20],[5,9,13,17]];
  function drawTracking(){
    if(!R.on||!trackOn.checked){track.style.display='none';return;}
    track.style.display='block';const dpr=Math.min(3,devicePixelRatio||1);
    const rect=track.getBoundingClientRect(),w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));
    if(track.width!==w||track.height!==h){track.width=w;track.height=h;}
    tx.clearRect(0,0,w,h);const p=R.perception||{},mesh=p.face?.mesh,skels=p.hand?.skeleton;
    const drawLine=(pts,indices,color)=>{tx.strokeStyle=color;tx.lineWidth=Math.max(.7,dpr*.7);tx.lineJoin='round';
      tx.beginPath();let first=true;for(const i of indices){const q=pts[i];if(!q)continue;
        const x=q[0]*w,y=q[1]*h;if(first){tx.moveTo(x,y);first=false;}else tx.lineTo(x,y);}tx.stroke();};
    if(mesh){tx.fillStyle='#c8f5ffb0';for(const q of mesh){tx.beginPath();tx.arc(q[0]*w,q[1]*h,Math.max(.38,dpr*.35),0,Math.PI*2);tx.fill();}
      for(const line of FACE_LINES)drawLine(mesh,line,'#b6f1ffc7');}
    for(const hand of skels||[]){for(const line of HAND_LINES)drawLine(hand,line,'#ffdba9d8');
      tx.fillStyle='#ffe6b5';for(const q of hand){tx.beginPath();tx.arc(q[0]*w,q[1]*h,Math.max(.8,dpr*.7),0,Math.PI*2);tx.fill();}}
  }
  function startWorker(){
    if(R.workers.face||R.workers.hand)return;
    const timer=setTimeout(()=>{if(!R.ready&&R.on)note('Some camera models timed out; voice still works.',true);},180000);
    for(const mode of ['face','hand']){
      const worker=new Worker(A('vision_worker.js'));
      R.workers[mode]=worker;
      worker.onmessage=e=>{
        const m=e.data;
        if(m.type==='ready'){
          R.readyModes[mode]=true;R.ready=R.readyModes.face&&R.readyModes.hand;
          if(R.ready){clearTimeout(timer);note('Camera ready — expressions and hands are tracked on your phone.');}
        }else if(m.type==='perception')applyPerception(m);
        else if(m.type==='error'){R.busy[mode]=false;R.measurements.errors.push(m.message);note('Camera tracking: '+m.message,true);}
      };
      worker.onerror=e=>{R.busy[mode]=false;note('Camera model error: '+e.message,true);};
      worker.postMessage({type:'init',mode,assets:{bundle:A('vision_bundle.js'),loader:A('vision_wasm_internal.js'),
        wasm:A('vision_wasm_internal.wasm'),face:A('face_landmarker.task'),gesture:A('gesture_recognizer.task')}});
    }
  }
  async function startCamera(){
    if(R.on||!cbox.checked)return;
    try{
      if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw Error('camera unavailable in this browser');
      const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480},
        frameRate:{ideal:30,max:30}},audio:false});
      R.stream=stream;cam.srcObject=stream;await cam.play();cam.style.display='block';R.on=true;drawTracking();
      note('Loading camera models…');startWorker();requestAnimationFrame(cameraLoop);
    }catch(e){note('Camera unavailable: '+e.message+' · voice still works.',true);}
  }
  function stopCamera(){
    R.on=false;R.ready=false;R.readyModes={face:false,hand:false};R.busy={face:false,hand:false};
    if(R.stream){R.stream.getTracks().forEach(t=>t.stop());R.stream=null;}
    cam.srcObject=null;cam.style.display='none';track.style.display='none';
    for(const w of Object.values(R.workers))w.terminate();R.workers={};
    note('Camera off.');
  }
  const shotCanvas=document.createElement('canvas');shotCanvas.width=320;shotCanvas.height=180;
  const shotCtx=shotCanvas.getContext('2d',{alpha:false});
  function cameraLoop(now){
    if(!R.on)return;
    requestAnimationFrame(cameraLoop);
    if(cam.readyState<2)return;
    for(const mode of ['face','hand'])if(R.readyModes[mode]&&!R.busy[mode]&&(!R.lastInfer||now-(R.lastInfer[mode]||0)>=33)){
      R.lastInfer=R.lastInfer||{};R.lastInfer[mode]=now;R.busy[mode]=true;
      createImageBitmap(cam,0,0,cam.videoWidth,cam.videoHeight,{resizeWidth:320,resizeHeight:240})
        .then(bmp=>{if(R.on&&R.workers[mode])R.workers[mode].postMessage({type:'frame',bmp,t:now,started:now},[bmp]);
          else{bmp.close();R.busy[mode]=false;}},e=>{R.busy[mode]=false;note('Camera frame: '+e.message,true);});
    }
    // A small unmirrored still, at most one per second, to the delegated vision backend.
    if(S.live&&S.stage==='live'&&now-R.lastShot>=1000){
      R.lastShot=now;
      shotCtx.drawImage(cam,0,0,320,180);
      const data=shotCanvas.toDataURL('image/jpeg',.45);
      if(data.length<120000){
        const p=R.perception||{},f=p.face||{},h=p.hand||{};
        const observation=(f.present?'Face '+(R.lastExpr||'neutral')+'. ':'')+
          (h.present?'Hand '+(h.wave?'waving':h.label||'visible')+'. ':'');
        sendCtl({type:'camera_frame',image:data,observation});R.measurements.stills++;
      }
    }
  }
  const oldGo=document.getElementById('go').onclick;
  document.getElementById('go').onclick=()=>{if(!S.live&&cbox.checked)startCamera();else if(S.live)stopCamera();oldGo();};
  cbox.onchange=()=>{if(cbox.checked&&S.live)startCamera();else if(!cbox.checked)stopCamera();};
  trackOn.onchange=drawTracking;
  addEventListener('pagehide',stopCamera);
  setInterval(()=>{if(R.on&&!S.live&&!S.starting)stopCamera();},600);
  const oldOnJSON=onJSON;
  const commands=[
    [/\bturn (?:your |its )?head (?:to )?(?:the )?(left|right)\b/,m=>[['head_yaw',m[1]==='left'?-1:1]]],
    [/\btilt (?:your |its )?head (?:to )?(?:the )?(left|right)\b/,m=>[['head_tilt',m[1]==='left'?-1:1]]],
    [/\bnod(?: your head)?\b/,()=>[['head_nod',1]]],
    [/\bshake (?:your |its )?head\b/,()=>[['head_shake',1]]],
    [/\bwiggl(?:e|ing) (?:your )?(?:(left|right|both) )?ears?\b/,m=>m[1]==='left'?[['left_ear_wiggle',1]]:
      m[1]==='right'?[['right_ear_wiggle',1]]:[['ears_wiggle',1]]],
    [/\b(?:flop|pop) (?:your )?(left|right) ear\b/,m=>[[m[1]+'_ear_flop',1]]],
    [/\b(?:lift|raise) (?:both of )?(?:your )?ears? up\b/,()=>[['left_ear_up',1],['right_ear_up',1]]],
    [/\b(?:lower|drop) (?:your )?(left|right) ear\b/,m=>[[m[1]+'_ear_down',1]]],
    [/\bwiggle (?:your )?nose\b/,()=>[['nose_wiggle',1]]],
    [/\bscrunch (?:your )?nose\b/,()=>[['nose_scrunch',1]]],
    [/\bsniff(?: the air)?\b/,()=>[['nose_sniff',1]]],
    [/\blook (up|down|left|right)(?: with your eyes)?\b/,m=>m[1]==='up'?[['eyes_y',-1]]:
      m[1]==='down'?[['eyes_y',1]]:[['eyes_x',m[1]==='left'?-1:1]]],
    [/\bwink (?:your )?(left|right) eye\b/,m=>[[m[1]+'_eye_wink',1]]],
    [/\bblink(?: both eyes)?\b/,()=>[['eyes_blink',1]]],
    [/\b(?:widen|open wide) (?:your )?eyes\b/,()=>[['eyes_widen',1]]],
    [/\bsquint(?: your eyes)?\b/,()=>[['eyes_squint',1]]],
    [/\braise (?:your )?(left|right) eye ?brow\b/,m=>[[m[1]+'_brow_raise',1]]],
    [/\b(?:furrow|lower) (?:your )?(left|right) eye ?brow\b/,m=>[[m[1]+'_brow_furrow',1]]],
    [/\bopen (?:your |its )?mouth\b/,()=>[['mouth_open',1]]],
    [/\b(?:stick|put) (?:your )?tongue out\b/,()=>[['tongue_out',1]]],
    [/\bpuff (?:out )?(?:your )?cheeks\b/,()=>[['cheeks_puff',1]]],
    [/\braise (?:your )?(left|right) arm\b/,m=>[[m[1]+'_arm_raise',1]]],
    [/\braise both (?:your )?arms\b/,()=>[['left_arm_raise',1],['right_arm_raise',1]]],
    [/\bwave (?:your )?(left|right) (?:paw|hand)\b/,m=>[[m[1]+'_arm_wave',1]]],
    [/\bpoint (?:with )?(?:your )?(left|right) (?:paw|hand)\b/,m=>[[m[1]+'_paw_point',1]]],
    [/\bclap (?:your )?(?:paws|hands)\b/,()=>[['clap',1]]],
    [/\bwag (?:your )?tail\b/,()=>[['tail_wag',1]]],
    [/\blean (?:your )?body (?:to )?(?:the )?(left|right)\b/,m=>[['body_lean',m[1]==='left'?-1:1]]],
    [/\bbounce (?:your )?body\b/,()=>[['body_bounce',1]]],
    [/\bsmile\b/,()=>[['mouth_smile',1]]],
    [/\bfrown\b/,()=>[['mouth_frown',1]]],
  ];
  function transcriptAction(delta){
    if(R.ignoreCommandUntil&&performance.now()<R.ignoreCommandUntil)return;
    R.transcriptBuffer=(R.transcriptBuffer+delta.toLowerCase().replace(/[^a-z0-9% ]+/g,' '))
      .replace(/\s+/g,' ').slice(-240);
    if(/\b(?:don t|dont|do not|stop)\b/.test(R.transcriptBuffer.slice(-60))){
      for(const c of Object.values(R.channels)){c.repeat=false;c.until=performance.now();}
      R.transcriptBuffer='';R.ignoreCommandUntil=performance.now()+2200;return;}
    for(let pass=0;pass<4;pass++){
      let best=null;
      for(const [re,map] of commands){const m=re.exec(R.transcriptBuffer);
        if(m&&(!best||m.index<best.m.index))best={m,map};}
      if(!best)break;
      const {m,map}=best,parts=map(m),prefix=R.transcriptBuffer.slice(0,m.index);
      const repeat=/\bkeep\b/.test(prefix),now=performance.now();
      const percent=prefix.match(/(\d{1,3})\s*%/);
      const scale=percent?clamp(Number(percent[1])/100,0,1):/\b(?:slightly|a little|gently)\b/.test(prefix)?.35:
        /\bhalfway\b/.test(prefix)?.5:1;
      R.commandHeardAt=now;animate(parts.map(([part,amount])=>({part,amount:amount*scale,duration_ms:2500,repeat})),'voice transcript');
      R.commandLog.push({t:Math.round(now),phrase:m[0],parts:parts.map(x=>x[0]),source:'voice transcript'});
      R.lastLocalParts=now-R.lastLocalAt<3000?[...new Set([...R.lastLocalParts,...parts.map(x=>x[0])])]:parts.map(x=>x[0]);
      R.lastLocalAt=now;
      if(R.commandLog.length>200)R.commandLog.shift();
      R.transcriptBuffer=R.transcriptBuffer.slice(m.index+m[0].length);
    }
    const duration=R.transcriptBuffer.match(/\bfor (\d+|one|two|three|four|five|ten) seconds?\b/);
    if(duration&&R.lastLocalParts.length&&performance.now()-R.lastLocalAt<6000){
      const numbers={one:1,two:2,three:3,four:4,five:5,ten:10};
      const ms=Math.min(30000,1000*(numbers[duration[1]]||Number(duration[1])||3));
      for(const part of R.lastLocalParts)if(R.channels[part]){R.channels[part].repeat=false;R.channels[part].until=performance.now()+ms;}
      R.transcriptBuffer=R.transcriptBuffer.slice(duration.index+duration[0].length);
    }
  }
  onJSON=function(j){oldOnJSON(j);
    if(j.type==='action')setAction(j.action,j.source||'model',j.args||{});
    if(j.type==='transcript'&&j.who==='you')transcriptAction(String(j.text||''));
  };
  // Visible body reactions are renderer parameters; there is no full-stage canvas.
  function draw(now){
    requestAnimationFrame(draw);
    const p=R.perception||{},f=p.face||{},h=p.hand||{};
    const valid=R.on&&R.seenAt&&now-R.seenAt<650;
    const action=now<R.actionUntil?R.action:null;
    const expression=v('mouth_smile',now)>.2?'happy':v('mouth_frown',now)>.2?'concerned':
      action==='smile'?'happy':action==='frown'?'concerned':action==='surprised'?'surprised':
      R.actionUntil>now&&R.emotion!=='calm'?R.emotion:(valid?R.lastExpr:'calm');
    const gesture=action&&['wave','raise_hand','point','thumbs_up'].includes(action)?action:
      valid?(h.wave?'wave':h.thumbs_up?'thumbs_up':h.point?'point':h.hold_up?'point':''):'';
    const lookX=action==='look_left'?-1:action==='look_right'?1:
      v('eyes_x',now)||v('head_yaw',now)||(valid&&f.present?clamp((f.x-.5)*2,-1,1):0);
    const lookY=action==='look_up'?-1:action==='look_down'?1:
      v('eyes_y',now)||(valid&&f.present?clamp((f.y-.5)*1.7,-1,1):0);
    const nod=v('head_nod',now)*Math.sin(now/110)*.24;
    const shake=v('head_shake',now)*Math.sin(now/120)*.30;
    const talking=S.userEndCt!==null&&S.ctx&&S.ctx.currentTime-S.userEndCt<.3;
    const blink=(Math.sin(now/780)>.997||(valid&&f.blink>.65)||v('eyes_blink',now)>.2)?1:0;
    const ear=(side)=>{
      const x=side<0?'left':'right';
      const wiggle=v('ears_wiggle',now)+v(x+'_ear_wiggle',now);
      return {pose:clamp(v(x+'_ear_up',now)-v(x+'_ear_down',now)-v(x+'_ear_flop',now)
            +wiggle*Math.sin(now/105)*.45,-1,1),
        flick:clamp(Math.abs(wiggle*Math.sin(now/105)),0,1)};
    };
    const el=ear(-1),er=ear(1);
    const mouthOpen=Math.max(v('mouth_open',now),expression==='surprised'?.62:0,v('tongue_out',now)*.47);
    const armL=v('left_arm_raise',now),armR=v('right_arm_raise',now);
    const waveL=v('left_arm_wave',now),waveR=Math.max(v('right_arm_wave',now),gesture==='wave'?1:0);
    const pose={
      smile:clamp(Math.max(expression==='happy'?.72:0,v('mouth_smile',now)),0,1),
      brow_knit:clamp(Math.max(expression==='concerned'?.8:0,v('mouth_frown',now)*.5),0,1),
      head_yaw:clamp(lookX*.30+v('head_yaw',now)*.15+shake,-.5,.5),
      head_pitch:clamp(v('head_pitch',now)*.32-lookY*.17+nod+(talking?Math.sin(now/180)*.02:0),-.4,.4),
      head_roll:clamp(v('head_tilt',now)*.32,-.4,.4),gaze_x:clamp(lookX,-1,1),gaze_y:clamp(-lookY,-1,1),
      blink_l:Math.max(blink,v('left_eye_wink',now),v('eyes_squint',now)*.65),
      blink_r:Math.max(blink,v('right_eye_wink',now),v('eyes_squint',now)*.65),
      brow_l:clamp(v('left_brow_raise',now)-v('left_brow_furrow',now)+(expression==='surprised'?.6:0),-1,1),
      brow_r:clamp(v('right_brow_raise',now)-v('right_brow_furrow',now)+(expression==='surprised'?.6:0),-1,1),
      ear_l:el.pose,ear_r:er.pose,ear_flick_l:el.flick,ear_flick_r:er.flick,
      cheek:clamp(v('cheeks_puff',now)+v('cheeks_blush',now)*.6,0,1),
      nostril:clamp(Math.max(v('nose_wiggle',now)*(.5+.5*Math.sin(now/90)),v('nose_scrunch',now),
        v('nose_sniff',now)*(.5+.5*Math.sin(now/130))),0,1),
      lip_open:clamp(mouthOpen,0,1),teeth_gap:clamp(mouthOpen*.8,0,1),
      lip_width:clamp(v('mouth_smile',now)*.6-v('mouth_pout',now)*.85-v('mouth_frown',now)*.3,-1,1),
      lip_protrude:clamp(v('mouth_pout',now),0,1),tongue_dental:clamp(v('tongue_out',now),0,1),
      arm_l:clamp(armL,0,1),arm_r:clamp(Math.max(armR,gesture==='raise_hand'?1:0),0,1),
      wave_l:clamp(waveL,0,1),wave_r:clamp(waveR,0,1),
      point_l:clamp(v('left_paw_point',now),0,1),point_r:clamp(Math.max(v('right_paw_point',now),gesture==='point'?1:0),0,1),
      paw_open_l:clamp(v('left_paw_open',now),0,1),paw_open_r:clamp(v('right_paw_open',now),0,1),
      clap:clamp(v('clap',now),0,1),thumbs_up:clamp(Math.max(v('thumbs_up',now),gesture==='thumbs_up'?1:0),0,1),
      tail_wag:clamp(v('tail_wag',now),-1,1),tail_raise:clamp(v('tail_raise',now),0,1),
      body_lean:clamp(v('body_lean',now),-1,1),body_bounce:clamp(v('body_bounce',now),0,1)
    };
    if(R.pendingExpr){record('face',now-R.pendingExpr.t);R.pendingExpr=null;}
    if(R.pendingGesture){record('gesture',now-R.pendingGesture.t);R.pendingGesture=null;}
    if(R.commandPending){const q=R.commandPending;record('commands',now-(q.heardAt||q.t));R.commandPending=null;}
    const dt=R.poseAt?Math.min(100,now-R.poseAt):16;R.poseAt=now;
    for(const [k,target] of Object.entries(pose)){
      const prev=R.smoothPose[k]??target;
      const tau=/^(head_|body_|arm_|wave_|point_|tail_)/.test(k)?115:70;
      R.smoothPose[k]=prev+(target-prev)*(1-Math.exp(-dt/tau));
    }
    if(S.live&&now-R.lastPose>80){R.lastPose=now;sendCtl({type:'pose',pose:R.smoothPose});}
  }
  requestAnimationFrame(draw);
  window.__avatar.interactive={state:R,startCamera,stopCamera,setAction,perceive:applyPerception,
    transcriptAction,
    measurements:()=>({face:R.measurements.face,gesture:R.measurements.gesture,commands:R.measurements.commands,
      inference:R.measurements.inference,stills:R.measurements.stills,errors:R.measurements.errors,
      commandLog:R.commandLog})};
})();

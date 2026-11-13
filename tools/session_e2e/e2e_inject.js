/* e2e_inject.js — injected into the page by driver.cjs (Playwright add_init_script). The page's own code is untouched.
 *
 * 1. getUserMedia override: audio requests get a WebAudio MediaStreamDestination that the harness feeds per turn
 *    (window.__e2e.say(base64 int16 pcm, sampleRate)) over a -60 dBFS noise floor; video requests go to the real
 *    (fake-device) camera.
 * 2. window.__e2e.tap(): wraps AvatarVoice.createVoice so every voice callback (tool calls with args + results, transcripts,
 *    user-speech start/stop, responses, metrics, state) is logged with wall-clock ms; a 20 Hz sampler logs avatar audio
 *    level / speaking, the motion state (changes only), new action-log events, and any optional state object the page
 *    exposes: window.__avatar.e2eState() or .board() / .tv() / .teach() / .stations() / .program() (JSON, logged on change,
 *    kind "state:<name>"; {busy:true} in e2eState() keeps the harness waiting).
 * 3. window.__e2e.drain() returns the new log rows + sample rows since the last call.
 */
(() => {
  if (window.__e2e) return;
  const E = window.__e2e = {log: [], samples: [], li: 0, si: 0, ac: null, dest: null, says: [], pending: {}, last: {}, busy: false, tapped: false};
  const now = () => Date.now();
  const cut = (x, n) => { let s; try { s = typeof x === 'string' ? x : JSON.stringify(x); } catch (e) { s = String(x); } return s === undefined ? '' : (s.length > n ? s.slice(0, n) + '…' : s); };
  const push = (kind, o) => { E.log.push(Object.assign({w: now(), kind}, o || {})); };
  E.push = push;

  // ---------------------------------------------------------------- 1. the user's microphone
  function ensureAudio() {
    if (E.ac) return;
    const ac = E.ac = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 48000});
    E.dest = ac.createMediaStreamDestination();
    const n = ac.sampleRate * 2, buf = ac.createBuffer(1, n, ac.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * 0.0017;          // ~ -60 dBFS room floor
    const src = ac.createBufferSource(); src.buffer = buf; src.loop = true; src.connect(E.dest); src.start();
    ac.resume().catch(() => {});
  }
  const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async c => {
    if (!c || !c.audio) return gum(c);
    ensureAudio(); await E.ac.resume().catch(() => {});
    const s = new MediaStream();
    for (const t of E.dest.stream.getAudioTracks()) s.addTrack(t.clone());
    if (c.video) { const v = await gum({video: c.video}); for (const t of v.getVideoTracks()) s.addTrack(t); }
    push('mic_open', {tracks: s.getTracks().map(t => t.kind)});
    return s;
  };
  E.say = (b64, sr, label) => {
    ensureAudio();
    const bin = atob(b64), n = bin.length >> 1, f = new Float32Array(n);
    for (let i = 0; i < n; i++) { let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); if (v > 32767) v -= 65536; f[i] = v / 32768; }
    const buf = E.ac.createBuffer(1, n, sr); buf.getChannelData(0).set(f);
    const src = E.ac.createBufferSource(); src.buffer = buf; src.connect(E.dest);
    const w0 = now(); src.start();
    const rec = {start_w: w0, dur_ms: Math.round(n / sr * 1000), label: label || ''}; E.says.push(rec); E.curSay = rec;
    push('user_audio_start', {label: rec.label, dur_ms: rec.dur_ms});
    setTimeout(() => push('user_audio_end', {label: rec.label}), rec.dur_ms);
    return rec;
  };

  // ---------------------------------------------------------------- 2. the tap
  E.tap = () => {
    if (E.tapped) return true;
    const AV = window.AvatarVoice;
    if (!AV || !AV.createVoice) return false;
    E.tapped = true;
    const orig = AV.createVoice;
    AV.createVoice = function (o) {
      const o2 = Object.assign({}, o);
      const wrap = (name, fn) => { const f = o[name]; o2[name] = function () { try { fn.apply(null, arguments); } catch (e) {} return f ? f.apply(this, arguments) : undefined; }; };
      o2.onCall = function (name, args, phase, info) {
        const id = info && (info.call_id || info.id) || '';
        push('tool', {name, phase, id, args: cut(args, 700)});
        if (phase === 'start') E.pending[id || name] = now();
        let r; try { r = o.onCall ? o.onCall.apply(this, arguments) : undefined; } catch (e) { push('tool', {name, phase: 'throw', id, err: cut(String(e && e.message || e), 200)}); throw e; }
        if (phase === 'done' || phase === 'early') push('tool_result', {name, phase, id, result: cut(r, 900)});
        if (phase === 'done') delete E.pending[id || name];
        return r;
      };
      wrap('onTranscript', (who, text, fin, info) => {
        const k = who === 'you' ? 'user' : 'avatar';
        if (fin) { push('tx', {who: k, text: cut(text, 600), final: true}); E.last[k + '_open'] = false; }
        else if (!E.last[k + '_open']) { E.last[k + '_open'] = true; push('tx_start', {who: k, text: cut(text, 60)}); }
      });
      wrap('onUserSpeech', ph => push('user_speech', {ph}));
      wrap('onResponse', (ph, info) => push('response', {ph, info: cut(info, 200)}));
      wrap('onMetric', m => { if (m && m.type === 'turn') push('metric_turn', {m: cut(m, 500)}); else if (m && m.type === 'usage') push('usage', {m: cut(m, 400)}); });
      wrap('onState', (s, d) => push('voice_state', {state: s, detail: cut(d, 200)}));
      wrap('onInterrupt', () => push('interrupt', {}));
      const v = orig.call(this, o2);
      E.voiceObj = v;
      return v;
    };
    // sampler
    const seen = new WeakSet(); let lastSig = '', mst = 0; const lastOpt = {};
    setInterval(() => {
      const A = window.__avatar; if (!A) return;
      const D = A.D || {}; const w = now();
      let rms = 0, voiced = 0, spk = 0, urms = 0;
      try { const av = D.play && D.play.avatarVoice(); if (av) { rms = av.rms || 0; voiced = av.voiced ? 1 : 0; } } catch (e) {}
      try { spk = D.voice && D.voice.speaking && D.voice.speaking() ? 1 : 0; } catch (e) {}
      try { const uv = D.voice && D.voice.userVoice && D.voice.userVoice(); urms = uv ? (uv.rms || 0) : 0; } catch (e) {}
      let act = 0;
      if (++mst % 2 === 0) {
        try {
          const st = A.mstate && A.mstate();
          if (st) {
            const names = (st.active || []).map(a => typeof a === 'string' ? a : (a && (a.name || a.kind || a.id)) || '?');
            act = names.length; E.activeNames = names;
            const sig = JSON.stringify([st.posture, st.facing, names, st.holding, (st.props || []).length]);
            if (sig !== lastSig) { lastSig = sig; push('state', {posture: st.posture, facing: st.facing, active: names, holding: cut(st.holding, 160), at: cut(st.place || st.at || st.station || null, 80)}); }
          }
        } catch (e) {}
        try {
          const evs = D.alogEvents || [];
          for (const e of evs) { if (typeof e !== 'object' || seen.has(e)) continue; seen.add(e); if (e.src === 'cospeech' || e.src === 'life') continue;
            push('motion', {name: e.name, mkind: e.kind, src: e.src, phase: e.phase, mid: e.id, reason: cut(e.reason, 120), args: cut(e.args, 200)}); }
        } catch (e) {}
        let busy = act > 0;
        for (const k of ['e2eState', 'board', 'tv', 'teach', 'stations', 'program']) {
          let fn = A[k] || (D[k] && (D[k].state || D[k].snapshot));
          if (typeof fn !== 'function') continue;
          try { const val = fn.call(A[k] ? A : D[k]); const s = cut(val, 1500); if (s !== lastOpt[k]) { lastOpt[k] = s; push('state:' + k, {v: s}); }
            if (k === 'e2eState' && val && val.busy) busy = true; } catch (e) {}
        }
        E.busy = busy;
      }
      E.samples.push([w, +rms.toFixed(4), voiced, spk, +urms.toFixed(4), E.busy ? 1 : 0, Object.keys(E.pending).length]);
    }, 50);
    return true;
  };
  E.drain = () => {
    const r = {log: E.log.slice(E.li), samples: E.samples.slice(E.si)};
    E.li = E.log.length; E.si = E.samples.length; return r;
  };
  E.peek = () => {
    const s = E.samples[E.samples.length - 1] || [0, 0, 0, 0, 0, 0, 0];
    return {w: now(), rms: s[1], voiced: s[2], spk: s[3], urms: s[4], busy: s[5], pending: s[6], active: E.activeNames || [], tools: E.log.filter(x => x.kind === 'tool').slice(-6).map(x => x.name + ':' + x.phase)};
  };
  // console lines the page prints (boards / TV / stations builders log here): the driver also forwards page.on('console')
  window.addEventListener('error', e => push('page_error', {msg: cut(e.message, 300)}));
})();

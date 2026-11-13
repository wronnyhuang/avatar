#!/usr/bin/env node
// speech — does the action-log note cost first-audio latency? gpt-realtime-2.1, AUDIO in/out over WebSocket,
// server_vad 300 ms (the phone's settings), the muse persona. Two sessions interleaved turn by turn:
//   A = memory persona + a ~900-char "Your body's action log" note sent as he starts talking (the page's flush)
//   B = the baseline persona, no notes.
// Each turn streams a TTS question in real time (20 ms chunks) + silence; latency = end of his voiced audio →
// first response.output_audio.delta. Writes <out>/latency.json and prints an estimated cost (list price).
//   OPENAI_API_KEY=... node tools/actionmem/rt_latency.cjs [out_dir]     (default <repo>/runs/actionmem; PYTHON = interpreter for app/rtc.py)
'use strict';
const fs = require('fs'), path = require('path'), {execFileSync} = require('child_process');
const ROOT = path.resolve(__dirname, '..', '..');
const OUT = process.argv[2] || path.join(ROOT, 'runs', 'actionmem');
fs.mkdirSync(OUT, {recursive: true});
const KEY = process.env.OPENAI_API_KEY;
const PY = process.env.PYTHON || 'python3';
const sessFor = mem => JSON.parse(execFileSync(PY, ['-c', `import rtc, json; print(json.dumps(rtc.realtime_session('Jolly','plush muse','verse','server:300','','parallel', ${JSON.stringify(mem)}, 'muse')))`], {cwd: path.join(ROOT, 'app')}).toString());
const QS = ['What did you just do?', 'Tell me a short joke.', 'What are you holding right now?', "How's your day going?", 'Did you wave at me?',
  "What's your favorite color?", 'Are you still sitting at the desk?', 'Tell me something fun about clouds.'];
const NOTE = "Silent context (not words he said) - never read aloud, quote, describe or mention it: [Your body's action log, 1:12 into the call. The complete record of what you did on screen — if something is not in it, you did NOT do it.] 0:20, 52 s ago: put on the headphones (asked); 0:24, 48 s ago: pulled the chair over (asked); 0:28, 44 s ago: pulled the desk over (asked); 0:31, 41 s ago: sat down at the desk (asked); 0:35, 37 s ago: slid the laptop closer (asked); 0:38, 34 s ago: opened the laptop (asked); 0:41, 31 s ago: typed on the laptop for a while (asked), then stopped; 1:02, 10 s ago: turned back to face him (asked). Right now: sitting, facing him, looking at him, both hands empty (holding nothing), wearing the headphones, at the desk, the laptop is open.";
const RATES = {text_in: 4, audio_in: 32, cached: 0.5, text_out: 24, audio_out: 64};
const usd = u => { const i = u.input_token_details || {}, o = u.output_token_details || {}, c = i.cached_tokens_details || {};
  return (Math.max(0, (i.text_tokens || 0) - (c.text_tokens || 0)) * RATES.text_in + Math.max(0, (i.audio_tokens || 0) - (c.audio_tokens || 0)) * RATES.audio_in +
    ((c.text_tokens || 0) + (c.audio_tokens || 0)) * RATES.cached + (o.text_tokens || 0) * RATES.text_out + (o.audio_tokens || 0) * RATES.audio_out) / 1e6; };

async function tts(text) {
  const r = await fetch('https://api.openai.com/v1/audio/speech', {method: 'POST', headers: {Authorization: 'Bearer ' + KEY, 'content-type': 'application/json'},
    body: JSON.stringify({model: 'gpt-4o-mini-tts', voice: 'alloy', input: text, response_format: 'pcm'})});
  if (!r.ok) throw new Error('tts ' + r.status);
  return Buffer.from(await r.arrayBuffer());     // 24 kHz int16 mono
}
function open(sess) {
  return new Promise((res, rej) => {
    const ws = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1', ['realtime', 'openai-insecure-api-key.' + KEY]);
    const S = {ws, usd: 0, onev: null};
    ws.onmessage = e => { let m; try { m = JSON.parse(e.data); } catch (x) { return; } if (m.type === 'response.done' && m.response && m.response.usage) S.usd += usd(m.response.usage); if (S.onev) S.onev(m); };
    ws.onerror = () => rej(new Error('ws'));
    ws.onopen = () => { ws.send(JSON.stringify({type: 'session.update', session: {type: 'realtime', instructions: sess.instructions, output_modalities: ['audio'],
      audio: sess.audio, tools: sess.tools, tool_choice: 'auto'}})); setTimeout(() => res(S), 800); };
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function turn(S, pcm, note) {
  if (note) S.ws.send(JSON.stringify({type: 'conversation.item.create', item: {type: 'message', role: 'system', content: [{type: 'input_text', text: note}]}}));
  let first = null, done = null, txt = '';
  S.onev = m => { if (m.type === 'response.output_audio.delta' && first === null) first = Date.now(); if (m.type === 'response.output_audio_transcript.delta') txt += m.delta || '';
    if (m.type === 'response.done') done = Date.now(); };
  const CH = 960, t0 = Date.now();     // 20 ms of 24 kHz int16
  // find the last voiced sample (end of his speech)
  let lastV = 0; for (let i = 0; i < pcm.length; i += 2) if (Math.abs(pcm.readInt16LE(i)) > 500) lastV = i;
  let endT = null;
  const total = pcm.length + 24000 * 2 * 2;   // + 2 s silence
  for (let off = 0, k = 0; off < total; off += CH, k++) {
    const chunk = off < pcm.length ? pcm.subarray(off, Math.min(pcm.length, off + CH)) : Buffer.alloc(CH);
    S.ws.send(JSON.stringify({type: 'input_audio_buffer.append', audio: chunk.toString('base64')}));
    if (endT === null && off + CH > lastV) endT = Date.now();
    const due = t0 + (k + 1) * 20; const w = due - Date.now(); if (w > 0) await sleep(w);
    if (first) break;
  }
  for (let i = 0; i < 200 && !done; i++) await sleep(50);
  await sleep(600);
  return {lat_ms: first ? first - endT : null, said: txt};
}
(async () => {
  const pcms = [];
  for (const q of QS) pcms.push(await tts(q));
  const A = await open(sessFor('notes')), B = await open(sessFor('off'));
  const res = {A: [], B: []};
  for (let i = 0; i < QS.length; i++) {
    const order = i % 2 ? [['B', B, null], ['A', A, NOTE]] : [['A', A, NOTE], ['B', B, null]];
    for (const [k, S, note] of order) { const r = await turn(S, pcms[i], note); res[k].push(Object.assign({q: QS[i]}, r)); console.log(k, QS[i], r.lat_ms, '|', r.said.slice(0, 90)); }
  }
  A.ws.close(); B.ws.close();
  const med = a => { const s = a.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
  const p90 = a => { const s = a.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(0.9 * s.length))] : null; };
  const sum = {notes: {median: med(res.A.map(r => r.lat_ms)), p90: p90(res.A.map(r => r.lat_ms)), n: res.A.length},
    off: {median: med(res.B.map(r => r.lat_ms)), p90: p90(res.B.map(r => r.lat_ms)), n: res.B.length}, usd: +(A.usd + B.usd + 0.01).toFixed(4)};
  fs.writeFileSync(path.join(OUT, 'latency.json'), JSON.stringify({sum, res}, null, 1));
  console.log(JSON.stringify(sum));
})().catch(e => { console.error(e); process.exit(1); });

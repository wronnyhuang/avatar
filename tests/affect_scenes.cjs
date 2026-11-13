// joy: emotion readability eval scenes. The page on a dev server instance (tools/devinst.sh) in headless Chromium (?voice=mock): per scene the user's
// line arrives as a transcript, the mock vendor plays the reply WAV (<AVATAR_HOME>/mock/<id>.wav) through the real voice
// graph (lip sync, co-speech, the playout tap) and the reply text streams as realtime transcript deltas, so the server's
// affect judge (app/affect.py, Haiku through the claude CLI door) runs exactly as live. The stage is screenshotted every --every ms.
//   node tests/affect_scenes.cjs --url http://127.0.0.1:18474/ --key K --scenes scenes.json --out DIR [--only id,id] [--q 540p]
'use strict';
const fs = require('fs'), path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const CHROME = process.env.CHROME_PATH || process.env.CHROME || undefined;   // undefined = playwright's own chromium
const a = {q: '540p', every: 250, w: 960, h: 540, shotS: 11, gapS: 4};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
fs.mkdirSync(a.out, {recursive: true});
const sleep = ms => new Promise(r => setTimeout(r, ms));
let scenes = JSON.parse(fs.readFileSync(a.scenes, 'utf8'));
if (a.only) { const o = a.only.split(','); scenes = scenes.filter(s => o.includes(s.id)); }
(async () => {
  const browser = await chromium.launch({executablePath: CHROME, headless: true, args: ['--no-sandbox', '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream', '--mute-audio', '--autoplay-policy=no-user-gesture-required', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', '--enable-unsafe-swiftshader']});
  const ctx = await browser.newContext({viewport: {width: +a.w, height: +a.h + 160}, permissions: ['camera', 'microphone']});
  await ctx.addInitScript(q => { try { localStorage.setItem('avatar.q', q); } catch (e) {} }, a.q);
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(String(e.message).slice(0, 300)));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text().slice(0, 300)); });
  const u = new URL(a.url); u.searchParams.set('k', a.key); u.searchParams.set('voice', 'mock'); u.searchParams.set('char', 'muse'); u.searchParams.set('camera', '0');
  u.searchParams.set('mock', scenes[0].id + '.wav'); u.searchParams.set('mock_lat_ms', '300');
  if (a.extra) for (const kv of a.extra.split('&')) { const [k, v] = kv.split('='); u.searchParams.set(k, v); }
  await page.goto(u.toString(), {waitUntil: 'load', timeout: 60000});
  await sleep(1500);
  await page.click('#go', {timeout: 15000});
  await page.waitForFunction(() => window.__avatar && window.__avatar.D && window.__avatar.D.voice && window.__avatar.S.live, null, {timeout: 240000});
  for (let i = 0; i < 240; i++) {
    const fps = await page.evaluate(() => { const m = window.__avatar.S.lastMsg || {}; return m.fps || 0; }).catch(() => 0);
    if (fps >= 3) break; await sleep(500);
    if (i === 239) { console.error('NO VIDEO'); await browser.close(); process.exit(2); }
  }
  await sleep(3000);
  const stage = await page.$('#stage');
  for (const sc of scenes) {
    const dir = path.join(a.out, sc.id); fs.mkdirSync(dir, {recursive: true});
    // a clean start: the previous scene's feeling let go (eval isolation; the live app decays on its own)
    await page.evaluate(() => { try { const M = window.__avatar.D.motion; if (M && M.affectReset) M.affectReset(); window.__avatar.command({action: 'expression', name: 'neutral'}, 'test'); } catch (e) {} });
    await sleep(+a.gapS * 1000);
    const tl = [];
    const at0 = await page.evaluate(() => { const L = window.__avatar.D.affLast; return L ? L.at : null; });   // the previous scene's verdict is not this one's
    const t0 = Date.now();
    // his line, then the end of his turn (the mock vendor answers with this scene's WAV), then the reply text deltas
    await page.evaluate(async ([user, wav, key]) => {
      const A = window.__avatar, V = A.D.voice, ctx = A.S.ctx;
      const r = await fetch('mock/' + encodeURIComponent(wav) + '?k=' + encodeURIComponent(key), {cache: 'no-store'});
      const buf = await ctx.decodeAudioData(await r.arrayBuffer());
      V.feed({type: 'conversation.item.input_audio_transcription.completed', transcript: user});
      V.feed({type: 'test.mock_user_end', bufs: [{url: wav, buf}]});
    }, [sc.u, sc.id + '.wav', a.key]);
    const rid = 'resp_' + sc.id + '_' + Date.now();
    const words = sc.r.split(/(\s+)/);
    const feeder = (async () => {
      await sleep(250);   // the realtime text runs a little ahead of the audio (300 ms mock latency)
      for (let i = 0; i < words.length; i += 4) {
        const d = words.slice(i, i + 4).join('');
        await page.evaluate(([d2, r2]) => window.__avatar.D.voice.feed({type: 'response.output_audio_transcript.delta', response_id: r2, item_id: r2, delta: d2}), [d, rid]);
        await sleep(120);
      }
      await page.evaluate(([t2, r2]) => window.__avatar.D.voice.feed({type: 'response.output_audio_transcript.done', response_id: r2, item_id: r2, transcript: t2}), [sc.r, rid]);
    })();
    let n = 0;
    while (Date.now() - t0 < +a.shotS * 1000) {
      const t = Date.now();
      try {
        await stage.screenshot({path: path.join(dir, `f${String(n).padStart(3, '0')}.png`)});
        const st = await page.evaluate(() => { const D = window.__avatar.D; let cam = null; try { const L = D.lastRig || {}; cam = L.cam_zoom; } catch (x) {}
          return {aff: D.affLast || null, voiced: !!(D.play && D.play.avatarVoice().voiced), cam, affState: D.motion && D.motion.affectState ? D.motion.affectState() : null}; });
        tl.push({n, t: (t - t0) / 1000, ...st}); n++;
      } catch (e) { errors.push('shot: ' + e.message.slice(0, 120)); }
      const w = +a.every - (Date.now() - t); if (w > 0) await sleep(w);
    }
    await feeder;
    const aff = tl.map(x => x.aff).filter(x => x && x.at !== at0).pop() || null;
    const firstVoiced = (tl.find(x => x.voiced) || {}).t ?? null;
    const lastVoiced = ([...tl].reverse().find(x => x.voiced) || {}).t ?? null;
    fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify({scene: sc, frames: n, every: +a.every, aff, firstVoiced, lastVoiced, tl}, null, 1));
    console.log(JSON.stringify({id: sc.id, frames: n, aff: aff && {feel: aff.feel, i: aff.intensity, g: aff.gesture, ms: aff.ms, v: aff.v, a: aff.a, d: aff.d, label: aff.label}, voiced: [firstVoiced, lastVoiced]}));
  }
  await page.click('#go', {timeout: 5000}).catch(() => {});
  await sleep(1500);
  fs.writeFileSync(path.join(a.out, 'errors.json'), JSON.stringify([...new Set(errors)].slice(0, 40), null, 1));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });

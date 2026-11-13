// Does recording cost the session anything? (rec) A/B on the real page: ?rec=0 vs ?rec=1, alternating runs,
// same fake camera / mic, mock voice. Per run (after a warm-up): displayed avatar fps, frames late/dropped, the video
// round trip (D.rtt — the render path), perception face fps, main-thread long tasks (> 50 ms), the recorder's own draw ms.
//   node tests/rec_ab.cjs --url http://127.0.0.1:PORT/ --key K --cam clip.y4m --mic clip.wav [--runs 2] [--seconds 20]
'use strict';
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const a = {runs: 2, seconds: 20, warm: 8, char: 'muse', chrome: process.env.CHROME_PATH || undefined, mock: 'mock_01_hi.wav,mock_04_fact.wav,mock_02_food.wav'};
for (let i = 2; i < process.argv.length; i += 2) a[process.argv[i].replace(/^--/, '')] = process.argv[i + 1];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const med = xs => { const s = xs.filter(x => x != null).sort((p, q) => p - q); return s.length ? s[Math.floor(s.length / 2)] : null; };

async function once(rec) {
  const b = await chromium.launch({executablePath: a.chrome, headless: true, args: ['--no-sandbox', '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${a.cam}`, `--use-file-for-fake-audio-capture=${a.mic}`,
    '--autoplay-policy=no-user-gesture-required', '--mute-audio', '--disable-background-timer-throttling', '--disable-renderer-backgrounding']});
  const ctx = await b.newContext({viewport: {width: 412, height: 915}, deviceScaleFactor: 2, permissions: ['camera', 'microphone']});
  const p = await ctx.newPage();
  await p.addInitScript(() => { window.__lt = []; try { new PerformanceObserver(l => { for (const e of l.getEntries()) window.__lt.push(e.duration); }).observe({type: 'longtask', buffered: true}); } catch (e) {} });
  const u = new URL(a.url);
  for (const [k, v] of Object.entries({k: a.key, voice: 'mock', mock: a.mock, auto: '1', char: a.char, rec: rec ? '1' : '0', attention: '0'})) u.searchParams.set(k, v);
  await p.goto(u.toString(), {waitUntil: 'load', timeout: 60000});
  await p.waitForFunction(() => window.__avatar && window.__avatar.S && window.__avatar.S.live, null, {timeout: 60000});
  await sleep(+a.warm * 1000);
  const snap = () => p.evaluate(() => { const A = window.__avatar, S = A.S, D = A.D, ps = D.perc && D.perc.stats ? D.perc.stats() : null, r = A.rec ? A.rec() : null;
    return {t: performance.now(), drawn: S.drawn, late: S.st.late, dropped: S.st.dropped, rtt: D.rtt.slice(), face: ps && ps.models.face ? ps.models.face.fps : null,
      lt: window.__lt.slice(), rec: r && r.cur ? {p50: r.cur.draw_ms_p50, p95: r.cur.draw_ms_p95, fps: r.cur.fps} : null}; });
  const s0 = await snap();
  const faces = [];
  for (let i = 0; i < +a.seconds; i += 5) { await sleep(5000); faces.push((await snap()).face); }
  const s1 = await snap();
  await b.close();
  const dt = (s1.t - s0.t) / 1000, rtt = s1.rtt.slice(s0.rtt.length);
  const lt = s1.lt.slice(s0.lt.length);
  return {rec, fps: +((s1.drawn - s0.drawn) / dt).toFixed(1), late: s1.late - s0.late, dropped: s1.dropped - s0.dropped,
    rtt_p50: med(rtt), rtt_p90: rtt.length ? rtt.slice().sort((x, y) => x - y)[Math.floor(rtt.length * 0.9)] : null,
    face_fps: +(med(faces) || 0).toFixed(1), longtasks: lt.length, longtask_ms: Math.round(lt.reduce((x, y) => x + y, 0)), recDraw: s1.rec};
}
(async () => {
  const rows = [];
  for (let i = 0; i < +a.runs; i++) for (const rec of (i % 2 ? [1, 0] : [0, 1])) { const r = await once(rec); rows.push(r); console.log(JSON.stringify(r)); }
  const agg = rec => { const R = rows.filter(r => r.rec === rec), m = k => med(R.map(r => r[k]));
    return {fps: m('fps'), late: m('late'), dropped: m('dropped'), rtt_p50: m('rtt_p50'), rtt_p90: m('rtt_p90'), face_fps: m('face_fps'), longtasks: m('longtasks'), longtask_ms: m('longtask_ms')}; };
  console.log('rec=0', JSON.stringify(agg(0)));
  console.log('rec=1', JSON.stringify(agg(1)));
})();

// The Gemini mic path wired end to end in a real browser (voice.js 'gemini' mode, fake token door + fake WebSocket, an
// oscillator mic at a 24 kHz AudioContext): default = the anti-aliased resampler, micFir:false (?micfir=0) = the old
// sample picking. Checks the socket gets 320-sample audio/pcm;rate=16000 chunks, never the worklet's cfg note, and the
// 10 kHz tone's alias at 6 kHz is gone (NEW) / present (OLD).
//   PLAYWRIGHT_CORE=<.../playwright-core> CHROME_PATH=<chrome> node tests/mic_gemini_page_test.cjs
'use strict';
const fs = require('fs'), path = require('path');
let chromium;
try { ({chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core')); } catch (e) { console.log('SKIP: playwright-core not found (set PLAYWRIGHT_CORE=)'); process.exit(0); }
const CHROME = [process.env.CHROME_PATH, '/opt/google/chrome/chrome', '/usr/bin/google-chrome', '/usr/bin/chromium'].find(p => p && fs.existsSync(p));
(async () => {
  if (!CHROME) { console.log('SKIP: no Chrome (set CHROME_PATH=)'); process.exit(0); }
  const b = await chromium.launch({executablePath: CHROME, headless: true, args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required', '--allow-file-access-from-files']});
  let fail = 0;
  for (const fir of [null, false]) {
    const p = await b.newPage();
    await p.goto('file://' + path.join(__dirname, 'mic_gemini_page.html'));
    const r = await p.evaluate(f => window.run(f), fir);
    const db = 20 * Math.log10(Math.max(r.a6k, 1e-9) / r.a1k);
    const ok = r.n >= 40 && r.lens.length === 1 && r.lens[0] === 320 && r.mimes.join() === 'audio/pcm;rate=16000' && r.nonAudio === 0 &&
      (fir === false ? db > -6 && r.mic_rs && r.mic_rs.fir === false : db < -55 && r.mic_rs && r.mic_rs.fir === true && r.mic_rs.taps === 118);
    if (!ok) fail++;
    console.log(`${ok ? 'PASS' : 'FAIL'} ${fir === false ? 'micFir:false (old picking)' : 'default (anti-aliased)'}: ${r.n} chunks of ${r.lens} samples, ${r.mimes}; 10 kHz alias at 6 kHz = ${db.toFixed(1)} dB re the 1 kHz tone; stats.mic_rs ${JSON.stringify(r.mic_rs)}`);
    await p.close();
  }
  await b.close();
  console.log(fail ? `${fail} FAILED` : '2/2 passed');
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });

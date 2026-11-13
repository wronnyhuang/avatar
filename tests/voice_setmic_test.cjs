// rec: voice.js setMic in a real browser — after a mid-session mic switch the vendor hears the NEW mic, through
// the gate (gate stays, its input changes) and without it (the sender's track is replaced); the session is not redialed.
//   node tests/voice_setmic_test.cjs [--chrome /opt/google/chrome/chrome]
'use strict';
const path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_CORE || 'playwright-core');
const chrome = process.argv[3] || process.env.CHROME_PATH || undefined;   // undefined = playwright's own chromium
(async () => {
  const b = await chromium.launch({executablePath: chrome, headless: true, args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required', '--allow-file-access-from-files']});
  let fail = 0;
  for (const gate of [true, false]) {
    const p = await b.newPage();
    await p.goto('file://' + path.join(__dirname, 'voice_setmic_page.html'));
    const r = await p.evaluate(g => window.run(g), gate);
    const ok = r.ok && r.before.hz > 380 && r.before.hz < 500 && r.after.hz > 1250 && r.after.hz < 1400 && r.state === 'connecting' &&
      (gate ? r.gateOn && r.sameTrack : r.isB) && r.events.mic_switch === 1;
    if (!ok) fail++;
    console.log((ok ? 'PASS' : 'FAIL') + ` gate=${gate}: vendor hears ${r.before.hz} Hz -> ${r.after.hz} Hz after setMic` +
      (gate ? `, gate kept (same sender track ${r.sameTrack})` : `, sender track replaced (${r.isB})`) + `, session ${r.state}, no redial`);
    await p.close();
  }
  await b.close();
  console.log(fail ? `${fail} FAILED` : '2/2 passed');
  process.exit(fail ? 1 : 0);
})();

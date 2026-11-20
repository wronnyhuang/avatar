const AM = require('../../../app/motion.js');
function run(useHook) {
  const m = AM.createMotion({});
  m.setGesture({live: true});
  const sent = [];
  if (useHook) m.setClipHook((name, G) => (name === 'wave' || name === 'cover_eyes') ? (sent.push(name), {slug: name, dur: 2.0, owns: ['arm_r', 'fingers_r']}) : null, n => n === 'cover_eyes' || n === 'wave');
  let out = [];
  for (let i = 0; i < 120; i++) {
    if (i === 10) out.push(['cmd', JSON.stringify(m.command({action: 'gesture', name: 'wave'}, 'voice'))]);
    if (i === 60) out.push(['cmd2', JSON.stringify(m.command({action: 'gesture', name: 'cover_eyes'}, 'voice'))]);
    const r = m.step(1 / 30, {speaking: true});
    if (i % 10 === 5) out.push([i, (+r.hand_l_y).toFixed(3), (+r.hand_r_y).toFixed(3), (+r.gest_l).toFixed(2), (+r.gest_r).toFixed(2), (+r.smile).toFixed(2)]);
  }
  console.log(useHook ? 'HOOK' : 'NOHOOK', sent, out.map(x => x.join(' ')).join('\n'));
}
run(false); run(true);

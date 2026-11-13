// Test-only native landmark replay through the unchanged deployed perception functions.
// Never loaded by the avatar page. No browser or DOM emulation.
const fs = require('fs'), vm = require('vm'), readline = require('readline');
const source = process.argv[2];
let text = fs.readFileSync(source, 'utf8');
const marker = 'return {start, stop, setPreview, setMesh, latest: () => S.latest, stats, events: S.events,';
if (!text.includes(marker)) throw Error('perception replay injection point changed');
text = text.replace(marker, `return {_replay: frame => {
  S.running=true; S.ready={face:true,hand:true,pose:true};
  if(frame.motionFrame != null) S.motionFrame=frame.motionFrame;
  if(frame.face) onFace(frame.face);
  if(frame.hand) onHand(frame.hand);
  if(frame.pose) onPose(frame.pose);
  emitFrame(frame.t/1000);
  return S.latest;
}, start, stop, setPreview, setMesh, latest: () => S.latest, stats, events: S.events,`);
const sandbox = {module:{exports:{}}, performance, console, require};
vm.runInNewContext(text, sandbox, {filename:source});
const p = sandbox.module.exports.createPerception({pose:true});
const rl = readline.createInterface({input:process.stdin, crlfDelay:Infinity});
rl.on('line', line => {
 const raw=JSON.parse(line), features=p._replay(raw);
 // Camera mesh visualization isn't part of the motion controller input.
 if(features.face) delete features.face.mesh;
 process.stdout.write(JSON.stringify({t:raw.t/1000, features, events:features.events})+'\n');
});

// node tests/muse_planner_regressions_test.cjs — regressions: (1) the half-understood rule re-parses a clause from the words he
// said, not phrase tokens ('a bit' -> 'alittle'); (2) a late speech-start report with its true start keeps one request one
// utterance (dance-05: server 11 s late, transcript match 13.8 s after the fast matcher); (3) the muse's hop is a real
// small jump (both feet clearly off the ground), still under a jump; other characters' hop unchanged. Exit code = failures.
'use strict';
const fs = require('fs'), path = require('path');
const {makeMuse, museFiles, Moves} = require('./muse_lib.cjs');
const load = require('./load.cjs');
const {Intent, IntentDedupe: D} = load('app/intent.js');
let fails = 0;
const check = (n, ok, d) => { console.log((ok ? 'PASS ' : 'FAIL ') + n + (d ? '  — ' + d : '')); if (!ok) fails++; };
const rules = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'app', 'intent_rules.json')));
const m = Intent.create(rules, {act: true, scene: museFiles().props});
// (1) my own phrasings with phrase-token words in a later clause
for (const t of ['grab the laptop and push it a little bit closer', 'pick up the ball and toss it a bit higher',
  'take the mug and put it down for a sec']) {
  const hs = m.parse(t, {final: true}); const u = m.understood(t);
  const ws = []; for (const h of hs) ws.push(h.cmd.verb || h.cmd.name);
  check(`understood "${t}" when every clause parses`, hs.length < 2 || u, JSON.stringify(ws) + ' understood=' + u);
}
check('still half-understood: a clause that means nothing', !m.understood('jump and rotate backward all the way over'));
// (1b) put-back words on a furniture move mean its own spot; a plain 'push the chair back' stays 'back'
{ const t1 = m.parse('shove the desk back to where it came from', {final: true}).map((h) => h.cmd), t2 = m.parse('push the chair back', {final: true}).map((h) => h.cmd);
  check('"... back to where it came from" on a furniture move -> target home; plain "back" unchanged', t1.length === 1 && t1[0].target === 'home' && t2.length === 1 && t2[0].target === 'back', JSON.stringify([t1, t2])); }
// (2) dedupe across a late server speech-start
const dance = {action: 'move', name: 'dance'};
D.reset(); D.userSpoke(1120); D.seen(dance, 3600, 'matcher'); D.userSpoke(14000, 1000);
check('late report with true start: transcript match 13.8 s later is the same request', D.seen(dance, 17400, 'caption') === true);
check('... and so is the model call after it', D.seen(Object.assign({style: 'silly'}, dance), 17800, 'model') === true);
D.userSpoke(25000, 24800); check('a new utterance asking again still runs', D.seen(dance, 26500, 'matcher') === false);
D.reset(); D.userSpoke(1120); D.seen(dance, 3600, 'matcher'); D.userSpoke(12000, 9000);
check('a new utterance reported late (start after the command) still runs', D.seen(dance, 12500, 'caption') === false);
// (2b) hand_over: the matcher's 'other' paw and the model's named paw are one request (no mid-pass replacement)
D.reset(); D.userSpoke(1000); D.seen({action: 'object', verb: 'hand_over', object: {name: 'headphones'}, side: 'other'}, 3000, 'matcher');
check('hand_over other (matcher) then hand_over left (model): duplicate, not a replacement', D.seen({action: 'object', verb: 'hand_over', object: {ref: 'it'}, side: 'left'}, 4500, 'model') === true && !(D.last && D.last.conflict));
D.reset(); D.userSpoke(1000); D.seen({action: 'object', verb: 'pick_up', object: {name: 'ball'}, side: 'right'}, 3000, 'matcher');
check('pick_up with a different paw from the model still replaces', D.seen({action: 'object', verb: 'pick_up', object: {name: 'ball'}, side: 'left'}, 4500, 'model') === false && !!(D.last && D.last.conflict));
D.reset(); D.userSpoke(1000); D.seen({action: 'object', verb: 'pull', object: {name: 'chair'}, target: 'me'}, 3000, 'matcher');
check("pull chair 'me' (matcher) then pull chair 'closer' (model): one request", D.seen({action: 'object', verb: 'pull', object: {name: 'chair'}, target: 'closer', side: 'both'}, 3600, 'model') === true && !(D.last && D.last.conflict));
D.reset(); D.userSpoke(1000); D.seen({action: 'object', verb: 'push', object: {name: 'chair'}, target: 'away'}, 3000, 'matcher');
check("push 'away' then 'closer' still disagree", D.seen({action: 'object', verb: 'push', object: {name: 'chair'}, target: 'closer'}, 3600, 'model') === false);
// (3) hop heights on the muse
function span(cmd) {
  const T = makeMuse(3, {character: 'muse'}); T.step(10, false); const k0 = T.rows.length; T.m.command(Object.assign({action: 'move'}, cmd)); T.step(150, true);
  const ys = T.rows.slice(k0).map((r) => r.root_y || 0); return Math.max(...ys) - Math.min(...ys);
}
const hop = span({name: 'hop'}), small = span({name: 'hop', height: 'small'}), jump = span({name: 'jump'}), bounce = span({name: 'bounce'});
check('muse hop is a real small jump, above a bounce, under a jump', small > bounce && hop > bounce && hop < jump * 0.75, `hop ${hop.toFixed(3)} small ${small.toFixed(3)} bounce ${bounce.toFixed(3)} jump ${jump.toFixed(3)}`);
console.log(fails ? fails + ' failed' : 'all passed'); process.exit(fails);

// planner parity reference: exec_stream.cjs parseStream (the planner streaming bench, read-only) on recorded stream.json files.
//   node tests/planner_parse_ref.cjs <exec_stream.cjs> <stream.json>...  -> one JSON line per file: {path, parse}
// `parse` = the block exec_stream.cjs's main prints (poses with t to 3 decimals), without the player simulation.
'use strict';
const fs = require('fs');
const [execPath, ...files] = process.argv.slice(2);
const {parseStream} = require(require('path').resolve(execPath));
for (const f of files) {
  const S = JSON.parse(fs.readFileSync(f, 'utf8'));
  const parse = parseStream(S.chunks || [], S.t_done_ms ?? (S.chunks && S.chunks.length ? S.chunks[S.chunks.length - 1][0] : 0));
  const out = {n_poses: parse.poses.length, first_token_ms: parse.first_token_ms, first_pose_ms: parse.poses.length ? parse.poses[0].ms : null,
    complete_ms: parse.complete_ms, complete_how: parse.complete_how, found_poses_key: parse.found_poses_key, shift_s: parse.shift_s,
    dropped: parse.dropped.length, salvaged: parse.salvaged.length, unknown: parse.errors,
    poses: parse.poses.map(p => ({ms: p.ms, t: +p.t.toFixed(3), set: p.set})), plan_s: parse.poses.length ? +parse.poses[parse.poses.length - 1].t.toFixed(3) : 0};
  process.stdout.write(JSON.stringify({path: f, parse: out}) + '\n');
}

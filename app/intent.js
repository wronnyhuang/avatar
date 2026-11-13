/* Fast local intent matcher — the JS twin of app/intent.py (same engine, same intent_rules.json,
   pinned by tests/intent_vectors.json). Plain IIFE: works as a <script> (window.Intent, window.IntentDedupe)
   and under node (module.exports).

   const M = Intent.create(rules)          // rules = parsed intent_rules.json (the server can inline it)
   M.parse(text, {final, stable, nowMs})   // -> [{cmd, clause, rule, open}]
   const st = M.stream(); st.update(text, final, nowMs) -> [{cmd, clause, rule, partial}]  (fires once per utterance)
   IntentDedupe.seen(cmd, nowMs)           // true = the same body part was already commanded within ~3 s

   Side convention: "your left paw" = the character's own left = rig suffix 'r'; cmd.side is a rig suffix
   ('l'|'r'|'both'), cmd.own the character's own side. */
(function (root) {
  'use strict';
  const STABLE_MS = 200, REPEAT_WINDOW_MS = 1500, CTX_MS = 20000;
  const OWN = {l: 'right', r: 'left', both: 'both'};
  const r3 = (x) => Math.floor(x * 1000 + 0.5) / 1000;
  const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  const clone = (o) => JSON.parse(JSON.stringify(o));

  function normalize(text) {
    let t = String(text).toLowerCase().replace(/’/g, "'").replace(/‘/g, "'").replace(/'/g, '');
    t = t.replace(/[.!?;:]+/g, ' | ');   // commas are not clause breaks (ASR has none; captions do)
    t = t.replace(/[^a-z0-9| ]+/g, ' ');
    return t.split(' ').filter((w) => w);
  }

  function compile(data) {
    const R = {d: data, alias: data.alias, names: new Set(data.names), pre: new Set(data.pre),
      anchors: new Set(data.anchors), split: new Set(data.split), keep: new Set(data.keep), gerund: data.gerund,
      mods: data.mods, classes: Object.assign({}, data.classes, {emo: data.emo}), emoAlias: data.emo_alias,
      adjust: data.adjust, ranges: data.ranges, unit: new Set(data.unit_params), noscale: data.noscale,
      phr: {}, rules: [], nameParts: {}};
    data.phrases.forEach(([src, dst], i) => {
      const toks = src.split(' ');
      (R.phr[toks[0]] = R.phr[toks[0]] || []).push([toks.length, i, toks, dst]);
    });
    for (const k of Object.keys(R.phr)) {
      R.phr[k].sort((a, b) => (b[0] - a[0]) || (a[1] - b[1]));
      R.phr[k] = R.phr[k].map((e) => [e[2], e[3]]);
    }
    for (const r of data.rules) {
      const pats = [];
      for (const p of r.p) {
        const at = p.indexOf(' => ');
        const body = at >= 0 ? p.slice(0, at) : p, extra = at >= 0 ? p.slice(at + 4) : '';
        const fixed = {};
        if (extra) for (const kv of extra.split(' ')) { const e = kv.indexOf('='); fixed[e >= 0 ? kv.slice(0, e) : kv] = e >= 0 ? kv.slice(e + 1) : ''; }
        const elems = body.split(' ').map((tok) => { const opt = tok.startsWith('?'); return [opt, (opt ? tok.slice(1) : tok).split('/')]; });
        pats.push({src: p, elems, fixed, wait: (r.wait || []).includes(p)});
      }
      R.rules.push({r, pats});
    }
    for (const r of data.rules) {
      const c = r.c;
      if (c.name && !String(c.name).startsWith('$') && c.parts && !has(R.nameParts, c.name)) R.nameParts[c.name] = c.parts;
    }
    return R;
  }

  function applyPhrases(R, toks) {
    const out = [];
    let i = 0;
    while (i < toks.length) {
      let hit = null;
      for (const [ptoks, dst] of (has(R.phr, toks[i]) ? R.phr[toks[i]] : [])) {
        let ok = i + ptoks.length <= toks.length;
        for (let k = 0; ok && k < ptoks.length; k++) if (toks[i + k] !== ptoks[k]) ok = false;
        if (ok) { hit = [ptoks.length, dst]; break; }
      }
      if (hit) { out.push(hit[1]); i += hit[0]; } else { out.push(toks[i]); i += 1; }
    }
    return out;
  }

  function tokenize(R, text) {
    return applyPhrases(R, normalize(text).map((w) => (has(R.alias, w) ? R.alias[w] : w)));
  }

  function altMatch(R, alt, tok) {
    if (alt.startsWith('$')) {
      const cls = alt.slice(1), tab = R.classes[cls] || {};
      if (has(tab, tok)) return [true, cls, tab[tok]];
      return [false, null, null];
    }
    return [tok === alt, null, null];
  }

  function match(R, elems, toks, i, j, caps, out) {
    if (j === elems.length) { out.push([i, Object.assign({}, caps)]); return; }
    const [opt, alts] = elems[j];
    if (i < toks.length) {
      for (const alt of alts) {
        const [ok, key, val] = altMatch(R, alt, toks[i]);
        if (ok) {
          const c2 = Object.assign({}, caps);
          if (key && !has(c2, key)) c2[key] = val;
          match(R, elems, toks, i + 1, j + 1, c2, out);
        }
      }
    }
    if (opt) match(R, elems, toks, i, j + 1, caps, out);
  }

  function tail(R, toks) {
    const m = {int: 1.0, rep: null, hold: null};
    for (const t of toks) {
      if (R.names.has(t)) continue;
      if (!has(R.mods, t)) return null;
      const mod = R.mods[t];
      if (mod[0] === 'int') m.int *= mod[1];
      else if (mod[0] === 'rep') m.rep = mod[1];
      else if (mod[0] === 'hold') m.hold = mod[1];
    }
    return m;
  }

  function clampP(R, k, v) {
    let lo = -1.0, hi = 1.0;
    if (has(R.ranges, k)) [lo, hi] = R.ranges[k];
    else if (R.unit.has(k)) { lo = 0.0; hi = 1.0; }
    return Math.min(hi, Math.max(lo, v));
  }

  function scaled(R, k) {
    for (const s of R.noscale) {
      if ((s.startsWith('_') && k.endsWith(s)) || (!s.startsWith('_') && k.startsWith(s))) return false;
    }
    return true;
  }

  function build(R, rule, caps, mods) {
    const c = clone(rule.c);
    let side;
    if (has(caps, 'side')) side = caps.side;
    else if (rule.pside && ['paw', 'ear', 'eye', 'brow'].some((k) => caps[k] === 2)) side = 'both';
    else side = has(rule, 'side') ? rule.side : null;
    let lr = caps.lr || caps.dir || null;
    if (c.side === '$lrside') {
      if (lr !== 'left' && lr !== 'right') return null;
      side = lr === 'left' ? 'r' : 'l';
    }
    if (lr === null && rule.lrdefault) lr = rule.lrdefault;
    const sides = side === 'both' ? ['l', 'r'] : (side === 'l' || side === 'r' ? [side] : []);
    const k = rule.noint ? 1.0 : mods.int;
    if (has(c, 'targets')) {
      const t2 = {};
      for (const key of Object.keys(c.targets)) {
        let val = c.targets[key];
        const keys = key.includes('$') ? sides.map((s) => key.split('$').join(s)) : [key];
        if (val === '$yaw' || val === '$lat' || val === '$roll') {
          if (lr !== 'left' && lr !== 'right') return null;
          const mag = {$yaw: 0.55, $lat: 0.6, $roll: 0.5}[val];
          val = lr === 'left' ? mag : -mag;
        }
        for (const kk of keys) {
          const v = Number(val) * (scaled(R, kk) ? k : 1.0);
          t2[kk] = r3(clampP(R, kk, v));
        }
      }
      if (!Object.keys(t2).length) return null;
      c.targets = t2;
    }
    if (Array.isArray(c.parts)) {
      const ps = [];
      for (const p of c.parts) {
        for (const pp of (p.includes('$') ? sides.map((s) => p.split('$').join(s)) : [p])) if (!ps.includes(pp)) ps.push(pp);
      }
      c.parts = ps;
    }
    if (c.side === '$' || c.side === '$lrside') c.side = side;
    if (c.side === 'l' || c.side === 'r' || c.side === 'both') c.own = OWN[c.side];
    if (c.target === '$dir') {
      if (!lr) return null;
      c.target = lr;
    }
    if (c.action === 'look' && (c.target === 'left' || c.target === 'right')) c.screen_side = c.target === 'left' ? 'r' : 'l';
    if (c.name === '$emo') {
      const emo = caps.emo || R.emoAlias[caps._pat || ''];
      if (!emo) return null;
      c.name = emo;
    }
    if (c.action === 'expression') c.intensity = r3(Math.min(1.0, Math.max(0.2, (has(c, 'intensity') ? c.intensity : 0.8) * k)));
    else if (c.action === 'gesture' && k !== 1.0) c.intensity = r3(Math.min(1.5, Math.max(0.3, k)));
    if (mods.rep && c.action === 'gesture') c.repeat = mods.rep;
    const holdable = ['pose', 'gesture', 'expression', 'look'].includes(c.action);
    if (mods.hold !== null && mods.hold !== undefined && holdable) c.hold = mods.hold;
    if (caps._keep && holdable) c.hold = true;
    return c;
  }

  function clauses(R, toks) {
    const out = [];
    let cur = [], start = true, anch = false;
    for (const t of toks.concat(['|'])) {
      if (t === '|' || R.split.has(t) || (R.anchors.has(t) && cur.length)) {
        if (cur.length) out.push([cur, start, anch]);
        start = (t === '|') || (start && !cur.length);
        if (t === '|') start = true;
        cur = []; anch = false;
        if (R.anchors.has(t)) { cur = [t]; anch = true; }
        continue;
      }
      cur.push(t);
    }
    return out;
  }

  function stripPre(R, toks) {
    let i = 0;
    while (i < toks.length && (R.pre.has(toks[i]) || R.names.has(toks[i]))) i += 1;
    return i;
  }

  function matchClause(R, toks) {
    const i = stripPre(R, toks), rest = toks.slice(i);
    if (!rest.length) return ['pre'];
    const tries = [[rest, false]];
    if (R.keep.has(rest[0]) && rest.length > 1) {
      let r2 = rest.slice(1);
      if (r2.length && r2[0] === 'on') r2 = r2.slice(1);
      if (r2.length) {
        if (has(R.gerund, r2[0])) r2[0] = R.gerund[r2[0]];
        tries.push([applyPhrases(R, r2), true]);
      }
    }
    for (const [rt, keep] of tries) {
      for (const rule of R.rules) {
        if (rule.r.whole) continue;
        for (const pat of rule.pats) {
          const ends = [];
          match(R, pat.elems, rt, 0, 0, {}, ends);
          for (const [end, caps] of ends) {
            const mods = tail(R, rt.slice(end));
            if (mods === null) continue;
            Object.assign(caps, pat.fixed);
            caps._pat = pat.src;
            if (keep) caps._keep = true;
            return [rule, pat, caps, mods, end === rt.length, rt.slice(0, end)];
          }
        }
      }
    }
    return null;
  }

  function adjustOf(R, toks) {
    const i = stripPre(R, toks), rest = toks.slice(i);
    if (!rest.length) return null;
    if (has(R.adjust, rest[0]) && tail(R, rest.slice(1)) !== null) return R.adjust[rest[0]];
    return null;
  }

  function sortKeys(o) {
    if (Array.isArray(o)) return o.map(sortKeys);
    if (o && typeof o === 'object') { const r = {}; for (const k of Object.keys(o).sort()) r[k] = sortKeys(o[k]); return r; }
    return o;
  }
  const canon = (cmd) => JSON.stringify(sortKeys(cmd));

  // ============================================================================ intent: whole-body moves + props
  // The twin of app/intent.py's act grammar (same rules "act" section, same order, same results; pinned by
  // tests/intent_vectors.json). See intent.py for the commented version.
  const MOVE_FAM = {spin: 'spin', twirl: 'spin', turn: 'turn', turn_away: 'turn', turn_back: 'turn', backflip: 'flip',
    frontflip: 'flip', flip: 'flip', cartwheel: 'flip', jump: 'jump', hop: 'jump', bounce: 'jump', walk: 'walk', step: 'walk',
    sit: 'posture', sit_down: 'posture', stand: 'posture', stand_up: 'posture', get_up: 'posture', lie_down: 'posture',
    roll_over: 'posture', dance: 'dance', bow: 'bow', wiggle: 'wiggle', shake_off: 'wiggle'};
  const OBJ_FAM = {pick_up: 'pick_up', put_down: 'put', put_back: 'put', place: 'put', drop: 'put', hold_up: 'show',
    show: 'show', offer: 'show', inspect: 'inspect', turn: 'turn', flip: 'turn', hand_over: 'hand_over', toss: 'toss',
    catch: 'toss', shake: 'shake', hug: 'hug', smell: 'smell', put_on: 'wear', take_off: 'wear', pull: 'furniture',
    push: 'furniture', bring: 'furniture', slide: 'furniture', sit_in: 'seat', sit_at: 'seat', stand_up: 'seat',
    open: 'open', close: 'open', use: 'work', work_on: 'work', type_on: 'work', stop: 'work'};
  const LEGACY_MOVE = {spin: 'spin', jump: 'jump', dance: 'dance', bow: 'bow', turn_around: 'turn_away', turn: 'turn'};
  const JUMPY = ['jump', 'hop', 'bounce', 'backflip', 'frontflip', 'flip'];
  const HOMO = ['to', 'too', 'for', 'won', 'tree'];
  const DESC_KEYS = ['name', 'category', 'color', 'size', 'ordinal', 'pos'];
  const isAlpha = (t) => /^[a-z]+$/.test(t), isDigit = (t) => /^[0-9]+$/.test(t);
  const nkeys = (o) => (o && typeof o === 'object' ? Object.keys(o).length : 0);
  const seqEq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  function seqAt(toks, i, seq) {
    if (i + seq.length > toks.length) return false;
    for (let k = 0; k < seq.length; k++) if (toks[i + k] !== seq[k]) return false;
    return true;
  }
  const actElems = (pat) => pat.split(' ').map((tok) => { const opt = tok.startsWith('?'); return [opt, (opt ? tok.slice(1) : tok).split('/')]; });
  function actFirsts(elems, j) {
    j = j || 0;
    if (j >= elems.length) return [new Set(), true];
    const [opt, alts] = elems[j];
    const lit = new Set(); let wild = false;
    for (const a of alts) { if (a.startsWith('<') || a.startsWith('@')) wild = true; else lit.add(a); }
    if (opt) { const [l2, w2] = actFirsts(elems, j + 1); for (const x of l2) lit.add(x); wild = wild || w2; }
    return [lit, wild];
  }
  function actIndex(items) {
    const idx = {}, wild = [];
    items.forEach((it, k) => {
      const [lit, w] = actFirsts(it.elems);
      if (w) wild.push(k);
      for (const t of lit) (idx[t] = has(idx, t) ? idx[t] : []).push(k);
    });
    return [idx, wild];
  }
  function compileAct(a) {
    const byLen = (arr) => arr.sort((x, y) => y[0].length - x[0].length);
    const A = {a, cls: a.cls, madj: a.madj, nums: a.nums, qty: byLen(Object.entries(a.qty).map(([k, v]) => [k.split(' '), v])),
      colors: Object.assign({}, a.colors), sizes: a.sizes, ords: a.ords, shades: new Set(a.shades), nouns: Object.assign({}, a.nouns),
      multi: byLen(Object.entries(a.nouns_multi).map(([k, v]) => [k.split(' '), v])), generic: a.generic, anyn: a.anyn,
      places: Object.assign({}, a.places), pre: new Set(a.pre), fillers: new Set(a.fillers), gerund: a.gerund, neg: new Set(a.neg),
      styleStop: new Set(a.style_stop), frames: byLen(a.frames.map(([f, k]) => [f.split(' '), k])), pats: [], mods: [], names: new Set()};
    for (const e of a.moves) A.pats.push({kind: 'm', src: e[0], elems: actElems(e[0]), name: e[1], set: e.length > 2 ? e[2] : {}});
    for (const e of a.objs) A.pats.push({kind: 'o', src: e[0], elems: actElems(e[0]), verb: e[1], flags: e[2], set: e.length > 3 ? e[3] : {}});
    for (const e of a.mods) A.mods.push({src: e[0], elems: actElems(e[0]), kind: e[1], set: e[2], only: e.length > 3 ? e[3] : null});
    [A.patIdx, A.patWild] = actIndex(A.pats);
    [A.modIdx, A.modWild] = actIndex(A.mods);
    A.verbFirst = new Set(Object.keys(A.patIdx));
    return A;
  }
  function cands(idx, wild, tok) {
    const ks = has(idx, tok) ? idx[tok].slice() : [];
    if (!wild.length) return ks;
    return Array.from(new Set(ks.concat(wild))).sort((x, y) => x - y);
  }
  function addScene(A, cfg) {
    for (const c of (Array.isArray(cfg) ? cfg : [cfg])) {
      if (!c || typeof c !== 'object') continue;
      for (const p of (c.props || [])) {
        const words = (p.aliases || []).concat([String(p.id || '').replace(/_/g, ' '), String(p.name || '')]);
        for (let w of words) {
          w = normalize(w).join(' ').replace(/\|/g, '').trim();
          if (!w) continue;
          let ws = w.split(' ');
          while (ws.length > 1 && (has(A.colors, ws[0]) || has(A.sizes, ws[0]) || A.shades.has(ws[0]) || ['the', 'a', 'an'].includes(ws[0]))) ws = ws.slice(1);
          if (ws.length === 1) {
            if (!has(A.colors, ws[0]) && !has(A.generic, ws[0]) && !has(A.anyn, ws[0]) && !has(A.nouns, ws[0])) A.nouns[ws[0]] = ws[0];
          } else if (ws.length <= 3 && !A.multi.some((m) => seqEq(m[0], ws))) A.multi.push([ws, ws.join('_')]);
        }
        for (const col of [p.color].concat(p.colors || [])) if (typeof col === 'string' && /^[A-Za-z]+$/.test(col) && !has(A.colors, col.toLowerCase())) A.colors[col.toLowerCase()] = col.toLowerCase();
      }
      for (const pl of (c.places || [])) {
        for (const w of (pl.aliases || []).concat([String(pl.id || '')])) {
          const ws = normalize(w.replace(/_/g, ' ')).join(' ').split(' ');
          if (ws.length === 1 && ws[0] && !has(A.places, ws[0])) A.places[ws[0]] = ws[0];
        }
      }
    }
    A.multi.sort((x, y) => y[0].length - x[0].length);
  }
  function numSpans(A, toks, i) {
    const out = [];
    if (i >= toks.length) return out;
    for (const [seq, v] of A.qty) if (seqAt(toks, i, seq)) out.push([i + seq.length, v]);
    const t = toks[i];
    if (isDigit(t) && +t > 0 && +t <= 100) out.push([i + 1, parseInt(t, 10)]);
    else if (has(A.nums, t)) {
      if (!HOMO.includes(t) || (i + 1 < toks.length && ['times', 'time'].includes(toks[i + 1]))) out.push([i + 1, A.nums[t]]);
    }
    return out;
  }
  const pawAt = (A, toks, i) => (i < toks.length && has(A.cls.paw, toks[i]) ? 1 : 0);
  function destSpans(A, toks, i) {
    const out = [], n = toks.length;
    let j = i;
    if (j < n && toks[j] === 'over') j++;
    if (j < n && toks[j] === 'across') j++;
    if (j >= n || !['to', 'into', 'in', 'onto'].includes(toks[j])) return out;
    j++;
    if (j < n && ['your', 'the', 'my', 'ur'].includes(toks[j])) j++;
    if (j < n && ['other', 'left', 'right'].includes(toks[j])) {
      const side = toks[j]; j++;
      if (pawAt(A, toks, j)) out.push([j + 1, side]);
      out.push([j, side]);
    }
    return out;
  }
  const PLACE_PRE = [['on', 'top', 'of'], ['on', 'to'], ['on'], ['onto'], ['upon'], ['in'], ['into'], ['inside'], ['at'], ['to'],
    ['over', 'on'], ['over', 'to'], ['under'], ['underneath'], ['off']];
  const PLACE2 = {'side table': 'side_table', 'desk top': 'desk', 'coffee table': 'table', 'toy box': 'box'};
  function placeSpans(A, toks, i) {
    const out = [], n = toks.length;
    for (const pre of PLACE_PRE) {
      if (!seqAt(toks, i, pre)) continue;
      let j = i + pre.length;
      if (j < n && ['the', 'your', 'my', 'that', 'this', 'a', 'ur'].includes(toks[j])) j++;
      while (j < n && (A.shades.has(toks[j]) || has(A.colors, toks[j]) || has(A.sizes, toks[j]))) j++;
      if (j + 1 < n && has(PLACE2, toks[j] + ' ' + toks[j + 1])) out.push([j + 2, PLACE2[toks[j] + ' ' + toks[j + 1]]]);
      if (j < n && has(A.places, toks[j])) out.push([j + 1, A.places[toks[j]]]);
      if (pre.length === 1 && pre[0] === 'in' && j < n && toks[j] === 'front') return [];
    }
    return out;
  }
  const PRON = [[['whatever', 'youre', 'holding'], {ref: 'it'}], [['whatever', 'you', 'are', 'holding'], {ref: 'it'}],
    [['what', 'youre', 'holding'], {ref: 'it'}], [['what', 'you', 'are', 'holding'], {ref: 'it'}],
    [['whatever', 'your', 'holding'], {ref: 'it'}], [['what', 'your', 'holding'], {ref: 'it'}],
    [['whatever', 'you', 'have'], {ref: 'it'}], [['whatever', 'youve', 'got'], {ref: 'it'}],
    [['the', 'other', 'ones'], {ref: 'other'}], [['the', 'other', 'one'], {ref: 'other'}],
    [['both', 'of', 'them'], {ref: 'all'}], [['all', 'of', 'them'], {ref: 'all'}], [['one', 'of', 'them'], {ref: 'any'}],
    [['all', 'of', 'it'], {ref: 'all'}], [['another', 'one'], {ref: 'other'}], [['other', 'one'], {ref: 'other'}],
    [['the', 'other'], {ref: 'other'}], [['that', 'one'], {ref: 'that', _sel: 1}], [['this', 'one'], {ref: 'that', _sel: 1}],
    [['them', 'all'], {ref: 'all'}], [['either', 'one'], {ref: 'any'}], [['any', 'one'], {ref: 'any'}], [['it'], {ref: 'it'}],
    [['that'], {ref: 'that'}], [['this'], {ref: 'that'}], [['them'], {ref: 'all'}], [['those'], {ref: 'all'}],
    [['these'], {ref: 'all'}], [['em'], {ref: 'all'}], [['both'], {ref: 'all'}], [['everything'], {ref: 'all'}],
    [['all'], {ref: 'all'}], [['one'], {ref: 'any'}], [['something'], {ref: 'any'}], [['anything'], {ref: 'any'}],
    [['either'], {ref: 'any'}]];
  const DETS = [[['one', 'of', 'the'], 'oneof'], [['one', 'of', 'your'], 'oneof'], [['one', 'of', 'my'], 'oneof'],
    [['one', 'of', 'those'], 'oneof'], [['all', 'of', 'the'], 'all'], [['all', 'the'], 'all'], [['both', 'of', 'the'], 'all'],
    [['both', 'the'], 'all'], [['both', 'your'], 'all'], [['all', 'your'], 'all'], [['the'], 'the'], [['a'], 'a'], [['an'], 'a'],
    [['your'], 'the'], [['my'], 'the'], [['ur'], 'the'], [['that'], 'the'], [['this'], 'the'], [['these'], 'the'],
    [['those'], 'the'], [['some'], 'a'], [['any'], 'a'], [['another'], 'another'], [['both'], 'all'], [['his'], 'the'],
    [['her'], 'the'], [['our'], 'the'], [['their'], 'the']];
  const NP_PP = [[['on', 'your', 'left'], {pos: 'left'}], [['on', 'your', 'right'], {pos: 'right'}],
    [['to', 'your', 'left'], {pos: 'left'}], [['to', 'your', 'right'], {pos: 'right'}],
    [['on', 'my', 'left'], {pos: 'screen_left'}], [['on', 'my', 'right'], {pos: 'screen_right'}],
    [['on', 'the', 'left'], {pos: 'screen_left'}], [['on', 'the', 'right'], {pos: 'screen_right'}],
    [['in', 'the', 'middle'], {pos: 'middle'}], [['in', 'the', 'center'], {pos: 'middle'}], [['in', 'the', 'centre'], {pos: 'middle'}]];
  const descOk = (d) => !!d && typeof d === 'object' && DESC_KEYS.some((k) => has(d, k));
  function npHead(A, toks, j) {
    const out = [];
    for (const [seq, v] of A.multi) if (seqAt(toks, j, seq)) out.push([j + seq.length, v]);
    if (j < toks.length) {
      const t = toks[j];
      if (has(A.nouns, t)) out.push([j + 1, A.nouns[t]]);
      if (has(A.generic, t)) out.push([j + 1, '@' + A.generic[t] + (t.endsWith('s') && t !== 'stuff' ? '+' : '')]);
      if (has(A.anyn, t)) out.push([j + 1, '*' + (A.anyn[t] ? '+' : '')]);
    }
    return out;
  }
  function npSpans(A, toks, i) {
    const n = toks.length, out = [];
    if (i >= n) return out;
    for (const [seq, d] of PRON) if (seqAt(toks, i, seq)) out.push([i + seq.length, Object.assign({}, d)]);
    const dets = [[i, null]];
    for (const [seq, dv] of DETS) if (seqAt(toks, i, seq)) { dets.push([i + seq.length, dv]); break; }
    for (const [j0, det] of dets) npAdj(A, toks, j0, det, {}, 0, out);
    const seen = new Set(), res = [];
    const sorted = out.map((x, k) => [x, k]).sort((p, q) => (q[0][0] - p[0][0]) || (p[1] - q[1])).map((p) => p[0]);
    for (const [e, d] of sorted) if (!seen.has(e)) { seen.add(e); res.push([e, d]); }
    return res;
  }
  function npAdj(A, toks, j, det, d, unk, out) {
    const n = toks.length;
    for (const [e, head] of npHead(A, toks, j)) npFinish(A, toks, e, det, d, head, out);
    if (j >= n || j > 40) return;
    const t = toks[j];
    let nd = null;
    const sd = (k, v) => { nd = Object.assign({}, d); if (!has(nd, k)) nd[k] = v; };
    if (has(A.colors, t)) sd('color', A.colors[t]);
    else if (has(A.sizes, t)) sd('size', A.sizes[t]);
    else if (has(A.ords, t)) sd('ordinal', A.ords[t]);
    else if (t === 'other') { nd = Object.assign({}, d); nd.ref = 'other'; }
    else if ((t === 'left' || t === 'right') && det !== null) sd('pos', 'screen_' + t);
    else if (['middle', 'center', 'centre'].includes(t) && det !== null) sd('pos', 'middle');
    else if (A.shades.has(t) || t === 'and') nd = Object.assign({}, d);
    if (nd !== null && nkeys(nd) < 6) npAdj(A, toks, j + 1, det, nd, unk, out);
    else if (nd === null && det !== null && unk < 2 && isAlpha(t) && !A.neg.has(t) && !A.styleStop.has(t) && !A.pre.has(t) && !has(A.cls.paw, t)) {
      npAdj(A, toks, j + 1, det, Object.assign({}, d, {_unk: 1}), unk + 1, out);
    }
  }
  function npFinish(A, toks, e, det, d0, head, out) {
    let d = Object.assign({}, d0);
    const plural = head.endsWith('+');
    head = head.replace(/\++$/, '');
    if (head.startsWith('@')) d.category = head.slice(1);
    else if (head === '*') { if (d._unk) return; }
    else d.name = head;
    delete d._unk;
    if (det === 'another') d.ref = 'other';
    if (plural && det !== 'oneof' && !has(d, 'ordinal')) d.ref = d.ref === 'other' ? d.ref : 'all';
    if (det === 'all') d.ref = 'all';
    const base = Object.assign({}, d);
    if (head === '*' && !DESC_KEYS.some((k) => has(d, k)) && !has(d, 'ref')) {
      if (det === 'a' || det === 'oneof') d.ref = 'any';
      else if (det === 'the') { d.ref = 'that'; d._one = 1; }
      else return;
    }
    out.push([e, d]);
    for (const [seq, pp] of NP_PP) if (seqAt(toks, e, seq)) out.push([e + seq.length, Object.assign({}, base, pp)]);
    for (const [e2, pl] of placeSpans(A, toks, e)) {
      if (['on', 'from', 'in', 'off'].includes(toks[e]) && pl !== 'head') out.push([e2, Object.assign({}, head === '*' ? base : d, {where: pl})]);
    }
    if (e + 1 < toks.length && toks[e] === 'from' && ['the', 'your'].includes(toks[e + 1])) {
      for (const [e2, pl] of placeSpans(A, ['on'].concat(toks.slice(e + 1)), 0)) out.push([e + e2, Object.assign({}, head === '*' ? base : d, {where: pl})]);
    }
  }
  function actMatch(A, elems, toks, i, j, caps, out) {
    if (j === elems.length) { out.push([i, caps]); return; }
    const [opt, alts] = elems[j], n = toks.length;
    const capList = (cls, v) => { const c2 = Object.assign({}, caps); c2['@' + cls] = (caps['@' + cls] || []).concat([v]); return c2; };
    for (const alt of alts) {
      if (alt === '<O>') { for (const [e, d] of npSpans(A, toks, i)) actMatch(A, elems, toks, e, j + 1, Object.assign({}, caps, {O: d}), out); }
      else if (alt === '<N>') { for (const [e, v] of numSpans(A, toks, i)) actMatch(A, elems, toks, e, j + 1, Object.assign({}, caps, {N: v}), out); }
      else if (alt === '<S>') { for (const [e, v] of destSpans(A, toks, i)) actMatch(A, elems, toks, e, j + 1, Object.assign({}, caps, {S: v}), out); }
      else if (alt === '<P>') { for (const [e, v] of placeSpans(A, toks, i)) actMatch(A, elems, toks, e, j + 1, Object.assign({}, caps, {P: v}), out); }
      else if (alt.startsWith('@')) {
        const cls = alt.slice(1), tab = A.cls[cls] || {};
        if (cls === 'style') {
          if (i < n && isAlpha(toks[i]) && !A.styleStop.has(toks[i]) && !A.neg.has(toks[i])) actMatch(A, elems, toks, i + 1, j + 1, capList(cls, toks[i]), out);
          continue;
        }
        for (const L of [3, 2, 1]) {
          if (i + L <= n) { const key = toks.slice(i, i + L).join(' '); if (has(tab, key)) actMatch(A, elems, toks, i + L, j + 1, capList(cls, key), out); }
        }
      } else if (i < n && toks[i] === alt) actMatch(A, elems, toks, i + 1, j + 1, caps, out);
    }
    if (opt) actMatch(A, elems, toks, i, j + 1, caps, out);
  }
  function actMods(A, toks, i, kind, name, depth) {
    depth = depth || 0;
    if (i >= toks.length) return [];
    if (depth > 12) return null;
    if (A.names.has(toks[i])) return actMods(A, toks, i + 1, kind, name, depth + 1);
    for (const k of cands(A.modIdx, A.modWild, toks[i])) {
      const m = A.mods[k];
      if (!m.kind.includes(kind) || (m.only !== null && !m.only.includes(name))) continue;
      const ends = [];
      actMatch(A, m.elems, toks, i, 0, {}, ends);
      for (const [e, caps] of ends) {
        if (e <= i) continue;
        const rest = actMods(A, toks, e, kind, name, depth + 1);
        if (rest !== null) return [[m, caps]].concat(rest);
      }
    }
    return null;
  }
  const clsVal = (A, cls, key) => { const tab = A.cls[cls] || {}; return has(tab, key) ? tab[key] : key; };
  function actVal(A, v, caps) {
    if (!(typeof v === 'string' && v.startsWith('$'))) return v;
    const k = v.slice(1), n = has(caps, 'N') ? caps.N : null;
    if (k === 'N') return n;
    if (k === 'Ns') return n === null ? null : n * 1000;
    if (k === 'Nm') return n === null ? null : n * 60000;
    if (k === 'O' || k === 'P' || k === 'S') return has(caps, k) ? caps[k] : null;
    if (k === 'slr') { const lr = (caps['@lr'] || [null])[0]; return lr === null ? null : 'screen_' + clsVal(A, 'lr', lr); }
    const lst = caps['@' + k];
    return lst && lst.length ? clsVal(A, k, lst[0]) : null;
  }
  function actApply(A, cmd, sets, caps, x) {
    for (const k of Object.keys(sets)) {
      const v = actVal(A, sets[k], caps);
      if (v === null || v === undefined) continue;
      if (x && k.startsWith('_')) x[k] = v; else cmd[k] = v;
    }
  }
  function scrub(d) {
    const r = {};
    for (const k of Object.keys(d)) if (!k.startsWith('_')) r[k] = (d[k] && typeof d[k] === 'object' && !Array.isArray(d[k])) ? scrub(d[k]) : d[k];
    return r;
  }
  const clampI = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.trunc(+v)));
  function buildAct(A, pat, caps, mods, st) {
    let cmd;
    if (pat.kind === 'm') {
      let name = pat.name;
      if (name === '$mn') name = clsVal(A, 'mn', (caps['@mn'] || ['spin'])[0]);
      cmd = {action: 'move', name};
      actApply(A, cmd, pat.set, caps);
      for (const a of (caps['@madj'] || []).concat(caps['@madjb'] || [])) {
        const mv = has(A.madj, a) ? A.madj[a] : {};
        for (const k of Object.keys(mv)) if (k !== '_k') cmd[k] = mv[k];
      }
      for (const [m, mc] of mods) actApply(A, cmd, m.set, mc);
      if (has(cmd, '_flip')) { if (['flip', 'backflip', 'frontflip'].includes(cmd.name)) cmd.name = cmd._flip; delete cmd._flip; }
      if (cmd.name === 'flip') { cmd.name = cmd.dir === 'forward' ? 'frontflip' : 'backflip'; if (cmd.dir === 'forward' || cmd.dir === 'back') delete cmd.dir; }   // plain flip = backflip
      if (cmd.name === 'turn_away' && has(cmd, 'count')) cmd.name = 'spin';
      if (cmd.name === 'turn' && !has(cmd, 'dir')) return null;
      if (has(cmd, 'height') && !JUMPY.includes(cmd.name)) delete cmd.height;
      if (has(cmd, 'style') && !has(pat.set, 'style') && !['dance', 'walk', 'bow', 'lie_down'].includes(cmd.name) && !mods.some(([m]) => has(m.set, 'style'))) delete cmd.style;
      if (has(cmd, 'count')) cmd.count = clampI(cmd.count, 1, 20);
      if (has(cmd, 'distance')) cmd.distance = clampI(cmd.distance, 0, 20);
    } else {
      cmd = {action: 'object', verb: pat.verb};
      const flags = pat.flags;
      if (has(caps, 'O')) cmd.object = Object.assign({}, caps.O);
      if (has(caps, 'S')) cmd.side = caps.S;
      const x = {};
      actApply(A, cmd, pat.set, caps, x);
      for (const [m, mc] of mods) actApply(A, cmd, m.set, mc, x);
      const nonfill = mods.some(([m]) => nkeys(m.set) > 0);
      let o = has(cmd, 'object') ? cmd.object : null;
      if (o !== null) {
        const one = o._one; delete o._one;
        if (one && has(cmd, 'side') && nkeys(o) === 1 && o.ref === 'that') { delete cmd.object; o = null; }
      }
      if (flags.includes('N') && !descOk(o) && !(has(cmd, 'side') && o === null)) return null;
      let sel = false;
      if (nkeys(o)) { sel = !!o._sel || o.ref === 'other'; delete o._sel; }
      if (flags.includes('n') && !descOk(o) && !nonfill && !sel && !(st.ctx && nkeys(o))) return null;
      if (x._ref && !has(cmd, 'object')) cmd.object = {ref: x._ref};
      let v = cmd.verb;
      if (v === 'place') {
        if (x._pb) v = 'put_back';
        else if (has(cmd, 'place')) v = 'place';
        else if (x._down) v = 'put_down';
        else if (x._hi) v = 'hold_up';
        else return null;
      } else if (v === 'put_down' && has(cmd, 'place')) v = 'place';
      // "push the chair back where it was" — 'where it was / belongs / came from' on a furniture move means its own
      // spot ('home'), not 'back from the desk' (the 'back' mod matched first and the put-back words were dropped)
      else if (x._pb && ['pull', 'push', 'bring', 'slide'].includes(v)) cmd.target = 'home';
      if ((v === 'pick_up' || v === 'hold_up') && x._hi) v = 'hold_up';
      if ((v === 'pick_up' || v === 'hold_up') && x._cam) v = 'show';
      if (v === 'turn' && x._flip) { v = 'flip'; delete cmd.amount; }
      if (v === 'slide' && cmd.target && typeof cmd.target === 'object' && nkeys(cmd.target) === 1 && has(cmd.target, 'name') &&
          has(A.places, cmd.target.name) && pat.src.startsWith('move')) {
        cmd.place = A.places[cmd.target.name]; delete cmd.target; v = 'place';
      }
      if (x._needtarget && !has(cmd, 'target') && !has(cmd, 'place')) return null;
      if (cmd.place === 'head' && ['place', 'put_down', 'put_on', 'put_back'].includes(v)) { v = 'put_on'; delete cmd.place; }
      if (['pull', 'push', 'bring', 'slide'].includes(v) && has(cmd, 'target') && has(cmd, 'place')) delete cmd.place;
      const PV = ['place', 'put_back', 'drop', 'put_down'];
      if (!PV.includes(v) && has(cmd, 'place') && cmd.place !== 'near') {
        if (v === 'slide' && !has(cmd, 'target')) v = 'place';
        else return null;
      }
      if (has(cmd, 'near') && !PV.includes(v)) return null;
      if (has(cmd, 'place') && cmd.place === 'near' && v === 'put_down') v = 'place';
      if (v === 'put_on' && cmd.place === 'head') delete cmd.place;
      cmd.verb = v;
      if (has(cmd, 'count')) cmd.count = clampI(cmd.count, 1, 20);
      delete cmd.speed; delete cmd.duration_ms;
    }
    if (st.speed && !has(cmd, 'speed') && cmd.action === 'move') cmd.speed = st.speed;
    if (st.keep) cmd.hold = true;
    if (st.side && cmd.action === 'object' && !has(cmd, 'side')) cmd.side = st.side;
    return scrub(cmd);
  }
  const PLACE_VERBS = ['place', 'put_down', 'put_back', 'drop', 'slide', 'pull', 'push', 'bring'];
  function actAt(A, toks, i, st) {
    if (i >= toks.length) return null;
    for (const k of cands(A.patIdx, A.patWild, toks[i])) {
      const pat = A.pats[k], ends = [];
      actMatch(A, pat.elems, toks, i, 0, {}, ends);
      for (const [e, caps] of ends) {
        if (e <= i) continue;
        if (PLACE_VERBS.includes(pat.verb) && caps.O && has(caps.O, 'where')) continue;
        let key = pat.verb || pat.name;
        if (key === '$mn') key = clsVal(A, 'mn', (caps['@mn'] || ['spin'])[0]);
        const mods = actMods(A, toks, e, pat.kind, key);
        if (mods === null) continue;
        const cmd = buildAct(A, pat, caps, mods, st);
        if (cmd !== null) return cmd;
      }
    }
    return null;
  }
  const LIKE_PREV = ['you', 'u', 'ya', 'so', 'can', 'could', 'would', 'and', 'then', 'just', 'okay', 'ok', 'um', 'uh'];
  function actClean(A, toks) {
    const out = [];
    for (let k = 0; k < toks.length; k++) {
      const t = toks[k];
      if (t === 'like') {   // "can you like spin", "like two steps" (kept: "dance like a robot")
        const nx = k + 1 < toks.length ? toks[k + 1] : '', pv = out.length ? out[out.length - 1] : '';
        if (!out.length || A.pre.has(pv) || A.names.has(pv) || LIKE_PREV.includes(pv) || /^\d+$/.test(nx) || has(A.nums, nx) ||
            (nx === 'a' && k + 2 < toks.length && ['couple', 'few', 'bunch'].includes(toks[k + 2]))) continue;
        out.push(t);
        continue;
      }
      if (A.fillers.has(t)) continue;
      if (out.length && out[out.length - 1] === t && ['the', 'a', 'your', 'it', 'my', 'to', 'and'].includes(t)) continue;
      out.push(t);
    }
    return out;
  }
  const RUNON_NOT = ['up', 'down', 'back', 'over', 'it', 'the', 'a', 'to', 'on', 'in', 'with', 'your', 'me', 'around', 'off',
    'out', 'away', 'here', 'one', 'that', 'this', 'them', 'for', 'at', 'of', 'my', 'an', 'another', 'some', 'both', 'all',
    'everything', 'something', 'please', 'now', 'again', 'really', 'very', 'super', 'so'];
  const actHead = (c) => (c.action || '') + '/' + String(c.name || c.verb || '');
  // negation scope, spoken order != done order (intent.py parse_act)
  const NEG_START = [['do', 'not'], ['dont'], ['no'], ['never'], ['not'], ['skip'], ['without'], ['hold', 'off', 'on'],
    ['hold', 'off'], ['forget', 'about'], ['instead', 'of']];
  const NEG_RESUME = ['just', 'but', 'instead', 'rather', 'only'];
  const NEG_RESUME2 = [['how', 'about'], ['what', 'about'], ['can', 'you'], ['could', 'you'], ['lets']];
  const NEG_MID = ['dont', 'not', 'never'];
  const ORDER_WORDS = {after: -1, once: -1, before: 1};
  const ORDER_SKIP = ['you', 'u', 'youve', 'youre', 'ya', 'we', 'i'];
  function negAt(toks, i) {
    for (const q of NEG_START) if (seqAt(toks, i, q)) return q.length;
    return 0;
  }
  const isPronIt = (o) => !!o && Object.keys(o).length === 1 && (o.ref === 'it' || o.ref === 'that');
  function parseAct(A, toks0, names, depth, ctx, ger0) {
    names = names || new Set(); depth = depth || 0;
    let toks = actClean(A, toks0);
    const st = ctx ? {ctx: true} : {};
    if (ger0) st.ger = true;
    let i = 0;
    const n = toks.length;
    while (i < n) {
      const t = toks[i];
      if (st.ger && has(A.gerund, t)) { toks = toks.slice(0, i).concat([A.gerund[t]], toks.slice(i + 1)); st.ger = false; }
      const cmd = actAt(A, toks, i, st);
      if (cmd !== null) return [cmd];
      const ng = negAt(toks, i);
      if (ng) {
        for (let k = i + ng; k < n; k++) {
          if (NEG_RESUME.includes(toks[k]) || NEG_RESUME2.some((q) => seqAt(toks, k, q))) {
            const k1 = NEG_RESUME.includes(toks[k]) ? k + 1 : k;
            const r = depth < 3 ? parseAct(A, toks.slice(k1), names, depth + 1, ctx, true) : null;
            return r && r.length ? r : [];
          }
        }
        return [];
      }
      if (depth < 3) {
        for (let k = i + 1; k < n; k++) {
          if (NEG_MID.includes(toks[k]) || (toks[k] === 'no' && k > i + 1)) {
            const kc = ['but', 'and'].includes(toks[k - 1]) ? k - 1 : k;
            const left = actAt(A, toks.slice(0, kc), i, st);
            if (left !== null) {
              let out = [left];
              for (let j = k + 1; j < n; j++) {
                if (NEG_RESUME.includes(toks[j])) { out = out.concat(parseAct(A, toks.slice(j + 1), names, depth + 1, true, true) || []); break; }
              }
              return out;
            }
            break;
          }
        }
        for (let k = i; k < n; k++) {
          if (!has(ORDER_WORDS, toks[k])) continue;
          let j = k + 1;
          while (j < n && ORDER_SKIP.includes(toks[j])) j++;
          const right = j < n ? parseAct(A, toks.slice(j), names, depth + 1, ctx, true) : null;
          if (!right || !right.length) continue;
          if (k === i) return right;
          const left = actAt(A, toks.slice(0, k), i, st);
          if (left === null) continue;
          return ORDER_WORDS[toks[k]] < 0 ? right.concat([left]) : [left].concat(right);
        }
      }
      if (depth < 3) {
        for (let k = n - 1; k > i; k--) {
          const cut = ['and', 'then', 'andthen', 'than'].includes(toks[k]);
          if (!cut && (!A.verbFirst.has(toks[k]) || RUNON_NOT.includes(toks[k]))) continue;
          const left = actAt(A, toks.slice(0, k), i, st);
          if (left === null) continue;
          const right = parseAct(A, cut ? toks.slice(k + 1) : toks.slice(k), names, depth + 1, true);
          if (right && right.length && left.object && Object.keys(left.object).length === 1 && left.object.ref === 'any' && right[0].action === 'move') continue;
          if (right && right.length && actHead(right[0]) !== actHead(left)) return [left].concat(right);
        }
        // a topicalised object ("the laptop, open it" / "laptop pick up")
        for (const [e, d] of npSpans(A, toks, i)) {
          if (!descOk(d) || e >= n || has(d, 'where')) continue;
          let rest = toks.slice(e);
          while (rest.length && (A.pre.has(rest[0]) || names.has(rest[0]))) rest = rest.slice(1);
          let done = null;
          for (const tail of [rest, rest.concat(['it'])]) {
            const r = tail.length ? parseAct(A, tail, names, depth + 1, true) : null;
            if (r && r.length === 1 && r[0].action === 'object' && (r[0].object === undefined || r[0].object === null || isPronIt(r[0].object))) {
              done = [Object.assign({}, r[0], {object: scrub(d)})]; break;
            }
          }
          if (done) return done;
          break;
        }
      }
      if (A.pre.has(t) || names.has(t)) { i++; continue; }
      let hit = null;
      for (const [seq, kind] of A.frames) if (seqAt(toks, i, seq)) { hit = [seq, kind]; break; }
      if (hit) {
        const [seq, kind] = hit;
        i += seq.length;
        if (kind !== 'fast' && kind !== 'slow') st.ger = true;   // a gerund after any frame
        if (kind === 'keep') st.keep = true;
        if (kind === 'fast' || kind === 'slow') st.speed = kind;
        continue;
      }
      if (t === 'use' || t === 'using') {
        let j = i + 1, both = false, side = null;
        if (j < n && ['your', 'the', 'my', 'both'].includes(toks[j])) { both = toks[j] === 'both'; j++; }
        if (j < n && ['left', 'right', 'other'].includes(toks[j])) { side = toks[j]; j++; }
        else if (j < n && ['both', 'two'].includes(toks[j])) { side = 'both'; j++; }
        else if (both) side = 'both';
        if (side && pawAt(A, toks, j) && j + 1 < n && ['to', 'and'].includes(toks[j + 1])) { st.side = side; i = j + 2; continue; }
      }
      return null;
    }
    return null;
  }
  const REL_HEAD = [['stop'], ['relax'], ['never', 'mind'], ['nevermind'], ['thats', 'enough'], ['enough'], ['okay', 'stop'],
    ['ok', 'stop'], ['cut', 'it', 'out'], ['quit', 'it'], ['stop', 'stop'], ['no', 'stop'], ['wait', 'stop']];
  const REL_NEG = ['dont', 'no', 'not', 'never', 'you', 'do', 'stop', 'quit', 'please'];
  function actRelease(A, raw) {
    const toks = actClean(A, raw);
    for (const h of REL_HEAD) {
      if (seqAt(toks, 0, h) && toks.length > h.length) {
        const rest = toks.slice(h.length);
        const nm = seqEq(h, ['never', 'mind']) || seqEq(h, ['nevermind']);
        if (REL_NEG.includes(rest[0]) || (nm && ['the', 'that', 'it', 'about'].includes(rest[0]))) {
          if (rest[0] === 'you' && !(rest.length > 1 && ['dont', 'can', 'do', 'need'].includes(rest[1]))) continue;
          return {action: 'release', parts: 'all'};
        }
      }
    }
    return null;
  }
  const HOLD_ON = [['dont', 'let', 'go'], ['dont', 'let', 'it', 'go'], ['dont', 'drop', 'it'], ['keep', 'holding', 'it'],
    ['keep', 'holding', 'on'], ['hold', 'on', 'to', 'it'], ['hold', 'it', 'tight'], ['dont', 'let', 'go', 'of', 'it'],
    ['and', 'dont', 'let', 'go'], ['keep', 'it'], ['keep', 'holding', 'them']];
  const GAZE_FREE = [['dont', 'look', 'at', 'me'], ['do', 'not', 'look', 'at', 'me'], ['dont', 'look'], ['no', 'looking'],
    ['dont', 'look', 'at', 'us'], ['and', 'dont', 'look', 'at', 'me'], ['but', 'dont', 'look', 'at', 'me']];
  function actAttach(hits, c) {
    if (!hits.length) return false;
    const prev = hits[hits.length - 1].cmd;
    if (prev.action !== 'move' && prev.action !== 'object') return false;
    if (c.action === 'hold') { prev.hold = true; return true; }
    if (c.action === 'look' && c.target === 'user' && prev.action === 'move') { prev.gaze = 'user'; return true; }
    return false;
  }
  function actCarry(A, raw, prev, names) {
    const toks = actClean(A, raw);
    let i = 0;
    while (i < toks.length && (A.pre.has(toks[i]) || names.has(toks[i]) || ['then', 'also', 'now'].includes(toks[i]))) i++;
    for (const [e, d] of npSpans(A, toks, i)) {
      if (!descOk(d) && d.ref !== 'other') continue;
      const mods = actMods(A, toks, e, 'o', prev.verb);
      if (mods === null) continue;
      const c = {action: 'object', verb: prev.verb, object: d}, x = {};
      for (const [m, mc] of mods) actApply(A, c, m.set, mc, x);
      if (['place', 'put_back'].includes(prev.verb) && has(prev, 'place') && !has(c, 'place')) c.place = prev.place;
      delete c.speed; delete c.duration_ms;
      return [scrub(c)];
    }
    return null;
  }
  // "take the headphones off and put them back" — a plural pronoun after the request named ONE thing (a plural
  // noun is still one thing) is that thing: {ref: 'it'} (motion.js binds 'it' to the object of the step before it, even
  // while that step is still queued), not {ref: 'all'}. Only when no "all / both / every…" was said.
  const PRON_MANY = ['them', 'those', 'these', 'em', 'they'], ALL_WORDS = ['all', 'both', 'everything', 'every', 'each'];
  const concreteObj = (o) => !!o && typeof o === 'object' && !o.ref && !!(o.id || o.name || o.color || o.category);
  const objSig = (o) => JSON.stringify([o.id || '', o.name || '', o.color || '', o.category || '', o.ordinal || '']);
  function actPronouns(out, words) {
    const ws = words || [];
    const many = ws.some((w) => PRON_MANY.includes(w)) && !ws.some((w) => ALL_WORDS.includes(w));
    const seen = [];
    for (const h of out) {
      const c = h.cmd;
      if (!c || c.action !== 'object') continue;
      const o = c.object;
      if (concreteObj(o)) { seen.push(o); continue; }
      if (!seen.length || nkeys(o) !== 1) continue;
      if (o.ref === 'all' && many && new Set(seen.map(objSig)).size === 1) c.object = {ref: 'it'};
    }
    return out;
  }
  function actPost(hits, words) {
    return actPronouns(actPost0(hits), words);
  }
  function actPost0(hits) {
    const out = [];
    for (const h of hits) {
      const c = h.cmd, last = out.length ? out[out.length - 1] : null;
      if (last && c.action === 'move' && c.name === 'turn_back' && last.cmd.action === 'move' && last.cmd.name === 'turn_away' &&
          !last.cmd.hold && last._raw && ['around', 'round'].includes(last._raw[last._raw.length - 1])) { out[out.length - 1] = h; continue; }
      if (last && c.action === 'object' && c.verb === 'catch' && last.cmd.action === 'object' && last.cmd.verb === 'toss') continue;
      out.push(h);
    }
    return out;
  }
  function applyPhrasesSpans(R, toks) {
    const out = [], raw = [];
    let i = 0;
    while (i < toks.length) {
      let hit = null;
      for (const [ptoks, dst] of (has(R.phr, toks[i]) ? R.phr[toks[i]] : [])) if (seqAt(toks, i, ptoks)) { hit = [ptoks.length, dst]; break; }
      if (hit) { out.push(hit[1]); raw.push(toks.slice(i, i + hit[0])); i += hit[0]; }
      else { out.push(toks[i]); raw.push([toks[i]]); i += 1; }
    }
    return [out, raw];
  }
  function clauseSpans(R, toks) {
    const out = [];
    let cur0 = 0, n = 0, start = true, anch = false;
    const all = toks.concat(['|']);
    for (let k = 0; k < all.length; k++) {
      const t = all[k];
      if (t === '|' || R.split.has(t) || (R.anchors.has(t) && n)) {
        if (n) out.push([cur0, cur0 + n, start, anch]);
        start = (t === '|') || (start && !n);
        if (t === '|') start = true;
        cur0 = k + 1; n = 0; anch = false;
        if (R.anchors.has(t)) { cur0 = k; n = 1; anch = true; }
        continue;
      }
      n += 1;
    }
    return out;
  }
  // ============================================================================ whole-body intent end (engine)

  function create(rulesData, opts) {
    opts = opts || {};
    const R = compile(rulesData);
    const self = {R, last: null};
    // whole-body intent: opts.act === false -> the base matcher exactly; opts.scene = config.props (or a list) adds prop nouns
    if (rulesData.act && opts.act !== false) {
      self.A = compileAct(rulesData.act);
      self.A.names = new Set(rulesData.names);
      if (opts.scene) addScene(self.A, opts.scene);
      if (opts.scene) setScene(opts.scene);   // IntentDedupe compares names through the same scene
    } else self.A = null;
    self.setScene = function (scene) { if (!self.A) return; self.A = compileAct(rulesData.act); self.A.names = new Set(rulesData.names); if (scene) addScene(self.A, scene); setScene(scene || null); };
    self.parse = function (text, opt) {
      opt = opt || {};
      const final = opt.final !== undefined ? !!opt.final : true;
      const stable = opt.stable !== undefined ? !!opt.stable : true;
      const nowMs = opt.nowMs || 0;
      const toks = tokenize(R, text);
      const hits = [];
      if (final || stable) {
        const flat = toks.filter((t) => t !== '|');
        for (const rule of R.rules) {
          if (!rule.r.whole) continue;
          for (const pat of rule.pats) {
            const ends = [];
            match(R, pat.elems, flat, 0, 0, {}, ends);
            if (ends.some((e) => e[0] === flat.length)) {
              const c = build(R, rule.r, {}, {int: 1.0, rep: null, hold: null});
              return [{cmd: c, clause: 0, rule: rule.r.id, open: !final}];
            }
          }
        }
      }
      let cls, craw = null;
      if (self.A) {                                                     // whole-body intent: clauses keep their raw tokens
        const [pt, raws] = applyPhrasesSpans(R, normalize(text).map((w) => (has(R.alias, w) ? R.alias[w] : w)));
        const spans = clauseSpans(R, pt);
        cls = spans.map(([a, b, st, an]) => [pt.slice(a, b), st, an]);
        craw = spans.map(([a, b]) => [].concat(...raws.slice(a, b)));
      } else cls = clauses(R, toks);
      let prevOk = false, negPrev = false;
      cls.forEach(([ct, sstart, anch], idx) => {
        const isLast = idx === cls.length - 1;
        const eligible = sstart || anch || prevOk || negPrev;
        prevOk = false; negPrev = false;
        if (!eligible) return;
        if (self.A) {
          const r = actClause(craw[idx], hits, idx, isLast, final, stable);
          if (r === 'wait') return;
          if (r === 'neg') negPrev = true;   // nothing fired; the legacy rules still see it, the next clause runs
          else if (r) { prevOk = true; return; }
        }
        const adj = adjustOf(R, ct);
        if (adj !== null) {
          if (self.last && nowMs - self.last[3] <= CTX_MS) {
            const [rule, caps, mods] = self.last;
            const m2 = Object.assign({}, mods);
            m2.int = Math.min(1.6, Math.max(0.3, mods.int * adj));
            const c = build(R, rule, caps, m2);
            if (c) { hits.push({cmd: c, clause: idx, rule: rule.id, open: isLast && !final, _ctx: [rule, caps, m2]}); prevOk = true; }
          }
          return;
        }
        const m = matchClause(R, ct);
        if (m === null) return;
        if (m[0] === 'pre') { prevOk = true; return; }
        const [rule, pat, caps, mods] = m;
        if (rule.r.final && !final) return;
        if (isLast && !final && !stable && (pat.wait || (rule.r.wait || []).includes(m[5].join(' ')))) return;
        const c = build(R, rule.r, caps, mods);
        if (c === null) return;
        if (self.A && actAttach(hits, c)) { prevOk = true; return; }   // whole-body intent: "... and stay like that"
        hits.push({cmd: c, clause: idx, rule: rule.r.id, open: isLast && !final, _ctx: [rule.r, caps, mods]});
        prevOk = true;
      });
      if (self.A) {
        const h2 = actPost(hits, normalize(text));
        for (const h of h2) delete h._raw;
        return h2;
      }
      return hits;
    };
    function actClause(raw, hits, idx, isLast, final, stable) {
      const A = self.A, prev0 = hits.length ? hits[hits.length - 1].cmd : null;
      if (prev0 && prev0.action === 'object') {
        const cl = actClean(A, raw);
        if (HOLD_ON.some((h) => seqEq(h, cl))) { prev0.hold = true; return true; }
      }
      const rel = actRelease(A, raw);
      if (rel !== null) { hits.push({cmd: rel, clause: idx * 8, rule: 'act:release', open: isLast && !final}); return true; }
      let cmds = parseAct(A, raw, R.names, 0, !!(prev0 && prev0.action === 'object'), hits.length > 0);
      const neg = !!cmds && cmds.length === 0;   // a negated clause
      if (cmds && !cmds.length) cmds = null;
      if (!cmds) {
        const prev = prev0;
        if (prev && prev.action === 'object') cmds = actCarry(A, raw, prev, R.names);
        if (!cmds && prev && prev.action === 'move' && GAZE_FREE.some((g) => seqEq(g, raw))) { prev.gaze = 'free'; return true; }
      }
      if (!cmds) return neg ? 'neg' : false;
      if (isLast && !final && !stable) return 'wait';
      cmds.forEach((c, k) => {
        hits.push({cmd: c, clause: idx * 8 + k, rule: 'act:' + c.action + '/' + String(c.verb || c.name),
          open: isLast && !final, _raw: cmds.length === 1 ? raw : null});
      });
      return true;
    }
    // every clause with content words produced a command (or is a negation /
    // correction) — else the sentence is only half understood and the voice model's call is the one that counts
    // (same rule as app/intent.py Matcher.understood)
    const FILLER = new Set(('please thanks thank you me now okay ok so like just for a an the to and then can could would will muse hey hi oh um uh ' +
      "right again too also it that this i my your yeah yes well go ahead let's lets do").split(' '));
    const NEG = new Set(["don't", 'dont', 'not', 'never', 'no', 'stop', 'without', 'instead', 'actually']);
    self.understood = function (text) {
      const [pt, raws] = applyPhrasesSpans(R, normalize(text).map((w) => (has(R.alias, w) ? R.alias[w] : w)));
      const spans = clauseSpans(R, pt);
      const content = (ws) => ws.filter((w) => w !== '|' && !FILLER.has(w) && w.length > 1);
      if (spans.length <= 1) {
        const ws = content(pt);
        if (ws.length <= 6 || ws.some((w) => NEG.has(w))) return true;
        // a long one-clause request joined by and/then ("take the eraser from the tray and hold it up so I
        // can see it") is understood when every and/then part with content words parses on its own (live c1: 7 content
        // words -> 'half understood' -> only the voice model's pick_up ran, never the show)
        const said = [].concat(...raws).filter((w) => w !== '|'), parts = [[]];
        for (const w of said) { if (w === 'and' || w === 'then') parts.push([]); else parts[parts.length - 1].push(w); }
        const real = parts.filter((p) => content(p).length);
        return real.length > 1 && real.every((p) => self.parse(p.join(' '), {final: true, stable: true}).length > 0);
      }
      for (const sp of spans) {
        const ws = pt.slice(sp[0], sp[1]).filter((w) => w !== '|');
        if (ws.some((w) => NEG.has(w)) || !content(ws).length) continue;
        // re-parse the clause from the words he said, not the phrase tokens ('a bit' -> 'alittle' is no word the
        // parser reads back, so "get the laptop and slide it a bit closer" read as half understood and nothing ran)
        const said = [].concat(...raws.slice(sp[0], sp[1])).filter((w) => w !== '|');
        if (!self.parse(said.join(' '), {final: true, stable: true}).length) return false;
      }
      return true;
    };
    self.commit = function (hit, nowMs) {
      const ctx = hit._ctx;
      if (ctx && ['pose', 'expression', 'gesture'].includes(ctx[0].c.action) && !ctx[0].noint) self.last = [ctx[0], ctx[1], ctx[2], nowMs];
      else if (hit.cmd && hit.cmd.action === 'release') self.last = null;   // a released pose is no longer what "more" refers to
    };
    // the pose "more" / "higher" would adjust has been superseded (the voice model started another action, or
    // he said something else that was no command): a later "more" must not re-apply it
    self.forgetContext = function () { self.last = null; };
    self.stream = function () {
      const st = {text: '', changedMs: 0, emitted: {}, recent: []};
      st.update = function (text, final, nowMs) {
        text = String(text).trim();
        if (text !== st.text) { st.text = text; st.changedMs = nowMs; }
        const stable = (nowMs - st.changedMs) >= STABLE_MS;
        const out = [];
        if (text) {
          for (const h0 of self.parse(text, {final, stable, nowMs})) {
            const cj = canon(h0.cmd);
            if (st.emitted[h0.clause] === cj) continue;
            if (st.recent.some(([t, c]) => c === cj && nowMs - t < REPEAT_WINDOW_MS) && !has(st.emitted, h0.clause)) {
              st.emitted[h0.clause] = cj; continue;
            }
            const prev = has(st.emitted, h0.clause) ? st.emitted[h0.clause] : null;
            st.emitted[h0.clause] = cj;
            st.recent.push([nowMs, cj]);
            self.commit(h0, nowMs);
            const h = Object.assign({}, h0); delete h._ctx; h.partial = !final;
            if (prev) h.replaces = JSON.parse(prev);   // a refinement of this clause's earlier command
            out.push(h);
          }
        }
        st.recent = st.recent.filter(([t]) => nowMs - t < REPEAT_WINDOW_MS);
        if (final) { st.text = ''; st.emitted = {}; }
        return out;
      };
      return st;
    };
    return self;
  }

  // ------------------------------------------------------------------------------------------- dedupe
  const PART_PREFIX = [['hand_l', 'hand_l'], ['hand_r', 'hand_r'], ['point_l', 'hand_l'], ['point_r', 'hand_r'],
    ['thumb_l', 'hand_l'], ['thumb_r', 'hand_r'], ['paw_open_l', 'hand_l'], ['paw_open_r', 'hand_r'],
    ['arm_l', 'hand_l'], ['arm_r', 'hand_r'], ['wave_l', 'hand_l'], ['wave_r', 'hand_r'],
    ['head_', 'head'], ['gaze_', 'gaze'], ['blink_', 'eyes'], ['eye_wide', 'eyes'], ['squint', 'eyes'],
    ['brow', 'brows'], ['ear_l', 'ear_l'], ['ear_r', 'ear_r'], ['ear_flick_l', 'ear_l'],
    ['ear_flick_r', 'ear_r'], ['nose', 'nose'], ['nostril', 'nose'], ['mouth', 'mouth'], ['jaw', 'mouth'],
    ['tongue', 'tongue'], ['cheek', 'cheeks'], ['shoulder', 'shoulders'], ['tail', 'tail'],
    ['body', 'body'], ['smile', 'expression'], ['clap', 'hand_l'], ['thumbs_up', 'hand_l']];
  function partOf(param) {
    for (const [pre, part] of PART_PREFIX) if (param.startsWith(pre)) return part;
    return param;
  }
  let defaultR = null;
  function cmdKeys(cmd, R) {
    R = R || defaultR || {nameParts: {}};
    const a = cmd.action;
    if (a === 'release') {
      const ps = cmd.parts;
      return (ps === 'all' || !Array.isArray(ps)) ? ['release:all'] : ps.map((p) => 'release:' + p).sort();
    }
    if (a === 'hold') return ['hold'];
    if (a === 'look') return ['gaze'];
    if (a === 'expression') return ['expression'];
    // whole-body intent: whole-body moves and props dedupe by action family (a matcher spin vs the model's twirl = one request)
    if (a === 'move') return ['move:' + (has(MOVE_FAM, cmd.name) ? MOVE_FAM[cmd.name] : String(cmd.name))];
    if (a === 'object') {
      const f = has(OBJ_FAM, cmd.verb) ? OBJ_FAM[cmd.verb] : String(cmd.verb);
      return f === 'seat' ? ['move:posture', 'object:seat'] : ['object:' + f];
    }
    if ((a === 'gesture' || a === 'pose') && has(LEGACY_MOVE, cmd.name)) return ['move:' + MOVE_FAM[LEGACY_MOVE[cmd.name]]];
    let parts = cmd.parts;
    if (!parts || !parts.length) {
      const ps = new Set();
      for (const k of Object.keys(cmd.targets || {})) ps.add(partOf(k));
      if (!ps.size && has(R.nameParts, cmd.name)) {
        const side = cmd.side;
        const sides = (side === 'both' || side === undefined || side === null) ? ['l', 'r'] : [side];
        for (const p of R.nameParts[cmd.name]) for (const pp of (p.includes('$') ? sides.map((s) => p.split('$').join(s)) : [p])) ps.add(pp);
      }
      parts = ps.size ? Array.from(ps).sort() : [String(cmd.name || a)];
    }
    return Array.from(new Set(parts.map((p) => p.replace('_open', '')))).sort();
  }

  /* IntentDedupe.seen(cmd, nowMs, source): call for EVERY command before acting on it. source = 'matcher' (the server's
     mic matcher), 'caption' (the page matcher on final transcripts), 'model' (the realtime model's tool calls), ...
     Returns true = drop it: ANOTHER source drove an overlapping body part within WINDOW_MS (the model's function
     call ~1 s after the matcher already acted, or the caption matcher re-reading the same words), or the SAME source
     repeats an identical command. A same-source command with different content on the same part passes (a matcher
     refinement "raise your paw" -> "... a little", or the model chaining wave -> point). */
  /* whole-body intent: userSpoke(t) (the voice model's speech_started) marks a NEW utterance. Another source's command
     overlapping a record is a duplicate only while no new utterance began after that record — or when it is the same
     request anyway (same action + name/verb family): a matcher "spin" is not repeated by the model's later spin call,
     while "now jump" said after it still runs even if the matcher missed it. Without userSpoke the base rule holds.
     last = {dup, rec} of the latest seen(); attach(res) keeps motion.command()'s result on that record so a dropped
     duplicate tool call can answer with what the first one actually did. */
  const reqKey = (c) => {
    if (c.action === 'move') return 'move:' + (has(MOVE_FAM, c.name) ? MOVE_FAM[c.name] : c.name);
    if (c.action === 'object') return 'object:' + (has(OBJ_FAM, c.verb) ? OBJ_FAM[c.verb] : c.verb);
    if ((c.action === 'gesture' || c.action === 'pose') && has(LEGACY_MOVE, c.name)) return 'move:' + MOVE_FAM[LEGACY_MOVE[c.name]];
    return c.action + ':' + String(c.name || c.target || '');
  };
  /* the voice model's move/object call that DISAGREES with what another source (the fast matcher, the
     caption matcher) already ran for the same request — another move name, another object, verb, count, side, target or
     place — is not a duplicate: it replaces that job (seen() returns false, last.conflict = the replaced record; page.html
     runCommand sends it with motion.command's `replaces: <that job's id>`). A model call that agrees (same request,
     synonyms such as spin/twirl, args it simply left out) stays deduped. Only the model overrides; a later matcher /
     caption read never overrides the model. */
  const ACT_SYN = {twirl: 'spin', sit_down: 'sit', stand_up: 'stand', get_up: 'stand', hop: 'jump', bounce: 'jump',
    flip: 'backflip', step: 'walk', hold_up: 'show', drop: 'put_down', bring: 'pull', work_on: 'type_on', use: 'type_on'};
  const actSyn = (n) => (has(ACT_SYN, n) ? ACT_SYN[n] : n);
  const isActCmd = (c) => !!c && (c.action === 'move' || c.action === 'object');
  const argStr = (v) => (v === undefined || v === null ? null : (typeof v === 'object' ? String(v.name || v.id || v.ref || JSON.stringify(v)) : String(v)).toLowerCase());
  const POSTURE = {sit: 'sit', sit_down: 'sit', sit_in: 'sit', sit_at: 'sit', stand: 'stand', stand_up: 'stand', get_up: 'stand'};
  /* the same thing under two names (the matcher's "headset" vs the model's "pink headphones", "computer" vs
     "laptop", "side_table" vs "side table") is not a disagreement: with the scene (IntentDedupe.setScene(config.props))
     object / place / target names resolve to the scene's prop and place ids, and two descriptions agree when they can
     mean the same one. Without a scene, or a name the scene doesn't know: the raw comparison (spaces = underscores). */
  let SCENE = null;
  const sceneToks = (v) => String(v || '').toLowerCase().replace(/[_\-]+/g, ' ').split(/\s+/).filter((w) => w && !['the', 'a', 'an', 'my', 'your', 'one', 'that', 'this', 'those', 'these', 'of', 'on', 'onto', 'in', 'at', 'to', 'top', 'thing', 'things', 'item', 'object', 'stuff'].includes(w));
  const sing = (w) => (w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w);
  function setScene(cfg) {
    SCENE = null;
    const cs = (Array.isArray(cfg) ? cfg : [cfg]).filter((c) => c && typeof c === 'object');
    if (!cs.length) return;
    const props = [], places = [];
    for (const c of cs) {
      for (const p of (c.props || [])) {
        const words = new Set(); for (const w of [p.id, p.name, p.kind, p.category].concat(p.aliases || [])) for (const t of sceneToks(w)) { words.add(t); words.add(sing(t)); }
        const colors = new Set([p.color].concat(p.colors || []).filter((x) => typeof x === 'string').map((x) => x.toLowerCase()));
        props.push({id: String(p.id), words, colors});
      }
      for (const pl of (c.places || [])) {
        const words = new Set(); for (const w of [pl.id, pl.name].concat(pl.aliases || [])) for (const t of sceneToks(w)) { words.add(t); words.add(sing(t)); }
        places.push({id: String(pl.id), words});
      }
    }
    SCENE = {props, places};
  }
  function sceneIds(desc, kind) {   // -> [ids] it can mean, or null (no scene / nothing to resolve / unknown)
    if (!SCENE || desc === undefined || desc === null) return null;
    const d = typeof desc === 'object' ? desc : {name: String(desc)};
    const L = kind === 'place' ? SCENE.places : SCENE.props;
    if (d.id && L.some((q) => q.id === String(d.id))) return [String(d.id)];
    const ts = sceneToks(d.name || d.object || d.place || d.category || ''), col = d.color ? String(d.color).toLowerCase() : null;
    if (!ts.length && !col) return null;
    const ids = L.filter((q) => (!col || (q.colors && q.colors.has(col))) && ts.every((t) => q.words.has(t) || q.words.has(sing(t)) || (q.colors && q.colors.has(t)))).map((q) => q.id);
    return ids.length ? ids : null;
  }
  const meet = (x, y) => x.some((v) => y.includes(v));
  const flatStr = (v) => (v === null ? null : v.replace(/[_\s-]+/g, ' ').trim());
  // 'me' / 'closer' / 'over' / 'here' are one destination for the engine (furnGoal: toward him, near where it
  // stands): the model's 'closer' after the matcher's 'me' replaced a chair pull mid-walk ("pull the chair over here")
  const NEAR_ME = new Set(['me', 'closer', 'over', 'here', 'over_here', 'to_me', 'toward_me', 'towards_me', 'nearer', 'near_me']);
  function sameWhere(x, y) {   // place / target: the same scene place or thing (or the same words)
    for (const kind of ['place', 'prop']) { const ix = sceneIds(x, kind), iy = sceneIds(y, kind); if (ix && iy) return meet(ix, iy); }
    const fx = flatStr(argStr(x)), fy = flatStr(argStr(y));
    return fx === fy || (NEAR_ME.has(String(fx).replace(/\s+/g, '_')) && NEAR_ME.has(String(fy).replace(/\s+/g, '_')));
  }
  function actAgree(a, b) {   // a = the record that ran, b = the model's call
    // the model's plain "sit" after the matcher's "sit at the desk" is the same request, less specific: agrees;
    // the model's "sit in the chair" after the matcher's plain "sit" is more specific: replaces it
    if (a.action === 'object' && b.action === 'move' && has(POSTURE, a.verb) && POSTURE[a.verb] === POSTURE[b.name]) return true;
    if (a.action !== b.action) return false;
    if (a.action === 'move' ? actSyn(a.name) !== actSyn(b.name) : actSyn(a.verb) !== actSyn(b.verb)) return false;
    for (const k of ['count', 'dir', 'side', 'target', 'place', 'style']) {
      // hand_over always ends in the other paw — 'other' vs the paw the model names is the same request (the model's
      // 'left' replaced the matcher's 'other' mid-pass and the headphones went back to the right paw)
      if (k === 'side' && a.action === 'object' && actSyn(a.verb) === 'hand_over') continue;
      const x = argStr(a[k]), y = argStr(b[k]);
      if (x !== null && y !== null && x !== y && !(k === 'style') && !((k === 'target' || k === 'place') && sameWhere(a[k], b[k]))) return false;
    }
    const oa = a.object, ob = b.object;
    if (oa && ob) {
      const ia = sceneIds(oa, 'prop'), ib = sceneIds(ob, 'prop');
      if (ia && ib) { if (!meet(ia, ib)) return false; }
      else for (const k of ['id', 'name', 'color']) if (oa[k] && ob[k] && String(oa[k]).toLowerCase() !== String(ob[k]).toLowerCase()) return false;
      if (oa.ref === 'other' && ob.ref && ob.ref !== 'other' && !ob.name) return false;
    }
    return true;
  }
  function poseOpposes(a, b) {
    if (!a || !b || a.action !== 'pose' || b.action !== 'pose' || !a.targets || !b.targets) return false;
    for (const k of Object.keys(a.targets)) { const x = +a.targets[k], y = +b.targets[k]; if (Math.abs(x) > 0.05 && Math.abs(y) > 0.05 && x * y < 0) return true; }
    return false;
  }
  const IntentDedupe = {
    WINDOW_MS: 3000, TURN_MS: 30000, LATE_MS: 2500, recs: [], speechAt: null, last: null,
    reset() { this.recs = []; this.speechAt = null; this.last = null; },
    // a speech-start signal that lands within LATE_MS after a command (the server VAD ran 2.6 s behind his voice in
    // eval runs) is that same utterance reported late, not a new one: it must not cut the command's record loose.
    // startedMs = when that speech really began, when the reporter knows it (gpt-realtime's audio_start_ms on the
    // page's clock): a report 11 s late (eval dance-05) is then still the same utterance, and LATE_MS's guess is not needed.
    // A record lives while he has not started speaking again, up to TURN_MS (was 10 s: the late transcript match of that
    // run came 13.8 s after the fast matcher and danced a second time).
    userSpoke(nowMs, startedMs) { const t = nowMs === undefined ? (typeof performance !== 'undefined' ? performance.now() : Date.now()) : nowMs;
      let s;
      if (Number.isFinite(startedMs) && startedMs <= t) s = startedMs;
      else {
        const late = this.recs.filter((r) => t >= r.t && t - r.t < this.LATE_MS && (this.speechAt === null || r.t >= this.speechAt));
        s = late.length ? Math.min(t, ...late.map((r) => r.t)) : t;
      }
      this.speechAt = this.speechAt === null ? s : Math.max(this.speechAt, s); },
    seen(cmd, nowMs, source) {
      if (nowMs === undefined) nowMs = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      source = source || cmd.source || 'model';
      // (doubled actions): one request per UTTERANCE across the fast matcher, the caption
      // matcher and the voice model — a record lives until he starts speaking again (up to TURN_MS), not just WINDOW_MS:
      // the model's call / the transcript match arrive 1-4 s after he stops, often > 3 s after the matcher fired mid-sentence
      this.recs = this.recs.filter((r) => nowMs - r.t < this.WINDOW_MS || (nowMs - r.t < this.TURN_MS && this.speechAt !== null && this.speechAt <= r.t));
      const keys = cmdKeys(cmd), c2 = Object.assign({}, cmd); delete c2.source;
      const cj = canon(c2), rk = reqKey(c2);
      let conflict = null;
      for (const r of this.recs) {
        if (!r.keys.some((k) => keys.includes(k))) continue;
        const sameUtt = this.speechAt === null || r.t >= this.speechAt;
        const fresh = nowMs - r.t < this.WINDOW_MS;   // same-source exact repeats and cross-utterance same-family: 3 s only
        if ((r.source !== source && (sameUtt || (fresh && r.rk === rk))) || (fresh && r.cj === cj)) {
          if (r.cj !== cj && r.source !== source && source === 'model' && r.source !== 'model' && isActCmd(c2) && isActCmd(r.cmd) && !actAgree(r.cmd, c2)) { conflict = r; break; }
          // the model's pose drives the same part the OPPOSITE way (matcher: head up; model: head down) -> not
          // the same request, whatever the utterance bookkeeping says: the model's runs and replaces the matcher's record
          if (r.source !== source && source === 'model' && r.source !== 'model' && poseOpposes(r.cmd, c2)) { conflict = r; break; }
          this.last = {dup: true, rec: r}; return true;
        }
      }
      const rec = {t: nowMs, keys, cj, rk, source, res: null, cmd: c2};
      if (conflict) { this.recs = this.recs.filter((r) => r !== conflict); rec.replaced = conflict; }
      this.recs.push(rec);
      this.last = {dup: false, rec, conflict};
      return false;
    },
    attach(res) { if (this.last && !this.last.dup && this.last.rec) this.last.rec.res = res; },
    // the command could not be executed (e.g. motion.js has no such gesture): forget it so the next source's
    // version of the same request (the realtime model's tool call) is not dropped as a duplicate
    forget(cmd, source) {
      const c2 = Object.assign({}, cmd); delete c2.source;
      const cj = canon(c2);
      for (let i = this.recs.length - 1; i >= 0; i--) {
        if (this.recs[i].cj === cj && (!source || this.recs[i].source === source)) { this.recs.splice(i, 1); return true; }
      }
      return false;
    },
  };

  IntentDedupe.agree = actAgree;
  IntentDedupe.setScene = setScene;
  const Intent = {create, normalize, tokenize: (R, t) => tokenize(R, t), canon, cmdKeys, partOf, MOVE_FAM, OBJ_FAM,
    setDefaultRules(data) { defaultR = compile(data); }, STABLE_MS};
  if (typeof module !== 'undefined' && module.exports) module.exports = {Intent, IntentDedupe};
  else { root.Intent = Intent; root.IntentDedupe = IntentDedupe; }
})(typeof window !== 'undefined' ? window : globalThis);

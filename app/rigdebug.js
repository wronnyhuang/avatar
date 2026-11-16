/* rigdebug.js — the live RIG DEBUG panel of the avatar page: a menu that opens up the rig and shows every control
 * live, who drives it and what the avatar sees. Contract: docs/RIG-DEBUG.md.
 *
 * window.AvatarRigDebug = {mount(ctx), open(), close(), toggle(), isOpen(), onRender(j), snapshot()}
 *   mount({D, S, Q, sendCtl})   page.html, once at load: wires #rigbtn (and ?rig=1 opens it). Builds and runs nothing.
 *   onRender(j)                 the hub's {"type":"rigdbg", fps, cols, rows | unsupported} (page onJSON)
 *   snapshot()                  window.__avatar.rigdbg(): {open, paused, collapsed, ticks, lip, owners, dials, log, perc}
 * + planner: {mode, source, model, program, ms, warnings, errors, targets, ...} (also closed)
 * ctx.planner {get, set, last, warm} -> the "Planner" row at the top of the meters (current | semantic API: the next
 *   command) and the latest commanded action: source, model, the semantic program as it streams, timings, warnings,
 *   errors, resolved live targets. Painted in the slow frame only when its data changed; nothing while closed.
 *
 * ZERO COST WHEN CLOSED: no panel DOM until the first open, D.rdbg === null while closed (every page hook is a one-line
 * `if (D.rdbg) D.rdbg.X(...)`), no timers / listeners / probe capture while closed. Open: tick() (rig timer, right after
 * motion.step) copies the 77 values into a ring and asks for ONE animation frame; everything else (motion.probe(), the
 * motion's action events, perception, DOM) runs in that frame, writes only changed cells, skips groups scrolled out of
 * view, and only the header while collapsed. The page sends the hub nothing new except {"type":"rigdbg","on":true|false}
 * (open / close; again at a new session while open) — the renderer then reports its lip values per frame (contract §1).
 *
 * Instance hooks (D.rdbg while open): tick(rig, rt), shown(t) (the frame on screen), ev(name, detail) (avatarEvent),
 *   cmd(cmd, source, info) (runCommand), ctl(obj) (sendCtl).
 * Motion probe (contract §2, app/motion.js): m.setProbe(on), m.probe(), m.meta() — FEATURE-DETECTED; without it the
 *   panel shows the values with "owner unknown" and reads the driving strip from m.debug().
 */
(function () {
  'use strict';
  const raf = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : f => setTimeout(f, 16);
  const caf = typeof cancelAnimationFrame === 'function' ? cancelAnimationFrame : clearTimeout;
  const NH = 160;                  // history ring: ~5.3 s at the 30 Hz rig tick (sparklines, "moving")
  const MOVE_BACK = 9;             // ~300 ms at 30 Hz
  // a session's first seconds, while the page locks its video clock to the voice, the panel runs LIGHT (header every
  // frame; a new session: no meter refresh / probe for WARM_MS; just opened: a full refresh at most every WARM_FULL_MS): with the panel open at Start the desktop drawer's full-rate refresh slowed that
  // lock (sync max 123-184 ms vs 76-96 ms in the first second, equal from ~4 s on)
  const WARM_MS = 4000, WARM_FULL_MS = 500, OPEN_WARM_MS = 1500;
  const LOG_CAP = 200, LIP_ROWS = 300;
  const LS = {h: 'avatar.rig.h', w: 'avatar.rig.w', fold: 'avatar.rig.fold'};
  const LIP_DEFAULT = ['jaw', 'round', 'spread', 'press', 'labdent', 'tongue', 'cheek', 'voice', 'lip_open', 'lip_upper', 'lip_lower',
    'lip_width', 'lip_protrude', 'lip_press', 'lip_tuck', 'teeth_gap', 'tongue_tip', 'tongue_dental', 'tongue_back'];

  // ------------------------------------------------------------------ owners: layer -> plain words + colour
  const OWN = {
    life: ['life', '#8b98a9'], fidget: ['fidget', '#9aa7b8'], gaze: ['gaze follow', '#4fd1c5'], mirror: ['mirroring', '#f78fb3'],
    feeling: ['feeling', '#d88cff'], reflex: ['reflex', '#ffa657'], speech: ['speech', '#79b8ff'], cospeech: ['speech gestures', '#a5a5ff'],
    listen: ['listener', '#7ee787'], think: ['thinking', '#c297ff'], program: ['program', '#f2cc60'], planner: ['planner move', '#e3b341'],
    move: ['body move', '#b4e05a'], cmd: ['commanded', '#ff7b72'], look: ['commanded look', '#56d4f5'], hold: ['hold', '#d2a679'],
    prior: ['learned prior', '#bc8cff'], learned: ['learned (hub)', '#a8c66c'], lips: ['lip sync (renderer)', '#ff9bce'],
  };
  const UNKNOWN = ['owner unknown', '#6b7685'];
  const ownWord = by => (OWN[by] || [by ? String(by) : UNKNOWN[0]])[0];
  const ownCol = by => (OWN[by] || UNKNOWN)[1];
  // gesture sources (motion.js G.src) -> layers, for the fallback driving strip and the log
  const SRC_LAYER = {cmd: 'cmd', ui: 'cmd', voice: 'cmd', matcher: 'cmd', reflex: 'reflex', speech: 'speech', cospeech: 'cospeech',
    listen: 'listen', think: 'think', program: 'program', life: 'life', fidget: 'fidget', affect: 'feeling', auto: 'life'};
  const CMD_SRC = {voice: 'voice', 'voice-early': 'voice (early)', matcher: 'his words (hub)', 'matcher-llm': 'his words (hub LLM)',
    caption: 'his words (page)', ui: 'page', affect: 'feeling', planner: 'planner'};
  // a reflex and the perception event that triggers it: one log line "wave detected → wave-back (reflex)"
  const REFLEX_OF = {wave_back: 'wave', look_back: 'look_back', perk: 'face_appear', look_where_pointed: 'point', curious_look: 'object_up',
    startle: 'approach_fast', nod_back: 'nod', quizzical: 'head_shake'};

  // ------------------------------------------------------------------ groups
  const GROUPS = [
    ['head', 'Head', p => /^head_/.test(p)],
    ['eyes', 'Eyes & gaze', p => /^(gaze_[xy]|blink_[lr]|eye_wide|squint)$/.test(p)],
    ['brows', 'Brows & face', p => /^(brow_|nostril$|nose_wiggle$)/.test(p)],
    ['mouth', 'Mouth & lips', (p, M) => M.lips.has(p) || /^(smile|mouth_open|tongue_out)$/.test(p)],
    ['torso', 'Torso', p => /^(breath$|body_)/.test(p)],
    ['arms', 'Arms & hands', p => /^(hand_|thumb_|shoulder_|point_|arm_[lr]$|wave_[lr]$|paw_open_|clap$|thumbs_up$)/.test(p)],
    ['ears', 'Ears & tail', p => /^(ear_|tail_)/.test(p)],
  ];
  const MORE = [
    ['feel', 'Feelings & effects', (p, M) => M.emo.has(p)],
    ['root', 'Root & camera', p => /^(root_|squash$|stand$|cam_|gaze_free$|fingers_)/.test(p)],
    ['feet', 'Feet', p => /^foot_/.test(p)],
    ['props', 'Props', p => /^prop\d+_/.test(p)],
    ['gest', 'Gesture network', p => /^gest_[lr]$/.test(p)],
  ];

  // ------------------------------------------------------------------ helpers
  const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const lsGet = k => { try { return localStorage.getItem(k); } catch (e) { return null; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} };
  const hexA = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; };
  const short = (s, n) => { s = String(s == null ? '' : s); return s.length > n ? s.slice(0, n - 1) + '…' : s; };
  const num = (v, d) => (Number.isFinite(+v) ? (+v).toFixed(d === undefined ? 2 : d) : '–');
  function h(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }
  function call(f, dflt) { try { return f(); } catch (e) { return dflt; } }

  // static param lists when no motion object exists yet (panel opened before Start); motion.js order
  function staticAll(L, rig) {
    const t = ['root_x', 'root_y', 'root_z', 'root_yaw', 'root_pitch', 'root_roll', 'squash', 'gaze_free', 'cam_zoom', 'cam_x', 'cam_y', 'stand', 'fingers_l', 'fingers_r'];
    const pr = k => { for (const a of ['x', 'y', 'z', 'rx', 'ry', 'rz', 'hl', 'hr', 'hh', 's']) t.push('prop' + k + '_' + a); };
    for (let k = 0; k < 8; k++) pr(k);
    t.push('foot_l_x', 'foot_l_y', 'foot_l_z', 'foot_r_x', 'foot_r_y', 'foot_r_z');
    for (let k = 8; k < 16; k++) pr(k);
    const R = L.RANGE || {}, seen = new Set(rig.concat(t));
    const emo = Object.keys(R).filter(p => !seen.has(p) && !/^gest_/.test(p));
    return rig.concat(t, emo, L.GEST_PARAMS || ['gest_l', 'gest_r']);
  }
  // the param lists / ranges: motion.meta() when the probe exists, else the motion object + AvatarMotionLib
  function metaOf(m) {
    const L = (typeof window !== 'undefined' && (window.AvatarMotionLib || window.AvatarMotion)) || {};
    const mt = m && typeof m.meta === 'function' ? call(() => m.meta(), null) : null;
    const rig = ((mt && mt.params) || (m && m.params) || L.RIG_PARAMS || []).slice();
    const all = ((mt && mt.allParams) || (m && m.allParams) || staticAll(L, rig)).slice();
    const ranges = Object.assign({}, L.RANGE || {}, (mt && mt.ranges) || {});
    const lips = new Set((mt && mt.lipsync) || L.LIPSYNC || LIP_DEFAULT);
    const legacy = new Set((mt && mt.legacy) || L.LEGACY || []);
    const periodic = new Set((mt && mt.periodic) || all.filter(p => /^root_(yaw|pitch|roll)$|^prop\d+_r[xyz]$/.test(p)));
    for (const p of lips) if (!ranges[p]) ranges[p] = p === 'lip_width' ? [-1, 1] : [0, 1];
    for (const p of legacy) if (!ranges[p]) ranges[p] = [0, 1];   // = motion meta()'s fallback (static path at open)
    for (const p of all) if (!ranges[p] && !periodic.has(p)) ranges[p] = [-1, 1];
    const rigSet = new Set(rig);
    const emo = new Set(all.filter(p => !rigSet.has(p) && !/^(root_|squash$|stand$|cam_|gaze_free$|fingers_|foot_|prop\d+_|gest_)/.test(p)));
    return {rig, all, ranges, lips, legacy, periodic, emo, probe: !!(mt && m && typeof m.probe === 'function'), layers: (mt && mt.layers) || null};
  }

  // ------------------------------------------------------------------ module state
  let C = null;                     // {D, S, Q, sendCtl} from mount
  let P = null;                     // the open panel instance (=== C.D.rdbg)
  let ui = null;                    // the built DOM (kept, hidden, after the first close)
  const keep = {collapsed: false, more: false, tab: 'rig', unsupported: null, planFold: false, fold: new Set((lsGet(LS.fold) || '').split(',').filter(Boolean))};

  // ------------------------------------------------------------------ CSS (injected at the first open)
  const CSS = `
#rigbtn[aria-pressed=true]{border-color:var(--acc,#3d7cf0);background:#1b2433}
#rigpanel{position:fixed;z-index:35;display:flex;flex-direction:column;background:#11151c;color:var(--fg,#e8ecf1);border:1px solid var(--line,#262c36);font:13px/1.3 system-ui,-apple-system,Roboto,sans-serif;box-shadow:0 -10px 28px #000b;overscroll-behavior:contain;-webkit-tap-highlight-color:transparent}
#rigpanel button{font:600 13px system-ui,-apple-system,Roboto,sans-serif;color:var(--fg,#e8ecf1);background:var(--card,#171b22);border:1px solid var(--line,#262c36);border-radius:10px;cursor:pointer;touch-action:manipulation}
#rigpanel button:focus-visible{outline:2px solid #fff;outline-offset:2px}
html.rd-sheet #rigpanel{left:0;right:0;bottom:0;height:var(--rdh,42vh);border-radius:16px 16px 0 0;border-width:1px 0 0;padding:0 env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
html.rd-sheet body{padding-bottom:calc(var(--rdh,42vh) + 84px)}
html.rd-sheet #stage{position:sticky;top:0;z-index:25}
html.rd-sheet #dock{bottom:calc(var(--rdh,42vh) + 8px);z-index:45}
html.rd-wide #rigpanel{top:0;right:0;bottom:0;width:var(--rdw,400px);border-width:0 0 0 1px;box-shadow:-10px 0 28px #0009;padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) 0}
html.rd-wide body{margin-left:max(0px,calc((100vw - var(--rdw,400px) - 760px) / 2));margin-right:calc(var(--rdw,400px) + max(0px,calc((100vw - var(--rdw,400px) - 760px) / 2)))}
html.rd-wide #dock{left:calc((100vw - var(--rdw,400px)) / 2);width:min(760px,calc(100vw - var(--rdw,400px)))}
html.rd-wide.rd-col #rigpanel{bottom:auto}
.rd-grip{position:absolute;z-index:3;touch-action:none}
html.rd-sheet .rd-grip{left:0;right:0;top:0;height:16px;cursor:ns-resize}
html.rd-sheet .rd-grip::after{content:"";position:absolute;left:50%;top:6px;width:38px;height:4px;margin-left:-19px;border-radius:2px;background:#4a5466}
html.rd-wide .rd-grip{left:-5px;top:0;bottom:0;width:10px;cursor:ew-resize}
html.rd-wide .rd-grip::after{content:"";position:absolute;left:3px;top:50%;width:4px;height:42px;margin-top:-21px;border-radius:2px;background:#3a4354}
.rd-head{flex:none;display:flex;align-items:center;gap:8px;padding:8px 8px 6px 12px;min-height:50px;touch-action:none;user-select:none;-webkit-user-select:none}
html.rd-sheet .rd-head{padding-top:14px}
.rd-title{flex:1 1 auto;min-width:0;display:flex;align-items:center;gap:8px;cursor:pointer}
.rd-title b{font-size:15px}
.rd-dot{width:9px;height:9px;border-radius:50%;flex:none;background:#5b6677}
.rd-dot.live{background:var(--ok,#2fb77a);box-shadow:0 0 0 3px rgba(47,183,122,.18)}
.rd-dot.paused{background:var(--warn,#e0a526)}
#rd-sum{flex:1 1 auto;min-width:0;color:var(--dim,#93a0b0);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rd-tabs{display:none;gap:0;flex:none}
html.rd-sheet .rd-tabs{display:flex}
html.rd-sheet #rigpanel:not(.rd-col) #rd-sum{display:none}
#rigpanel .rd-tabs button{border-radius:0;min-height:36px;padding:0 12px;font-size:13px;color:var(--dim,#93a0b0)}
#rigpanel .rd-tabs button:first-child{border-radius:10px 0 0 10px}
#rigpanel .rd-tabs button:last-child{border-radius:0 10px 10px 0;border-left:0}
#rigpanel .rd-tabs button[aria-selected=true]{color:var(--fg,#e8ecf1);background:#1f2a3d;border-color:#35507e}
.rd-btns{flex:none;display:flex;gap:6px}
#rigpanel .rd-btns button{width:40px;height:40px;padding:0;display:flex;align-items:center;justify-content:center}
#rigpanel .rd-btns svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}
#rd-pause[aria-pressed=true]{background:#3a2e12;border-color:#7a5d1c;color:#ffe29a}
#rigpanel.rd-col .rd-strips,#rigpanel.rd-col .rd-main,#rigpanel.rd-col .rd-logbox,#rigpanel.rd-col .rd-tabs{display:none!important}
.rd-strips{flex:none;border-top:1px solid var(--line,#262c36);padding:5px 0 4px}
.rd-strip{display:flex;flex-wrap:wrap;gap:4px 5px;padding:2px 10px;align-items:center}
html.rd-sheet .rd-strip{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none}
html.rd-sheet .rd-strip::-webkit-scrollbar{display:none}
.rd-strip>em{font-style:normal;font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;color:#6f7b8c;margin-right:2px;flex:none}
.rd-chip{flex:none;display:inline-flex;align-items:center;gap:5px;font-size:12px;padding:2px 8px;border-radius:999px;background:#1a2029;border:1px solid #29313d;white-space:nowrap;font-variant-numeric:tabular-nums}
.rd-chip i{width:7px;height:7px;border-radius:50%;background:#4a5466;flex:none}
.rd-chip i.on{background:var(--ok,#2fb77a)}
.rd-chip.dim{color:#7d8899}
.rd-main{flex:1 1 auto;min-height:0;overflow-y:auto;overscroll-behavior:contain;border-top:1px solid var(--line,#262c36);-webkit-overflow-scrolling:touch}
#rigpanel[data-tab=events] .rd-main{display:none}
html.rd-sheet #rigpanel[data-tab=rig] .rd-logbox{display:none}
.rd-note{padding:6px 12px;color:var(--dim,#93a0b0);font-size:11.5px}
.rd-note code{font:11px ui-monospace,Menlo,monospace;color:#c9d3df}
.rd-group{border-top:1px solid #1e242e}
.rd-ghead{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:8px;padding:6px 10px 6px 12px;min-height:34px;background:#161b23;cursor:pointer;user-select:none;-webkit-user-select:none}
.rd-gname{font-weight:650;font-size:13px}
.rd-gname::before{content:"";display:inline-block;width:7px;height:7px;margin:0 8px 1px 0;border-right:2px solid #7d8899;border-bottom:2px solid #7d8899;transform:rotate(45deg);transition:transform .12s}
.rd-group.fold .rd-gname::before{transform:rotate(-45deg)}
.rd-group.fold .rd-dials{display:none}
.rd-gn{color:#6f7b8c;font-size:11.5px}
.rd-gown{margin-left:auto;font-size:11.5px;font-weight:650;padding:2px 9px;border-radius:999px;background:#222a35;color:#c9d3df;white-space:nowrap;max-width:55%;overflow:hidden;text-overflow:ellipsis}
.rd-sub{padding:5px 12px 3px;font-size:11.5px;color:var(--dim,#93a0b0);border-top:1px dashed #252c37}
.rd-sub b{color:#ff9bce;font-weight:650}
.rd-dial{display:grid;grid-template-columns:minmax(0,7.4em) 3.5em minmax(36px,1fr) 8.7em;align-items:center;gap:0 8px;padding:2px 10px 2px 12px;min-height:24px;cursor:pointer;transition:background-color .6s}
.rd-dial[data-src=renderer]{grid-template-columns:minmax(0,7.4em) 3.5em minmax(36px,1fr) 10.4em}
.rd-dial.moving{background:rgba(61,124,240,.09);box-shadow:inset 3px 0 0 var(--c,#3d7cf0);transition:background-color .08s}
.rd-dial.rest .rd-own{opacity:.45}.rd-dial.rest .rd-val{color:#7d8899}
.rd-dial.open{background:#18202c}
.rd-name{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;color:#c9d3df;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rd-name em{font:600 9.5px system-ui,sans-serif;font-style:normal;color:#9a8358;border:1px solid #4d4128;border-radius:4px;padding:0 3px;margin-left:5px;vertical-align:1px}
.rd-val{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;text-align:right;font-variant-numeric:tabular-nums;color:#e8ecf1}
.rd-bar{position:relative;isolation:isolate;height:9px;border-radius:5px;background:#212833;overflow:hidden}
.rd-bar i{position:absolute;left:0;top:0;bottom:0;width:100%;transform-origin:0 0;transform:scaleX(0);background:var(--c,#3d7cf0);border-radius:2px}
.rd-bar b{position:absolute;top:0;bottom:0;width:1px;background:#6b7685;z-index:1}
.rd-bar u{position:absolute;top:-1px;bottom:-1px;width:3px;margin-left:-1px;background:#fff;border-radius:1px;z-index:2;box-shadow:0 0 0 1px #000a;text-decoration:none}
.rd-own{font-size:10.5px;line-height:1.35;font-weight:600;padding:1px 5px;border-radius:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--c,#8b98a9);background:var(--cb,rgba(139,152,169,.12));justify-self:start;max-width:100%}
.rd-dial.rd-wait .rd-own{grid-column:2 / -1;justify-self:stretch;white-space:normal;font-weight:500;color:#7d8899;background:transparent;padding-left:0}
.rd-dial.rd-wait .rd-val,.rd-dial.rd-wait .rd-bar{display:none}
.rd-spark{display:block;width:calc(100% - 24px);height:48px;margin:2px 12px 6px;background:#0d1117;border:1px solid #232a35;border-radius:6px}
#rd-more{display:block;width:calc(100% - 24px);margin:10px 12px;min-height:40px;text-align:left;padding:0 12px}
#rd-more::before{content:"";display:inline-block;width:7px;height:7px;margin:0 10px 2px 0;border-right:2px solid currentColor;border-bottom:2px solid currentColor;transform:rotate(-45deg)}
#rd-more[aria-expanded=true]::before{transform:rotate(45deg)}
.rd-pslot{padding:3px 12px 0;font:600 11.5px system-ui,sans-serif;color:#9aa7b8}
.rd-empty{padding:4px 12px 8px;color:#6f7b8c;font-size:12px}
.rd-logbox{flex:0 0 28%;min-height:0;display:flex;flex-direction:column;border-top:1px solid var(--line,#262c36)}
html.rd-sheet .rd-logbox{flex:1 1 auto}
.rd-loghead{flex:none;display:flex;align-items:center;gap:8px;padding:6px 12px;font-weight:650;font-size:13px;background:#161b23}
.rd-loghead span{color:#6f7b8c;font-weight:500;font-size:11.5px}
#rd-log{flex:1 1 auto;min-height:0;overflow-y:auto;margin:0;padding:2px 0 8px;list-style:none;overscroll-behavior:contain}
.rd-ev{display:flex;gap:8px;padding:3px 12px;font-size:12.5px;line-height:1.35;border-left:3px solid transparent}
.rd-ev time{flex:none;font:11.5px ui-monospace,SFMono-Regular,Menlo,monospace;color:#6f7b8c;padding-top:1px}
.rd-ev[data-kind=perc]{border-left-color:#4fd1c5}.rd-ev[data-kind=reflex]{border-left-color:#ffa657}.rd-ev[data-kind=cmd]{border-left-color:#ff7b72}
.rd-ev[data-kind=voice]{border-left-color:#79b8ff}.rd-ev[data-kind=planner]{border-left-color:#e3b341}.rd-ev[data-kind=gaze]{border-left-color:#56d4f5}
.rd-ev[data-kind=feeling]{border-left-color:#d88cff}.rd-ev[data-kind=motion]{border-left-color:#a5a5ff}.rd-ev[data-kind=info]{border-left-color:#4a5466;color:#9aa7b8}
.rd-plan{border-bottom:1px solid #1e242e;background:#131820}
.rd-phead{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;padding:7px 10px 7px 12px;min-height:40px}
.rd-phead .rd-gname{cursor:pointer;user-select:none;-webkit-user-select:none}
.rd-plan.fold .rd-phead .rd-gname::before{transform:rotate(-45deg)}
.rd-plan.fold .rd-pbody{display:none}
.rd-seg{display:inline-flex;flex:none}
#rigpanel .rd-seg button{min-height:32px;padding:0 11px;border-radius:0;font-size:12.5px;color:var(--dim,#93a0b0)}
#rigpanel .rd-seg button:first-child{border-radius:9px 0 0 9px}
#rigpanel .rd-seg button:last-child{border-radius:0 9px 9px 0;border-left:0}
#rigpanel .rd-seg button[aria-pressed=true]{color:#1a1200;background:#e3b341;border-color:#e3b341}
#rigpanel .rd-seg button:disabled{opacity:.45;cursor:default}
#rd-psum{flex:1 1 12em;min-width:0;font-size:11.5px;color:var(--dim,#93a0b0);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rd-pbody{padding:0 12px 8px;font-size:12px}
.rd-pcmd{color:#e8ecf1;margin:2px 0 5px;overflow-wrap:anywhere}
.rd-pkv{display:grid;grid-template-columns:max-content 1fr;gap:1px 10px;font-variant-numeric:tabular-nums}
.rd-pkv dt{color:#7d8899}.rd-pkv dd{margin:0;font:11.5px ui-monospace,SFMono-Regular,Menlo,monospace;color:#c9d3df;overflow-wrap:anywhere}
.rd-pkv dd.warn{color:#ffd37a}.rd-pkv dd.err{color:#ff8f87}
.rd-prog{margin:6px 0 0;padding:6px 8px;max-height:38vh;overflow:auto;background:#0d1117;border:1px solid #232a35;border-radius:6px;font:11.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:#d6e2ef;white-space:pre-wrap;overflow-wrap:anywhere;user-select:text;-webkit-user-select:text;cursor:text}
.rd-pnote{color:#7d8899;font-size:11.5px;margin-top:4px}
@media (max-width:899px){.rd-dial{min-height:28px}}
html.rd-wide .rd-chip{font-size:11.5px;padding:1px 7px}html.rd-wide .rd-strip{gap:3px 4px}
@media (prefers-reduced-motion:reduce){.rd-dial,.rd-dial.moving,.rd-gname::before{transition:none}}`;

  const ICON = {
    pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>',
    play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5l12 7-12 7z"/></svg>',
    down: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>',
    up: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 15l6-6 6 6"/></svg>',
    right: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>',
    left: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  };

  // ------------------------------------------------------------------ DOM build (first open only)
  function build() {
    if (ui) return ui;
    if (!document.getElementById('rd-css')) { const st = h('style'); st.id = 'rd-css'; st.textContent = CSS; document.head.appendChild(st); }
    const root = h('section'); root.id = 'rigpanel'; root.hidden = true;
    root.setAttribute('role', 'region'); root.setAttribute('aria-label', 'Rig debug panel');
    root.innerHTML =
      '<div class="rd-grip" aria-hidden="true"></div>' +
      '<div class="rd-head"><div class="rd-title" title="Tap to fold / unfold; drag to resize"><span class="rd-dot"></span><b>Rig</b><span id="rd-sum"></span></div>' +
      '<div class="rd-tabs" role="tablist"><button type="button" role="tab" data-tab="rig" aria-selected="true">Meters</button><button type="button" role="tab" data-tab="events" aria-selected="false">Events</button></div>' +
      '<div class="rd-btns"><button type="button" id="rd-pause" aria-pressed="false" aria-label="Freeze the panel (the avatar keeps going)" title="Freeze the panel (the avatar keeps going)">' + ICON.pause + '</button>' +
      '<button type="button" id="rd-collapse" aria-expanded="true" aria-label="Fold the panel" title="Fold the panel"></button>' +
      '<button type="button" id="rd-close" aria-label="Close the rig panel" title="Close">' + ICON.close + '</button></div></div>' +
      '<div class="rd-strips"><div class="rd-strip" id="rd-drive"><em>driving</em></div><div class="rd-strip" id="rd-perc"><em>sees</em></div></div>' +
      '<div class="rd-main"></div>' +
      '<div class="rd-logbox"><div class="rd-loghead">Events <span id="rd-logn"></span></div><ol id="rd-log" aria-live="off"></ol></div>';
    document.body.appendChild(root);
    const q = s => root.querySelector(s);
    ui = {root, head: q('.rd-head'), grip: q('.rd-grip'), title: q('.rd-title'), dot: q('.rd-dot'), sum: q('#rd-sum'), pause: q('#rd-pause'),
      collapse: q('#rd-collapse'), close: q('#rd-close'), drive: q('#rd-drive'), perc: q('#rd-perc'), main: q('.rd-main'),
      logbox: q('.rd-logbox'), log: q('#rd-log'), logn: q('#rd-logn'), tabs: [...root.querySelectorAll('.rd-tabs button')]};
    // perception chips (fixed keys; tests read #rd-perc [data-k])
    for (const k of ['face', 'hands', 'pose', 'pos', 'attn', 'handup', 'expr', 'mic', 'speak', 'render']) {
      const c = h('span', 'rd-chip dim'); c.dataset.k = k; c.innerHTML = '<i></i><span>' + k + ' –</span>'; ui.perc.appendChild(c);
    }
    ui.driveBox = h('span'); ui.driveBox.style.display = 'contents'; ui.drive.appendChild(ui.driveBox);
    // buttons
    ui.pause.addEventListener('click', () => { if (P) setPaused(!P.paused); });
    ui.collapse.addEventListener('click', () => setCollapsed(!keep.collapsed));
    ui.close.addEventListener('click', () => close());
    for (const b of ui.tabs) b.addEventListener('click', () => setTab(b.dataset.tab));
    // taps in the meters: a dial -> its sparkline; a group header -> fold; More
    ui.main.addEventListener('click', e => {
      if (!P) return;
      const pb = e.target.closest('.rd-seg button'); if (pb) { setPlannerMode(pb.dataset.v); return; }
      if (e.target.closest('.rd-phead .rd-gname')) { keep.planFold = !keep.planFold; if (ui.plan) ui.plan.classList.toggle('fold', keep.planFold); return; }
      if (e.target.closest('.rd-plan')) return;
      const more = e.target.closest('#rd-more'); if (more) { setMore(!keep.more); return; }
      const gh = e.target.closest('.rd-ghead'); if (gh) { toggleFold(gh.parentNode); return; }
      const d = e.target.closest('.rd-dial'); if (d && P.byEl.has(d)) toggleSpark(P.byEl.get(d));
    });
    wireDrag();
    return ui;
  }

  // the meter rows of the 77 (and, on first "More", the extras)
  function buildMeters() {
    const M = P.M, main = ui.main;
    main.textContent = '';
    buildPlanner(main);                                        // the Planner row + section, at the top
    main.appendChild(h('div', 'rd-note', '<code>_l</code> / <code>_r</code> = screen-left / screen-right (the character\'s own right / left). Tap a row for its last 5 s; tap a group to fold it.'));
    const placed = new Set();
    P.groups = [];
    for (const [id, name, test] of GROUPS) {
      let ps = M.rig.filter(p => !placed.has(p) && test(p, M));
      ps.forEach(p => placed.add(p));
      if (id === 'mouth') ps = ps.filter(p => !M.lips.has(p)).concat(ps.filter(p => M.lips.has(p)));
      if (id === 'arms') ps = ps.filter(p => !M.legacy.has(p)).concat(ps.filter(p => M.legacy.has(p)));
      P.groups.push(mkGroup(id, name, ps, main, false));
    }
    const rest = M.rig.filter(p => !placed.has(p));
    if (rest.length) P.groups.push(mkGroup('other', 'Other', rest, main, false));
    const mb = h('button', '', 'More: feelings &amp; effects, root &amp; camera, feet, props, gesture network');
    mb.type = 'button'; mb.id = 'rd-more'; mb.setAttribute('aria-expanded', 'false'); main.appendChild(mb);
    ui.moreBtn = mb;
    ui.moreBox = h('div'); ui.moreBox.id = 'rd-moreb'; ui.moreBox.hidden = true; main.appendChild(ui.moreBox);
    P.moreBuilt = false;
  }
  function buildMore() {
    if (P.moreBuilt) return;
    P.moreBuilt = true;
    const M = P.M, rigSet = new Set(M.rig);
    const ex = M.all.filter(p => !rigSet.has(p));
    for (const [id, name, test] of MORE) {
      const ps = ex.filter(p => test(p, M));
      if (id === 'props') { P.props = mkGroup('props', name, [], ui.moreBox, true); P.props.slots = new Map(); P.groups.push(P.props); continue; }
      if (ps.length) P.groups.push(mkGroup(id, name, ps, ui.moreBox, true));
    }
    refillExtras();
  }
  // the extras are sampled only while "More" is open: their history restarts flat at the newest value
  function refillExtras() {
    const r = P.rig || {};
    for (let i = P.nRig; i < P.NP; i++) { const x = r[P.names[i]], v = typeof x === 'number' && x === x ? x : 0; for (let k = 0; k < NH; k++) P.hist[k * P.NP + i] = v; }
  }
  function mkGroup(id, name, ps, parent, more) {
    const el = h('div', 'rd-group'); el.dataset.g = id;
    if (keep.fold.has(id)) el.classList.add('fold');
    el.innerHTML = '<div class="rd-ghead"><span class="rd-gname">' + esc(name) + '</span><span class="rd-gn">' + (ps.length ? ps.length : '') + '</span><span class="rd-gown">–</span></div><div class="rd-dials"></div>';
    parent.appendChild(el);
    const g = {id, name, el, own: el.querySelector('.rd-gown'), box: el.querySelector('.rd-dials'), dials: [], vis: true, more, ownTxt: ''};
    let sub = null;
    for (const p of ps) {
      if (P.M.lips.has(p) && !sub) {
        sub = h('div', 'rd-sub', '<b>Lip sync</b> · <span>set by the renderer — waiting</span>'); g.box.appendChild(sub); g.lipSub = sub.querySelector('span');
      }
      g.dials.push(mkDial(p, g.box));
    }
    if (P.io) P.io.observe(el);
    return g;
  }
  function mkDial(p, parent) {
    const M = P.M, i = P.idx.get(p), [lo, hi] = M.ranges[p] || [-1, 1], periodic = M.periodic.has(p), lip = M.lips.has(p);
    const signed = !periodic && lo < 0 && hi > 0;
    const el = h('div', 'rd-dial'); el.dataset.p = p; if (lip) el.dataset.src = 'renderer';
    el.innerHTML = '<span class="rd-name" title="' + esc(p) + (periodic ? ' (turns, shown mod 1)' : ' [' + lo + ', ' + hi + ']') + '">' + esc(p) +
      (M.legacy.has(p) ? '<em>legacy</em>' : '') + '</span><span class="rd-val">–</span><span class="rd-bar">' +
      (signed ? '<b style="left:' + (100 * -lo / (hi - lo)).toFixed(2) + '%"></b>' : '') + '<i></i></span><span class="rd-own">–</span>';
    parent.appendChild(el);
    const d = {p, i, el, lo, hi, signed, periodic, lip, span: periodic ? 1 : (hi - lo), z: signed ? -lo / (hi - lo) : 0,
      val: el.querySelector('.rd-val'), bar: el.querySelector('.rd-bar'), fill: el.querySelector('.rd-bar i'), own: el.querySelector('.rd-own'),
      c: {t: '', a: -1, b: -1, own: '', ownTitle: '', col: '', mv: false, wait: null, rt: -2}, movUntil: 0, spark: null, rcol: -1};
    d.dec = d.span <= 0.5 ? 3 : 2;
    P.byEl.set(el, d); P.dials.set(p, d);
    return d;
  }

  // ------------------------------------------------------------------ open / close
  function open() {
    if (!C) return false;
    if (P) { if (P.collapsed && !keep.collapsed) setCollapsed(false); return true; }
    build();
    const D = C.D;
    P = {D, S: C.S, ticks: 0, paused: false, collapsed: keep.collapsed, openedAt: performance.now(), m: null, M: null, probeOn: false,
      pr: null, prErr: null, dbg: null, dbgAt: 0, names: [], idx: new Map(), nRig: 0, NP: 0, hist: null, hpos: 0, histN: 0,
      dials: new Map(), byEl: new WeakMap(), groups: [], sparks: new Set(), log: [], logPend: [], logShown: 0, rafId: 0, dirty: false,
      lip: {rows: [], n: 0, fps: 0, cols: null, colIdx: null, last_t: null, unsupported: keep.unsupported && keep.unsupported === (C.S && C.S.sid), msgs: 0},
      shownT: null, sid: null, sentOn: null, seenEv: new WeakSet(), seenDE: new WeakSet(), slowAt: 0, headAt: 0, hzT: performance.now(), hzN: 0, hz: 0,
      lastGaze: undefined, lastFeel: undefined, lastFlags: undefined, needAttach: false, rig: null, perc: {}, moreBuilt: false, props: null, io: null, mq: null};
    P.tick = tick; P.shown = shown; P.ev = perceptionEv; P.cmd = commandEv; P.ctl = ctlEv; P.onRender = onRender;
    setupParams(D.motion || null);
    if (typeof IntersectionObserver === 'function') {
      P.io = new IntersectionObserver(es => { for (const e of es) { const g = P && P.groups.find(x => x.el === e.target); if (g) g.vis = e.isIntersecting; } }, {root: ui.main, rootMargin: '120px 0px'});
      for (const g of P.groups) P.io.observe(g.el);
    }
    ui.root.hidden = false;
    D.rdbg = P;
    layout();
    P.mq = window.matchMedia ? window.matchMedia('(min-width: 900px)') : null;
    P.onMq = () => layout();
    P.onResize = () => { if (P && document.documentElement.classList.contains('rd-sheet')) applyHeight(curH()); };
    if (P.mq) { if (P.mq.addEventListener) P.mq.addEventListener('change', P.onMq); else if (P.mq.addListener) P.mq.addListener(P.onMq); }
    window.addEventListener('resize', P.onResize);
    setTab(keep.tab); setCollapsed(keep.collapsed, true); setMore(keep.more, true);
    const btn = document.getElementById('rigbtn'); if (btn) btn.setAttribute('aria-pressed', 'true');
    attach();
    if (!P.M) setupParams(null);            // no session yet -> build the meters NOW from the library's lists, not at Start
    P.warmUntil = performance.now() + OPEN_WARM_MS; P.warmFullMs = WARM_FULL_MS;   // spread the first paints after the build
    addLog('info', 'panel opened' + (P.M.probe ? '' : ' (this build has no motion probe: values only, owners unknown)'));
    // the renderer's lip values: on now if a session runs (else at the first tick of the next one)
    sessionCheck();
    if (P.lip.unsupported) lipStatus();
    paintStatic();
    return true;
  }
  function close() {
    if (!P) return false;
    const D = C.D, S = C.S;
    if (P.m && typeof P.m.setProbe === 'function') call(() => P.m.setProbe(false));
    if (P.sentOn && S && S.live && S.sid === P.sentOn) call(() => C.sendCtl({type: 'rigdbg', on: false}));
    if (P.rafId) caf(P.rafId);
    if (P.io) P.io.disconnect();
    if (P.mq) { if (P.mq.removeEventListener) P.mq.removeEventListener('change', P.onMq); else if (P.mq.removeListener) P.mq.removeListener(P.onMq); }
    window.removeEventListener('resize', P.onResize);
    D.rdbg = null; P = null;
    ui.root.hidden = true;
    for (const c of ['rd-sheet', 'rd-wide', 'rd-col']) document.documentElement.classList.remove(c);
    document.documentElement.style.removeProperty('--rdh'); document.documentElement.style.removeProperty('--rdw');
    const btn = document.getElementById('rigbtn'); if (btn) btn.setAttribute('aria-pressed', 'false');
    return true;
  }
  const toggle = () => (P ? (close(), false) : open());

  // params + ring for this motion object (rebuilt only if its param list differs)
  function setupParams(m) {
    const M = metaOf(m);
    const same = P.M && P.M.rig.join() === M.rig.join() && P.M.all.join() === M.all.join();
    P.M = M;
    if (same) { rerange(); return; }
    const rigSet = new Set(M.rig);
    P.names = M.rig.concat(M.all.filter(p => !rigSet.has(p)));
    P.nRig = M.rig.length; P.NP = P.names.length;
    P.idx = new Map(P.names.map((p, i) => [p, i]));
    P.hist = new Float32Array(NH * P.NP); P.hpos = 0; P.histN = 0;
    P.dials = new Map(); P.byEl = new WeakMap(); P.sparks.clear();
    if (P.io) P.io.disconnect();
    buildMeters();
    if (keep.more) { buildMore(); ui.moreBox.hidden = false; }
    lipCols();
  }
  // the same param list with other ranges (meters built at open from the library's lists): fix the rows in place
  function rerange() {
    for (const d of P.dials.values()) {
      const r = P.M.ranges[d.p]; if (!r || d.periodic || (r[0] === d.lo && r[1] === d.hi)) continue;
      const [lo, hi] = r, signed = lo < 0 && hi > 0;
      Object.assign(d, {lo, hi, signed, span: hi - lo, z: signed ? -lo / (hi - lo) : 0}); d.dec = d.span <= 0.5 ? 3 : 2;
      let zb = d.bar.querySelector('b');
      if (signed) { if (!zb) { zb = h('b'); d.bar.insertBefore(zb, d.bar.firstChild); } zb.style.left = (100 * -lo / (hi - lo)).toFixed(2) + '%'; } else if (zb) zb.remove();
      const nm = d.el.querySelector('.rd-name'); if (nm) nm.title = d.p + ' [' + lo + ', ' + hi + ']';
      if (d.c) { d.c.a = d.c.b = d.c.t = undefined; }
    }
  }
  // the motion object of this session (startMotion builds a new one per session): probe on, its old events seen
  function attach() {
    const m = P.D.motion || null;
    if (m === P.m) return;
    if (P.m && typeof P.m.setProbe === 'function') call(() => P.m.setProbe(false));
    P.m = m;
    if (m) setupParams(m);
    P.probeOn = !!(m && typeof m.probe === 'function' && typeof m.setProbe === 'function');
    if (P.probeOn && !P.paused) call(() => m.setProbe(true));
    P.pr = null; P.lastGaze = undefined; P.lastFeel = undefined; P.lastFlags = undefined;
    // the last 10 s of the motion's own action log (what just happened), then only new ones
    if (m && typeof m.events === 'function') { const evs = call(() => m.events(10), []) || []; for (const e of evs) motionEv(e); }
    const DE = P.D.events || []; for (const e of DE) P.seenDE.add(e);
  }
  function sessionCheck() {
    const S = C.S;
    if (!S || !S.live || !S.sid) return;
    if (P.sid !== S.sid) {
      P.sid = S.sid; P.lip.rows = []; P.lip.last_t = null; P.lip.live = false;
      P.warmUntil = performance.now() + WARM_MS; P.warmFullMs = Infinity;   // header only while this session's clock locks
      P.lip.unsupported = keep.unsupported === S.sid;
      lipStatus();
    }
    if (P.sentOn !== S.sid) { P.sentOn = S.sid; call(() => C.sendCtl({type: 'rigdbg', on: true})); }
  }

  // ------------------------------------------------------------------ layout: bottom sheet (< 900 px) / right drawer
  function wide() { return P && P.mq ? P.mq.matches : (window.innerWidth || 0) >= 900; }
  function curH() { const v = +lsGet(LS.h); return v > 0 ? v : Math.round(0.42 * (window.innerHeight || 800)); }
  function maxH() {
    const vh = window.innerHeight || 800, st = document.getElementById('stage');
    const stH = st ? st.getBoundingClientRect().height + 6 : 0;
    const dock = document.getElementById('dock'), dkH = dock ? Math.max(58, dock.getBoundingClientRect().height) + 20 : 78;
    return Math.round(clamp(vh - stH - dkH, 0.25 * vh, 0.85 * vh));
  }
  function applyHeight(px) {
    const vh = window.innerHeight || 800;
    const v = Math.round(clamp(px, 0.25 * vh, maxH()));
    if (!keep.collapsed) document.documentElement.style.setProperty('--rdh', v + 'px');
    return v;
  }
  function applyWidth(px) {
    const vw = window.innerWidth || 1200;
    const v = Math.round(clamp(px, 320, Math.max(320, Math.min(760, vw - 420))));
    document.documentElement.style.setProperty('--rdw', v + 'px');
    return v;
  }
  function layout() {
    if (!P) return;
    const R = document.documentElement.classList, w = wide();
    R.toggle('rd-wide', w); R.toggle('rd-sheet', !w);
    if (w) { applyWidth(+lsGet(LS.w) || 400); document.documentElement.style.removeProperty('--rdh'); }
    else { document.documentElement.style.removeProperty('--rdw'); if (keep.collapsed) document.documentElement.style.setProperty('--rdh', 'calc(56px + env(safe-area-inset-bottom))'); else applyHeight(curH()); }
    setCollapsed(keep.collapsed, true);
    P.logShown = -1; markDirty();
  }
  function wireDrag() {
    let st = null;
    const down = (e, kind) => {
      if (!P || (e.button !== undefined && e.button !== 0)) return;
      if (e.target.closest('button')) return;
      st = {kind, x: e.clientX, y: e.clientY, h: ui.root.getBoundingClientRect().height, w: ui.root.getBoundingClientRect().width, moved: false, id: e.pointerId, el: e.currentTarget};
      try { e.currentTarget.setPointerCapture(e.pointerId); } catch (x) {}
    };
    const move = e => {
      if (!st || e.pointerId !== st.id) return;
      const dx = e.clientX - st.x, dy = e.clientY - st.y;
      if (!st.moved && Math.hypot(dx, dy) < 6) return;
      st.moved = true; e.preventDefault();
      if (wide()) { if (st.kind === 'grip' || st.kind === 'edge') lsSet(LS.w, String(applyWidth(st.w - dx))); }
      else {
        if (keep.collapsed && dy < -12) { setCollapsed(false); st.h = 56; }
        if (!keep.collapsed) lsSet(LS.h, String(applyHeight(st.h - dy)));
      }
    };
    const up = e => {
      if (!st || e.pointerId !== st.id) return;
      const s = st; st = null;
      try { s.el.releasePointerCapture(e.pointerId); } catch (x) {}
      if (!s.moved && (s.kind === 'head' || s.kind === 'head-wide' || (s.kind === 'grip' && !wide()))) {
        if (e.target.closest && e.target.closest('.rd-tabs,.rd-btns')) return;
        setCollapsed(!keep.collapsed);
      }
    };
    ui.grip.addEventListener('pointerdown', e => down(e, 'grip'));
    ui.head.addEventListener('pointerdown', e => { if (!e.target.closest('.rd-tabs,.rd-btns')) down(e, wide() ? 'head-wide' : 'head'); });
    for (const el of [ui.grip, ui.head]) { el.addEventListener('pointermove', move); el.addEventListener('pointerup', up); el.addEventListener('pointercancel', () => { st = null; }); }
  }
  function setCollapsed(on, quiet) {
    keep.collapsed = !!on;
    if (!ui) return;
    ui.root.classList.toggle('rd-col', keep.collapsed);
    document.documentElement.classList.toggle('rd-col', keep.collapsed);
    ui.collapse.setAttribute('aria-expanded', String(!keep.collapsed));
    const w = P ? wide() : false;
    ui.collapse.innerHTML = keep.collapsed ? (w ? ICON.down : ICON.up) : (w ? ICON.up : ICON.down);
    ui.collapse.setAttribute('aria-label', keep.collapsed ? 'Unfold the panel' : 'Fold the panel to its header');
    ui.collapse.title = ui.collapse.getAttribute('aria-label');
    if (P) {
      P.collapsed = keep.collapsed;
      if (!w) { if (keep.collapsed) document.documentElement.style.setProperty('--rdh', 'calc(56px + env(safe-area-inset-bottom))'); else applyHeight(curH()); }
      if (!quiet) { P.logShown = -1; markDirty(); }
    }
  }
  function setPaused(on) {
    if (!P) return;
    P.paused = !!on;
    ui.pause.setAttribute('aria-pressed', String(P.paused));
    ui.pause.innerHTML = P.paused ? ICON.play : ICON.pause;
    ui.pause.setAttribute('aria-label', P.paused ? 'Resume the panel' : 'Freeze the panel (the avatar keeps going)');
    ui.pause.title = ui.pause.getAttribute('aria-label');
    if (P.probeOn && P.m) call(() => P.m.setProbe(!P.paused));   // no capture while frozen
    addLog('info', P.paused ? 'panel frozen (the avatar keeps going)' : 'panel live again');
    paintHead(performance.now(), true);
    if (!P.paused) markDirty();
    else flushLog(true);
  }
  function setTab(t) {
    keep.tab = t === 'events' ? 'events' : 'rig';
    if (!ui) return;
    ui.root.dataset.tab = keep.tab;
    for (const b of ui.tabs) b.setAttribute('aria-selected', String(b.dataset.tab === keep.tab));
    if (P) { P.logShown = -1; markDirty(); }
  }
  function setMore(on, quiet) {
    keep.more = !!on;
    if (!P || !ui.moreBtn) return;
    if (keep.more) { if (P.moreBuilt) refillExtras(); else buildMore(); }
    ui.moreBox.hidden = !keep.more;
    ui.moreBtn.setAttribute('aria-expanded', String(keep.more));
    if (!quiet) markDirty();
  }
  function toggleFold(gel) {
    const id = gel.dataset.g, f = !gel.classList.contains('fold');
    gel.classList.toggle('fold', f);
    if (f) keep.fold.add(id); else keep.fold.delete(id);
    lsSet(LS.fold, [...keep.fold].join(','));
    markDirty();
  }
  function logVisible() { return P && !P.collapsed && (wide() || keep.tab === 'events'); }

  // ------------------------------------------------------------------ hooks (D.rdbg.*) — cheap; work happens in the frame
  function tick(rig, rt) {
    P.ticks++; P.hzN++;
    if (P.paused || !rig) return;
    if (P.D.motion !== P.m) P.needAttach = true;          // a new session's motion object: attached in the frame
    if (P.sentOn !== (P.S && P.S.sid)) sessionCheck();
    const n = P.moreBuilt && keep.more ? P.NP : P.nRig, names = P.names, H = P.hist;
    const base = (P.hpos = (P.hpos + 1) % NH) * P.NP;
    for (let i = 0; i < n; i++) { const x = rig[names[i]]; H[base + i] = typeof x === 'number' && x === x ? x : 0; }
    P.histN++; P.rt = rt; P.rig = rig;
    markDirty();
  }
  function shown(t) { P.shownT = t; }
  function markDirty() { if (P && !P.rafId) P.rafId = raf(frame); }
  function perceptionEv(name, detail) {
    const d = detail || {};
    const side = s => (s === 'l' ? 'screen-left' : s === 'r' ? 'screen-right' : '');
    let txt;
    switch (name) {
      case 'wave': txt = 'wave detected' + (d.source || d.side ? ' (' + [d.source, side(d.side)].filter(Boolean).join(', ') + ')' : ''); break;
      case 'look_back': txt = 'he looked back' + (Number.isFinite(+d.away_s) ? ' (away ' + num(d.away_s, 1) + ' s)' : ''); break;
      case 'expression_change': txt = 'his expression: ' + (d.from || '?') + ' → ' + (d.to || '?') + (Number.isFinite(+d.intensity) ? ' (' + num(d.intensity) + ')' : ''); break;
      case 'gesture': txt = 'his gesture "' + short(d.name || '?', 40) + '"'; break;
      case 'attention': txt = 'attention: ' + short(d.what || d.verdict || '?', 80) + (d.worth_reply ? ' (worth a reply)' : ''); break;
      case 'turn_bid': txt = 'turn bid: ' + (d.bid || '?') + (d.cue ? ' (' + String(d.cue).replace(/_/g, ' ') + ')' : ''); break;
      case 'face_appear': txt = 'face appeared' + (Number.isFinite(+d.away_s) ? ' (away ' + num(d.away_s, 1) + ' s)' : ''); break;
      case 'face_lost': txt = 'face lost'; break;
      case 'attention_lost': txt = 'he looked away'; break;
      case 'object_up': txt = 'he holds something up' + (side(d.side) ? ' (' + side(d.side) + ')' : ''); break;
      case 'point': txt = 'he points' + (d.at ? ' at ' + d.at : '') + (side(d.side) ? ' (' + side(d.side) + ')' : ''); break;
      case 'thumbs_up': txt = 'thumbs up'; break;
      case 'nod': txt = 'he nodded'; break;
      case 'head_shake': txt = 'he shook his head'; break;
      case 'approach_fast': txt = 'something came at the camera fast'; break;
      default: txt = String(name).replace(/_/g, ' ') + (d.why ? ': ' + short(d.why, 60) : '');
    }
    addLog('perc', txt, name);
  }
  function commandEv(cmd, source) {
    if (!cmd) return;
    addLog('cmd', 'command' + (CMD_SRC[source] || source ? ' (' + (CMD_SRC[source] || source) + ')' : '') + ': ' + cmdText(cmd), 'cmd');
  }
  function ctlEv(o) {
    if (!o || o.type !== 'call') return;
    const r = o.ran, ok = r === true || (r && typeof r === 'object' && r.ok !== false);
    const res = r === undefined ? '' : ok ? ' → ' + (r && (r.did || r.status) ? short(r.did || r.status, 60) : r && r.queued ? 'queued' : r && r.dup ? 'already doing it' : 'ok') : ' → failed' + (r && r.reason ? ': ' + short(r.reason, 60) : '');
    addLog('voice', 'voice tool: ' + o.name + '(' + argText(o.args) + ')' + res, 'call');
  }
  function onRender(j) {
    if (j && j.unsupported) {
      keep.unsupported = (C && C.S && C.S.sid) || true;
      if (P) { P.lip.unsupported = true; lipStatus(); addLog('info', "this renderer build doesn't report its lip values"); }
      return;
    }
    if (!P || !j || !Array.isArray(j.rows)) return;
    const L = P.lip;
    L.msgs++;
    if (Array.isArray(j.cols) && (!L.cols || L.cols.join() !== j.cols.join())) { L.cols = j.cols.slice(); lipCols(); }
    if (j.fps) L.fps = +j.fps;
    for (const r of j.rows) {
      if (!Array.isArray(r) || r.length < 2) continue;
      const last = L.rows[L.rows.length - 1];
      if (last && r[1] < last[1] - 1) L.rows = [];          // a new timeline (session restart)
      L.rows.push(r); L.n++; L.last_t = r[1];
    }
    if (L.rows.length > LIP_ROWS) L.rows.splice(0, L.rows.length - LIP_ROWS);
    if (L.unsupported) { L.unsupported = false; keep.unsupported = null; }
    if (!L.live) { L.live = true; lipStatus(); }
  }
  function lipCols() {
    const L = P.lip;
    L.colIdx = L.cols ? new Map(L.cols.map((c, i) => [c, i + 2])) : null;
    for (const d of P.dials.values()) d.rcol = L.colIdx && L.colIdx.has(d.p) ? L.colIdx.get(d.p) : -1;
  }
  function lipStatus() {
    if (!P) return;
    const L = P.lip, txt = L.unsupported ? "set by the renderer (this renderer build doesn't report them)" : L.live ? 'renderer values for the frame on screen' : 'set by the renderer — waiting';
    for (const g of P.groups) if (g.lipSub) g.lipSub.textContent = txt;
    markDirty();
  }

  // ------------------------------------------------------------------ the frame: probe, events, DOM (changed cells only)
  function frame() {
    if (!P) return;
    P.rafId = 0;
    const now = performance.now();
    if (now - P.hzT >= 1000) { P.hz = Math.round(P.hzN * 1000 / (now - P.hzT)); P.hzN = 0; P.hzT = now; }
    if (P.paused) { paintHead(now); return; }
    if (now < (P.warmUntil || 0)) {        // light mode (a new session's first seconds / just opened)
      if (now - (P.warmFull || 0) < (P.warmFullMs || WARM_FULL_MS)) { paintHead(now); return; }
      P.warmFull = now;
    }
    if (P.needAttach) { P.needAttach = false; attach(); }
    // the probe: once per frame with a new tick (every 4th tick while folded: the header only)
    if (P.probeOn && P.m && P.prTick !== P.ticks && (!P.collapsed || P.ticks - (P.prTick || 0) >= 4)) {
      P.prTick = P.ticks;
      try { P.pr = P.m.probe(); P.prErr = null; } catch (e) { P.pr = null; P.prErr = String(e && e.message || e).slice(0, 80); }
      probeDiff();
    }
    if (now - P.slowAt >= 220) {          // ~4.5 Hz: action events, gaze returns, the strips, group chips, props slots
      P.slowAt = now;
      pollEvents();
      if (!P.probeOn && P.m && typeof P.m.debug === 'function' && now - P.dbgAt > 400) { P.dbgAt = now; P.dbg = call(() => P.m.debug(), null); }
      if (!P.collapsed) { paintDrive(); paintPerc(); paintGroups(); if (P.props && keep.more) propSlots(); if (keep.tab === 'rig' || wide()) paintPlanner(); }
    }
    paintHead(now);
    if (P.collapsed) return;
    if (keep.tab === 'rig' || wide()) {
      const row = lipRow();
      for (const g of P.groups) {
        if (!g.vis || g.el.classList.contains('fold') || (g.more && !keep.more)) continue;
        for (const d of g.dials) paintDial(d, now, row);
        if (g.slots) for (const s of g.slots.values()) if (!s.el.hidden) for (const d of s.dials) paintDial(d, now, row);
      }
      for (const d of P.sparks) paintSpark(d);
    }
    if (logVisible()) flushLog();
  }
  function cur(i) { return P.hist[P.hpos * P.NP + i]; }
  function back(i, k) { return P.hist[((P.hpos + NH - k) % NH) * P.NP + i]; }
  // the spread of the last ~300 ms (an oscillation as fast as the window still counts as moving)
  function spread(i) {
    if (P.histN <= MOVE_BACK) return 0;
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k <= MOVE_BACK; k++) { const v = P.hist[((P.hpos + NH - k) % NH) * P.NP + i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    return hi - lo;
  }
  function ownerOf(p) {
    if (P.M.lips.has(p)) return 'lips';
    if (!P.probeOn) return null;
    const o = P.pr && P.pr.owners ? P.pr.owners[p] : null;
    return o && o.by ? o.by : 'life';
  }
  function lipRow() {
    const L = P.lip, rows = L.rows; if (!rows.length) return null;
    const t = P.shownT;
    if (t === null || t === undefined || !Number.isFinite(t)) return rows[rows.length - 1];
    let best = null, bd = 1e9;
    for (let i = rows.length - 1; i >= 0; i--) { const r = rows[i], dd = Math.abs(r[1] - t); if (dd < bd) { bd = dd; best = r; } if (r[1] < t - 1) break; }
    const tol = 0.75 / (L.fps || 24);
    if (best && bd <= tol) return best;
    // no row for the frame on screen (yet): the newest row at or before it, else the newest
    for (let i = rows.length - 1; i >= 0; i--) if (rows[i][1] <= t) return rows[i];
    return rows[rows.length - 1];
  }
  function rowBack(row, s) {
    const rows = P.lip.rows, t = row[1] - s;
    for (let i = rows.length - 1; i >= 0; i--) if (rows[i][1] <= t) return rows[i];
    return null;
  }
  function paintDial(d, now, row) {
    const c = d.c;
    let v, by, wait = null, rv = null;
    if (d.lip) {
      if (row && d.rcol >= 0) { v = +row[d.rcol]; by = 'lips'; const rb = rowBack(row, 0.3); if (rb && Math.abs(v - rb[d.rcol]) > 0.02 * d.span) d.movUntil = now + 600; }
      else wait = P.lip.unsupported ? "set by the renderer (this renderer build doesn't report them)" : 'set by the renderer — waiting';
    } else {
      v = cur(d.i); by = ownerOf(d.p);
      if (spread(d.i) > 0.02 * d.span) d.movUntil = now + 600;
      if (row && d.rcol >= 0) rv = +row[d.rcol];
    }
    if (wait !== c.wait) {
      c.wait = wait; d.el.classList.toggle('rd-wait', !!wait);
      if (wait) { d.own.textContent = wait; d.own.title = wait; c.own = wait; d.el.style.removeProperty('--c'); d.el.style.removeProperty('--cb'); c.col = ''; }
      else c.own = '';
    }
    if (wait) { if (c.mv) { c.mv = false; d.el.classList.remove('moving'); } return; }
    const mv = d.movUntil > now;
    if (mv !== c.mv) { c.mv = mv; d.el.classList.toggle('moving', mv); }
    // value
    let t;
    if (d.periodic) { const f = v - Math.floor(v); t = num(v, 2); v = f; } else t = Math.abs(v) < 0.5 * Math.pow(10, -d.dec) ? (0).toFixed(d.dec) : v.toFixed(d.dec);
    if (t !== c.t) { c.t = t; d.val.textContent = t; }
    // bar: [a, b] of the track, centre zero for signed ranges
    const p = d.periodic ? clamp(v, 0, 1) : clamp((v - d.lo) / d.span, 0, 1);
    const a = Math.round(Math.min(d.z, p) * 400), b = Math.round(Math.max(d.z, p) * 400);
    if (a !== c.a || b !== c.b) { c.a = a; c.b = b; d.fill.style.transform = 'translateX(' + (a / 4) + '%) scaleX(' + ((b - a) / 400) + ')'; }
    // owner tag
    const word = by === null ? UNKNOWN[0] : ownWord(by);
    if (word !== c.own) { c.own = word; d.own.textContent = word; }
    const col = by === null ? UNKNOWN[1] : ownCol(by);
    if (col !== c.col) { c.col = col; d.el.style.setProperty('--c', col); d.el.style.setProperty('--cb', hexA(col, 0.14)); }
    const o = P.pr && P.pr.owners && P.pr.owners[d.p];
    const rest = !d.lip && P.probeOn && !o && Math.abs(v) < 1e-4;      // zero and nobody touches it
    if (rest !== c.rest) { c.rest = rest; d.el.classList.toggle('rest', rest); }
    const tt = by === 'lips' ? 'lip sync (renderer)' : o ? ownWord(o.by) + ' ' + num(o.w) + (o.also && o.also.length ? ' · also ' + o.also.map(x => ownWord(x[0]) + ' ' + num(x[1])).join(', ') : '') : word;
    if (tt !== c.ownTitle) { c.ownTitle = tt; d.own.title = tt; }
    // the renderer's final value of a non-lip column: a white tick on the bar
    const rp = rv === null || !Number.isFinite(rv) ? -1 : Math.round(clamp((rv - d.lo) / d.span, 0, 1) * 400);
    if (rp !== c.rt) {
      c.rt = rp;
      if (rp < 0) { if (d.rtick) d.rtick.hidden = true; }
      else {
        if (!d.rtick) { d.rtick = h('u'); d.bar.appendChild(d.rtick); }
        d.rtick.hidden = false; d.rtick.style.left = (rp / 4) + '%'; d.rtick.title = 'rendered ' + num(rv, 3);
        d.bar.title = 'page ' + t + ' · rendered ' + num(rv, 3);
      }
    }
  }
  function toggleSpark(d) {
    if (d.spark) { d.spark.remove(); d.spark = null; P.sparks.delete(d); d.el.classList.remove('open'); return; }
    const cv = h('canvas', 'rd-spark'); d.el.after(cv); d.spark = cv; P.sparks.add(d); d.el.classList.add('open');
    paintSpark(d);
  }
  function paintSpark(d) {
    const cv = d.spark; if (!cv || !cv.isConnected) return;
    const dpr = Math.min(3, window.devicePixelRatio || 1), W = Math.max(60, Math.round(cv.clientWidth * dpr)), Hh = Math.round(48 * dpr);
    if (cv.width !== W || cv.height !== Hh) { cv.width = W; cv.height = Hh; }
    const g = cv.getContext('2d'); g.clearRect(0, 0, W, Hh);
    const pad = 4 * dpr, yOf = v => Hh - pad - (clamp(d.periodic ? v - Math.floor(v) : (v - d.lo) / d.span, 0, 1)) * (Hh - 2 * pad);
    g.strokeStyle = '#2a3240'; g.lineWidth = 1; g.beginPath();
    if (d.signed) { const y = Math.round(yOf(0)) + 0.5; g.moveTo(0, y); g.lineTo(W, y); }
    g.moveTo(0, pad); g.lineTo(W, pad); g.moveTo(0, Hh - pad); g.lineTo(W, Hh - pad); g.stroke();
    const col = d.c.col || '#3d7cf0';
    g.strokeStyle = col; g.lineWidth = 1.6 * dpr; g.beginPath();
    let last = null;
    if (d.lip) {
      const rows = P.lip.rows, row = lipRow(); if (!row || d.rcol < 0) return;
      const t1 = row[1], t0 = t1 - 5; let first = true;
      for (const r of rows) { if (r[1] < t0 || r[1] > t1) continue; const x = (r[1] - t0) / 5 * W, y = yOf(+r[d.rcol]); if (first) { g.moveTo(x, y); first = false; } else g.lineTo(x, y); last = [x, y]; }
    } else {
      const n = Math.min(NH, P.histN);
      for (let k = n - 1; k >= 0; k--) { const x = (1 - k / (NH - 1)) * W, y = yOf(back(d.i, k)); if (k === n - 1) g.moveTo(x, y); else g.lineTo(x, y); last = [x, y]; }
    }
    g.stroke();
    if (last) { g.fillStyle = col; g.beginPath(); g.arc(last[0] - dpr, last[1], 2.5 * dpr, 0, 6.3); g.fill(); }
    g.fillStyle = '#6f7b8c'; g.font = (10 * dpr) + 'px ui-monospace,Menlo,monospace'; g.textBaseline = 'top';
    g.fillText(d.periodic ? '1' : String(d.hi), 3 * dpr, pad + dpr); g.textBaseline = 'bottom'; g.fillText(d.periodic ? '0' : String(d.lo), 3 * dpr, Hh - pad - dpr);
    g.textBaseline = 'top'; g.textAlign = 'right'; g.fillText(d.lip ? 'last 5 s (renderer)' : 'last 5 s → now', W - 10 * dpr, pad + dpr); g.textAlign = 'left';
  }
  // group chips: the dominant non-life owner by weight (lips only when nothing else drives the mouth)
  function groupOwner(g) {
    if (!P.probeOn) return g.dials.some(d => d.lip) && P.lip.live ? 'lips' : null;
    const W = {}, own = P.pr && P.pr.owners;
    let lips = false;
    const add = d => {
      if (d.lip) { lips = lips || P.lip.live; return; }
      const o = own && own[d.p]; if (!o) return;
      if (o.by && o.by !== 'life') W[o.by] = (W[o.by] || 0) + (+o.w || 0);
      if (o.also) for (const [l, w] of o.also) if (l && l !== 'life' && l !== 'lips') W[l] = (W[l] || 0) + 0.5 * (+w || 0);
    };
    for (const d of g.dials) add(d);
    if (g.slots) for (const s of g.slots.values()) for (const d of s.dials) add(d);
    let best = null, bw = 0.02;
    for (const k in W) if (W[k] > bw) { bw = W[k]; best = k; }
    return best || (lips && g.id === 'mouth' && avatarTalking() ? 'lips' : 'life');
  }
  function paintGroups() {
    for (const g of P.groups) {
      const by = groupOwner(g), t = by === null ? UNKNOWN[0] : ownWord(by);
      g.by = by;
      if (t !== g.ownTxt) { g.ownTxt = t; g.own.textContent = t; const col = by === null ? UNKNOWN[1] : ownCol(by); g.own.style.color = col; g.own.style.background = hexA(col, 0.15); }
    }
  }
  function propSlots() {
    const G = P.props;
    for (let k = 0; k < 16; k++) {
      const ps = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'hl', 'hr', 'hh', 's'].map(a => 'prop' + k + '_' + a).filter(p => P.idx.has(p));
      if (!ps.length) continue;
      const any = ps.some(p => Math.abs(cur(P.idx.get(p))) > 1e-4);
      let s = G.slots.get(k);
      if (any && !s) {
        const el = h('div'); el.innerHTML = '<div class="rd-pslot">prop ' + k + '</div>';
        const after = [...G.slots.entries()].filter(([j]) => j > k).sort((a, b) => a[0] - b[0])[0];
        G.box.insertBefore(el, after ? after[1].el : (G.empty && G.empty.isConnected ? G.empty : null));
        s = {el, dials: [], zeroAt: 0}; G.slots.set(k, s);
        for (const p of ps) s.dials.push(mkDial(p, el));
      }
      if (s) { if (any) { s.zeroAt = 0; s.el.hidden = false; } else if (!s.zeroAt) s.zeroAt = performance.now(); else if (performance.now() - s.zeroAt > 3000) s.el.hidden = true; }
    }
    const n = [...G.slots.values()].filter(s => !s.el.hidden).length;
    if (!G.empty) { G.empty = h('div', 'rd-empty', 'no prop in use'); G.box.appendChild(G.empty); }
    G.empty.hidden = n > 0;
    G.el.querySelector('.rd-gn').textContent = n ? n + ' in use' : '';
  }

  // ------------------------------------------------------------------ strips + header
  function avatarTalking() {
    const D = P.D;
    if (P.pr && typeof P.pr.speaking === 'boolean') return P.pr.speaking;
    return !!call(() => (D.voice && D.voice.speaking && D.voice.speaking()) || (D.play && D.play.avatarVoice && D.play.avatarVoice().voiced), false);
  }
  function chip(txt, col, title) {
    const st = col ? ' style="color:' + col + ';border-color:' + hexA(col, 0.35) + ';background:' + hexA(col, 0.1) + '"' : '';
    return '<span class="rd-chip"' + st + (title ? ' title="' + esc(title) + '"' : '') + '>' + esc(txt) + '</span>';
  }
  function driveData() {
    const r = P.pr, items = [];
    if (r) {
      if (r.speaking) items.push(['talking', ownCol('speech')]);
      if (r.listening) items.push(['he is talking', ownCol('listen')]);
      if (r.gaze) items.push(['gaze → ' + (r.gaze.target || r.gaze.mode || '?') + (r.gaze.focus && r.gaze.focus !== r.gaze.target ? ' (' + r.gaze.focus + ')' : ''), ownCol('gaze')]);
      for (const g of (r.gestures || []).slice().sort((a, b) => (b.w || 0) - (a.w || 0)).slice(0, 5)) {
        const lay = SRC_LAYER[g.src] || g.src;
        items.push([String(g.name).replace(/^expr:/, 'face: ') + ' · ' + ownWord(lay) + ' ' + num(g.w), ownCol(lay), (g.params || []).join(', ')]);
      }
      if (r.holds && r.holds.length) items.push(['hold ×' + r.holds.length, ownCol('hold'), r.holds.map(x => x.p).join(', ')]);
      for (const k of r.kf || []) items.push(['planner: ' + short(k.description || k.id, 36) + ' (' + (k.phase || '') + ' ' + num(k.w) + ')', ownCol('planner'), k.description]);
      if (r.move) items.push(['body move: ' + r.move.name + ' ' + num(r.move.w), ownCol('move')]);
      if (r.look) items.push(['commanded look: ' + (typeof r.look === 'object' ? JSON.stringify(r.look) : r.look), ownCol('look')]);
      if (r.affect && r.affect.label) items.push(['feeling: ' + r.affect.label + (Number.isFinite(+r.affect.k) ? ' ' + num(r.affect.k) : ''), ownCol('feeling')]);
      if (r.listenerOn) items.push(['listener on', ownCol('listen')]);
      if (r.programOn) items.push(['program on', ownCol('program')]);
      if (r.learned) items.push(['learned (hub) mode', ownCol('learned')]);
    } else if (P.dbg) {
      const g = P.dbg;
      if (g.speaking) items.push(['talking', ownCol('speech')]);
      if (g.listening) items.push(['he is talking', ownCol('listen')]);
      if (g.cmdLook) items.push(['commanded look: ' + g.cmdLook, ownCol('look')]);
      if (g.focus) items.push(['gaze focus: ' + g.focus, ownCol('gaze')]);
      else if (g.around) items.push(['looking around: ' + g.around, ownCol('life')]);
      for (const s of (g.gestures || []).slice(0, 5)) { const [n, src] = String(s).split(':'); const lay = SRC_LAYER[src] || src; items.push([n.replace(/^expr:/, 'face: ') + ' · ' + ownWord(lay), ownCol(lay)]); }
      if (g.holds && g.holds.length) items.push(['hold ×' + g.holds.length, ownCol('hold'), g.holds.join(', ')]);
      for (const k of g.kf || []) if (k) items.push(['planner: ' + short(k.description || k.id, 36) + ' (' + (k.phase || '') + ')', ownCol('planner')]);
      if (g.listener) items.push(['listener on', ownCol('listen')]);
      if (g.program) items.push(['program on', ownCol('program')]);
      if (g.prior) items.push(['learned prior', ownCol('prior')]);
    }
    const D = P.D;
    if (D.learned) items.push([D.unified ? 'unified (hub) mode' : 'learned (hub) mode', ownCol('learned')]);
    if (!items.length) items.push([P.m ? 'life only' : 'not running — press Start', '']);
    return items;
  }
  function paintDrive() {
    const items = driveData();
    const html = items.map(x => chip(x[0], x[1], x[2])).join('') + (P.prErr ? chip('probe error: ' + P.prErr, '#e5534b') : '') + (!P.probeOn && P.m ? chip('no motion probe in this build', '#6b7685') : '');
    if (html !== P.driveHtml) { P.driveHtml = html; ui.driveBox.innerHTML = html; }
  }
  function percData() {
    const D = P.D, S = P.S, f = D.features || null, face = f && f.face, out = {};
    const st = D.perc && D.perc.stats ? call(() => D.perc.stats(), null) : null;
    const hz = m => (st && st.models && st.models[m] ? Math.round(+st.models[m].fps || 0) + ' Hz' : '–');
    const hands = f && Array.isArray(f.hands) ? f.hands.filter(x => (x.age ?? 0) < 0.5) : [];
    out.face = [!!(face && face.present), 'face ' + hz('face')];
    out.hands = [hands.length > 0, 'hands ' + hz('hand') + (hands.length ? ' ×' + hands.length : '')];
    out.pose = [!!(f && f.pose && f.pose.present), 'pose ' + hz('pose')];
    const deg = r => (r >= 0 ? '+' : '−') + Math.abs(Math.round(r * 57.3)) + '°';
    out.pos = [!!(face && face.present), face && face.present && Number.isFinite(+face.ax) ? 'him x ' + deg(+face.ax) + ' y ' + deg(+face.ay || 0) : 'him –'];
    out.attn = [!!(f && f.attending), 'attending ' + (f ? (f.attending ? 'yes' : 'no') + (Number.isFinite(+f.attention) ? ' ' + num(f.attention) : '') : '–')];
    const G = P.m && P.m.state ? call(() => P.m.state.greetD, null) : null;
    out.handup = [!!(G && G.ev > 0.05), 'hand-up ' + (G ? num(G.ev) : '–')];
    const ua = D.uaff && D.uaff.state ? call(() => D.uaff.state(), null) : null, est = ua && ua.est;
    const xs = face && face.xp && face.xp.state;
    out.expr = [!!(est && est.label && est.label !== 'neutral'), 'his face ' + (est ? est.label + ' ' + num(est.conf) : xs ? xs : '–')];
    const uv = D.voice && D.voice.userVoice ? call(() => D.voice.userVoice(), null) : null;
    out.mic = [!!(uv && uv.voiced), 'mic ' + (uv ? num(uv.rms, 3) + (uv.voiced ? ' voice' : '') : '–') + (S && S.micMuted ? ' (muted)' : '')];
    out.speak = [avatarTalking(), 'avatar ' + (avatarTalking() ? 'speaking' : 'quiet')];
    const r = S && S.renderStats;
    out.render = [!!(r && r.fps), 'render ' + (r && r.fps != null ? num(r.fps, 0) + ' fps' : '–')];
    return out;
  }
  function paintPerc() {
    const o = percData(); P.perc = o;
    for (const c of ui.perc.children) {
      const k = c.dataset && c.dataset.k; if (!k || !o[k]) continue;
      const [on, txt] = o[k], key = (on ? 1 : 0) + txt;
      if (c._k === key) continue;
      c._k = key; c.firstChild.className = on ? 'on' : ''; c.lastChild.textContent = txt; c.classList.toggle('dim', !on);
    }
  }
  function headSummary() {
    if (P.paused) return 'frozen';
    if (!P.m || !(P.S && P.S.live)) return 'not running — press Start';
    const W = {};
    if (P.pr && P.pr.owners) for (const p in P.pr.owners) { const o = P.pr.owners[p]; if (o.by && o.by !== 'life' && o.by !== 'lips') W[o.by] = (W[o.by] || 0) + (+o.w || 0); }
    const top = Object.keys(W).sort((a, b) => W[b] - W[a]).slice(0, 2).map(ownWord);
    const gz = P.pr && P.pr.gaze ? 'gaze → ' + (P.pr.gaze.target || P.pr.gaze.mode) : '';
    return [P.hz + ' Hz', top.length ? top.join(', ') : (P.probeOn ? 'life' : 'owners unknown'), gz].filter(Boolean).join(' · ');
  }
  function paintHead(now, force) {
    if (!force && now - P.headAt < 250) return;
    P.headAt = now;
    const live = !!(P.S && P.S.live && P.m);
    const cls = 'rd-dot' + (P.paused ? ' paused' : live ? ' live' : '');
    if (ui.dot.className !== cls) ui.dot.className = cls;
    const s = P.warmFullMs === Infinity && now < (P.warmUntil || 0) ? 'session starting · meters resume in a moment' : headSummary();
    if (s !== P.sumTxt) { P.sumTxt = s; ui.sum.textContent = s; }
    const n = P.log.length, lt = n ? n + (n >= LOG_CAP ? '+' : '') : '';
    if (lt !== P.lognTxt) { P.lognTxt = lt; ui.logn.textContent = n ? '(' + lt + ', newest first)' : ''; ui.tabs[1].textContent = n ? 'Events ' + lt : 'Events'; }
  }
  function paintStatic() { paintHead(performance.now(), true); paintDrive(); paintPerc(); paintGroups(); paintPlanner(); markDirty(); }

  // ------------------------------------------------------------------ event log
  function clock(tp) {
    const t0 = (P.D.startedPerf && P.S && (P.S.live || P.D.startedPerf < tp)) ? P.D.startedPerf : P.openedAt;
    const s = Math.max(0, (tp - t0) / 1000), m = Math.floor(s / 60);
    return String(m).padStart(2, '0') + ':' + (s - 60 * m).toFixed(1).padStart(4, '0');
  }
  function addLog(kind, text, name, tp) {
    if (!P) return null;
    const e = {tp: tp === undefined ? performance.now() : tp, kind, text: short(text, 160), name: name || '', el: null};
    // kept in time order: a motion event polled a little late carries its own (earlier) time
    let i = P.log.length; while (i > 0 && P.log[i - 1].tp > e.tp) i--;
    P.log.splice(i, 0, e); P.logPend.push(e);
    if (P.log.length > LOG_CAP) { const gone = P.log.splice(0, P.log.length - LOG_CAP); for (const g of gone) if (g.el) g.el.remove(); }
    markDirty();
    return e;
  }
  function flushLog(force) {
    if (!P || !ui) return;
    if (P.paused && !force) return;
    if (P.logShown === -1) {             // the log became visible: rebuild it from the kept entries
      ui.log.textContent = ''; P.logPend = [];
      for (const e of P.log) e.el = null;
      const fr = document.createDocumentFragment();
      for (let i = P.log.length - 1; i >= 0; i--) fr.appendChild(logEl(P.log[i]));
      ui.log.appendChild(fr); P.logShown = 1;
      return;
    }
    if (!P.logPend.length) return;
    // newest first: each new row goes right after the closest newer row already shown (or at the top)
    const pend = P.logPend.filter(e => !e.el).sort((x, y) => x.tp - y.tp); P.logPend = [];
    for (const e of pend) {
      const i = P.log.indexOf(e); if (i < 0) continue;
      let j = i + 1; while (j < P.log.length && !P.log[j].el) j++;
      const li = logEl(e);
      ui.log.insertBefore(li, j < P.log.length ? P.log[j].el.nextSibling : ui.log.firstChild);
    }
  }
  function logEl(e) {
    const li = h('li', 'rd-ev'); li.dataset.kind = e.kind;
    li.innerHTML = '<time>' + clock(e.tp) + '</time><span>' + esc(e.text) + '</span>';
    e.el = li; return li;
  }
  function setLogText(e, text) { e.text = short(text, 160); if (e.el) e.el.lastChild.textContent = e.text; }
  // the motion's own action log (D.motion.events): gestures, reflexes, moves, objects, planner jobs, commands' outcomes
  function pollEvents() {
    const m = P.m;
    if (m && typeof m.events === 'function') { const evs = call(() => m.events(1.5), []) || []; for (const e of evs) motionEv(e); }
    const DE = P.D.events;                 // gaze returns go straight into D.events (gazereturn.js), not through avatarEvent
    if (DE && DE.length) for (let i = DE.length - 1; i >= 0; i--) {
      const e = DE[i]; if (P.seenDE.has(e)) break; P.seenDE.add(e);
      if (e && e.name === 'gaze_return') { const d = e.detail || {}; addLog('perc', 'he looked back after ' + num(d.away_s, 1) + ' s → ' + (d.kind || 'cue') + (d.phrase ? ' "' + short(d.phrase, 40) + '"' : d.cue ? ' (' + d.cue + ')' : ''), 'gaze_return', e.t); }
    }
  }
  function motionEv(e) {
    if (!e || P.seenEv.has(e)) return;
    P.seenEv.add(e);
    const tp = Number.isFinite(e.wall) ? performance.now() - (Date.now() - e.wall) : undefined;
    const lay = SRC_LAYER[e.src] || e.src || '';
    const nm = String(e.name || '').replace(/_/g, ' ');
    const why = e.reason ? ' — ' + short(e.reason, 60) : '';
    switch (e.kind) {
      case 'reflex': {
        const from = REFLEX_OF[e.name], tw = Date.now();
        const hit = from && P.log.slice(-12).reverse().find(x => x.name === from && !x.linked && (performance.now() - x.tp) < 2500 + (tw - (e.wall || tw)));
        const rn = nm.replace(/ /g, '-');
        if (hit) { hit.linked = true; setLogText(hit, hit.text + ' → ' + rn + ' (reflex)'); }
        else addLog('reflex', rn + ' (reflex)' + (e.did ? ': ' + e.did : ''), 'reflex', tp);
        return;
      }
      case 'kf': {
        const desc = (e.args && e.args.description) || nm;
        const ph = {ready: 'planned', start: 'started', end: 'finished', fail: 'failed', interrupted: 'cut short'}[e.phase] || e.phase;
        addLog('planner', 'planner move ' + ph + ': ' + short(desc, 80) + (e.phase === 'end' && e.planned_s ? ' (' + num(e.planned_s, 1) + ' s)' : '') + why, 'kf', tp);
        return;
      }
      case 'gesture': case 'expression': {
        if (e.phase === 'start') addLog('motion', (e.kind === 'expression' ? 'face ' : 'gesture ') + nm + ' (' + ownWord(lay) + ')', 'gesture', tp);
        else if (e.phase === 'interrupted') addLog('motion', (e.kind === 'expression' ? 'face ' : 'gesture ') + nm + ' cut short', 'gesture', tp);
        else if (e.phase === 'fail') addLog('motion', e.kind + ' ' + nm + ' failed' + why, 'gesture', tp);
        return;
      }
      case 'move': case 'object': {
        const ph = {start: 'started', end: 'finished', fail: 'failed', interrupted: 'cut short', queued: 'queued'}[e.phase] || e.phase || '';
        addLog('motion', (e.kind === 'move' ? 'body move ' : 'object ') + ph + ': ' + (e.did ? short(e.did, 90) : nm) + why, e.kind, tp);
        return;
      }
      case 'config': addLog('info', e.did || 'config', 'config', tp); return;
      default:
        if (e.phase === 'fail') addLog('motion', e.kind + ' ' + nm + ' failed' + why, e.kind, tp);
        else if (e.did) addLog('motion', e.kind + ': ' + short(e.did, 90), e.kind, tp);
    }
  }
  // gaze target / feeling / mode flags: diffs of probe()
  function probeDiff() {
    const r = P.pr; if (!r) return;
    const gz = r.gaze ? String(r.gaze.target || r.gaze.mode || '') : '';
    if (P.lastGaze !== undefined && gz !== P.lastGaze) {
      const last = P.log[P.log.length - 1];
      if (last && last.name === 'gaze' && performance.now() - last.tp < 700) setLogText(last, 'gaze → ' + gz);   // a flicker: one line
      else addLog('gaze', 'gaze → ' + gz, 'gaze');
    }
    P.lastGaze = gz;
    const fl = r.affect && r.affect.label ? r.affect.label : '';
    if (P.lastFeel !== undefined && fl !== P.lastFeel) addLog('feeling', fl ? 'feeling → ' + fl + (Number.isFinite(+r.affect.k) ? ' (' + num(r.affect.k) + ')' : '') : 'feeling eased out', 'feeling');
    P.lastFeel = fl;
    const flags = [r.listenerOn ? 'listener' : '', r.programOn ? 'program' : '', r.learned ? 'learned' : ''].filter(Boolean).join(',');
    if (P.lastFlags !== undefined && flags !== P.lastFlags) addLog('info', 'layers on: ' + (flags || 'rules only'), 'flags');
    P.lastFlags = flags;
  }

  // ------------------------------------------------------------------ text helpers
  function argText(a) {
    if (!a || typeof a !== 'object') return '';
    return short(Object.entries(a).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => k + ': ' + (typeof v === 'object' ? JSON.stringify(v) : v)).join(', '), 70);
  }
  function cmdText(c) {
    const a = c.action, side = c.side && c.side !== 'right' ? ' (' + c.side + ')' : '';
    const obj = o => (o ? (typeof o === 'string' ? o : [o.color, o.name || o.ref || o.category].filter(Boolean).join(' ')) : '');
    switch (a) {
      case 'gesture': return 'gesture ' + String(c.name || '?').replace(/_/g, ' ') + side + (c.repeat > 1 ? ' ×' + c.repeat : '') + (c.hold ? ', hold' : '');
      case 'look': return 'look ' + (typeof c.target === 'object' ? JSON.stringify(c.target) : c.target) + (c.hold ? ', hold' : '');
      case 'expression': return 'expression ' + c.name + (Number.isFinite(+c.intensity) ? ' ' + num(c.intensity) : '') + (c.hold ? ', hold' : '');
      case 'pose': return 'pose ' + short(Object.keys(c.targets || {}).join(', '), 60) + (c.hold ? ', hold' : '');
      case 'release': return 'release ' + (Array.isArray(c.parts) ? c.parts.join(', ') : c.parts || 'all');
      case 'move': return 'move ' + String(c.name || '?').replace(/_/g, ' ') + (c.dir ? ' ' + c.dir : '') + (c.count > 1 ? ' ×' + c.count : '');
      case 'object': return String(c.verb || '?').replace(/_/g, ' ') + ' ' + obj(c.object) + (c.target ? ' → ' + (typeof c.target === 'object' ? c.target.name : c.target) : '') + (c.place ? ' → ' + c.place : '');
      case 'hold': return 'hold that pose';
      default: return short(a + ' ' + JSON.stringify(Object.assign({}, c, {action: undefined})), 70);
    }
  }

  // ------------------------------------------------------------------ the Planner row + section
  // ctx.planner = {get() -> 'current'|'semantic'|'off', set(x), last() -> perform.js semLast(), warm() -> the hub's semwarm
  // answer}. Built with the meters (first open); painted in the slow frame only when its data changed. While the panel is
  // closed nothing here runs (the page / perform.js keep only the latest record; snapshot() reads it on demand).
  const PL_WORD = {current: 'current', semantic: 'semantic API'};
  function buildPlanner(main) {
    const el = h('div', 'rd-plan' + (keep.planFold ? ' fold' : ''));
    el.innerHTML = '<div class="rd-phead"><span class="rd-gname" title="Tap to fold / unfold">Planner</span>' +
      '<span class="rd-seg" role="group" aria-label="Which planner writes commanded moves (the next command)">' +
      '<button type="button" data-v="current" aria-pressed="false">current</button><button type="button" data-v="semantic" aria-pressed="false">semantic API</button></span>' +
      '<span id="rd-psum"></span></div><div class="rd-pbody"><div class="rd-pcmd"></div><dl class="rd-pkv"></dl><pre class="rd-prog" hidden></pre><div class="rd-pnote"></div></div>';
    main.appendChild(el);
    ui.plan = el; ui.psum = el.querySelector('#rd-psum'); ui.pseg = [...el.querySelectorAll('.rd-seg button')];
    ui.pcmd = el.querySelector('.rd-pcmd'); ui.pkv = el.querySelector('.rd-pkv'); ui.pprog = el.querySelector('.rd-prog'); ui.pnote = el.querySelector('.rd-pnote');
    P.planSig = null;
  }
  function setPlannerMode(v) {
    const pl = C && C.planner; if (!pl || typeof pl.set !== 'function') return;
    const was = call(() => pl.get(), null), now = call(() => pl.set(v), null);
    if (P && now && now !== was) addLog('planner', 'planner → ' + (PL_WORD[now] || now) + ' (from the next command)', 'planner');
    if (P) { P.planSig = null; paintPlanner(); }
  }
  const dms = v => (v !== null && v !== undefined && Number.isFinite(+v) ? Math.round(+v) + ' ms' : '–');
  function plannerData() {
    const pl = C && C.planner; if (!pl) return null;
    const mode = call(() => pl.get(), 'current'), L = call(() => pl.last(), null), W = call(() => (pl.warm ? pl.warm() : null), null);
    if (!L) return {mode, source: null, program: '', ms: {}, warnings: [], errors: [], targets: {}, last: null, warm: W};
    const pm = L.page_ms || {}, hm = L.hub_ms || {}, hub = L.hub || {}, c0 = pm.call;
    const rel = k => (Number.isFinite(pm[k]) && Number.isFinite(c0) ? pm[k] - c0 : null);
    const ms = {heard_to_plan: Number.isFinite(pm.heard) && Number.isFinite(pm.plan_sent) ? pm.plan_sent - pm.heard : null,
      call_to_plan: Number.isFinite(pm.plan_sent) && Number.isFinite(c0) ? pm.plan_sent - c0 : null,
      hub_first_token: hm.first_token ?? null, hub_first_stmt: hm.first_stmt ?? null, hub_first_pose: hm.first_pose ?? hub.ms_first_pose ?? null,
      hub_done: hm.done ?? hub.ms_done ?? null, page_first_pose: rel('first_pose'), ready: rel('ready'), first_motion: rel('first_motion'), end: rel('end'),
      motion_s: L.motion_s ?? null};
    return {mode, cmdMode: L.mode, id: L.id, description: L.description, phase: L.phase, outcome: L.outcome, source: L.source, model: L.model, cached: !!L.cached,
      program: L.program || '', ms, warnings: L.warnings || [], errors: (L.errors || []).concat(L.error ? [L.error] : []), targets: L.targets || {},
      summary: L.summary || null, sem_unavailable: !!L.sem_unavailable, warm: W, last: L};
  }
  function paintPlanner() {
    if (!P || !ui.plan) return;
    const d = plannerData(); if (!d) { ui.plan.hidden = true; return; }
    const sig = JSON.stringify([d.mode, d.id, d.phase, d.outcome, d.source, d.model, d.program.length, d.ms, d.warnings.length, d.errors.length, d.sem_unavailable, d.warm && d.warm.available]);
    if (sig === P.planSig) return;
    P.planSig = sig;
    for (const b of ui.pseg) { b.setAttribute('aria-pressed', String(b.dataset.v === d.mode)); b.disabled = d.mode === 'off'; }
    const T = d.ms.first_motion !== null ? 'motion +' + Math.round(d.ms.first_motion) + ' ms' : d.ms.page_first_pose !== null ? 'pose +' + Math.round(d.ms.page_first_pose) + ' ms' : '';
    ui.psum.textContent = d.mode === 'off' ? 'planner path off (?planner=0)' : d.id ? [d.cmdMode === 'semantic' ? 'semantic' : 'current', d.source, d.model, T].filter(Boolean).join(' · ') : 'no commanded action yet';
    ui.pcmd.textContent = d.id ? '"' + short(d.description, 200) + '" — ' + (d.outcome || d.phase || '') : '';
    const rows = [];
    const kv = (k, v, cls) => rows.push('<dt>' + esc(k) + '</dt><dd' + (cls ? ' class="' + cls + '"' : '') + '>' + esc(v) + '</dd>');
    if (d.id) {
      kv('planner', (PL_WORD[d.cmdMode] || d.cmdMode) + (d.cmdMode !== d.mode ? ' (next: ' + (PL_WORD[d.mode] || d.mode) + ')' : ''));
      kv('source', [d.source || '–', d.model, d.cached ? 'cached' : ''].filter(Boolean).join(' · '));
      if (d.ms.heard_to_plan !== null) kv('heard → plan sent', dms(d.ms.heard_to_plan));
      kv('call → plan sent', dms(d.ms.call_to_plan));
      if (d.cmdMode === 'semantic') { kv('hub first token', dms(d.ms.hub_first_token)); kv('hub first statement', dms(d.ms.hub_first_stmt)); }
      kv('hub first pose', dms(d.ms.hub_first_pose));
      kv('first pose on the page', dms(d.ms.page_first_pose));
      kv('ready', dms(d.ms.ready));
      kv('first motion on the rig', dms(d.ms.first_motion));
      kv('hub done', dms(d.ms.hub_done));
      kv('motion end', dms(d.ms.end));
      kv('motion length', d.ms.motion_s !== null ? num(d.ms.motion_s, 1) + ' s' : '–');
      if (d.summary && d.cmdMode !== 'semantic') kv('summary', short(d.summary, 200));
      if (d.warnings.length) kv('warnings', d.warnings.slice(0, 8).join(' · '), 'warn');
      if (d.errors.length) kv('errors', d.errors.slice(0, 8).join(' · '), 'err');
      const tg = Object.entries(d.targets || {});
      if (tg.length) kv('targets', tg.slice(0, 12).map(([k, v]) => k + (Array.isArray(v) ? ' (' + v.map(x => Math.round(+x)).join(', ') + ')' : '')).join('  '));
    }
    ui.pkv.innerHTML = rows.join('');
    const prog = d.cmdMode === 'semantic' ? d.program : '';
    ui.pprog.hidden = !prog; if (ui.pprog.textContent !== prog) ui.pprog.textContent = prog;
    const notes = [];
    if (d.sem_unavailable || (d.warm && d.warm.available === false)) notes.push('semantic API not available for this character' + (d.warm && d.warm.error ? ' (' + short(d.warm.error, 80) + ')' : ''));
    if (d.cmdMode === 'semantic' && d.id && !prog && d.source !== 'library' && !d.sem_unavailable) notes.push('waiting for the program…');
    if (d.source === 'library') notes.push('a known move (library): no program');
    ui.pnote.textContent = notes.join(' · ');
  }
  function plannerSnap() {
    const d = plannerData(); if (!d) return null;
    return {mode: d.mode, source: d.source, model: d.model || null, program: d.program, ms: d.ms, warnings: d.warnings, errors: d.errors, targets: d.targets,
      id: d.id || null, description: d.description || null, command_mode: d.cmdMode || null, sem_unavailable: d.sem_unavailable || !!(d.warm && d.warm.available === false)};
  }

  // ------------------------------------------------------------------ test hook
  function snapshot() {
    if (!P) return {open: false, paused: false, collapsed: keep.collapsed, ticks: 0, lip: {rows: 0, last_t: null, unsupported: !!keep.unsupported}, owners: {}, dials: {}, log: [], perc: {}, planner: plannerSnap()};
    const owners = {}, dials = {}, now = performance.now(), row = lipRow();
    for (const g of P.groups) owners[g.id] = g.by === null ? UNKNOWN[0] : ownWord(g.by || groupOwner(g));
    for (const [p, d] of P.dials) {
      let v = null;
      if (d.lip) v = row && d.rcol >= 0 ? +row[d.rcol] : null; else v = Math.round(cur(d.i) * 1e4) / 1e4;
      const by = d.lip ? (row && d.rcol >= 0 ? ownWord('lips') : P.lip.unsupported ? "set by the renderer (this renderer build doesn't report them)" : 'set by the renderer — waiting') : (ownerOf(p) === null ? UNKNOWN[0] : ownWord(ownerOf(p)));
      // "moving" from the history too: a group scrolled out of view is not painted, its flag is still true
      const mv = d.movUntil > now || (!d.lip && spread(d.i) > 0.02 * d.span);
      dials[p] = {v, by, moving: mv};
    }
    return {open: true, paused: P.paused, collapsed: P.collapsed, ticks: P.ticks, probe: P.probeOn, layout: wide() ? 'drawer' : 'sheet',
      lip: {rows: P.lip.n, last_t: P.lip.last_t, unsupported: !!P.lip.unsupported, shown_t: P.shownT},
      owners, dials, log: P.log.slice(-LOG_CAP).reverse().map(e => ({t: clock(e.tp), text: e.text, kind: e.kind})),
      perc: Object.fromEntries(Object.entries(P.perc || {}).map(([k, v]) => [k, v[1]])), planner: plannerSnap()};
  }

  function mount(ctx) {
    C = ctx;
    const btn = document.getElementById('rigbtn');
    if (btn && !btn._rd) { btn._rd = true; btn.addEventListener('click', () => toggle()); }
    if (C.Q && C.Q.get && C.Q.get('rig') === '1') {
      if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => open(), {once: true}); else open();
    }
  }

  window.AvatarRigDebug = {mount, open, close, toggle, isOpen: () => !!P, onRender, snapshot};
})();

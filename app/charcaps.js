/* charcaps.js — rig: what a character HAS (body, scene, teacher tools, feelings ...), for the page's gates.
   Mirrors app/charcaps.py. The hub sends config.caps (app.py config_msg) for a character that has any; before the
   config arrives (only the picker's key is known) the static table below answers. config.body / config.props also
   imply 'body' / 'scene'. Muse keeps every answer it had; the other 11 characters have none of these caps.
   API: CharCaps.has(cfg, key, cap) — true when the config (cfg.caps / cfg.body / cfg.props) OR the static table for
   `key` says so; pass null for either side to ask only the other. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.CharCaps = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const ALL = ['body', 'scene', 'act', 'memory', 'speech', 'hands', 'affect', 'tv', 'surfaces', 'marks'];
  const STATIC = {muse: ALL, juno: ALL, junor: ALL,
    gekkie: ALL, hooty: ALL, doughlad: ALL, bonbon: ALL, pneu: ALL, goober: ALL, jade: ALL, ruby: ALL, flurry: ALL};   // the mascots (app/mascots.py)
  function fromCfg(cfg) {
    const out = new Set();
    if (!cfg || typeof cfg !== 'object') return out;
    if (Array.isArray(cfg.caps)) for (const c of cfg.caps) out.add(String(c));
    if (cfg.body) out.add('body');
    if (cfg.props) out.add('scene');
    return out;
  }
  function fromKey(key) { return new Set(STATIC[key] || []); }
  function has(cfg, key, cap) { return fromCfg(cfg).has(cap) || (key ? fromKey(key).has(cap) : false); }
  return {ALL, STATIC, has, fromCfg, fromKey};
});

// Load a browser IIFE/UMD module from app/ in node (the daemon package.json says "type":"module").
const fs = require('fs'), path = require('path');
module.exports = function load(rel) {
  const src = fs.readFileSync(path.join(__dirname, '..', rel), 'utf8');
  const mod = {exports: {}};
  new Function('module', 'exports', src)(mod, mod.exports);
  return mod.exports;
};

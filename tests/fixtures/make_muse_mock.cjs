// node tests/fixtures/make_muse_mock.cjs — writes muse_body_mock.json + muse_props_mock.json (contract §0/§2 shapes) for the
// motion tests until renderer/animals/muse.{body,props}.json exist. The reach grid is a smooth nonlinear stand-in arm.
const fs = require('fs'), path = require('path');
const body = {character: 'muse', H: 1.0, com: [0, 0.52, 0], base_y: 0, extents: {min: [-0.3, 0, -0.24], max: [0.3, 1.0, 0.24]},
  stand: true, legs: {hip_l: [-0.09, 0.36, 0], hip_r: [0.09, 0.36, 0], thigh: 0.19, shin: 0.16, foot_l: [-0.09, 0, 0.03], foot_r: [0.09, 0, 0.03]},
  max_step: 0.22, seat: {below_hip: 0.06}, head: [0, 0.8, 0.1], notes: 'motion mock'};
const sg = s => s === 'l' ? -1 : 1;
function grasp(s, h) {   // a nonlinear stand-in arm: shoulder at (±0.2, 0.6, 0), rest paw by the hip
  const rest = [sg(s) * 0.22, 0.42, 0.14], L = 0.34;
  return [rest[0] + sg(s) * L * h[0] * (1 - 0.06 * h[1] * h[1]), rest[1] + L * h[1] * (1 - 0.08 * h[0] * h[0]) - 0.02 * h[2] * h[2],
    rest[2] + L * 0.9 * h[2] * (1 - 0.1 * Math.max(0, h[1])) + 0.03 * L * h[1]];
}
const ax = []; for (let i = 0; i < 9; i++) ax.push(+(-1.2 + 2.6 * i / 8).toFixed(4));
const reach = {};
for (const s of ['l', 'r']) { const pts = []; for (const x of ax) for (const y of ax) for (const z of ax) pts.push(...grasp(s, [x, y, z]).map(v => +v.toFixed(5))); reach[s] = {axes: {x: ax, y: ax, z: ax}, pts}; }
const props = {character: 'muse',
  places: [
    {id: 'floor', name: 'the floor', aliases: ['floor', 'ground'], center: [0, 0, 0.3], size: [4, 3], top_y: 0, spots: [[-0.3, 0, 0.45], [0.3, 0, 0.45], [0, 0, 0.6]]},
    {id: 'desk', name: 'the desk', aliases: ['desk', 'table', 'desk top'], prop: 'desk', center: [1.15, 0.5, -0.35], size: [0.7, 0.4], top_y: 0.42, spots: [[1.0, 0.42, -0.42], [1.2, 0.42, -0.3], [1.3, 0.42, -0.45], [1.12, 0.42, -0.2]]},
    {id: 'chair', name: 'the chair', aliases: ['chair', 'seat'], prop: 'chair', center: [-0.75, 0.2, 0.05], size: [0.26, 0.26], top_y: 0.2, spots: [[-0.75, 0.2, 0.06]]},
    {id: 'side_table', name: 'the side table', aliases: ['side table', 'little table', 'stool', 'shelf'], center: [-0.62, 0.4, -0.45], size: [0.3, 0.3], top_y: 0.4, spots: [[-0.62, 0.4, -0.42], [-0.55, 0.4, -0.5]]}],
  props: [
    {slot: 0, id: 'headphones', name: 'headphones', color: 'pink', colors: ['pink'], aliases: ['headphones', 'headset', 'cans'], category: 'wearable', size: 0.07, mass: 'light', grip: 'power', worn: true,
      home: {place: 'side_table', pos: [-0.62, 0.45, -0.42], rot: [0, 0, 0]}, throwable: false, facing: [0, 0, 1]},
    {slot: 1, id: 'laptop', name: 'laptop', color: 'silver', colors: ['silver', 'grey'], aliases: ['laptop', 'computer', 'notebook'], category: 'device', size: 0.12, mass: 'medium', grip: 'two_hand',
      home: {place: 'desk', pos: [1.2, 0.44, -0.3], rot: [0, -0.32, 0]}, throwable: false, facing: [0, 0, 1],
      lid: {hinge: [0, 0.02, -0.1], length: 0.2, open_deg: 110}, keyboard: [0, 0.025, 0.02], screen: [0, 0.12, -0.12]},
    {slot: 2, id: 'mug', name: 'blue mug', color: 'blue', colors: ['blue'], aliases: ['mug', 'cup'], category: 'cup', size: 0.05, mass: 'light', grip: 'power', home: {place: 'desk', pos: [1.3, 0.47, -0.45], rot: [0, 0, 0]}, throwable: false},
    {slot: 3, id: 'ball', name: 'red ball', color: 'red', colors: ['red'], aliases: ['ball', 'toy'], category: 'toy', size: 0.05, mass: 'light', grip: 'power', home: {place: 'floor', pos: [0.32, 0.05, 0.42], rot: [0, 0, 0]}, throwable: true},
    {slot: 4, id: 'book', name: 'green book', color: 'green', colors: ['green'], aliases: ['book'], category: 'book', size: 0.06, mass: 'light', grip: 'pinch', home: {place: 'side_table', pos: [-0.55, 0.42, -0.5], rot: [0, 0.1, 0]}, throwable: false},
    {slot: 5, id: 'chair', name: 'chair', aliases: ['chair', 'seat'], category: 'furniture', furniture: true, free: true, size: 0.16,
      home: {place: 'floor', pos: [-0.75, 0.0, 0.05], rot: [0, 0.25, 0]}, seat: {center: [0, 0.2, 0.0], top_y: 0.2, front: [0, 0, 1]}, handles: [[-0.1, 0.45, -0.12], [0.1, 0.45, -0.12]]},
    {slot: 6, id: 'desk', name: 'desk', aliases: ['desk', 'table'], category: 'furniture', furniture: true, free: true, size: 0.35,
      home: {place: 'floor', pos: [1.15, 0.0, -0.35], rot: [0, -0.32, 0]}, front: [0, 0, 1], depth: 0.4, width: 0.7, handles: [[-0.3, 0.4, 0.2], [0.3, 0.4, 0.2]], chair_spot: [0, 0, 0.4]}],
  reach};
fs.writeFileSync(path.join(__dirname, 'muse_body_mock.json'), JSON.stringify(body, null, 1));
fs.writeFileSync(path.join(__dirname, 'muse_props_mock.json'), JSON.stringify(props));
console.log('wrote muse mocks');

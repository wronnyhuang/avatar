// ---------------------------------------------------------------------------
//  engine/props.glsl — props (the muse scene, docs/INTENT-RULES.md sections 0-2): the objects of a character's
//  scene as part of the CHARACTER's own SDF (same march, sun, soft shadow, AO, TAA and DoF as the character; no
//  overlay), the attach logic of the prop{k}_* columns, and the painter.  Included after the animal (scene_main.glsl).
//
//  Generated tables (engine/props_defs.py, from renderer/animals/<c>.props.json), k = slot 0..PROPS_N-1:
//    PR_KIND[k] (PK_*), PR_HOME[k] (home centre, scene units), PR_ROT[k] (home rotation, radians: X then Y then Z,
//    world axes), PR_SIZE[k] (scale: the model's unit frame -> scene units), PR_BND[k] (bounding radius, unit frame),
//    PR_COL/PR_COL2[k] (albedo), PR_GA[k] (grip anchor, unit frame), PR_EXT[k] (half extents, unit frame), PR_FREE[k]
//    (1 = furniture: never attached), PROPS_H (scene units per body height), PROPS_HP_W / PROPS_HP_ARC (headphone fit).
//  Column semantics (each slot): x y z (H) rx ry rz (turns) hl hr (paw attach) hh (head attach, worn) s (articulation).
//   free:  centre = home + H*(x,y,z) (world axes); rotation = R(rx,ry,rz)*home rotation, about its own centre.  A
//          piece of furniture turned by ry about its vertical centre line stays on the floor.
//   held:  the paw's grasp point (charGrasp, char-local -> world through the root) is where the object's grip anchor
//          meets the palm: the anchor sits out along the palm normal by the object's extent that way (no gap, no
//          overlap), offset by H*(x,y,z); rotation = an upright, camera-facing frame (the body's up through the root)
//          times R(rx,ry,rz)*home rotation — so a picked-up object stays the way it stood and the turns are in words
//          the page uses: ry = turn it round (about the paw's up), rx = tip toward / away from the camera, rz = roll.
//          hl and hr both > 0 = a two-paw hold or a hand-over: the weighted blend of the two.
//   worn:  hh > 0 = seated on the head frame (charHeadFrame: headphones over the crown, cups on the sides), + H*(x,y,z)
//          and R(rx,ry,rz) in the head frame.  Mixed weights blend free -> held -> worn.
// ---------------------------------------------------------------------------
#ifndef PROPS_N
#define PROPS_N 0
#endif

// ------------------------------------------------------------------ the root transform (body agent) — identity fallback
#ifndef ROOT_XFORM
mat3 gRootR = mat3(1.0);
vec3 rootFwd(vec3 pl){ return pl; }
vec3 rootInv(vec3 pw){ return pw; }
#endif

#if PROPS_N > 0 && defined(SA_SKIN)
// test scaffolding (props JSON "renderer": {"grasp_fn": "propsShapeGrasp", "head_fn": "propsShapeHead"}): the shapes'
// mitten hands (engine/shape_arms.glsl) and head, so the props layer can be exercised on a shipped character with a
// test props file (AVATAR_PROPS_JSON).  No shipped character has a props file: this is never compiled for them.
void propsShapeGrasp(int side, out vec3 pos, out mat3 frame){
  frame = gSAH[side];
  pos = gSAW[side] + frame*vec3(0.0, 0.50*SA_HAND, 0.30*SA_HAND);
}
void propsShapeHead(out vec3 pos, out mat3 frame){ pos = saToWorld(vec3(0.0)); frame = transpose(gHR); }
#endif

// the character's grasp and head frames (char-local).  The props file names them (PROPS_GRASP_FN / PROPS_HEAD_FN,
// e.g. museGrasp / museHeadFrame); a character may instead define CHAR_GRASP + charGrasp().  Fallbacks below.
#ifdef CHAR_GRASP
#ifndef PROPS_GRASP_FN
#define PROPS_GRASP_FN charGrasp
#endif
#endif
#ifndef PROPS_GRASP_FN
void charGraspDefault(int side, out vec3 pos, out mat3 frame){
  pos = vec3(side == 0 ? -0.5 : 0.5, -1.0, 0.4);
  frame = mat3(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(side == 0 ? 1.0 : -1.0, 0.0, 0.0));
}
#define PROPS_GRASP_FN charGraspDefault
#endif
#ifndef PROPS_HEAD_FN
void charHeadFrameDefault(out vec3 pos, out mat3 frame){ pos = vec3(0.0); frame = mat3(1.0); }
#define PROPS_HEAD_FN charHeadFrameDefault
#endif

#if PROPS_N > 0
#define PK_BALL 1
#define PK_MUG 2
#define PK_BOOK 3
#define PK_DUCK 4
#define PK_APPLE 10
#define PK_PLANT 21
#define PK_HEADPHONES 22
#define PK_CHAIR 23
#define PK_DESK 24
#define PK_LAPTOP 25
#define PK_CHALK 26          // scene: a stick of chalk (on the board's tray)
#define PK_ERASER 27         // scene: a felt board eraser
#ifndef PROPS_HP_W
#define PROPS_HP_W 0.80
#endif
#ifndef PROPS_HP_ARC
#define PROPS_HP_ARC 1.10
#endif
#ifndef PROPS_HP_WR
#define PROPS_HP_WR PROPS_HP_W     // the relaxed (unworn) cup spacing
#endif
#ifndef PROPS_HP_BZ
#define PROPS_HP_BZ 0.0            // the band's offset forward (head frame): clears ears on top of the head
#endif
#ifndef PROPS_PALM_SIGN
#define PROPS_PALM_SIGN 1.0        // -1 if the character's grasp frame z points out of the BACK of the hand
#endif
#ifndef PROPS_GRASP_OFS
#define PROPS_GRASP_OFS 0.0        // the grasp function's own offset off the palm (scene units): taken back here
#endif

mat3 prRX(float a){ float c=cos(a),s=sin(a); return mat3(1,0,0, 0,c,s, 0,-s,c); }
mat3 prRY(float a){ float c=cos(a),s=sin(a); return mat3(c,0,-s, 0,1,0, s,0,c); }
mat3 prRZ(float a){ float c=cos(a),s=sin(a); return mat3(c,s,0, -s,c,0, 0,0,1); }
mat3 prEuler(vec3 a){ return prRZ(a.z)*prRY(a.y)*prRX(a.x); }       // X first, then Y, then Z (world axes)
float prBox(vec3 p, vec3 b, float r){ vec3 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r; }
float prCyl(vec3 p, float r, float h){ vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h); return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); }

// ------------------------------------------------------------------ the models.  Unit frame: y up, z = its front
// (toward the camera at home); |q| <= PR_BND.  m = the part, for the painter; a = articulation 0..1 (prop{k}_s)
#ifdef PROPKIND_BALL
float pk_ball(vec3 q, float a, out float m){ m = 0.0; return length(q) - 0.98; }
#endif
#ifdef PROPKIND_MUG
float pk_mug(vec3 q, float a, out float m){
  q.y += 0.08;
  float body = prCyl(q, 0.62, 0.70) - 0.06;
  float inner = prCyl(q - vec3(0.0, 0.22, 0.0), 0.52, 0.70);
  float d = max(body, -inner);
  vec3 h = q - vec3(0.70, 0.02, 0.0);
  float hd = length(vec2(length(h.xy) - 0.34, h.z)) - 0.085;
  hd = max(hd, -h.x - 0.02);
  m = hd < d ? 1.0 : (inner > -0.02 && q.y > -0.40 ? 2.0 : 0.0);
  return min(d, hd);
}
#endif
#ifdef PROPKIND_BOOK
// a hardback, spine on the left (x -0.62); a = 0 shut .. 1 open flat, both pages facing the front (+z): the front
// half swings 180deg*a about the spine line, and the spine slides to the middle so the open book stays centred
float bookHalf(vec3 c, float side, out float pages){
  // a half lies on z in [0, side*0.19]: its board outside, its half of the page block inside
  float board = prBox(c - vec3(0.02, 0.0, side*0.170), vec3(0.64, 0.86, 0.022), 0.012);
  pages = prBox(c - vec3(0.04, 0.0, side*0.080), vec3(0.60, 0.82, 0.075), 0.006);
  return board;
}
float pk_book(vec3 q, float a, out float m){
  vec3 s = q - vec3(-0.62 + 0.62*a, 0.0, 0.0);
  vec3 f = s; f.xz = rot(3.14159*a)*f.xz; f.x -= 0.62;     // the front half, swung back about the spine
  vec3 b = s; b.x -= 0.62;
  float pf, pb;
  float df = bookHalf(f, 1.0, pf), db = bookHalf(b, -1.0, pb);
  float spine = max(length(s.xz - vec2(0.0, -0.095*a)) - 0.20, abs(s.y) - 0.86);
  spine = max(spine, mix(s.x - 0.02, s.z + 0.15, a));
  float d = min(min(df, db), spine);
  float pg = min(pf, pb);
  m = pg < d ? 1.0 : 0.0;
  return min(d, pg);
}
#endif
#ifdef PROPKIND_DUCK
float pk_duck(vec3 q, float a, out float m){
  float body = sdEllipsoid(q - vec3(0.0, -0.30, -0.05), vec3(0.66, 0.46, 0.80));
  body = smin(body, sdEllipsoid(q - vec3(0.0, -0.05, -0.62), vec3(0.30, 0.30, 0.28)), 0.25);
  float head = length(q - vec3(0.0, 0.42, 0.34)) - 0.40;
  float d = smin(body, head, 0.18);
  float beak = sdEllipsoid(q - vec3(0.0, 0.32, 0.80), vec3(0.20, 0.08, 0.20));
  m = beak < d ? 1.0 : 0.0;
  if(length(vec3(abs(q.x), q.yz) - vec3(0.20, 0.52, 0.66)) < 0.09) m = 2.0;
  return min(d, beak);
}
#endif
#ifdef PROPKIND_APPLE
float pk_apple(vec3 q, float a, out float m){
  vec3 p = q - vec3(0.0, -0.06, 0.0);
  float r = length(p.xz);
  float d = sdEllipsoid(p, vec3(0.86, 0.78, 0.86));
  d += 0.16*exp(-r*r*14.0)*smoothstep(-0.2, 0.6, p.y) + 0.10*exp(-r*r*18.0)*smoothstep(0.1, -0.6, p.y);
  d *= 0.85;
  float stem = sdCapsule(q, vec3(0.0, 0.46, 0.0), vec3(0.06, 0.92, 0.02), 0.045);
  float leaf = sdEllipsoid(prRZ(-0.6)*(q - vec3(0.20, 0.82, 0.0)), vec3(0.24, 0.06, 0.12));
  m = stem < d ? 1.0 : 0.0;
  if(leaf < min(d, stem)) m = 2.0;
  return min(min(d, stem), leaf);
}
#endif
#ifdef PROPKIND_PLANT
// a little potted plant: a terracotta pot (bottom y -0.97), soil, five rounded leaves on short stalks
float pk_plant(vec3 q, float a, out float m){
  vec3 p = q - vec3(0.0, -0.55, 0.0);
  float t = clamp((p.y + 0.42)/0.84, 0.0, 1.0);
  float pot = max(length(p.xz) - mix(0.34, 0.46, t), abs(p.y) - 0.40) - 0.02;
  float rim = max(length(p.xz) - 0.52, abs(p.y - 0.36) - 0.06) - 0.02;
  pot = min(pot, rim);
  float soil = max(length(p.xz) - 0.44, abs(p.y - 0.38) - 0.03);
  float lv = 1e3;
  for(int i=0;i<5;i++){
    float fi = float(i);
    float an = fi*2.513 + 0.4;
    vec3 dir = normalize(vec3(cos(an)*0.62, 1.0 - 0.10*fi, sin(an)*0.62));
    vec3 o = vec3(0.0, -0.18, 0.0);
    float st = sdCapsule(q, o, o + dir*0.40, 0.028);
    vec3 l = q - (o + dir*(0.62 + 0.05*fi));
    vec3 ax = normalize(cross(dir, vec3(-sin(an), 0.0, cos(an))));
    vec3 lq = vec3(dot(l, ax), dot(l, dir), dot(l, cross(ax, dir)));
    lq.z -= 0.35*lq.x*lq.x;            // cupped
    float leaf = sdEllipsoid(lq, vec3(0.21, 0.30, 0.035));
    lv = min(lv, min(st, leaf*0.9));
  }
  float d = min(min(pot, soil), lv);
  m = lv <= d + 0.001 ? 1.0 : (soil <= d + 0.001 ? 2.0 : 0.0);
  return d;
}
#endif
#ifdef PROPKIND_HEADPHONES
// headphones as worn, unit frame = the head frame (x right, y up, z forward): the cups' inner faces at x = +-W on the
// ear line (y 0), the band an elliptic arc over the crown, top at y = ARC.  Parts: 0 shell, 1 cushions, 2 band/yokes
float hpArc(vec2 b, vec2 e){
  float k0 = length(b/e), k1 = length(b/(e*e));
  return abs(k0*(k0 - 1.0)/max(k1, 1e-4));
}
float pk_headphones(vec3 q, float a, out float m){
  // a = the band's spread: 0 relaxed (the spring band pulls the cups in: PROPS_HP_WR), 1 opened to the head (worn)
  float W = mix(PROPS_HP_WR, PROPS_HP_W, a), A = PROPS_HP_ARC;
  vec3 c = vec3(abs(q.x) - W, q.y, q.z);
  float cush = max(length(c.yz) - 0.22, abs(c.x - 0.05) - 0.05) - 0.05;      // the soft ring against the head
  float shell = max(length(c.yz) - 0.26, abs(c.x - 0.17) - 0.08) - 0.04;      // the cup
  // band: an ellipse arc from the cups' tops (x +-(W + 0.17), y 0.36) to the crown (0, A)
  vec2 e = vec2(W + 0.17, max(A - 0.36, 0.2));
  vec2 b = vec2(abs(q.x), q.y - 0.36);
  float arc = hpArc(b, e);
  float band = max(max(arc - 0.040, abs(q.z - PROPS_HP_BZ) - 0.10), -b.y) - 0.012;
  vec2 e2 = vec2(W + 0.11, max(A - 0.42, 0.2));
  float pad = max(max(hpArc(vec2(abs(q.x), q.y - 0.36), e2) - 0.035, abs(q.z - PROPS_HP_BZ) - 0.075), 0.06 - b.y) - 0.012;
  float yoke = sdCapsule(vec3(abs(q.x), q.yz), vec3(W + 0.17, 0.22, 0.0), vec3(W + 0.17, 0.40, 0.0), 0.040);
  float d = min(min(shell, cush), min(min(band, yoke), pad));
  m = (cush <= d + 0.002 || pad <= d + 0.002) ? 1.0 : ((band <= d + 0.002 || yoke <= d + 0.002) ? 2.0 : 0.0);
  return d;
}
#endif
#ifdef PROPKIND_CHAIR
// a small rounded chair, built in scene units (its scale is 1) from PROPS_CHAIR = (seat width, seat depth, seat
// height, backrest height), frame origin at the middle of its height: floor at y = -hb/2, front +z.  Parts: 0 wood,
// 1 upholstery
#ifndef PROPS_CHAIR
#define PROPS_CHAIR vec4(1.20, 0.95, 0.62, 1.55)
#endif
float pk_chair(vec3 q, float a, out float m){
  vec4 D = PROPS_CHAIR;
  float hw = 0.5*D.x, hd = 0.5*D.y, fl = -0.5*D.w;
  vec3 p = q - vec3(0.0, fl, 0.0);                          // from the floor
  float th = 0.06*D.z;                                       // frame thickness
  float seat = prBox(p - vec3(0.0, D.z - 0.11, 0.0), vec3(hw, 0.05, hd), 0.04);
  float cushion = prBox(p - vec3(0.0, D.z - 0.035, 0.03), vec3(hw - 0.05, 0.045, hd - 0.06), 0.045);
  float lr = 0.045;
  vec3 l = vec3(abs(p.x) - (hw - 0.08), p.y, abs(p.z) - (hd - 0.08));
  float leg = sdTaper(l, vec3(0.0, 0.0, 0.0), vec3(0.0, D.z - 0.12, 0.0), lr*0.85, lr*1.15);
  vec3 bp = vec3(abs(p.x) - (hw - 0.08), p.y, p.z + (hd - 0.07));
  float post = sdTaper(bp, vec3(0.0, D.z - 0.12, 0.0), vec3(0.0, D.w - 0.04, -0.06), lr*1.05, lr);
  float back = prBox(p - vec3(0.0, mix(D.z, D.w, 0.62), -(hd - 0.03) - 0.04), vec3(hw - 0.06, 0.22*(D.w - D.z), 0.04), 0.04);
  float rail = sdCapsule(p, vec3(-(hw - 0.08), D.w - 0.05, -(hd - 0.07) - 0.06), vec3(hw - 0.08, D.w - 0.05, -(hd - 0.07) - 0.06), 0.05);
  float frame = min(min(seat, leg), min(post, rail));
  float d = min(min(frame, cushion), back);
  m = (cushion <= d + 0.002 || back <= d + 0.002) ? 1.0 : 0.0;
  return d;
}
#endif
#ifdef PROPKIND_DESK
// a small writing desk, scene units (scale 1), PROPS_DESK = (width, depth, top height): frame origin at the middle of
// its height (floor y = -h/2, top face y = +h/2), front +z, a drawer under the right side.  Parts: 0 wood, 1 knob,
// 2 drawer front
#ifndef PROPS_DESK
#define PROPS_DESK vec3(1.90, 0.95, 1.10)
#endif
float pk_desk(vec3 q, float a, out float m){
  vec3 D = PROPS_DESK;
  float hw = 0.5*D.x, hd = 0.5*D.y, hh = 0.5*D.z;
  float top = prBox(q - vec3(0.0, hh - 0.04, 0.0), vec3(hw, 0.04, hd), 0.02);
  vec3 l = vec3(abs(q.x) - (hw - 0.10), q.y, abs(q.z) - (hd - 0.09));
  float leg = sdTaper(l, vec3(0.0, -hh, 0.0), vec3(0.0, hh - 0.08, 0.0), 0.038, 0.050);
  float apron = prBox(q - vec3(0.0, hh - 0.15, -0.02), vec3(hw - 0.12, 0.07, hd - 0.10), 0.01);
  float drawer = prBox(q - vec3(0.30*hw + 0.05, hh - 0.22, 0.02), vec3(0.36*hw, 0.10, hd - 0.10), 0.012);
  float knob = length(q - vec3(0.30*hw + 0.05, hh - 0.22, hd - 0.06)) - 0.035;
  float body = min(min(top, leg), min(apron, drawer));
  float d = min(body, knob);
  m = knob <= d + 0.001 ? 1.0 : (drawer <= d + 0.001 && q.z > hd - 0.13 ? 2.0 : 0.0);
  return d;
}
#endif
#ifdef PROPKIND_LAPTOP
// a laptop, shut: base x +-1, z +-0.68, y [-0.08, 0]; the lid on top y [0, 0.08].  The lid hinges at the base's back top
// edge (y 0, z -0.66): a = 0 shut .. 1 open (112 degrees).  Parts: 0 shell, 1 keys, 2 screen, 3 trackpad, 4 bezel
vec3 laptopLid(vec3 q, float a){
  vec3 h = q - vec3(0.0, 0.0, -0.66);
  h.yz = rot(a*1.95)*h.yz;             // open: the lid stands up from the hinge, tipped back past upright
  return h - vec3(0.0, 0.0, 0.66);     // lid frame = the base's frame when shut: y = 0 its inner face, z from the hinge (-0.66) to its front/top edge (0.68)
}
float pk_laptop(vec3 q, float a, out float m){
  float base = prBox(q - vec3(0.0, -0.04, 0.0), vec3(1.0, 0.04, 0.68), 0.02);
  vec3 lq = laptopLid(q, a);
  float lid = prBox(lq - vec3(0.0, 0.035, 0.0), vec3(1.0, 0.035, 0.68), 0.02);
  float d = min(base, lid);
  m = 0.0;
  if(lid < base){
    if(lq.y < 0.012) m = (abs(lq.x) < 0.92 && abs(lq.z - 0.02) < 0.58) ? 2.0 : 4.0;
  } else if(q.y > -0.012){
    // the key well runs to the front edge (a compact laptop: short plush arms reach its near rows from a chair)
    if(abs(q.x) < 0.88 && q.z > -0.30 && q.z < 0.60) m = 1.0;
    else if(abs(q.x) < 0.32 && q.z > -0.60 && q.z < -0.36) m = 3.0;      // a slim pad strip behind the keys
  }
  return d;
}
#endif
// ---- OBJECTS (PK 30-34): hat, glasses, gift box, watering can, toy block
#define PK_HAT 30
#define PK_GLASSES 31
#define PK_BOX 32
#define PK_WATERING_CAN 33
#define PK_BLOCK 34
float prSeg2(vec2 p, vec2 a, vec2 b){ vec2 pa = p - a, ba = b - a; return length(pa - ba*clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0)); }
#ifdef PROPKIND_HAT
// a straw sun hat: a hollow rounded crown (open underneath: it goes ON a head) and a wide brim drooping a little at
// its edge.  y from -0.31 (brim edge) to 0.28 (crown top); crown opening r 0.55.  Parts: 0 straw, 1 ribbon band + bow
float pk_hat(vec3 q, float a, out float m){
  float r = length(q.xz);
  float tp = 1.0 + 0.10*clamp(q.y - 0.03, -0.3, 0.3);                 // the crown narrows a little toward its top
  vec2 o = abs(vec2(r*tp, q.y - 0.03)) - vec2(0.60, 0.25) + 0.14;
  float outer = length(max(o, 0.0)) + min(max(o.x, o.y), 0.0) - 0.14;
  vec2 i = abs(vec2(r*tp, q.y + 0.04)) - vec2(0.55, 0.25) + 0.10;
  float inner = length(max(i, 0.0)) + min(max(i.x, i.y), 0.0) - 0.10;
  float crown = max(outer, -inner);
  float e = max(r - 0.55, 0.0);
  float yb = -0.205 - 0.17*e*e;                                          // the brim's mid surface, drooping outward
  float brim = max(max(abs(q.y - yb) - 0.024, r - 1.0), 0.53 - r)*0.92;
  float band = max(max(outer - 0.016, abs(q.y + 0.10) - 0.07), -inner);
  vec3 bq = q - vec3(0.0, -0.10, -0.60);                                  // a little bow at the back
  float bow = min(sdEllipsoid(vec3(abs(bq.x) - 0.10, bq.y, bq.z), vec3(0.11, 0.07, 0.05)), length(bq) - 0.055);
  bow = min(bow, sdCapsule(vec3(abs(bq.x), bq.yz), vec3(0.02, -0.02, 0.0), vec3(0.09, -0.16, -0.03), 0.022));
  float d = min(min(crown, brim), min(band, bow));
  m = min(band, bow) <= d + 0.001 ? 1.0 : 0.0;
  return d;
}
#endif
#ifdef PROPKIND_GLASSES
// round sunglasses, the lenses' centres at x +-0.46 (bridge at 0), front +z.  a = the arms: 0 folded behind the
// lenses .. 1 open (worn opens them: propsInit).  Open, the arm ends sit at x +-1.50, z -1.20 (the sides of a head).
// Parts: 0 frame, 1 lens
float pk_glasses(vec3 q, float a, out float m){
  vec3 s = vec3(abs(q.x), q.yz);
  vec3 l = s - vec3(0.46, 0.0, 0.0);
  float lr = length(l.xy);
  float ring = length(vec2(lr - 0.255, l.z)) - 0.036;
  float lens = max(lr - 0.26, abs(l.z + 0.004*lr*lr*0.0) - 0.016);
  float bridge = min(sdCapsule(s, vec3(0.0, 0.10, 0.0), vec3(0.22, 0.07, 0.0), 0.030), length(s - vec3(0.22, 0.07, 0.0)) - 0.030);
  vec3 h = s - vec3(0.72, 0.03, -0.06);
  float hinge = length(h + vec3(0.0, 0.0, 0.03)) - 0.055;
  h.xz = rot(2.20*(1.0 - a))*h.xz;                                      // folded: swung in behind the lenses
  float arm = min(sdCapsule(h, vec3(0.0), vec3(0.42, 0.01, -0.62), 0.027), sdCapsule(h, vec3(0.42, 0.01, -0.62), vec3(0.76, -0.05, -1.14), 0.027));
  float fr = min(min(ring, bridge), min(hinge, arm));
  m = lens < fr ? 1.0 : 0.0;
  return min(fr, lens);
}
#endif
#ifdef PROPKIND_BOX
// a gift box: a hollow body (y -0.90 .. 0.40, walls 0.08, its inside floor at y -0.82) and a cap lid (y 0.32 .. 0.62,
// a bow on top up to y ~0.86) hinged at the body's back top edge (y 0.42, z -0.90): a = 0 shut .. 1 open (110 deg).
// Parts: 0 body, 1 lid, 2 bow, 3 the body's inside, 4 the lid's underside
vec3 boxLid(vec3 q, float a){
  vec3 h = q - vec3(0.0, 0.42, -0.90);
  h.yz = rot(1.92*a)*h.yz;
  return h + vec3(0.0, 0.42, -0.90);
}
float pk_box(vec3 q, float a, out float m){
  float body = prBox(q - vec3(0.0, -0.25, 0.0), vec3(0.96, 0.65, 0.84), 0.035);
  float cav = prBox(q - vec3(0.0, 0.30, 0.0), vec3(0.88, 1.12, 0.76), 0.02);
  body = max(body, -cav);
  vec3 l = boxLid(q, a);
  float lin = prBox(l - vec3(0.0, 0.33, 0.0), vec3(0.975, 0.20, 0.855), 0.02);
  float lid = max(prBox(l - vec3(0.0, 0.47, 0.0), vec3(1.02, 0.15, 0.90), 0.035), -lin);
  vec3 b = l - vec3(0.0, 0.66, 0.0);
  float bow = length(b) - 0.085;
  vec3 bl = vec3(abs(b.x), b.yz);
  bl.xy = rot(-0.55)*(bl.xy - vec2(0.17, 0.07));
  bow = min(bow, sdEllipsoid(bl, vec3(0.19, 0.10, 0.09)));
  float d = min(min(body, lid), bow);
  // the insides (card): the body's cavity walls + floor, the underside of the lid
  m = bow <= d + 0.001 ? 2.0 : (lid <= d + 0.001 ? (lin < 0.015 ? 4.0 : 1.0) : (cav < 0.012 ? 3.0 : 0.0));
  return d;
}
#endif
#ifdef PROPKIND_WATERING_CAN
// a watering can, spout to its right (+x): a body (x -0.15, r 0.50, y -0.72 .. 0.20) under a low dome, a top
// handle arching over it (top y 0.72), the spout rising to a sprinkler rose (centre T, facing D).  a = the water flow
// 0..1: a stream from the rose falling under gravity in WORLD space (dn = world down in the unit frame), plus a few
// drops.  Parts: 0 metal, 1 handle + rose, 2 water
#define CAN_T vec3(1.06, 0.49, 0.0)
#define CAN_D vec3(0.653, 0.757, 0.0)
float pk_can(vec3 q, float a, vec3 dn, out float m){
  vec3 b = q - vec3(-0.15, -0.26, 0.0);
  float body = prCyl(b, 0.46, 0.42) - 0.04;
  body = smin(body, sdEllipsoid(q - vec3(-0.15, 0.17, 0.0), vec3(0.47, 0.14, 0.47)), 0.06);
  float spout = sdTaper(q, vec3(0.22, -0.46, 0.0), CAN_T - CAN_D*0.06, 0.095, 0.045);
  vec3 v = q - CAN_T;
  float al = dot(v, CAN_D);
  float rose = max(length(v - CAN_D*al) - mix(0.06, 0.12, clamp(al/0.05 + 0.5, 0.0, 1.0)), abs(al) - 0.05);
  vec2 hq = vec2(length(q.xy - vec2(-0.15, 0.16)) - 0.44, q.z);
  float handle = max(length(hq) - 0.055, 0.16 - q.y);
  float metal = smin(body, spout, 0.05);
  float d = min(metal, min(rose, handle));
  m = min(rose, handle) <= d + 0.001 ? 1.0 : 0.0;
  if(a > 0.001){
    // the stream: P(s) = T0 + D*0.42*s + dn*1.35*s^2, s in [0, 1] (a parabola, the way the water leaves the rose),
    // thinner as the flow drops; a cheap bounding sphere skips it away from the stream
    vec3 t0 = CAN_T + CAN_D*0.05;
    vec3 mid = t0 + CAN_D*0.21 + dn*0.34;
    float bs = length(q - mid) - 1.25;
    float w = 1e3;
    if(bs < 0.1){
      float rr = mix(0.025, 0.060, smoothstep(0.0, 0.6, a));
      vec3 p0 = t0;
      for(int i=1;i<=5;i++){
        float s = float(i)/5.0;
        vec3 p1 = t0 + CAN_D*0.42*s + dn*1.35*s*s;
        w = min(w, sdTaper(q, p0, p1, rr*1.05, rr*0.85));
        p0 = p1;
      }
      for(int i=0;i<3;i++){                                             // drops breaking off the end of it
        float s = 1.0 + 0.30*fract(float(i)/3.0 + uTime*1.7);
        vec3 pd = t0 + CAN_D*0.42*s + dn*1.35*s*s;
        w = min(w, length(q - pd) - rr*0.95);
      }
    } else w = bs;
    if(w < d){ d = w; m = 2.0; }
  }
  return d;
}
#endif
#ifdef PROPKIND_BLOCK
// a rounded wooden toy cube (half side 0.9); a letter on its faces is painted (PR_VAR: the slot's letter code)
float pk_block(vec3 q, float a, out float m){ m = 0.0; return prBox(q, vec3(0.90), 0.14); }
// the letter's 2D distance on a face (uv -1..1 across the face): 'A', 'B', else a star-ish dot
float prLetter(vec2 p, int c){
  if(c == 65){
    float d = min(prSeg2(vec2(abs(p.x), p.y), vec2(0.42, -0.58), vec2(0.0, 0.60)), prSeg2(p, vec2(-0.22, -0.16), vec2(0.22, -0.16)));
    return d - 0.11;
  }
  if(c == 66){
    float d = min(prSeg2(p, vec2(-0.34, -0.58), vec2(-0.34, 0.58)), min(prSeg2(p, vec2(-0.34, 0.58), vec2(0.02, 0.58)),
              min(prSeg2(p, vec2(-0.34, 0.0), vec2(0.06, 0.0)), prSeg2(p, vec2(-0.34, -0.58), vec2(0.06, -0.58)))));
    vec2 u = p - vec2(0.02, 0.29), w = p - vec2(0.06, -0.29);
    d = min(d, max(abs(length(u) - 0.29), -u.x));
    d = min(d, max(abs(length(w) - 0.29), -w.x));
    return d - 0.10;
  }
  return length(p) - 0.32;
}
#endif
#ifdef PROPKIND_CHALK
// scene: a stick of chalk along its own y (+-1), radius 0.22, the ends softly rounded and a little worn at +y
// (the writing tip).  Parts: 0 chalk
float pk_chalk(vec3 q, float a, out float m){
  m = 0.0;
  float r = 0.22 - 0.05*smoothstep(0.70, 1.0, q.y);
  return sdCapsule(q, vec3(0.0, -0.80, 0.0), vec3(0.0, 0.84, 0.0), r);
}
#endif
#ifdef PROPKIND_ERASER
// scene: a board eraser: an oak back (y 0..0.36, grip on top) on a grey felt pad (y -0.36..0, its -y face rubs
// the board).  x +-1, z +-0.42.  Parts: 0 wood, 1 felt
float pk_eraser(vec3 q, float a, out float m){
  float wood = prBox(q - vec3(0.0, 0.17, 0.0), vec3(0.98, 0.17, 0.40), 0.08);
  float felt = prBox(q - vec3(0.0, -0.18, 0.0), vec3(1.0, 0.18, 0.42), 0.05);
  m = felt < wood ? 1.0 : 0.0;
  return min(wood, felt);
}
#endif

// ------------------------------------------------------------------ slot state (per fragment, propsInit)
vec3  gPrC[16];      // centre, world (scene units)
mat3  gPrM[16];      // world -> unit frame rotation (unit q = gPrM*(p - c)/size)
float gPrArt[16];    // articulation (prop{k}_s)
float gPrRest[16];   // 1 = sitting on its home place (contact shade under it)
// PROPS_UNI (the frame passes): the per-frame poses are uniforms (rt_engine computes them once per frame in its
// p_ppose pass and reads them back), so no per-fragment prop state exists there (dynamically indexed per-fragment
// arrays live in slow local memory on the GPU).  Elsewhere (bake, probes, the pose passes) propsInit fills arrays.
#ifdef PROPS_UNI
// the slot loops of the frame passes run to a UNIFORM bound (= PROPS_N): with a constant bound the compiler
// unrolls them, one inlined copy of every model per slot per loop — at 14 slots that code (registers, i-cache) cost
// the character pass ~5 ms on the K620 even with the new props off screen
uniform int uPrN;
#define PR_LOOP_N uPrN
uniform vec3 uPrC[16];      // centre, world
uniform mat3 uPrM[16];      // world -> unit frame
uniform vec2 uPrAR[16];     // articulation, rest
#define PRC(k) uPrC[k]
#define PRART(k) uPrAR[k].x
#define PRREST(k) uPrAR[k].y
#define PRM(k) uPrM[k]
#else
#define PR_LOOP_N PROPS_N
#define PRC(k) gPrC[k]
#define PRART(k) gPrArt[k]
#define PRREST(k) gPrRest[k]
#define PRM(k) gPrM[k]
#endif

// the bounding radius / box of slot k as it is now: an articulated prop (lid open, arms out, water flowing) may reach
// further than shut (props_defs BBOX_ART); the others have the same numbers both ways
float prBnd(int k){ return PRART(k) > 0.001 ? PR_BND2[k] : PR_BND[k]; }
float propUnit(int k, vec3 q, out float m){
  float a = PRART(k);
  m = 0.0;
  switch(PR_KIND[k]){
#ifdef PROPKIND_BALL
    case PK_BALL: return pk_ball(q, a, m);
#endif
#ifdef PROPKIND_MUG
    case PK_MUG: return pk_mug(q, a, m);
#endif
#ifdef PROPKIND_BOOK
    case PK_BOOK: return pk_book(q, a, m);
#endif
#ifdef PROPKIND_DUCK
    case PK_DUCK: return pk_duck(q, a, m);
#endif
#ifdef PROPKIND_APPLE
    case PK_APPLE: return pk_apple(q, a, m);
#endif
#ifdef PROPKIND_PLANT
    case PK_PLANT: return pk_plant(q, a, m);
#endif
#ifdef PROPKIND_HEADPHONES
    case PK_HEADPHONES: return pk_headphones(q, a, m);
#endif
#ifdef PROPKIND_CHAIR
    case PK_CHAIR: return pk_chair(q, a, m);
#endif
#ifdef PROPKIND_DESK
    case PK_DESK: return pk_desk(q, a, m);
#endif
#ifdef PROPKIND_LAPTOP
    case PK_LAPTOP: return pk_laptop(q, a, m);
#endif
#ifdef PROPKIND_HAT
    case PK_HAT: return pk_hat(q, a, m);
#endif
#ifdef PROPKIND_GLASSES
    case PK_GLASSES: return pk_glasses(q, a, m);
#endif
#ifdef PROPKIND_BOX
    case PK_BOX: return pk_box(q, a, m);
#endif
#ifdef PROPKIND_WATERING_CAN
    case PK_WATERING_CAN: return pk_can(q, a, PRM(k)*vec3(0.0, -1.0, 0.0), m);
#endif
#ifdef PROPKIND_BLOCK
    case PK_BLOCK: return pk_block(q, a, m);
#endif
#ifdef PROPKIND_CHALK
    case PK_CHALK: return pk_chalk(q, a, m);
#endif
#ifdef PROPKIND_ERASER
    case PK_ERASER: return pk_eraser(q, a, m);
#endif
  }
  return 1e3;
}

// the floor's own steps (props file "floor_steps": a rug, a mat): a free prop that rests on the floor follows them, so
// a chair dragged off the rug does not float and one dragged onto it does not sink.  PROPS_FLOOR_STEP = vec4(x0, z0,
// x1, z1) (scene), PROPS_FLOOR_STEP_H its height; PR_ONFLOOR[k] = 1 for a prop whose home is on the floor
float propsFloorAt(vec2 xz){
#ifdef PROPS_FLOOR_STEP
  vec4 r = PROPS_FLOOR_STEP;
  vec2 e = min(xz - r.xy, r.zw - xz);
  return PROPS_FLOOR_STEP_H*smoothstep(-0.02, 0.02, min(e.x, e.y));
#else
  return 0.0;
#endif
}

// quaternion nlerp for the free <-> held <-> worn rotation ease (shorter arc)
vec4 prQuat(mat3 m){
  float tr = m[0][0] + m[1][1] + m[2][2];
  vec4 q;
  if(tr > 0.0){ float s = sqrt(tr + 1.0)*2.0; q = vec4((m[1][2] - m[2][1])/s, (m[2][0] - m[0][2])/s, (m[0][1] - m[1][0])/s, 0.25*s); }
  else if(m[0][0] > m[1][1] && m[0][0] > m[2][2]){ float s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])*2.0; q = vec4(0.25*s, (m[1][0] + m[0][1])/s, (m[2][0] + m[0][2])/s, (m[1][2] - m[2][1])/s); }
  else if(m[1][1] > m[2][2]){ float s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])*2.0; q = vec4((m[1][0] + m[0][1])/s, 0.25*s, (m[2][1] + m[1][2])/s, (m[2][0] - m[0][2])/s); }
  else { float s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])*2.0; q = vec4((m[2][0] + m[0][2])/s, (m[2][1] + m[1][2])/s, 0.25*s, (m[0][1] - m[1][0])/s); }
  return normalize(q);
}
mat3 prQuatM(vec4 q){
  float x = q.x, y = q.y, z = q.z, w = q.w;
  return mat3(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + z*w), 2.0*(x*z - y*w),
              2.0*(x*y - z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + x*w),
              2.0*(x*z + y*w), 2.0*(y*z - x*w), 1.0 - 2.0*(x*x + y*y));
}
mat3 prNlerp(mat3 a, mat3 b, float t){
  vec4 qa = prQuat(a), qb = prQuat(b);
  if(dot(qa, qb) < 0.0) qb = -qb;
  return prQuatM(normalize(mix(qa, qb, t)));
}

// the held pose of slot k in one paw (char-local in, char-local out: the caller applies the root)
void propHeldIn(int side, int k, mat3 Rcol, vec3 offs, out vec3 c, out mat3 R){
  vec3 gp; mat3 gf;
  PROPS_GRASP_FN(side, gp, gf);
  vec3 up = vec3(0.0, 1.0, 0.0);
  vec3 camL = transpose(gRootR)*(uCamPos - rootFwd(gp));          // the camera, seen in the character's frame
  vec3 fw = camL - up*dot(camL, up);
  fw = length(fw) > 1e-4 ? normalize(fw) : vec3(0.0, 0.0, 1.0);
  mat3 G = mat3(cross(up, fw), up, fw);
  R = G*Rcol;
  gp -= normalize(gf[2])*PROPS_GRASP_OFS;
  vec3 pn = normalize(gf[2])*PROPS_PALM_SIGN;
  vec3 nl = transpose(R)*(-pn);                                    // toward the palm, in the unit frame
  float sup = PR_SUPR[k] > 0 ? PR_GR[k] : dot(abs(nl), PR_EXT[k]) - dot(nl, PR_GA[k]);   // its extent toward the palm, from the anchor
  float sq = PR_GRIP[k] == 1 ? 0.0 : 0.03;                         // a power grip presses into the soft palm a little
  c = gp + pn*(max(sup, 0.04) - sq)*PR_SIZE[k] - R*PR_GA[k]*PR_SIZE[k] + G*offs;
}
#endif   // PROPS_N > 0

// a cheap lower bound of the CHARACTER's distance (world point in, world distance out), so points among the props
// skip the character's full SDF (the props widen the character pass to their pixels).  The props file names it
// ("renderer": {"char_bound": "muse"}); without one the bound is 0 (always the full SDF).  PROPS_BOUND_MUSE: a box
// (character-local) round the muse's joints, padded for the fur and the smooth blends, set up once per fragment
// after the character's init (animals/muse.glsl globals).
#if PROPS_N > 0 && defined(PROPS_BOUND_MUSE) && !defined(MUSE_DEBUG_SEAT)
vec3 gPbLo, gPbHi;
void propsBoundInit(){
  vec3 lo = gHO - vec3(1.0), hi = gHO + vec3(1.0);
#define PB_(c, r) { vec3 c_ = (c); lo = min(lo, c_ - vec3(r)); hi = max(hi, c_ + vec3(r)); }
  PB_(gMHip - M_HIP + vec3(0.0, 0.69, 0.0), 0.60)
  PB_(upperToLocal(vec3(0.0, 1.0, 0.0)), 0.62)
  for(int i=0;i<2;i++){
    PB_(gMSh[i], 0.22) PB_(gMEl[i], 0.22) PB_(gMWr[i], 0.22) PB_(gMBC[i], gMBR[i] + 0.07)
    PB_(gMHipJ[i], 0.23) PB_(gMKn[i], 0.23) PB_(gMAnk[i], 0.40)
  }
#undef PB_
  gPbLo = lo; gPbHi = hi;
}
float propsBoundMuse(vec3 pw){
  vec3 p = rootInv(pw);
  vec3 d = max(gPbLo - p, p - gPbHi);
  float r = length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
#ifdef ROOT_XFORM
  if(uRootOn > 0.5) r *= uRootLip;
#endif
  return r;
}
#define PROPS_CHAR_BOUND propsBoundMuse
// where the ray (world, t in world units) is inside the character's joint box: false = it never enters it
bool propsCharBox(vec3 ro, vec3 rd, out float t0, out float t1){
  vec3 o = rootInv(ro), d = rootInv(ro + rd) - o;
  vec3 inv = 1.0/(abs(d) + 1e-6)*sign(d + 1e-7);
  vec3 ta = (gPbLo - o)*inv, tb = (gPbHi - o)*inv;
  vec3 tn = min(ta, tb), tf = max(ta, tb);
  t0 = max(max(tn.x, tn.y), tn.z); t1 = min(min(tf.x, tf.y), tf.z);
  return t1 >= max(t0, 0.0);
}
#else
void propsBoundInit(){}
#define PROPS_CHAR_BOUND(p) 0.0
bool propsCharBox(vec3 ro, vec3 rd, out float t0, out float t1){ t0 = 0.0; t1 = 1e9; return true; }
#endif
bool propsCharBoxHit(vec3 ro, vec3 rd, float t1){ float a, b; return propsCharBox(ro, rd, a, b) && a < t1; }

void propsInit(){
#if PROPS_N > 0 && !defined(PROPS_UNI)
  for(int k=0;k<PROPS_N;k++){
    int b = PR_PX[k];
    vec3 offs = vec3(uP[b], uP[b+1], uP[b+2])*PROPS_H;
    vec3 tr = vec3(uP[b+3], uP[b+4], uP[b+5])*6.2831853;           // turns -> radians (periodic: no fract needed)
    float wl = clamp(uP[b+6], 0.0, 1.0), wr = clamp(uP[b+7], 0.0, 1.0);
    float wh = PR_PHH[k] >= 0 ? clamp(uP[PR_PHH[k]], 0.0, 1.0) : 0.0;
    gPrArt[k] = PR_PS[k] >= 0 ? clamp(uP[PR_PS[k]], 0.0, 1.0) : 0.0;
#ifdef PROPKIND_HEADPHONES
    if(PR_KIND[k] == PK_HEADPHONES) gPrArt[k] = max(gPrArt[k], wh);     // on the head the band is open to fit it
#endif
#ifdef PROPKIND_GLASSES
    if(PR_KIND[k] == PK_GLASSES) gPrArt[k] = max(gPrArt[k], wh);           // worn: the arms open round the head
#endif
    if(PR_FREE[k] > 0){ wl = 0.0; wr = 0.0; wh = 0.0; }            // furniture is never attached: the page moves it
    mat3 Rh = prEuler(PR_ROT[k]);
    mat3 Rcol = prEuler(tr)*Rh;
    vec3 c = PR_HOME[k] + offs; mat3 R = Rcol;
    if(PR_ONFLOOR[k] > 0) c.y += propsFloorAt(c.xz) - propsFloorAt(PR_HOME[k].xz);      // on the floor: follow its steps
    float wa = wl + wr, w = clamp(wa + wh, 0.0, 1.0);
    if(w > 0.0005){
      vec3 cH = vec3(0.0); mat3 RH = mat3(1.0);
      if(wa > 0.0005){
        vec3 cl = vec3(0.0), cr = vec3(0.0); mat3 Ml = mat3(1.0), Mr = mat3(1.0);
        if(wl > 0.0005) propHeldIn(0, k, Rcol, offs, cl, Ml);
        if(wr > 0.0005) propHeldIn(1, k, Rcol, offs, cr, Mr);
        cH = (cl*wl + cr*wr)/wa;
        RH = wr > wl ? Mr : Ml;
      }
      if(wh > 0.0005){
        vec3 hp; mat3 hf;
        PROPS_HEAD_FN(hp, hf);
        vec3 cW = hp + hf*offs;
        mat3 RW = hf*prEuler(tr);                                  // worn: the head frame IS its frame (no home turn)
        float f = wh/max(wa + wh, 1e-4);
        cH = wa > 0.0005 ? mix(cH, cW, f) : cW;
        RH = wa > 0.0005 ? prNlerp(RH, RW, f) : RW;
      }
      cH = rootFwd(cH); RH = gRootR*RH;                            // held / worn ride the whole-character root
      c = mix(c, cH, w);
      R = prNlerp(R, RH, w);
    }
    gPrC[k] = c; gPrM[k] = transpose(R);
    gPrRest[k] = (1.0 - w)*(1.0 - smoothstep(0.004, 0.03, abs(offs.y)));
  }
#endif
  propsBoundInit();
}

// which slots a ray can meet (bit k): the character passes set it per camera ray and per shadow ray, so the SDF
// loops skip every prop the ray never comes near (all slots by default); gPrT0/gPrT1 = where the ray is inside any
// masked prop's box.  gPrOn = false: the character's SDF without the props (its fur march: props are plain solids
// marched on their own, engine/scene_main.glsl PASS_CHAR)
int gPrMask = 0xFFFF;
// perf: the slots whose bounding sphere comes within r of p (bit k).  AO taps within r/2 of p see exactly the
// same distances with this mask as with every slot (a slot further off adds neither a model nor any occlusion)
int propsNearMask(vec3 p, float r){
  int msk = 0;
#if PROPS_N > 0
  for(int k=0;k<PR_LOOP_N;k++)
    if(length(p - PRC(k)) - prBnd(k)*PR_SIZE[k] <= r) msk |= 1 << k;
#endif
  return msk;
}
bool gPrOn = true;
float gPrT0 = 0.0, gPrT1 = 1e3;
int propsRayMask(vec3 ro, vec3 rd, float t1, float pad){
  int msk = 0;
  gPrT0 = 1e9; gPrT1 = -1e9;
#if PROPS_N > 0
  for(int k=0;k<PR_LOOP_N;k++){
    // the ray against the prop's own box (its unit frame, a conservative box + the pad)
    float sz = PR_SIZE[k];
    bool art = PRART(k) > 0.001;
    vec3 bc = art ? PR_BBC2[k] : PR_BBC[k], bh = art ? PR_BBH2[k] : PR_BBH[k];
    // perf: a sphere round the padded box first (most rays miss most props: a dot product, not a box test)
    vec3 oc = ro - PRC(k);
    float R = (length(bc) + length(bh))*sz + pad*1.7320508, b = dot(oc, rd), disc = b*b - dot(oc, oc) + R*R;
    if(disc < 0.0 || sqrt(disc) < b) continue;      // (missed, or wholly behind the ray)
    mat3 M = PRM(k);
    vec3 o = M*oc/sz - bc, d = M*rd;
    vec3 e = bh + vec3(pad/sz);
    vec3 inv = 1.0/(abs(d) + 1e-6)*sign(d + 1e-7);
    vec3 ta = (-e - o)*inv, tb = (e - o)*inv;
    vec3 tn = min(ta, tb), tf = max(ta, tb);
    float t0 = max(max(tn.x, tn.y), tn.z), t1b = min(min(tf.x, tf.y), tf.z);
    if(t1b >= max(t0, 0.0) && t0 < t1/sz){ msk |= 1 << k; gPrT0 = min(gPrT0, max(t0, 0.0)*sz); gPrT1 = max(gPrT1, t1b*sz); }
  }
#endif
  return msk;
}
#if PROPS_N > 0
int gPrId;          // slot of the nearest prop
float gPrPart;      // its part
vec3 gPrQ;          // unit-frame point
// pad: how near a prop's bounding sphere a point must be before its model runs (0.03 surfaces, 0.10 shadow rays)
float propsDistB(vec3 p, float pad){
  float d = 1e3;
  gPrId = -1;
  // perf: only the mask's set bits, lowest first (the slot order of a plain k loop): a ray that meets no prop
  // costs nothing here however many slots the props file has
  for(int mk = gPrMask & ((1 << PR_LOOP_N) - 1); mk != 0; ){
    int low = mk & (-mk);
    mk ^= low;
    int k = int(log2(float(low)) + 0.5);
    vec3 v = p - PRC(k);
    float s = PR_SIZE[k];
    float bnd = length(v) - prBnd(k)*s;
    if(bnd > pad){ d = min(d, bnd); continue; }      // bounding sphere: a far prop costs one length()
    vec3 q = PRM(k)*v/s;
    float m;
    float dk = propUnit(k, q, m)*s;
    if(dk < d){ d = dk; gPrId = k; gPrPart = m; gPrQ = q; }
  }
  return d;
}
float propsDist(vec3 p){ return propsDistB(p, 0.03); }
// the props alone along a ray (plain solids, the masked slots): the hit distance or 1e9.
// perf: the march and the hit's four normal taps share ONE propsDist call site (every call site inlines every
// prop model: the shader's size and registers cost every pixel of the character pass, prop or not), and the hit's
// slot / part / unit point / normal are kept (gPrHit*) for the painter, so it needs no SDF call of its own.  Same
// numbers as propsDist at the hit + propsNormal(hit).
int gPrHitId = -1;
float gPrHitPart = 0.0;
vec3 gPrHitQ = vec3(0.0), gPrHitN = vec3(0.0, 1.0, 0.0);
float propsMarch(vec3 ro, vec3 rd, float t){
  gPrHitId = -1;
  if(gPrMask == 0) return 1e9;
  t = max(t, gPrT0);
  int tap = -1, steps = 0;           // tap -1: marching; 0..3: the normal's taps round the hit
  vec3 p0 = vec3(0.0), n = vec3(0.0);
  for(int i=0;i<44;i++){
    vec3 k = 0.5773*(2.0*vec3(float(((tap+3)>>1)&1), float((tap>>1)&1), float(tap&1)) - 1.0);
    float h = propsDist(tap < 0 ? ro + rd*t : p0 + k*0.0012);
    if(tap < 0){
      if(h < max(0.0006, 0.0002*t)){                // ~a sixth of a 270-line pixel at the camera's distance
        gPrHitId = gPrId; gPrHitPart = gPrPart; gPrHitQ = gPrQ;
        p0 = ro + rd*t; tap = 0;
        continue;
      }
      t += h;
      steps++;
      if(t > gPrT1 || steps >= 40) break;
    } else {
      n += k*h;
      tap++;
      if(tap == 4){ gPrHitN = normalize(n); return t; }
    }
  }
  gPrHitId = -1;
  return 1e9;
}
vec3 propsNormal(vec3 p){
  vec3 n = vec3(0.0);
  for(int i=0;i<4;i++){            // one call site (a loop), not four inlined copies of every model
    vec3 k = 0.5773*(2.0*vec3(float(((i+3)>>1)&1), float((i>>1)&1), float(i&1)) - 1.0);
    n += k*propsDist(p + k*0.0012);
  }
  return normalize(n);
}
// shadow rays: the same models (soft shadows march few steps near them); perf: the same function as propsDist
float propsShadow(vec3 p){ return propsDistB(p, 0.10); }
// the live background pass reads the props' poses through PRC/PRM (PROPS_TEX), so it never runs the character's init
void propsFromPose(){}
// the props' sun shadow and contact occlusion on the SET (the live background pass: props are never baked).
// p = a set point (from the background depth).  Only points near a prop or whose sun ray passes one do any work.
vec2 propsOnSet(vec3 p){
  // per prop, only where it matters: contact AO within 1.5 bounds of it, and a short soft-shadow march of THAT prop
  // alone through the stretch of the sun ray inside its bounding sphere.  perf: the AO taps and the shadow
  // steps share one model call site (one loop: taps first, then the march) — the same numbers as two loops
  float sh = 1.0, ao = 1.0;
  for(int k=0;k<PR_LOOP_N;k++){
    vec3 oc = p - PRC(k);
    float s = PR_SIZE[k];
    float R = prBnd(k)*s;
    float m;
    float o2 = dot(oc, oc);
    bool doA = o2 < 2.25*R*R;
    float b = dot(oc, uSunDir), c = o2 - R*R, disc = b*b - c;
    float sq = sqrt(max(disc, 0.0)), t1 = -b + sq;
    bool doS = disc > 0.0 && t1 > 0.0;
    if(!doA && !doS) continue;
    float t = max(-b - sq, 0.02), occ = 0.0;
    int i = doA ? 0 : 3, iS = 0;
    for(int it=0; it<19; it++){
      bool A = i < 3;
      float h = 0.025 + 0.07*float(i);
      float dk = propUnit(k, PRM(k)*(A ? oc + vec3(0.0, h, 0.0) : oc + uSunDir*t)/s, m)*s;
      if(A){
        occ += max(h - dk, 0.0)/h*(0.55 - 0.15*float(i));
        i++;
        if(i == 3){ ao = min(ao, 1.0 - 0.6*clamp(occ, 0.0, 1.0)); if(!doS) break; }
      } else {
        sh = min(sh, 8.0*dk/t);
        t += clamp(dk, 0.015, 0.4);
        iS++;
        if(sh < 0.01 || t > t1 || iS >= 16) break;
      }
    }
  }
  sh = clamp(sh, 0.0, 1.0); sh = sh*sh*(3.0 - 2.0*sh);
  return vec2(sh, ao);
}

// ------------------------------------------------------------------ the painter
void prPlastic(inout Surf s, vec3 a){ s.alb = a; s.spec = 0.55; s.shin = 70.0; s.wrap = 0.18; s.sheen = 0.25; s.mirror = 0.05; }
void prGlaze(inout Surf s, vec3 a){ s.alb = a; s.spec = 0.9; s.shin = 140.0; s.wrap = 0.12; s.sheen = 0.20; s.mirror = 0.14; }
void prMatte(inout Surf s, vec3 a){ s.alb = a; s.spec = 0.06; s.shin = 12.0; s.wrap = 0.30; s.sheen = 0.35; s.mirror = 0.0; }
void prWaxy(inout Surf s, vec3 a){ s.alb = a; s.spec = 0.45; s.shin = 45.0; s.wrap = 0.30; s.sheen = 0.30; s.mirror = 0.03; }
void prWood(inout Surf s, vec3 a, vec3 q){
  float g = vnoise3(q*vec3(2.0, 2.0, 26.0))*0.6 + vnoise3(q*vec3(5.0, 40.0, 5.0))*0.4;
  s.alb = a*(0.80 + 0.32*g); s.spec = 0.20; s.shin = 36.0; s.wrap = 0.18; s.sheen = 0.2; s.mirror = 0.02;
}

// the painter of slot k at unit point q, part m (propsSurface: the nearest prop at p; the character pass paints its
// march's hit, gPrHit*, with no SDF call)
void propsPaint(int k, vec3 q, float m, vec3 p, inout Surf s){
  if(k < 0) return;
  int kind = PR_KIND[k];
  vec3 c1 = PR_COL[k], c2 = PR_COL2[k];
  vec3 n = s.n;
  s.fur = 0.0; s.trans = 0.0; s.emit = vec3(0.0); s.occ = 1.0; s.T = vec3(0.0, 1.0, 0.0);
  vec3 nl = PRM(k)*n;
  float g = vnoise3(q*9.0);
  if(kind == PK_BALL){
    float band = smoothstep(0.30, 0.27, abs(q.y));
    float line = smoothstep(0.035, 0.015, abs(abs(q.y) - 0.30));
    prPlastic(s, mix(mix(c1, c2, band), vec3(0.85, 0.82, 0.76), line)*(0.96 + 0.06*g));
  } else if(kind == PK_MUG){
    if(m > 1.5){ prGlaze(s, c2*0.92); s.occ = mix(0.45, 1.0, smoothstep(-0.30, 0.55, q.y)); }
    else {
      vec2 h = (q.xy - vec2(0.0, -0.02))*vec2(1.0, 1.15);
      h.x = abs(h.x);
      float heart = min(length(h - vec2(0.10, 0.06)) - 0.13, max(dot(h, vec2(0.72, 0.69)) - 0.12, -h.y - 0.24 + 0.9*h.x));
      vec3 a = c1;
      if(q.z > 0.3 && m < 0.5) a = mix(a, c2, smoothstep(0.012, -0.012, heart));
      prGlaze(s, a);
    }
  } else if(kind == PK_BOOK){
    if(m > 0.5){
      // pages: lines of print on an open spread, the page edges on a shut book
      float spread = step(0.3, PRART(k))*step(0.7, abs(nl.z));
      float print = step(0.55, fract(q.y*13.0))*step(abs(q.y), 0.70)*step(0.10, abs(q.x - (-0.62)))*step(abs(q.x + 0.62), 1.30);
      float lines = mix(0.86 + 0.10*sin((q.z + q.x)*260.0), 1.0 - 0.45*print, spread);
      prMatte(s, vec3(0.90, 0.87, 0.78)*lines);
    } else {
      vec3 a = c1*(0.92 + 0.10*vnoise3(q*40.0));
      float plate = step(abs(q.x - 0.06), 0.36)*step(abs(q.y - 0.36), 0.15)*step(0.1, q.z);
      float bands = step(q.x, -0.44)*(step(abs(q.y - 0.55), 0.04) + step(abs(q.y + 0.55), 0.04));
      a = mix(a, c2, clamp((plate*(1.0 - PRART(k)) + bands), 0.0, 1.0));
      prMatte(s, a);
    }
  } else if(kind == PK_DUCK){
    if(m > 1.5) prGlaze(s, vec3(0.015)); else if(m > 0.5) prPlastic(s, vec3(0.90, 0.32, 0.03)); else prPlastic(s, c1);
  } else if(kind == PK_APPLE){
    if(m > 1.5) prWaxy(s, vec3(0.12, 0.38, 0.05));
    else if(m > 0.5) prMatte(s, vec3(0.18, 0.10, 0.04));
    else {
      float st = vnoise3(vec3(atan(q.z, q.x)*3.0, q.y*14.0, 0.0));
      prWaxy(s, mix(c1, c2, 0.45*smoothstep(0.35, 0.9, st) + 0.25*smoothstep(0.3, -0.7, q.y))*(0.92 + 0.12*vnoise3(q*30.0)));
      s.spec = 0.65; s.shin = 55.0;
    }
  } else if(kind == PK_PLANT){
    if(m > 1.5) prMatte(s, vec3(0.10, 0.065, 0.04)*(0.8 + 0.4*g));
    else if(m > 0.5){ prWaxy(s, c2*(0.85 + 0.3*g)); s.trans = 0.5; s.transCol = c2; }
    else { prMatte(s, c1*(0.88 + 0.16*vnoise3(q*14.0))); s.spec = 0.10; }
  } else if(kind == PK_HEADPHONES){
    if(m > 1.5){ prGlaze(s, c2); s.spec = 1.1; s.mirror = 0.25; }                 // band + yokes: satin metal
    else if(m > 0.5){ prMatte(s, c1*0.45 + vec3(0.03)); s.sheen = 0.6; }          // cushions: soft
    else prPlastic(s, c1);
  } else if(kind == PK_CHAIR){
    if(m > 0.5){ prMatte(s, c2*(0.90 + 0.14*vnoise3(q*30.0))); s.sheen = 0.55; s.wrap = 0.35; }   // upholstery
    else prWood(s, c1, q);
  } else if(kind == PK_DESK){
    if(m > 1.5) prWood(s, c1*0.88, q.zyx);
    else if(m > 0.5){ prGlaze(s, vec3(0.80, 0.70, 0.40)); s.mirror = 0.3; }
    else prWood(s, c1, q.y > 0.58 ? q.zxy*vec3(1.0, 1.0, 0.4) : q);
  } else if(kind == PK_LAPTOP){
    prPlastic(s, c1); s.spec = 0.8; s.shin = 90.0; s.mirror = 0.12;               // aluminium-ish shell
    float open = PRART(k);
    if(m > 3.5){ prGlaze(s, vec3(0.03)); }
    else if(m > 2.5){ prGlaze(s, c1*0.85); s.spec = 0.3; }
    else if(m > 1.5){
      // the screen (lid frame: x across, z from the hinge up the lid): a soft glow with a simple UI when open
      vec3 lq = laptopLid(q, open);
      vec2 u = vec2(lq.x/0.92, (lq.z - 0.02)/0.58);       // -1..1 across the panel, +1 = its top edge
      vec3 bg = mix(vec3(0.22, 0.36, 0.64), vec3(0.55, 0.70, 0.95), 0.5 + 0.5*u.y);
      vec3 col = mix(bg, vec3(0.92, 0.94, 0.98), step(0.84, u.y)*0.85);
      float win = step(abs(u.x + 0.10), 0.72)*step(abs(u.y + 0.06), 0.66);
      col = mix(col, vec3(0.96, 0.96, 0.97), win);
      float row = floor(u.y*7.0);
      float txt = win*step(abs(u.x + 0.22), 0.50)*step(0.50, fract(u.y*7.0))*step(u.y, 0.48)*step(-0.62, u.y);
      txt *= step(u.x + 0.72, 0.95*fract(sin(row*12.9898)*43758.5) + 0.35);
      col = mix(col, vec3(0.35, 0.40, 0.50), txt*0.8);
      float pic = step(abs(u.x - 0.42), 0.13)*step(abs(u.y - 0.26), 0.16);
      col = mix(col, vec3(0.95, 0.55, 0.45), pic);
      float on = smoothstep(0.35, 0.7, open);
      s.alb = mix(vec3(0.02, 0.025, 0.03), col*0.20, on);
      s.emit = col*0.60*on;
      s.spec = 0.9; s.shin = 160.0; s.mirror = 0.15; s.wrap = 0.0; s.sheen = 0.0;
    } else if(m > 0.5){
      vec2 kq = vec2((q.x + 0.88)/0.1467, (q.z + 0.30)/0.1500);
      vec2 kc = fract(kq) - 0.5;
      float key = smoothstep(0.45, 0.36, max(abs(kc.x), abs(kc.y)));
      prPlastic(s, mix(vec3(0.05), vec3(0.17, 0.17, 0.18), key)); s.spec = 0.3;
    }
#ifdef PROPKIND_HAT
  } else if(kind == PK_HAT){
    if(m > 0.5){ prMatte(s, c2*(0.92 + 0.10*g)); s.sheen = 0.45; s.spec = 0.18; s.shin = 30.0; }     // ribbon: satin
    else {
      // woven straw: an over-under weave round the crown and out along the brim
      float r = length(q.xz), an = atan(q.z, q.x);
      float u = an*26.0, v = (r < 0.57 ? q.y : r)*30.0;
      float wv = sin(u)*sin(v*3.14159), ch = smoothstep(-0.4, 0.4, wv);
      float fib = vnoise3(vec3(u*0.7 + ch*7.0, v*4.0, 0.0));
      prMatte(s, c1*(0.86 + 0.12*ch + 0.12*fib)*(0.94 + 0.10*g));
      s.spec = 0.12; s.shin = 18.0; s.sheen = 0.30;
      s.occ = mix(0.75, 1.0, smoothstep(0.0, 0.35, abs(wv)));
      if(r < 0.545 && nl.y < -0.2) s.occ *= 0.55;                     // inside the crown
    }
#endif
#ifdef PROPKIND_GLASSES
  } else if(kind == PK_GLASSES){
    if(m > 0.5){
      // dark tinted lens: a glossy mirror-ish face with a soft blue-green tint toward the top
      vec3 l = vec3(abs(q.x) - 0.46, q.yz);
      prGlaze(s, mix(c1*0.6, c2*0.35, smoothstep(-0.25, 0.25, l.y)));
      s.spec = 1.4; s.shin = 160.0; s.mirror = 0.45; s.wrap = 0.0; s.sheen = 0.1;
      float glint = smoothstep(0.05, 0.0, abs(l.x + l.y*0.9 + 0.08))*smoothstep(0.24, 0.10, length(l.xy));
      s.emit = c2*0.10*glint;
    } else { prPlastic(s, c1); s.spec = 0.8; s.shin = 90.0; }
#endif
#ifdef PROPKIND_BOX
  } else if(kind == PK_BOX){
    vec3 bq = m > 0.5 && m < 2.5 ? boxLid(q, PRART(k)) : q;
    if(m > 1.5 && m < 2.5){ prMatte(s, c2*(0.95 + 0.06*g)); s.sheen = 0.5; s.spec = 0.25; }        // the bow: satin
    else if(m > 2.5){ prMatte(s, mix(c1*0.55, vec3(0.70, 0.62, 0.50), 0.55)*(0.9 + 0.1*g));         // inside: card
      s.occ = m > 3.5 ? 0.8 : mix(0.45, 1.0, smoothstep(-0.82, 0.30, q.y)); }
    else {
      // wrapping paper (faint diagonal stripes) crossed by a white ribbon: round the body's faces and over the lid
      float stripe = 0.5 + 0.5*sin((bq.x + bq.y + bq.z)*18.0);
      vec3 paper = c1*(0.90 + 0.08*stripe)*(0.96 + 0.06*g);
      vec3 bn = nl;
      if(m > 0.5) bn.yz = rot(1.92*PRART(k))*bn.yz;                   // the lid's normal in its own (shut) frame
      float rib = max(step(abs(bq.x), 0.13)*(1.0 - step(0.6, abs(bn.x))), step(abs(bq.z), 0.13)*(1.0 - step(0.6, abs(bn.z))));
      if(m < 0.5 && bn.y < -0.6) rib = 0.0;                            // not under the box
      prMatte(s, mix(paper, c2, rib)); s.spec = mix(0.15, 0.35, rib); s.sheen = mix(0.25, 0.5, rib);
      if(m < 0.5) s.occ = mix(0.75, 1.0, smoothstep(0.26, 0.20, q.y));   // the lid's shade on the body below it
    }
#endif
#ifdef PROPKIND_WATERING_CAN
  } else if(kind == PK_WATERING_CAN){
    if(m > 1.5){                                                         // water: clear, blue-ish, glinting
      prGlaze(s, vec3(0.30, 0.52, 0.80)); s.spec = 1.6; s.shin = 200.0; s.mirror = 0.35; s.wrap = 0.4;
      s.trans = 0.8; s.transCol = vec3(0.55, 0.80, 1.0); s.emit = vec3(0.05, 0.09, 0.14);
    } else if(m > 0.5){ prGlaze(s, c2*0.85); s.spec = 1.0; s.mirror = 0.22; }                         // handle, rose
    else {
      // painted metal: a few seams round the body, a little wear
      float seam = smoothstep(0.012, 0.0, abs(fract((q.y + 0.72)*3.4) - 0.5) - 0.48)*step(q.x, 0.35);
      prGlaze(s, c1*(0.92 + 0.10*vnoise3(q*14.0))*(1.0 - 0.18*seam)); s.spec = 0.75; s.shin = 80.0; s.mirror = 0.10;
    }
#endif
#ifdef PROPKIND_BLOCK
  } else if(kind == PK_BLOCK){
    // painted wood; the slot's letter (PR_VAR) in rgb2, raised a little (a darker rim round it), on every face
    vec3 aq = abs(q);
    vec2 uv = aq.z >= max(aq.x, aq.y) ? vec2(q.x*sign(q.z), q.y) : (aq.x >= aq.y ? vec2(-q.z*sign(q.x), q.y) : vec2(q.x, -q.z*sign(q.y)));
    float ld = prLetter(uv/0.80, PR_VAR[k]);
    float edge = smoothstep(0.62, 0.72, max(abs(uv.x), abs(uv.y)));
    vec3 a = c1*(0.86 + 0.18*vnoise3(q*vec3(3.0, 24.0, 3.0)))*(1.0 - 0.10*edge);
    a = mix(a, c2, smoothstep(0.03, -0.01, ld));
    a *= 1.0 - 0.35*smoothstep(0.05, 0.0, abs(ld - 0.02));                       // the letter's raised rim
    prWood(s, vec3(1.0), q); s.alb = a; s.spec = 0.30; s.shin = 40.0;
#endif
  } else if(kind == PK_CHALK){          // scene: dusty, matte, a touch of powder on the worn tip
    prMatte(s, c1*(0.93 + 0.10*vnoise3(q*vec3(14.0, 4.0, 14.0))));
    s.spec = 0.02; s.sheen = 0.15; s.wrap = 0.35;
  } else if(kind == PK_ERASER){
    if(m > 0.5){ prMatte(s, c2*(0.85 + 0.25*vnoise3(q*vec3(30.0, 6.0, 30.0)))); s.sheen = 0.55; }   // felt (chalk-dusted)
    else prWood(s, c1, q.zxy*vec3(1.0, 1.0, 0.5));
  } else {
    prPlastic(s, c1);
  }
  // resting on its place: the contact shade under it (the set's baked AO cannot see a movable object)
  float rest = PRREST(k);
  if(rest > 0.0 && PR_FREE[k] == 0){
    float lowY = (p.y - (PRC(k).y - 0.97*PR_EXT[k].y*PR_SIZE[k]))/PR_SIZE[k];
    float ao = mix(0.45, 1.0, smoothstep(0.0, 0.45, lowY))*mix(0.6, 1.0, smoothstep(-0.85, 0.1, n.y));
    s.occ *= mix(1.0, ao, rest);
  }
}
void propsSurface(vec3 p, vec3 rd, inout Surf s){
  propsDist(p);
  propsPaint(gPrId, gPrQ, gPrPart, p, s);
}
#endif   // PROPS_N > 0

#if PROPS_N == 0
// no props file (the 11 shipped characters): the character pass's prop hooks are constants the compiler folds away
float propsDist(vec3 p){ return 1e3; }
float propsMarch(vec3 ro, vec3 rd, float t){ return 1e9; }
vec3 propsNormal(vec3 p){ return vec3(0.0, 1.0, 0.0); }
#endif

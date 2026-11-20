#ifdef P_FX_BLUSH
#define FXV_BLUSH uP[P_FX_BLUSH]
#else
#define FXV_BLUSH 0.0
#endif
#ifdef P_FX_GREEN
#define FXV_GREEN uP[P_FX_GREEN]
#else
#define FXV_GREEN 0.0
#endif
#ifdef P_FX_PALE
#define FXV_PALE uP[P_FX_PALE]
#else
#define FXV_PALE 0.0
#endif
#ifdef P_FX_SWEAT
#define FXV_SWEAT uP[P_FX_SWEAT]
#else
#define FXV_SWEAT 0.0
#endif
#ifdef P_FX_TEAR
#define FXV_TEAR uP[P_FX_TEAR]
#else
#define FXV_TEAR 0.0
#endif
#ifdef P_FX_VEIN
#define FXV_VEIN uP[P_FX_VEIN]
#else
#define FXV_VEIN 0.0
#endif
// ---------------------------------------------------------------------------
// engine/mascot_body.glsl — the second half of a mascot on the muse skeleton (see engine/mascot_rig.glsl).
//  Included LAST by animals/<name>.glsl, after the character defined its hooks:
//
//    float mascotHead(vec3 q)                   the head host, head space (ears / crest / hat joined in)
//    float mascotTorso(vec3 p)                  the trunk, character-local (gMHip, localToUpper(), gBreath)
//    void  mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s)
//                                               albedo + material by gMPart (0 head 1 body 2 arm 3 hand 4 leg 5 foot
//                                               7 extra), gMSide, gMLimbT, gMPalmW, gMSoleW; r = rest-pose point
//  optional (define the macro, then the function):
//    MR_EXTRA       float mascotExtra(vec3 p, out vec3 rest)   one more solid, character-local: a tail (part 7)
//    MR_HEAD_EXTRA  float mascotHeadBare(vec3 q)               a hard solid on the head that the lips never carve (a hat brim)
//    MR_FACE_POST   void mascotFacePost(vec3 q, vec3 n, inout Surf s)   paint over the eyes / lips (a monocle)
//    MR_INIT_HOOK   void mascotInit()                          pose of the extras, end of charInit
//    MR_CUSTOM_BEND void mascotMouthBend()                     the resting mouth line
//  Shape numbers (defaults = the muse's):
// ---------------------------------------------------------------------------
#ifndef MR_ARM_R
#define MR_ARM_R vec3(0.115, 0.098, 0.088)       // arm radius at the shoulder, elbow, wrist
#endif
#ifndef MR_LEG_R
#define MR_LEG_R vec3(0.140, 0.122, 0.112)       // leg radius at the hip, knee, ankle
#endif
#ifndef MR_FOOT_C
#define MR_FOOT_C vec3(0.0, -0.050, 0.070)       // foot ellipsoid (foot frame, origin the ankle)
#endif
#ifndef MR_FOOT_R
#define MR_FOOT_R vec3(0.135, 0.100, 0.200)
#endif
#ifndef MR_PALM_R
#define MR_PALM_R vec3(0.112, 0.095, 0.064)
#endif
#ifndef MR_NFING
#define MR_NFING 4                               // fingers besides the thumb (3 or 4)
#endif
#ifndef MR_FING_R
#define MR_FING_R 0.029
#endif
#ifndef MR_FING_L
#define MR_FING_L 1.0                            // finger length, x the muse's
#endif
#ifdef MR_FING_PAD                               // defined: a round pad of this radius on every fingertip (a gekkie)
#define MR_FING_PAD_ON 1
#else
#define MR_FING_PAD_ON 0
#define MR_FING_PAD 0.0
#endif
#ifndef MR_THUMB_R
#define MR_THUMB_R 0.037
#endif
#ifndef MR_K_NECK
#define MR_K_NECK 0.16                           // smooth joins: head-trunk, trunk-arm, trunk-leg, arm-hand, leg-foot
#endif
#ifndef MR_K_ARM
#define MR_K_ARM 0.07
#endif
#ifndef MR_K_LEG
#define MR_K_LEG 0.08
#endif
#ifndef MR_K_HAND
#define MR_K_HAND 0.05
#endif
#ifndef MR_K_FOOT
#define MR_K_FOOT 0.06
#endif
#ifndef MR_K_EXTRA
#define MR_K_EXTRA 0.08
#endif
#ifndef MR_HEAD_BOUND
#define MR_HEAD_BOUND 0.92                       // the head's bounding sphere (head space, about the origin)
#endif
#ifndef MR_FACE_C
#define MR_FACE_C vec3(0.0, -0.05, 0.5)          // only points this near carry the eyes / brows / lips work
#endif
#ifndef MR_FACE_RAD
#define MR_FACE_RAD 0.55
#endif
#ifndef MR_CREASE_OCC
#define MR_CREASE_OCC 0.40                       // how dark the smooth joins (neck, armpits, hips) shade
#endif
#ifndef MR_CHEEK_R
#define MR_CHEEK_R 0.10
#endif
#ifndef MR_SHADOW_HEAD
#define MR_SHADOW_HEAD (M_HEADR*0.97)            // the head's shadow proxy (ellipsoid radii)
#endif

#ifdef MR_AO_ARM_PAD
bool gMAOPass = false;                                 // gekkie: true while engine/scene_main.glsl's charAO evaluates
#endif
float museArm(vec3 p, int i, out float tt){
#ifdef MR_ARM_FN
  return MR_ARM_FN(p, i, tt);                         // a character's own arm shape (Flurry's kinked twigs)
#endif
  vec3 a = gMSh[i], b = gMEl[i], c = gMWr[i];
  vec3 pa = p - a, ba = b - a;
  float h1 = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  float d1 = length(pa - ba*h1) - mix(MR_ARM_R.x, MR_ARM_R.y, h1);
  vec3 pb = p - b, cb = c - b;
  float h2 = clamp(dot(pb, cb)/dot(cb, cb), 0.0, 1.0);
  float d2 = length(pb - cb*h2) - mix(MR_ARM_R.y, MR_ARM_R.z, h2);
  tt = d1 < d2 ? 0.5*h1 : 0.5 + 0.5*h2;
#ifdef MR_RING_ARM
  d1 -= MR_RING_ARM(h1); d2 -= MR_RING_ARM(h2);          // stacked rings along the limb (a ring man)
  return smin(d1, d2, 0.02);
#else
  return smin(d1, d2, 0.05);
#endif
}
// the hand: palm pad + MR_NFING fingers + a thumb (the muse's mitten, its digits' curl logic)
float museHand(vec3 p, int i){
  mat3 F = gMHand[i];
  vec3 q = transpose(F)*(p - gMPalm[i]);              // x across, y to the fingertips, z out of the palm
  float s = i == 0 ? -1.0 : 1.0;
  float ts = -s;                                      // thumb side along x (mirror-symmetric hands)
#ifdef MR_HAND_SCALE
  q /= MR_HAND_SCALE;                                 // gekkie: a smaller, finer hand (distances scaled back below)
#endif
  float d = sdEllipsoid(q - vec3(0.0, -0.005, -0.012), MR_PALM_R);
  float sp = gMSpread[i];
#ifdef MR_SPREAD
  sp = MR_SPREAD(sp);                                 // a character's own finger splay (the gekkie fans them wide)
#endif
  for(int k=0;k<MR_NFING;k++){
    float c = gMCurl[i*5 + k];
#if MR_NFING == 3
    float fk = float(k)*1.5;
#else
    float fk = float(k);
#endif
    vec3 base = vec3(ts*(0.070 - 0.047*fk), 0.062 - 0.010*abs(fk - 1.5), -0.008);
    float len = (0.105 - 0.014*abs(fk - 1.2))*MR_FING_L;
    vec3 dir = normalize(vec3(ts*sp*(0.6 - 0.4*fk)*0.35, cos(c*1.45), sin(c*1.45)));     // curls toward the palm (+z)
    vec3 mid = base + dir*len*0.55;
    vec3 dir2 = normalize(vec3(dir.x, cos(c*2.7), sin(c*2.7)));
    vec3 tip = mid + dir2*len*0.5;
    float f = sdCapsule(q, base, mid, MR_FING_R);
    f = smin(f, sdCapsule(q, mid, tip, MR_FING_R*0.93), 0.010);
#if MR_FING_PAD_ON
    f = smin(f, length(q - tip - dir2*MR_FING_PAD*0.25) - MR_FING_PAD, 0.012);
#endif
    d = smin(d, f, 0.010);
  }
  float c = gMCurl[i*5 + 4];
  vec3 tb = vec3(ts*0.085, -0.030, -0.004);
  vec3 tdir = normalize(mix(vec3(ts*0.75, 0.60, 0.28), vec3(-ts*0.45, 0.35, 0.82), c));   // folds across the palm
  vec3 tt = tb + tdir*0.095*MR_FING_L;
  d = smin(d, sdCapsule(q, tb, tt, MR_THUMB_R), 0.016);
#if MR_FING_PAD_ON
  d = smin(d, length(q - tt - tdir*MR_FING_PAD*0.25) - MR_FING_PAD, 0.012);
#endif
  gMPalmW = smoothstep(0.012, -0.02, q.z + 0.045)*smoothstep(0.10, 0.05, length(q.xy));
#ifdef MR_HAND_SCALE
  d *= MR_HAND_SCALE;
#endif
  return d;
}
float museLeg(vec3 p, int i, out float tt){
  vec3 a = gMHipJ[i], b = gMKn[i], c = gMAnk[i];
  vec3 pa = p - a, ba = b - a;
  float h1 = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  float d1 = length(pa - ba*h1) - mix(MR_LEG_R.x, MR_LEG_R.y, h1);
  vec3 pb = p - b, cb = c - b;
  float h2 = clamp(dot(pb, cb)/dot(cb, cb), 0.0, 1.0);
  float d2 = length(pb - cb*h2) - mix(MR_LEG_R.y, MR_LEG_R.z, h2);
  tt = d1 < d2 ? 0.5*h1 : 0.5 + 0.5*h2;
#ifdef MR_RING_LEG
  d1 -= MR_RING_LEG(h1); d2 -= MR_RING_LEG(h2);
  return smin(d1, d2, 0.02);
#else
  return smin(d1, d2, 0.06);
#endif
}
#ifndef MR_FOOT_CUSTOM
float museFoot(vec3 p, int i, out float sole){
  vec3 q = transpose(gMFoot[i])*(p - gMAnk[i]);
  float d = sdEllipsoid(q - MR_FOOT_C, MR_FOOT_R);
  d = smax(d, -(q.y + M_ANK_Y - 0.002), 0.05);         // a flat soft sole
  sole = smoothstep(-0.10, -0.13, q.y);
  return d;
}
#endif

// ---- the whole character (local space)
void museEval(vec3 p){
  gMPart = 1; gMSoleW = 0.0; gMPalmW = 0.0; gMFace = 0.0; gMSide = 0; gMLimbT = 0.0;
  gMPanel = 0.0; gMHood = 0.0; gMCrease = 0.0;
#ifdef MR_HEAD_SCALE
  vec3 q = gHR*(p - gHO)/MR_HEAD_SCALE;                // gekkie: the whole head (face, eyes, lips) built at 1/scale
#else
  vec3 q = gHR*(p - gHO);
#endif
  gQ = q;
  float hb = length(q) - MR_HEAD_BOUND;
  float head, bare = 1e3;
  if(hb < 0.25){
    float host = mascotHead(q);
    float skin = host;
    if(length(q - MR_FACE_C) < MR_FACE_RAD){            // only the face carries the face's work
      gMFace = 1.0;
      skin = lipsEval(q, host);
      skin = faceEval(q, skin, host);
      bare = min(gLipBare, gFEye);
    } else { gFEye = 1e3; gFBrow = 1e3; gFLid = 1e3; gFOffLips = 1.0; }
#ifdef MR_HEAD_EXTRA
    bare = min(bare, mascotHeadBare(q));
#endif
    head = skin;
  } else {
    head = hb + 0.05; gFEye = 1e3; gFBrow = 1e3; gFLid = 1e3; gFOffLips = 1.0;
  }
#ifdef MR_HEAD_SCALE
  head *= MR_HEAD_SCALE; bare *= MR_HEAD_SCALE;
  gFEye *= MR_HEAD_SCALE; gFLid *= MR_HEAD_SCALE; gFBrow *= MR_HEAD_SCALE;   // face.glsl's paint tolerances are world units
#endif
#ifdef MR_BARE_EXTRA
  bare = min(bare, mascotBare(p));                     // a hard worn solid off the head (the gekkie's bow tie), local space
#endif
  gMHeadD = head;
  float body = mascotTorso(p);
  float d = smin(head, body, MR_K_NECK);
  gMCrease = clamp((min(head, body) - d)*25.0, 0.0, 1.0);
  vec3 rest = q;
  if(body < head){ gMPart = 1; rest = localToUpper(p) + vec3(3.0, 0.0, 0.0); }
  else gMPart = 0;
  float best = d;
  // arms + hands
  for(int i=0;i<2;i++){
    float tt;
    float a = museArm(p, i, tt);
#ifdef MR_AO_ARM_PAD
    if(gMAOPass) a += MR_AO_ARM_PAD;                   // gekkie: the AO pass sees the arms thinner / farther (no black armpits)
#endif
    float hd = length(p - gMBC[i]) - gMBR[i];
    float hn = hd < best + 0.05 ? museHand(p, i) : hd;
#ifdef MR_AO_ARM_PAD
    if(gMAOPass) hn += MR_AO_ARM_PAD;
#endif
    float limb = smin(a, hn, MR_K_HAND);
    if(limb < best){
      vec3 ax = normalize(gMWr[i] - gMSh[i]);
      gMPart = hn < a ? 3 : 2; gMSide = i; gMLimbT = tt;
      rest = gMPart == 3 ? transpose(gMHand[i])*(p - gMPalm[i])*1.6 + vec3(7.0 + float(i), 0.0, 0.0)
                         : vec3(tt*0.6, dot(p - gMSh[i], mPerp(ax)), dot(p - gMSh[i], cross(ax, mPerp(ax)))) + vec3(5.0 + float(i), 2.0, 0.0);
    }
    float dl = smin(d, limb, MR_K_ARM);
    gMCrease = max(gMCrease, clamp((min(d, limb) - dl)*45.0, 0.0, 1.0));
    d = dl;
    best = min(best, limb);
  }
  // legs + feet
  for(int i=0;i<2;i++){
    float tl;
    float leg = museLeg(p, i, tl);
    float sole;
    float ft = museFoot(p, i, sole);
    float limb = smin(leg, ft, MR_K_FOOT);
    if(limb < best){
      gMPart = ft < leg ? 5 : 4; gMSide = i; gMLimbT = tl;
      gMSoleW = gMPart == 5 ? sole : 0.0;
      vec3 ax = normalize(gMAnk[i] - gMHipJ[i]);
      rest = transpose(gMFoot[i])*(p - gMAnk[i]) + vec3(9.0 + 2.0*float(i), 0.0, 0.0);
      if(gMPart == 4) rest = vec3(dot(p - gMHipJ[i], ax), dot(p - gMHipJ[i], mPerp(ax)), dot(p - gMHipJ[i], cross(ax, mPerp(ax)))) + vec3(13.0 + float(i), 0.0, 0.0);
    }
    float dg = smin(d, limb, MR_K_LEG);
    gMCrease = max(gMCrease, clamp((min(d, limb) - dg)*40.0, 0.0, 1.0));
    d = dg;
    best = min(best, limb);
  }
#ifdef MR_EXTRA
  {
    vec3 er;
    float ex = mascotExtra(p, er);
    if(ex < best){ gMPart = 7; rest = er + vec3(17.0, 0.0, 0.0); }
    float de = smin(d, ex, MR_K_EXTRA);
    gMCrease = max(gMCrease, clamp((min(d, ex) - de)*30.0, 0.0, 1.0));
    d = de;
    best = min(best, ex);
  }
#endif
  gMSkin = d; gMBare = bare; gMRest = rest;
  gMFurW = 0.0;
}
void charEval(vec3 p){
  museEval(p);
  gcBase = min(gMSkin, gMBare - 0.0007); gcBare = gMBare; gcRest = gMRest;
  gcLen = 0.0;
}
float charTuft(){ return 1.0; }

float charShadow(vec3 p){
  vec3 q = gHR*(p - gHO);
#ifdef MR_SHADOW_HEAD_FN
#ifdef MR_HEAD_SCALE
  float d = mascotShadowHead(q/MR_HEAD_SCALE)*MR_HEAD_SCALE;   // ... built at 1/scale like the head itself
#else
  float d = mascotShadowHead(q);                       // a head that is no ellipsoid brings its own inscribed stand-in (head space)
#endif
#else
  float d = sdEllipsoid(q, MR_SHADOW_HEAD);
#endif
#ifdef MR_SHADOW_TORSO
  d = min(d, mascotShadowTorso(p));                    // a smooth stand-in (no self-shadow inside creases)
#else
  d = min(d, mascotTorso(p) + 0.02);
#endif
  for(int i=0;i<2;i++){
#ifdef MR_SHADOW_ARM_R
    // gekkie: thin arm / palm stand-ins (radii x upper, y fore, z palm), so an arm held in by the trunk does not smear it
    d = min(d, sdCapsule(p, gMSh[i], gMEl[i], MR_SHADOW_ARM_R.x));
    d = min(d, sdCapsule(p, gMEl[i], gMWr[i], MR_SHADOW_ARM_R.y));
    d = min(d, length(p - gMPalm[i]) - MR_SHADOW_ARM_R.z);
#else
    d = min(d, sdCapsule(p, gMSh[i], gMEl[i], MR_ARM_R.y*0.92));
    d = min(d, sdCapsule(p, gMEl[i], gMWr[i], MR_ARM_R.z*0.92));
    d = min(d, length(p - gMPalm[i]) - 0.095);
#endif
    d = min(d, sdCapsule(p, gMHipJ[i], gMKn[i], MR_LEG_R.y*0.92));
    d = min(d, sdCapsule(p, gMKn[i], gMAnk[i], MR_LEG_R.z*0.92));
    d = min(d, sdEllipsoid(transpose(gMFoot[i])*(p - gMAnk[i]) - MR_FOOT_C, MR_FOOT_R*0.9));
  }
#ifdef MR_EXTRA
  { vec3 er; d = min(d, mascotExtra(p, er) + 0.01); }
#endif
  return d;
}

// ---- for the props agent (character-local): the muse's, number for number
void museGrasp(int side, out vec3 pos, out mat3 frame){
  int i = side == 0 ? 0 : 1;
  frame = gMHand[i];                                   // x across the palm, y to the fingertips, z out of the palm
  pos = gMPalm[i] + frame[2]*0.070 + frame[1]*0.010;   // where a held thing's grip point sits, just off the palm
}
void museHeadFrame(out vec3 pos, out mat3 frame){
  frame = transpose(gHR);                              // head axes in local space: x right, y up, z out of the face
  pos = gHO + frame*vec3(0.0, 0.02, -0.03);
}

#ifdef MUSE_EMO
// ---- EMO-EXPR cartoon marks, painted on the 3D head (head space q: lit, glossy, animated on uTime).  Paint only:
// nothing here is marched, so they cost one shading call and nothing at all while their params are 0
float museDrop(vec2 p, float r, float h){
  // a drop pointing up: round bottom (radius r about the origin), its tip at y = h
  if(p.y <= 0.0) return length(p) - r;
  if(p.y >= h) return length(p - vec2(0.0, h));
  float t = p.y/h;
  return (abs(p.x) - r*(1.0 - t*t)*(1.0 - 0.30*t))*0.85;
}
void musePaintDrop(inout Surf s, vec2 p, float r, float h, float a, vec3 fill, vec3 rim){
  float d = museDrop(p, r, h);
  float m = smoothstep(0.003, -0.003, d)*a;
  if(m <= 0.0) return;
  vec3 c = mix(fill, rim, 0.85*smoothstep(-0.010, -0.003, d));          // a darker outline just inside the edge
  float hl = smoothstep(0.34*r, 0.16*r, length(p - vec2(-0.32*r, 0.12*r)));
  s.alb = mix(s.alb, c, m);
  s.emit = mix(s.emit, c*0.30 + vec3(1.0)*1.8*hl, m);
  s.n = normalize(mix(s.n, normalize(s.n + transpose(gHR)*vec3(p/(1.4*r), 0.0)), m));     // a little dome
  s.spec = mix(s.spec, 0.9, m); s.shin = mix(s.shin, 90.0, m);
  s.fur *= 1.0 - m; s.sheen *= 1.0 - m; s.occ = mix(s.occ, 1.0, m); s.mirror *= 1.0 - m;
}
void museEmoPaint(vec3 q, inout Surf s){
  float front = smoothstep(0.15, 0.35, q.z);
  float sx = q.x < 0.0 ? -1.0 : 1.0;
  // fx_pale: dread — the head goes pale and blue from the crown, dark gloom lines hang over the brow
  float pl = clamp(FXV_PALE, 0.0, 1.0);
  if(pl > 0.0){
    float top = smoothstep(-0.15, 0.40, q.y);
    float pf = pl*(0.30 + 0.70*top)*mix(0.55, 1.0, gMPanel);           // the face panel most, the coat less
    s.alb = mix(s.alb, s.alb*vec3(0.66,0.76,1.0), pf);
    s.emit *= mix(vec3(1.0), vec3(0.66,0.78,1.05), pf);
    float xs = q.x/0.072, id = floor(xs + 0.5), fx = abs(xs - id)*0.072;
    float y0 = 0.15 + 0.06*hash11(id*7.31 + 3.0);
    float ln = smoothstep(0.012, 0.006, fx)*smoothstep(y0, y0 + 0.08, q.y)*smoothstep(0.47, 0.40, q.y)
             *smoothstep(0.40, 0.33, abs(q.x))*front;
    s.alb = mix(s.alb, vec3(0.13,0.11,0.26), 0.88*pl*ln);
    s.emit *= 1.0 - 0.85*pl*ln; s.fur *= 1.0 - pl*ln; s.sheen *= 1.0 - pl*ln;
  }
  // fx_green ( EMO-R2; soft gradient rq): queasy — a sickly green tint IN the face material (albedo, before
  // shading, so it is lit like the skin): strongest over the cheeks and under the eyes, a little round the mouth,
  // fading smoothly to nothing well inside the panel edge (no decal edge, no level line).  0.3-0.5 = queasy.
  float gn = clamp(FXV_GREEN, 0.0, 1.0);
  if(gn > 0.0){
    vec2 u = vec2(q.x/0.50, (q.y + 0.015)/0.405);
    vec2 au = abs(u);
    float sq = au.x*au.x*au.x + au.y*au.y*au.y;
    float inner = smoothstep(0.90, 0.40, sq)*front;                  // 0 before the panel's own edge (0.90..1.04)
    vec2 c = vec2(abs(q.x), q.y);
    float cheek = exp(-dot((c - vec2(0.20, -0.07))*vec2(4.2, 5.0), (c - vec2(0.20, -0.07))*vec2(4.2, 5.0)));
    float under = exp(-dot((c - vec2(0.17, 0.03))*vec2(6.0, 9.0), (c - vec2(0.17, 0.03))*vec2(6.0, 9.0)));
    float mouth = exp(-dot((q.xy - vec2(0.0, -0.17))*vec2(4.0, 6.0), (q.xy - vec2(0.0, -0.17))*vec2(4.0, 6.0)));
    float w = clamp(0.55 + 0.65*cheek + 0.45*under + 0.30*mouth, 0.0, 1.0)*inner;
    float gf = smoothstep(0.0, 0.45, gn)*w;
    s.alb = mix(s.alb, s.alb*vec3(0.60, 0.96, 0.40), gf);                       // the green, in the material
    s.alb = mix(s.alb, s.alb*vec3(0.80, 0.86, 0.66), 0.45*gf*clamp(under + 0.6*mouth, 0.0, 1.0));   // olive hollows
    s.emit *= mix(vec3(1.0), vec3(0.80, 1.0, 0.70), 0.6*gf);                   // the painted blush cools with it
    // queasy lines: short wavy strokes above the brows, only for a strong reaction (>~0.7)
    float xs = q.x/0.060, id = floor(xs + 0.5);
    float fx = abs(xs - id + 0.22*sin(q.y*70.0 + id*1.7 - uTime*2.5))*0.060;
    float ln = smoothstep(0.0115, 0.0060, fx)*smoothstep(0.255, 0.300, q.y)*smoothstep(0.43, 0.36, q.y)
             *smoothstep(0.25, 0.19, abs(q.x))*front*smoothstep(0.65, 0.95, gn);
    s.alb = mix(s.alb, vec3(0.16, 0.34, 0.10), 0.9*ln*gMPanel);
    s.emit *= 1.0 - 0.8*ln*gMPanel; s.fur *= 1.0 - ln; s.sheen *= 1.0 - ln;
  }
  // fx_tear > ~0.6: a drop runs from under each eye down the cheek, a wet streak behind it
  float ta = smoothstep(0.55, 0.80, clamp(FXV_TEAR, 0.0, 1.0));
  if(ta > 0.0){
    vec2 c0 = vec2(sx*(gEyeC.x + 0.028), gEyeC.y - gEyeR - 0.020);
    float ph = fract(uTime*0.40 + (sx > 0.0 ? 0.0 : 0.53));
    vec2 c = c0 - vec2(-sx*0.010*ph, 0.035 + 0.25*ph*sqrt(ph));
    float streak = smoothstep(0.012, 0.005, abs(q.x - mix(c0.x, c.x, clamp((c0.y - q.y)/max(c0.y - c.y, 1e-3), 0.0, 1.0))))
                 *step(c.y, q.y)*step(q.y, c0.y)*front*ta;
    s.alb = mix(s.alb, s.alb*vec3(0.80,0.92,1.10), 0.6*streak);
    s.spec = mix(s.spec, 0.7, streak); s.shin = mix(s.shin, 80.0, streak); s.fur *= 1.0 - streak;
    s.emit += vec3(0.25,0.32,0.40)*0.35*streak;
    musePaintDrop(s, q.xy - c, 0.030, 0.070, ta*front*smoothstep(0.0, 0.10, ph)*smoothstep(1.0, 0.82, ph),
                  vec3(0.62,0.84,1.0), vec3(0.20,0.42,0.78));
  }
  // fx_sweat: a sweat drop at the screen-right temple, swaying a little
  float sw = clamp(FXV_SWEAT, 0.0, 1.0);
  if(sw > 0.0){
    float k = mix(0.6, 1.0, smoothstep(0.0, 0.7, sw));
    vec2 c = vec2(0.432, 0.150 - 0.010*sin(uTime*1.9));
    musePaintDrop(s, (q.xy - c)/k, 0.034, 0.095, smoothstep(0.0, 0.25, sw)*front, vec3(0.72,0.88,1.0), vec3(0.20,0.44,0.80));
  }
  // fx_vein: the anger cross mark on the screen-left upper forehead, pulsing
  float vn = clamp(FXV_VEIN, 0.0, 1.0);
  if(vn > 0.0){
    float pu = 0.5 + 0.5*sin(uTime*7.5);
    float sz = 0.054*(0.70 + 0.30*smoothstep(0.0, 0.6, vn))*(1.0 + 0.14*pu);
    vec2 a = abs((q.xy - vec2(-0.355, 0.292))/sz);
    float d = abs(length(a - vec2(1.0)) - 0.76) - 0.20;                    // four arcs bulging toward the middle
    d = max(d, max(a.x, a.y) - 1.0);
    float m = smoothstep(0.08, -0.04, d)*front*smoothstep(0.0, 0.25, vn);
    vec3 col = mix(vec3(0.80,0.05,0.07), vec3(0.42,0.01,0.03), smoothstep(-0.14, 0.02, d));
    s.alb = mix(s.alb, col, m); s.emit = mix(s.emit, col*(0.10 + 0.20*pu), m);
    s.spec *= 1.0 - m; s.wrap = mix(s.wrap, 0.1, m);
    s.fur *= 1.0 - m; s.sheen *= 1.0 - m; s.occ = mix(s.occ, 1.0, m);
  }
}
#endif

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  museEval(pw);
  vec3 n = s.n, q = gQ;
  vec3 r = gcRest;
  s.fur = 0.0; s.T = normalize(cross(n, vec3(0.0, 0.0, 1.0)) + vec3(1e-4));
  s.wrap = 0.15; s.sheen = 0.08; s.occ = 1.0; s.spec = 0.10; s.shin = 30.0; s.trans = 0.0; s.emit = vec3(0.0);
#ifdef LOOK_V3
  s.rough = 0.55; s.sheenK = 0.0; s.sss = 0.0; s.coat = 0.0; s.cavity = 1.0;
#endif
  gMPanel = float(gMPart == 0)*smoothstep(0.10, 0.30, q.z);        // the emotion paints: the front of the head
  mascotMaterial(pw, q, r, n, s);
#if defined(MUSE_EMO) && defined(MR_CHEEK)
  // fx_blush: a soft rosy patch on each cheek (MR_CHEEK = its centre, head space xy; MR_CHEEK_R its size)
  if(gMPart == 0){
    float fb = clamp(FXV_BLUSH, 0.0, 1.0);
    if(fb > 0.0){
      vec2 ck = vec2(abs(q.x), q.y) - MR_CHEEK;
      float bl = exp(-dot(ck, ck)/(MR_CHEEK_R*MR_CHEEK_R*(1.0 + 1.2*fb)))*smoothstep(0.05, 0.30, q.z);
      s.alb = mix(s.alb, vec3(0.90, 0.20, 0.24), min(0.75*fb, 0.8)*bl);
    }
  }
#endif
  // contours: the smooth joins (neck, armpits, hips) and the underside shade a little
  s.occ *= 1.0 - MR_CREASE_OCC*gMCrease;
  s.occ *= mix(0.78, 1.0, smoothstep(-0.75, 0.25, n.y));
#ifdef MR_FACE_ZMIN
  if(gMPart == 0 && gMFace > 0.5 && q.z > MR_FACE_ZMIN){   // the face paints stop at this head-space depth (never on the back of the head)
#else
  if(gMPart == 0 && gMFace > 0.5){
#endif
    lipsPaint(n, s);
    facePaint(q, rd, n, gMSkin, s);
#ifdef LOOK_V3
    s.rough = mix(s.rough, 0.40, gPLip); s.sss = mix(s.sss, 0.45, gPLip); s.sssCol = mix(s.sssCol, vec3(1.0, 0.30, 0.28), gPLip);
    s.cavity *= 1.0 - 0.45*gPCave;
    s.rough = mix(s.rough, 0.30, gPTongue); s.sss = mix(s.sss, 0.60, gPTongue); s.spec = max(s.spec, 0.25*gPTongue);
    s.rough = mix(s.rough, 0.25, gPTeeth);
#endif
#ifdef MR_FACE_POST
    mascotFacePost(q, n, s);
#endif
  }
#ifdef MUSE_EMO
  if(gMPart == 0) museEmoPaint(q, s);
#endif
}

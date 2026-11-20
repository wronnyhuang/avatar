// ---------------------------------------------------------------------------
// animals/jade.glsl — JADE, an original lizard girl in the muse's classroom (our own design, not a brand mascot).
//  A round, short-snouted head with big front-set golden eyes, a frill of soft crests over each eye like lashes (they lift
//  with her brows), cream freckles on a jade-green back, a mint belly, rosy cheeks, a small smile, a pink blossom tucked
//  behind her right "ear", padded fingers and toes, and a banded tail that curls up at the tip.
//  The skeleton, the hand map and the grasp are the muse's (engine/mascot_rig.glsl): it shares muse.props.json.
// ---------------------------------------------------------------------------
#define M_H 2.30
#define M_HEADC vec3(0.0, 1.68, 0.06)
#define M_HEADR vec3(0.52, 0.47, 0.54)
#define M_UPPER 0.33
#define M_FORE 0.30
#define MR_SH_X 0.320
#define MR_SH_Y 1.030
#define MR_HIP_X 0.170
#define MR_TOE_OUT 0.26
#define MR_HAND_THICK 0.06

#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.68, 0.06)
#define MUSE_EMO 1
#define EYE_R0 0.158
#define EYE_C0 vec3(0.250,0.085,0.360)
#define EYE_R gEyeR
#define EYE_C gEyeC
#define EYE_AU 0.80
#define EYE_AL (-0.95)
#define EYE_AC (-0.12)
#define EYE_LID_TH 0.014
#define EYE_SQUINT 0.50
#define EYE_IRIS 0.780
#define EYE_PUPIL 0.380
#define EYE_COL_A vec3(1.000,0.780,0.200)          // bright honey gold, glowing low in the iris
#define EYE_COL_B vec3(0.600,0.340,0.045)          // amber gold outer ring
#define EYE_COL_W vec3(0.880,0.860,0.760)
#define LINER_COL vec3(0.012,0.055,0.030)          // the lid's edge: deep jade
#define EYE_GLOW vec3(0.160,0.100,0.010)
#define EYE_CATCH 0.925
#define BROW_C vec3(0.250,0.300,0.0)
#define BROW_L 0.110
#define BROW_W 0.010
#define BROW_COL vec3(0.040,0.250,0.120)          // a faint ridge; the crests above it do the talking
#define BROW_H 0.006
#define BROW_RISE 0.060
#define MUSE_FROWN 0.40
#define BROW_SAD_TILT 1.4
#define LIP_O vec3(0.0,-0.205,0.470)
#define LIP_TILT 0.24
#define LIP_W 0.200
#define LIP_HU (0.020*LIPU)
#define LIP_HL (0.026*LIPU)
#define LIP_TU (0.004*LIPU)
#define LIP_TL (0.005*LIPU)
#define LIP_BOW 0.10
#define LIP_SQUARE 0.15
#define LIP_OPEN_Y 1.15
#define LIP_PUSH (0.055*LIPU)
#define LIP_SKIN 0.80
#define LIP_BLEND (0.010*LIPU)
#define LIP_TOOTH_Z (-0.20*LIPU)
#define LIP_TEETH_H ((0.006 + 0.030*gEmTeeth)*LIPU)
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(0.440,0.200,0.190)          // a soft rose lip
#define LIP_COL_L vec3(0.540,0.260,0.250)
#define LIP_COL_LINE vec3(0.090,0.030,0.035)
#define LIP_LINE 1.7
#define LIP_LINE_SOFT 0.40
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.40,0.08,0.11)
#define LIP_COL_DEEP vec3(0.080,0.014,0.024)
#define LIP_COL_GUM  vec3(0.60,0.20,0.24)
#define LIP_COL_TONGUE vec3(0.95,0.45,0.52)
#define LIP_COL_TOOTH vec3(0.60,0.20,0.24)
#define LIP_SPEC 0.30
#define LIP_SHIN 70.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.05
#define LIP_WRAP 0.30
#define LIP_EMIT vec3(0.05,0.02,0.02)
#define LIP_TONGUE_FREE 0.11
#pragma include face
#pragma include lips

#define MR_ARM_R vec3(0.112, 0.098, 0.080)
#define MR_REST_CURL 0.26
#define MR_LEG_R vec3(0.152, 0.118, 0.088)
#define MR_PALM_R vec3(0.106, 0.100, 0.044)
#define MR_FING_R 0.0240
#define MR_FING_L 1.58
#define MR_FING_PAD 0.0380
#define MR_THUMB_R 0.0260
#define MR_K_NECK 0.14
#define MR_K_ARM 0.06
#define MR_K_LEG 0.09
#define MR_K_HAND 0.04
#define MR_K_FOOT 0.05
#define MR_K_EXTRA 0.13
#define MR_HEAD_BOUND 0.86
#define MR_FACE_C vec3(0.0, 0.0, 0.30)
#define MR_FACE_RAD 0.66
#define MR_EXTRA 1
#define MR_INIT_HOOK 1
#define MR_FOOT_CUSTOM 1
#define MR_CUSTOM_BEND 1
#define MR_HEAD_EXTRA 1
#define MR_CREASE_OCC 0.18
#define MR_FACE_POST 1
#define MR_CHEEK vec2(0.270, -0.120)
#define MR_CHEEK_R 0.09
#define MR_BOUND_EXTRA PB_(gTail[2], 0.34) PB_(gTail[3], 0.28) PB_(gTail[4], 0.24) PB_(gTail[5], 0.22)
//@shadow-begin
#define MR_SHADOW_HEAD vec3(0.48, 0.42, 0.48)
//@shadow-end
#pragma include mascot_rig

const vec3 C_JADE   = vec3(0.012,0.175,0.110);      // the back and the sides: a blue-leaning jade
const vec3 C_JADE2  = vec3(0.050,0.370,0.225);      // sun-side scales
const vec3 C_MINT   = vec3(0.640,0.790,0.600);      // throat, belly, under the jaw: mint cream
const vec3 C_HANDJ  = vec3(0.520,0.700,0.540);      // fingers and toes: paler
const vec3 C_FRECK  = vec3(0.880,0.890,0.640);      // the freckles: cream
const vec3 C_PETAL  = vec3(0.960,0.300,0.500);      // the blossom: pink
const vec3 C_PETAL2 = vec3(1.000,0.700,0.780);      // petal edges: paler
const vec3 C_POLLEN = vec3(1.000,0.720,0.120);

vec3 gTail[6];
float gTailR[6];
vec3 gFlN, gFlT1, gFlT2;
void mascotInit(){
  // the tail: from the back of the pelvis down to the floor, round to screen-right and curling up and in at the tip.
  // tail_wag swings it (more toward the tip), tail_raise lifts it off the floor, tail_curl winds the tip in further.
  float wag = clamp(uP[P_TAIL_WAG], -1.0, 1.0), rs = clamp(uP[P_TAIL_RAISE], 0.0, 1.0), cu = clamp(uP[P_TAIL_CURL], 0.0, 1.0);
  vec3 b = gMHip - M_HIP;
  gFlN = normalize(vec3(-0.55, 0.42, 0.72)); gFlT1 = normalize(cross(gFlN, vec3(0.0, 1.0, 0.0))); gFlT2 = cross(gFlN, gFlT1);
  gTail[0] = vec3(0.00, 0.58, -0.22); gTail[1] = vec3(0.06, 0.32, -0.45); gTail[2] = vec3(0.30, 0.130, -0.48);
  gTail[3] = vec3(0.62, 0.120, -0.34); gTail[4] = vec3(0.82, 0.300, -0.26); gTail[5] = vec3(0.68, 0.450, -0.22);
  gTailR[0] = 0.180; gTailR[1] = 0.172; gTailR[2] = 0.148; gTailR[3] = 0.118; gTailR[4] = 0.086; gTailR[5] = 0.046;
  for(int k=1;k<6;k++){
    float w = float(k)/5.0;
    vec3 v = gTail[k] - gTail[0];
    v.xz = rot(-(0.55*wag*w*w + 0.9*cu*w*w*w))*v.xz;
    v.y += rs*(0.10 + 0.55*w*w) + 0.014*sin(uTime*1.1 + 2.0*w)*w;
    gTail[k] = gTail[0] + v;
    gTail[k].y = max(gTail[k].y, gTailR[k] + 0.004 - b.y);        // it lies ON the floor, never in it
  }
  for(int k=0;k<6;k++) gTail[k] += b;
}
void mascotMouthBend(){
  // a small sweet smile: the corners turn up at rest; a knit brow with no smile turns them down
  float op = smoothstep(0.0, 0.6, max(uP[P_LIP_OPEN], uP[P_MOUTH_OPEN]));
  float frown = clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  float rest = 0.120*mix(0.40, 1.0, smoothstep(0.0, 0.05, uP[P_SMILE]));
  gLBend += LIP_W*((rest + 0.08*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - 0.6*op) - MUSE_FROWN*frown*(1.0 - 0.15*op))*(1.0 - gLRnd);
}

// ---- the head (head space): a round cranium, a short soft muzzle, full cheeks, a mound round each big eye
float mascotHead(vec3 q){
  float d = sdEllipsoid(q - vec3(0.0, 0.050, -0.050), vec3(0.470, 0.400, 0.420));          // cranium
  d = smin(d, sdEllipsoid(q - vec3(0.0, -0.100, 0.170), vec3(0.340, 0.240, 0.330)), 0.16);  // muzzle
  d = smin(d, sdEllipsoid(q - vec3(0.0, -0.225, 0.050), vec3(0.300, 0.150, 0.330)), 0.12);  // jaw / throat
  vec3 qa = vec3(abs(q.x), q.yz);
  d = smin(d, sdEllipsoid(qa - vec3(0.300, -0.120, 0.180), vec3(0.150, 0.115, 0.150)), 0.10); // cheeks
  d = smin(d, length(qa - vec3(0.250, 0.085, 0.330)) - 0.183, 0.07);                       // the eye's mound (lids)
  d = smin(d, sdEllipsoid(q - vec3(0.0, 0.110, 0.320), vec3(0.150, 0.150, 0.165)), 0.09);    // the brow plate between the eyes
  d = smin(d, sdEllipsoid(qa - vec3(0.200, -0.185, 0.385), vec3(0.070, 0.055, 0.055)), 0.06); // smile cheeks at the mouth corners
  return d;
}

// ---- head extras (never carved by the lips): the crest "lashes" over each eye and the blossom
int gJx;          // 1 crest, 2 petal, 3 the blossom's heart
float gJcrest;    // 0 at a crest's root .. 1 at its tip
float jadeCrests(vec3 q){
  float s = q.x < 0.0 ? -1.0 : 1.0;
  float bu = clamp(q.x < 0.0 ? gBrowL : gBrowR, -1.0, 1.0), kn = gKnit;
  vec3 qa = vec3(abs(q.x), q.yz);
  vec3 c = gEyeC + vec3(0.0, 0.030*bu - 0.012*kn, -0.020);
  float R = gEyeR + 0.030;
  float d = 1e3; gJcrest = 0.0;
  for(int k=0;k<5;k++){
    float t = float(k)/4.0;
    float th = mix(1.15, 0.28, t) + 0.10*kn*(1.0 - t);                 // over the upper-outer lid, toward the outer corner
    vec3 dir = normalize(vec3(cos(th), sin(th), 0.42));
    vec3 a = c + dir*R;
    float L = mix(0.040, 0.066, t);
    // each crest sweeps up and out, curving outward like a lash: two segments
    vec3 d1 = normalize(vec3(cos(th) + 0.10, sin(th) + 0.35 + 0.20*bu, 0.05));
    vec3 d2 = normalize(vec3(cos(th) + 0.75, sin(th) + 0.15 + 0.20*bu, -0.25));
    vec3 m = a + d1*L*0.55, tip = m + d2*L*0.55;
    vec3 pa = qa - a, ba = m - a;
    float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
    float dk = length(pa - ba*h) - mix(0.016, 0.011, h);
    float hh = h*0.5;
    vec3 pb = qa - m, bb = tip - m;
    float h2 = clamp(dot(pb, bb)/dot(bb, bb), 0.0, 1.0);
    float dk2 = length(pb - bb*h2) - mix(0.011, 0.0055, h2);
    if(dk2 < dk){ dk = dk2; hh = 0.5 + 0.5*h2; }
    if(dk < d){ d = dk; gJcrest = hh; }
  }
  return d;
}
float gFlVein;
float jadeBlossom(vec3 q, out float heart, out float pr){
  vec3 F0 = vec3(-0.335, 0.355, -0.030);
  vec3 l = q - F0;
  l = vec3(dot(l, gFlT1), dot(l, gFlT2), dot(l, gFlN));
  if(length(l) > 0.30){ heart = 1e3; pr = 0.0; return length(l) - 0.22; }
  float d = 1e3; pr = 0.0;
  for(int k=0;k<5;k++){
    float a = float(k)*1.2566 + 0.30;
    vec2 cs = vec2(cos(a), sin(a));
    vec3 pl = vec3(dot(l.xy, cs), dot(l.xy, vec2(-cs.y, cs.x)), l.z);
    pl.x -= 0.092;
    pl.z -= 0.16*(pl.x + 0.04)*(pl.x + 0.04)/0.10 + 0.012;               // cupped: the tips lift outward
    pl.y *= 1.0 + 0.9*smoothstep(0.02, 0.10, pl.x)*0.0;
    float dp = sdEllipsoid(pl, vec3(0.098, 0.072, 0.023));
    dp += 0.004*smoothstep(0.05, 0.10, pl.x)*(1.0 - smoothstep(0.0, 0.02, abs(pl.y)))*0.0;
    if(dp < d){ d = dp; pr = clamp((pl.x + 0.092)/0.19, 0.0, 1.0); gFlVein = pl.y; }
  }
  heart = sdEllipsoid(l - vec3(0.0, 0.0, 0.022), vec3(0.042, 0.042, 0.030));
  return min(d, heart);
}
float mascotHeadBare(vec3 q){
  float heart, pr;
  float cr = jadeCrests(q);
  float fl = jadeBlossom(q, heart, pr);
  gJx = cr < fl ? 1 : (heart <= fl + 1e-4 ? 3 : 2);
  return min(cr, fl);
}

void mascotFacePost(vec3 q, vec3 n, inout Surf s){
  if(gFEye > 0.003) return;
  vec3 e = (vec3(abs(q.x), q.yz) - gEyeC)/gEyeR;
  float gold = smoothstep(0.08, 0.30, s.alb.r - s.alb.b)*smoothstep(0.10, 0.25, s.alb.r);
  float ang = atan(e.y, e.x);
  float st = vnoise2(vec2(ang*10.0, length(e.xy)*2.0)) + 0.5*vnoise2(vec2(ang*23.0, length(e.xy)*5.0));
  s.alb *= mix(1.0, 0.78 + 0.32*st, gold);
}
vec3 gNeckA, gNeckB;
float mascotTorso(vec3 p){
  vec3 b = p - (gMHip - M_HIP);
  float lo = sdEllipsoid(b - vec3(0.0, 0.640, -0.010), vec3(0.300, 0.300, 0.262));          // belly, pelvis space
  vec3 u = localToUpper(p);
  float br = 1.0 + 0.014*gBreath;
  float hi = sdEllipsoid((u - vec3(0.0, 0.905, 0.0))/vec3(br, 1.0, br), vec3(0.282, 0.290, 0.232));   // chest
  float d = smin(lo, hi, 0.15);
  d = smin(d, sdEllipsoid(u - vec3(0.0, 1.040, -0.010), vec3(0.325, 0.105, 0.190)), 0.11);  // sloping shoulders
  // the neck: from the chest up into the back of the head (it follows the head)
  vec3 na = upperToLocal(vec3(0.0, 1.060, -0.010));
  vec3 nb = gHO + transpose(gHR)*vec3(0.0, -0.210, -0.110);
  d = smin(d, sdTaper(p, na, nb, 0.262, 0.272), 0.14);
  return d;
}
float mascotExtra(vec3 p, out vec3 rest){
  float d = 1e3; float along = 0.0, acc = 0.0; vec3 ax = vec3(0.0, 0.0, -1.0), base = gTail[0];
  for(int k=0;k<5;k++){
    vec3 a = gTail[k], c = gTail[k+1];
    vec3 pa = p - a, ba = c - a;
    float L = length(ba);
    float h = clamp(dot(pa, ba)/(L*L), 0.0, 1.0);
    float dk = length(pa - ba*h) - mix(gTailR[k], gTailR[k+1], h);
    if(dk < d){ along = acc + h*L; ax = ba/L; base = a + ba*h; }
    d = smin(d, dk, 0.05);
    acc += L;
  }
  vec3 e1 = mPerp(ax), e2 = cross(ax, e1);
  rest = vec3(along, dot(p - base, e1), dot(p - base, e2));
  return d;
}
// the foot: a small flat pad with four slim toes that end in round pads
float museFoot(vec3 p, int i, out float sole){
  vec3 q = transpose(gMFoot[i])*(p - gMAnk[i]);
  float d = sdEllipsoid(q - vec3(0.0, -0.085, 0.040), vec3(0.080, 0.048, 0.112));
  float s = i == 0 ? -1.0 : 1.0;
  for(int k=0;k<4;k++){
    float a = (float(k) - 1.5)*0.40 + s*0.10;
    vec3 dir = vec3(sin(a), 0.0, cos(a));
    vec3 t0 = vec3(0.0, -0.105, 0.080) + dir*0.03, t1 = vec3(0.0, -0.114, 0.080) + dir*(0.140 - 0.012*abs(float(k) - 1.5));
    float f = sdCapsule(q, t0, t1, 0.019);
    f = smin(f, length(q - t1 - vec3(0.0, 0.004, 0.0)) - 0.028, 0.013);
    d = smin(d, f, 0.020);
  }
  d = smax(d, -(q.y + M_ANK_Y - 0.002), 0.012);        // flat on the floor
  sole = smoothstep(-0.118, -0.140, q.y);
  return d;
}

// ---- scales: a jittered cell pattern in the rest frame (it rides the skin).  x = distance to the cell centre 0..1,
// y = a random value per cell.  8 cells: evaluated once per pixel, in the shading only.
vec3 gkOff;
vec2 gkCells(vec3 x){
  vec3 ip = floor(x - 0.5), fp = x - ip;
  float best = 9.0, id = 0.0; vec3 off = vec3(0.0);
  for(int k=0;k<8;k++){
    vec3 g = vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1));
    vec3 h = hash33(ip + g);
    vec3 r = g + 0.22 + 0.56*h - fp;
    float dd = dot(r, r);
    if(dd < best){ best = dd; id = h.x; off = r; }
  }
  gkOff = -off;
  return vec2(sqrt(best), id);
}
// freckles: sparse soft cream dots on a coarse jittered grid
float jadeFreckles(vec3 x){
  vec3 ip = floor(x), fp = fract(x);
  vec3 h = hash33(ip + 17.0);
  float on = step(0.40, h.y);
  float rad = mix(0.20, 0.34, h.z);
  return on*smoothstep(rad, rad*0.55, length(fp - (0.25 + 0.5*h)));
}

void mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s){
  float jx = gMPart == 0 ? mascotHeadBare(q) : 1e3;
  if(jx < gMHeadD + 0.002 && jx <= gMBare + 0.001){
    if(gJx == 1){
      // the crests: the same jade, paler and a little translucent toward the tips
      vec3 alb = mix(C_JADE*0.9, vec3(0.008, 0.045, 0.028), smoothstep(0.15, 0.60, gJcrest));
      s.alb = alb; s.wrap = 0.30; s.sheen = 0.20; s.spec = 0.35; s.shin = 60.0; s.emit = alb*0.035;
      s.trans = 0.35*gJcrest; s.transCol = vec3(0.6, 1.0, 0.4);
#ifdef LOOK_V3
      s.rough = 0.34; s.coat = 0.30; s.sss = 0.40; s.sssCol = vec3(0.55, 0.95, 0.30); s.sheenK = 0.08; s.sheenCol = vec3(0.8, 1.0, 0.6);
#endif
      return;
    }
    float heart, pr;
    jadeBlossom(q, heart, pr);
    if(gJx == 3){
      // the blossom's heart: a pollen-yellow dome with a fine grain
      float g = vnoise3(q*180.0);
      s.alb = C_POLLEN*(0.80 + 0.30*g); s.wrap = 0.40; s.sheen = 0.10; s.spec = 0.10; s.shin = 20.0; s.emit = s.alb*0.04;
      s.n = normalize(n + 0.25*(vec3(vnoise3(q*220.0), vnoise3(q*220.0 + 5.1), vnoise3(q*220.0 + 9.7)) - 0.5));
#ifdef LOOK_V3
      s.rough = 0.80; s.sss = 0.30; s.sssCol = vec3(1.0, 0.7, 0.2); s.sheenK = 0.30; s.sheenCol = vec3(1.0, 0.9, 0.5);
#endif
      return;
    }
    // petals: soft pink, paler at the rim, fine veins, thin enough to glow when back-lit
    float vein = 0.10*smoothstep(0.006, 0.0, abs(gFlVein))*smoothstep(0.95, 0.3, pr)
               + 0.06*smoothstep(0.004, 0.0, abs(abs(gFlVein) - 0.022 - 0.03*pr))*smoothstep(0.2, 0.5, pr);
    vec3 alb = mix(C_PETAL, C_PETAL2, smoothstep(0.45, 1.0, pr))*(1.0 - vein) ;
    alb = mix(alb, vec3(0.98, 0.85, 0.55), 0.45*smoothstep(0.25, 0.0, pr));
    s.alb = alb; s.wrap = 0.45; s.sheen = 0.25; s.spec = 0.12; s.shin = 30.0; s.emit = alb*0.05;
    s.trans = 0.55; s.transCol = vec3(1.0, 0.55, 0.65);
#ifdef LOOK_V3
    s.rough = 0.55; s.sss = 0.60; s.sssCol = vec3(1.0, 0.45, 0.55); s.sheenK = 0.25; s.sheenCol = vec3(1.0, 0.85, 0.90);
#endif
    return;
  }
  // which way the skin faces in the part's own frame: the belly side (front, under) is pale
  vec3 u = localToUpper(pw);
  float belly = 0.0, hand = 0.0, back = 1.0;
  vec3 rs = r;
  if(gMPart == 0){
    // under the jaw and the lower lip: mint cream; the line runs just under the mouth and back along the jaw
    float jl = -0.250 - 0.05*smoothstep(0.10, 0.50, abs(q.x)) + 0.18*smoothstep(0.15, -0.35, q.z);
    belly = smoothstep(jl + 0.130, jl - 0.090, q.y)*smoothstep(0.42, 0.20, abs(q.x));
    back = smoothstep(-0.05, 0.25, q.y)*smoothstep(0.30, -0.10, q.z) + 0.5*smoothstep(0.20, 0.40, q.y);
  } else if(gMPart == 1){
    vec3 un = transpose(gMW)*n;
    vec2 bo = vec2(u.x/0.215, (u.y - 0.845)/0.315);
    belly = smoothstep(0.05, 0.50, un.z + 0.10*un.y)*smoothstep(1.10, 0.80, length(bo));
    float front = smoothstep(-0.05, 0.45, un.z);
    belly = max(belly, smoothstep(0.85, 1.05, u.y)*smoothstep(0.19, 0.11, abs(u.x))*front);           // a strip up to the throat
    belly = max(belly, smoothstep(1.00, 1.22, u.y)*smoothstep(0.26, 0.14, abs(u.x))*front);          // the throat
    back = smoothstep(0.0, -0.5, un.z);
    rs = u;
  } else if(gMPart == 2){
    hand = smoothstep(0.85, 1.0, gMLimbT)*0.25; back = 0.5;
  } else if(gMPart == 3){
    hand = 0.30; back = 0.0;
  } else if(gMPart == 4){
    hand = smoothstep(0.80, 1.0, gMLimbT)*0.25; back = 0.4;
  } else if(gMPart == 5){
    hand = 0.30; back = 0.0;
  } else if(gMPart == 7){
    belly = smoothstep(0.25, -0.35, n.y)*0.85; back = 1.0 - belly;
  }
  float faceF = gMPart == 0 ? smoothstep(0.05, 0.30, q.z)*smoothstep(0.40, 0.10, q.y) : 0.0;     // the face: fine, soft scales
  float cellSz = mix(mix(gMPart == 0 ? 0.017 : 0.020, 0.021, belly), 0.013, faceF);
  if(gMPart == 3 || gMPart == 5) cellSz = 0.015*1.6;            // hand / foot rest frames are scaled x1.6
  float wS = smoothstep(2.0, 4.2, cellSz/max(gMFp*(gMPart == 3 || gMPart == 5 ? 1.6 : 1.0), 1e-5));
  vec2 ce = gkCells(rs/cellSz);
  float dome = 1.0 - smoothstep(0.0, 0.62, ce.x);
  float groove = smoothstep(0.42, 0.66, ce.x);
  float big = vnoise3(rs*3.5), mid = vnoise3(rs*9.0 + 3.7);
  vec3 green = mix(C_JADE, C_JADE2, clamp(0.55 + 0.30*(big - 0.5) + 0.10*(ce.y - 0.5)*wS + 0.20*n.y, 0.0, 1.0));
  vec3 pale = C_MINT*(0.98 + 0.03*mid + 0.015*(ce.y - 0.5)*wS);
  vec3 alb = mix(green, pale, belly);
  alb = mix(alb, C_HANDJ*(0.92 + 0.14*mid), hand);
  if(gMPalmW > 0.0 || gMSoleW > 0.0) alb = mix(alb, C_MINT*vec3(0.95, 0.92, 0.88), 0.85*max(gMPalmW, gMSoleW));
  // the tail's bands: darker jade rings toward the tip
  if(gMPart == 7){
    alb = mix(alb, C_JADE2*1.10, 0.35*(1.0 - belly));
    float band = smoothstep(0.30, 0.10, abs(fract(r.x/0.135) - 0.5))*smoothstep(0.25, 0.55, r.x);
    alb *= 1.0 - 0.36*band*(1.0 - belly);
  }
  // the freckles: cream dots on the crown, the back, the outside of the arms and legs and the tail top
  float fk = jadeFreckles(rs/(gMPart == 0 ? 0.080 : 0.105))*clamp(back + 0.35*float(gMPart == 2 || gMPart == 4), 0.0, 1.0)*(1.0 - belly)*(1.0 - hand);
  if(gMPart == 0){
    vec2 ev = vec2(abs(q.x), q.y) - gEyeC.xy;
    fk *= smoothstep(gEyeR + 0.07, gEyeR + 0.12, length(ev));            // never on the lids
  }
  if(gMPart == 0){
    vec2 fz = vec2(abs(q.x) - 0.150, q.y + 0.075);
    float band = exp(-dot(fz/vec2(0.075, 0.040), fz/vec2(0.075, 0.040)))*smoothstep(0.30, 0.42, q.z);
    float dots = 0.0;
    for(int k=0;k<4;k++){
      vec2 c = vec2(0.100 + 0.034*float(k) + 0.010*sin(float(k)*7.3), -0.060 - 0.022*float(k & 1) + 0.008*cos(float(k)*5.1));
      dots = max(dots, smoothstep(0.011, 0.006, length(vec2(abs(q.x), q.y) - c)));
    }
    fk = max(fk, dots*smoothstep(0.30, 0.42, q.z)*(band > 0.0 ? 1.0 : 1.0));
  }
  alb = mix(alb, C_FRECK*(0.92 + 0.12*ce.y), 0.85*fk);
  // the grooves between scales are darker, the domes catch light
  alb *= 1.0 - (0.15 - 0.13*belly - 0.05*faceF)*groove*wS;
  if(gMPart == 0){
    // the lids: a paler mint-jade ring of fine skin round each eye
    vec2 ev = vec2(abs(q.x), q.y) - gEyeC.xy;
    float ring = smoothstep(gEyeR + 0.075, gEyeR + 0.010, length(ev))*smoothstep(0.0, 0.20, q.z);
    alb = mix(alb, vec3(0.120, 0.470, 0.230), 0.22*ring);
    // the brow and the bridge between the eyes: a touch lighter, so the face reads as one clean panel
    float bridge = smoothstep(0.20, 0.0, abs(q.x))*smoothstep(-0.10, 0.10, q.y)*smoothstep(0.32, 0.22, q.y)*smoothstep(0.15, 0.35, q.z);
    alb = mix(alb, C_JADE2*1.06, 0.55*bridge);
    // rosy cheeks: a soft pink flush under each eye, outside the mouth
    vec2 ck = vec2(abs(q.x), q.y) - vec2(0.275, -0.135);
    float blush = exp(-dot(ck, ck)/(0.075*0.075))*smoothstep(0.10, 0.30, q.z);
    alb = mix(alb, vec3(0.95, 0.40, 0.46), 0.45*blush);
    // nostrils: two small dark dots on the muzzle
    float no = smoothstep(0.019, 0.008, length(vec2(abs(q.x) - 0.058, (q.y + 0.075)*1.4)))*smoothstep(0.38, 0.46, q.z);
    alb = mix(alb, vec3(0.015, 0.050, 0.025), 0.90*no);
    s.occ *= 1.0 - 0.5*no;
  }
  s.alb = alb;
  vec3 lean = gkOff - n*dot(gkOff, n);
  s.n = normalize(n + lean*(0.70 - 0.45*belly - 0.35*faceF)*smoothstep(0.15, 0.60, ce.x)*wS);
  {
    vec3 bp = rs*26.0; float e0 = vnoise3(bp);
    vec3 g = vec3(vnoise3(bp + vec3(0.25, 0.0, 0.0)), vnoise3(bp + vec3(0.0, 0.25, 0.0)), vnoise3(bp + vec3(0.0, 0.0, 0.25))) - e0;
    g -= s.n*dot(g, s.n);
    s.n = normalize(s.n - g*(0.55*(1.0 - wS) + 0.15)*(1.0 - 0.6*faceF)*(1.0 - hand*0.5));
  }
  s.occ = 1.0 - 0.08*groove*wS;
  s.wrap = 0.28; s.sheen = 0.24;
  s.spec = mix(0.44, 0.28, belly)*(0.55 + 0.45*dome*wS + 0.45*(1.0 - wS)); s.shin = 60.0;
  s.emit = alb*vec3(0.032, 0.042, 0.030);
#ifdef LOOK_V3
  s.rough = mix(0.30, 0.44, groove*wS) + 0.05*belly;
  s.coat = 0.50*(1.0 - 0.6*groove*wS)*(1.0 - 0.4*belly)*(0.65 + 0.35*dome*wS + 0.35*(1.0 - wS));
  s.sss = mix(0.34, 0.44, max(belly, hand)); s.sssCol = mix(vec3(0.40, 0.95, 0.45), vec3(1.0, 0.82, 0.55), max(belly, hand));
  s.sheenK = 0.10; s.sheenCol = vec3(0.75, 1.0, 0.75);
  s.cavity = 1.0 - 0.15*groove*wS;
#endif
  s.trans = 0.10*hand; s.transCol = vec3(0.8, 0.9, 0.5);
}
#pragma include mascot_body

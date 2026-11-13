// ---------------------------------------------------------------------------
//  animals/monkey.glsl — a young storybook monkey.  Head space: origin at the skull centre, +x screen
//  right, +y up, +z toward the camera.  Units: skull radius ~0.6.  See docs/ANIMAL_GUIDE.md for the API.
//
//  THE MOUTH RIG reads the eleven lip_* columns directly (docs/V2.md); the legacy jaw/round/... are not read.
//  A monkey's mouth is the most mobile of the animals: a long upper lip that lifts off the teeth and funnels
//  forward, a free lower lip, corners that travel round the muzzle.  Upper and lower muzzle blend into one
//  closed form (the cheek web), and the APERTURE is cut through it: two arcs through the two corners, an upper
//  arc rising gAu and a lower one falling gAl, so the two lips are independent.  Both 0 is a line; shallow
//  arcs are a lens with real corners; arcs as high as the half width, with the corners rounded (gRc), are an
//  O.  Each lip is its own volume around that curve, riding the muzzle surface, with its own height, fullness
//  and depth.  The jaw hinges on teeth_gap and carries the chin, the lower teeth and the tongue; the upper
//  teeth stay on the skull.  Teeth are a row of soft sculpted teeth on an arch, in a gum, shaded by the lip.
//  A small opening is a dark slit first (gDark, gShow); colour and teeth come up with room.
//  Round three, docs/V3.md: the lips are flesh, not paint (gSect runs across each lip).  A press pales, fills and thickens them,
//  and they peel: gStick holds the opening in at the middle while the seam stays shut out to the corners, read from the
//  press value alone since a frame has no history.  Chin, cheeks, nose wings and the folds follow the lips.  The arches lean
//  toward the camera so the far corner is not a dark band.  Reads smirk and squint.
// ---------------------------------------------------------------------------
const vec3  NECK   = vec3(0.0,-0.60,-0.14);    // head rotates about this
const vec3  HINGE  = vec3(0.0,-0.10, 0.02);    // jaw hinge
const float Y0     = -0.272;                   // mouth line at the front of the muzzle
const float ZF     = 0.815;                    // front of the muzzle at the mouth line
const float W0     = 0.245;                    // half width of the mouth
const vec3  NOSEP  = vec3(0.0,-0.062,0.862);   // nose tip
const vec3  POLE   = vec3(0.0,-0.02,1.00);     // fur flows away from here
const vec3  EYEC   = vec3(0.258,0.125,0.440);  // mirrored
const float EYER   = 0.194;
const float EYETILT= -0.06;
const float YAW0   = -0.23;                    // resting 3/4 turn, toward screen left
const vec3  EARB   = vec3(0.670,0.085,-0.03);  // ear centre (mirrored)
const float MAXJAW = 0.40;                     // radians at teeth_gap=1
const float TIP_U  = Y0 - 0.016;               // upper biting edge, on the skull
const float TIP_L  = Y0 - 0.006;               // lower biting edge at teeth_gap 0, in the jaw frame: a small overbite
const vec3  ARCH_U = vec3(0.245,0.270,0.472);  // dental arch in plan: half x, half z, centre z.  Broad and U-shaped, a monkey's
const vec3  ARCH_L = vec3(0.215,0.240,0.478);
const float F_K = 13.0, F_L = 3.6;             // tuft density across / along the flow

// ---- per-fragment pose (filled by charInit)
mat3  gHR;          // world -> head
mat3  gBodyR;       // world -> head at rest (body fur + paint frame)
vec3  gHO;          // head-space origin offset
float gJawA, gGap, gRound, gPress, gSmile, gCheekP, gNostril, gOpen;
float gLipO, gUp, gLo, gPk, gSp, gPro, gTuck, gTgTip, gTgDent, gTgBack;   // the lip_* columns, gated by press
float gKx, gKz;             // the skin's squeeze and forward travel around the mouth
float gSquint, gSmirk, gStick, gNoseW, gOpenK;   // lower lids, corners to one side, press-stick along the seam, nose wings, 0 closed .. 1 open
float gBlinkL, gBlinkR;
vec2  gGaze;
float gBrowL, gBrowR, gKnit;
mat3  gEarRL, gEarRR;
float gBreath, gSeal;       // gSeal 1 = mouth closed: the interior cannot show
float gYC, gHa, gW, gBend;  // the mouth opening: chord height, half height, half width, corner lift
float gAu, gAl, gWi, gRc, gNoseY;          // upper rise, lower fall, inner half chord, corner rounding, nose travel
float gRu, gCu, gRl, gCl;                  // the two arcs: radius, centre's distance from the chord
float gHU, gHL, gTU, gTL, gDU, gDL;        // lip height, fullness, depth of the section centre, each lip
float gCornMin, gCornT, gCapU, gPad, gTub; // corners: thickness left, 0 lens .. 1 ring, pad
float gHoleM, gDark, gShow, gTunnel;       // carve margin, dark slit 1 .. 0, teeth hidden 0 .. 1, deep behind a pucker
float gTgDim, gToTip, gToGap, gTgFwd, gTipUY, gTipLY;

mat3 rotX(float a){ float c=cos(a),s=sin(a); return mat3(1,0,0, 0,c,s, 0,-s,c); }
mat3 rotY(float a){ float c=cos(a),s=sin(a); return mat3(c,0,-s, 0,1,0, s,0,c); }
mat3 rotZ(float a){ float c=cos(a),s=sin(a); return mat3(c,s,0, -s,c,0, 0,0,1); }
float sq(float x){ return x*x; }
float bell(float x){ x = clamp(1.0 - x*x, 0.0, 1.0); return x*x; }

// ---------------------------------------------------------------------------
//  THE ARMS AND HANDS (round three).  The torso is one rigid frame, b in monkeyEval: world yawed by BYAW.  Everything
//  below is posed once per fragment from the hand_* columns (docs/V3.md) and read by monkeyEval: a pure function of
//  the columns, no state.  Index 0 is screen left, 1 screen right.  Each arm is solved on the screen-right side and
//  mirrored, so "x" here always means away from the midline.
//    target    the column values around the rest wrist, then pushed out of the coat and the head, then held in reach.
//    shoulder  rides up and forward with the hand.  elbow: two-bone IK, its pole swinging out as the hand rises.
//    hand      forearm twist from hand_*_turn, wrist flex, five digits: palm slab, three phalanges, a thumb that opposes.
// ---------------------------------------------------------------------------
const float BYAW   = 0.20;                     // the torso's turn, against the head's
const float BONE_U = 0.76, BONE_F = 0.72;      // upper arm, forearm
const vec3  SHLD   = vec3(0.68,-1.00,-0.24);   // shoulder joint
const vec3  WRIST0 = vec3(0.62,-1.32,0.54);    // rest wrist: the hand lies on the thigh above the knee
const vec3  KNEE   = vec3(0.60,-1.73,0.76);
const float DIG_L[4]  = float[4](0.108, 0.036,-0.034,-0.100);    // knuckles across the palm, toward the thumb
const float DIG_D[4]  = float[4](0.286, 0.302, 0.290, 0.260);    // and along it
const float DIG_S[4]  = float[4](0.70, 0.12,-0.46,-1.00);        // splay
const vec3  DIG_B[4]  = vec3[4](vec3(0.128,0.098,0.076), vec3(0.144,0.108,0.080), vec3(0.134,0.100,0.076), vec3(0.104,0.078,0.066));   // phalanges
const float DIG_R[5]  = float[5](0.0335, 0.0350, 0.0330, 0.0290, 0.0400);     // base radius, the thumb last

vec3  gAS[2], gAE[2], gAW[2];     // shoulder, elbow, wrist
vec3  gAU[2], gAF[2], gAH[2];     // unit: down the upper arm, down the forearm, the elbow's hinge
mat3  gHB[2];                     // the hand: columns toward the thumb, toward the fingertips, out of the palm
vec3  gHBc[2], gHBh[2];           // its bounding box, in hand space
float gHBr[2];                    // how far a digit's joints stray from the chord, with its radius
vec3  gTP[2];                     // the thumb's pad faces this way
vec3  gDJ[40];                    // joints, hand*20 + digit*4 + joint; digit 4 is the thumb

// ---- the whole character moves with the body columns (docs/RIG.md): body_yaw / body_pitch / body_lean turn it about the
// seat, body_bounce lifts it.  Every entry point (charEval, charShadow, charSurface) maps the world point into the body's rest
// frame first, so head, arms, hands and coat ride along; head_* stay relative to the body.
const vec3 BODY_PIV = vec3(0.0,-2.00,-0.25);
mat3 gBodyM; vec3 gBodyT;
vec3 bodyXf(vec3 p){ return gBodyM*(p - BODY_PIV - gBodyT) + BODY_PIV; }
float gShrug[2];
float gThumbX[2], gPointX[2];
float gTgOut;
// ---- the tail.  Out from under the seat behind, round the screen-right hip and up beside it, so it reads from the
// front.  tail_wag swings it about its root (a POSITION, -1 back .. 1 forward), tail_raise lifts it, tail_curl rolls the
// tip into a hook.  Torso frame b (as the arms).
vec3 gTl[5]; vec4 gTlB; float gTailW;
const vec3 TAIL_ROOT = vec3(0.40,-1.98,-0.50);
void tailPose(){
  float wag = clamp(uP[P_TAIL_WAG], -1.0, 1.0), raise = clamp(uP[P_TAIL_RAISE], 0.0, 1.0), curl = clamp(uP[P_TAIL_CURL], 0.0, 1.0);
  float az = 0.16 + 0.42*wag;                       // the plane it hangs in: from +x toward the camera (+) or back (-)
  vec3 h = vec3(cos(az), 0.0, sin(az)), up = vec3(0.0,1.0,0.0);
  float a = -0.22 + 0.75*raise - 0.25*curl;
  vec2 c = vec2(0.0);
  gTl[0] = TAIL_ROOT;
  vec3 lo = TAIL_ROOT, hi = TAIL_ROOT;
  for(int i=0;i<4;i++){
    float fi = float(i);
    a += (0.10 + 0.18*fi) + curl*0.36*fi*fi;       // curl rolls the tip, the root stays
    c += vec2(cos(a), sin(a))*(0.50 - 0.04*fi);
    gTl[i+1] = TAIL_ROOT + h*c.x + up*c.y;
    lo = min(lo, gTl[i+1]); hi = max(hi, gTl[i+1]);
  }
  gTlB = vec4(0.5*(lo + hi), 0.5*length(hi - lo) + 0.16);
}
float tailD(vec3 b){
  float d = length(b - gTlB.xyz) - gTlB.w;
  for(int k=0;k<1 && d < 0.05;k++){
    d = sdTaper(b, gTl[0], gTl[1], 0.085, 0.078);
    d = smin(d, sdTaper(b, gTl[1], gTl[2], 0.078, 0.068), 0.04);
    d = smin(d, sdTaper(b, gTl[2], gTl[3], 0.068, 0.056), 0.04);
    d = smin(d, sdTaper(b, gTl[3], gTl[4], 0.056, 0.040), 0.04);
  }
  return d;
}

vec3 pushOut(vec3 p, vec3 c, vec3 r, float k){
  vec3 q = (p - c)/r;
  float l = length(q);
  return c + q*(smax(l, 1.0, k)/max(l, 1e-3))*r;
}

vec3 steer(vec3 f, vec3 W, vec3 c, vec3 r, float g){
  vec3 q = (W + f*0.54 - c)/r;
  return normalize(f + normalize(q/r)*g*(1.0 - smoothstep(0.90, 1.22, length(q))));
}

void fingerPose(int k, float open, float sx, vec3 W, vec3 t, vec3 f, vec3 n, out vec3 j0, out vec3 j1, out vec3 j2, out vec3 j3){
  float fk = float(k);
  float o = smoothstep(0.03*fk, 0.88 + 0.04*fk, open);
  float a1 = mix(0.40 + 0.11*fk, -0.05, o);
  float a2 = mix(0.74 + 0.06*fk, 0.07, o);
  float a3 = mix(0.44, 0.05, o);
  vec3 d0 = normalize(f + t*mix(0.035, 0.27, o)*DIG_S[k]);
  vec3 P = W + t*DIG_L[k] + f*DIG_D[k], B = DIG_B[k];
  j0 = vec3(sx*P.x, P.yz);
  P += (cos(a1)*d0 + sin(a1)*n)*B.x;               j1 = vec3(sx*P.x, P.yz);
  a1 += a2;  P += (cos(a1)*d0 + sin(a1)*n)*B.y;    j2 = vec3(sx*P.x, P.yz);
  a1 += a3;  P += (cos(a1)*d0 + sin(a1)*n)*B.z;    j3 = vec3(sx*P.x, P.yz);
}

void armPose(int i){
  float sx = i == 0 ? -1.0 : 1.0;
  // contract (docs/CONTRACT.md): the wrist's offset from rest in units of one full arm length, x outward
  const float L = 0.92*(BONE_U + BONE_F);
  float hx = clamp(uP[i == 0 ? P_HAND_L_X : P_HAND_R_X], -1.2, 1.4);
  float hy = clamp(uP[i == 0 ? P_HAND_L_Y : P_HAND_R_Y], -1.2, 1.4);
  float hz = clamp(uP[i == 0 ? P_HAND_L_Z : P_HAND_R_Z], -1.2, 1.4);
  float open = clamp(uP[i == 0 ? P_HAND_L_OPEN : P_HAND_R_OPEN], 0.0, 1.0);
  float turn = clamp(uP[i == 0 ? P_HAND_L_TURN : P_HAND_R_TURN], -1.0, 1.0);
  float thumb = clamp(uP[i == 0 ? P_THUMB_L : P_THUMB_R], 0.0, 1.0);
  float point = clamp(uP[i == 0 ? P_POINT_L : P_POINT_R], 0.0, 1.0);
  gThumbX[i] = thumb; gPointX[i] = point;

  // ---- the wrist target: the columns around the rest pose, then the anatomy
  vec3 T = WRIST0 + L*vec3(hx, hy, 0.85*hz);
  T.z += 0.10*smoothstep(0.0, 1.0, hy);                                   // a rising hand comes forward of the chest
  T.x += 0.26*smoothstep(0.3, 1.0, hy)*smoothstep(-0.1, 0.15, hx);        // ... and, unless it is reaching across, out past the ear
  T.y = smax(T.y, -1.37 - 0.85*smoothstep(1.05, 1.40, T.x), 0.06);        // the hand already lies on the thigh; only clear of it may it sink
  T = pushOut(T, vec3(0.0,-1.48,-0.24), vec3(0.92,0.98,0.84), 0.05);      // the torso and its coat
  T = pushOut(T, vec3(0.56,-1.86,0.30), vec3(0.42,0.46,0.70), 0.05);      // the thigh
  // the head, in its own frame: a hand may come IN FRONT of the face (eyes, chin), never into it.  Body frame here is the
  // screen-right solve: mirror the head point the same way
  vec3 Tw = vec3(sx*T.x, T.yz);
  Tw.xz = rot(-BYAW)*Tw.xz;                                               // torso frame -> world (monkeyEval: b.xz = rot(BYAW)*pw.xz)
  vec3 Th = gHR*(Tw - gHO) + NECK;
  // COVER_EYES: a hand reaching up and across in front of the face goes over its own side's eye, palm on it
  float cover = smoothstep(0.80, 0.98, hy)*smoothstep(-0.12, -0.30, hx)*smoothstep(0.05, 0.30, hz);
  Th = mix(Th, vec3(sx*0.24, EYEC.y - 0.25, EYEC.z + 0.38), cover);
  Th = pushOut(Th, vec3(0.0,-0.02,0.02), vec3(0.80,0.74,0.80), 0.05);     // skull, cheeks and the muzzle's sides
  Th = pushOut(Th, vec3(0.0,-0.22,0.50), vec3(0.34,0.30,0.48), 0.04);     // the muzzle
  Tw = transpose(gHR)*(Th - NECK) + gHO;
  Tw.xz = rot(BYAW)*Tw.xz;
  T = vec3(sx*Tw.x, Tw.yz);
  T.x = smax(T.x, 0.07, 0.06);                                            // each hand keeps to its own side: two meet at the midline

  float lift = smoothstep(-1.34, -0.72, T.y);
  float wide = smoothstep(0.75, 1.50, T.x);
  vec3 S = SHLD + vec3(-0.035,0.085,0.045)*lift + vec3(0.025,-0.010,0.0)*wide + vec3(0.0, 0.15, 0.02)*gShrug[i];

  // ---- two bones
  vec3 D = T - S;
  float d = max(length(D), 1e-3);
  vec3 e = D/d;
  float dm = -smax(-d, -0.965*(BONE_U + BONE_F), 0.10);
  dm = smax(dm, 0.34, 0.10);
  vec3 W = S + e*dm;
  float a = (BONE_U*BONE_U - BONE_F*BONE_F + dm*dm)/(2.0*dm);
  float h = sqrt(max(BONE_U*BONE_U - a*a, 1e-4));
  float fold = max(1.0 - smoothstep(0.42, 0.86, dm), 1.0 - smoothstep(0.26, 0.60, T.x));   // a folded or inboard arm opens its elbow clear of the ribs
  float low = 1.0 - smoothstep(-1.46, -1.30, T.y);
  vec3 pole = vec3(0.54 + 0.56*wide + 0.75*fold + 0.30*low, -0.84 + 0.20*lift + 0.30*wide + 0.40*low, -0.30 - 0.10*lift + 0.20*fold) - vec3(0.10*turn, 0.0, 0.0);
  pole = normalize(pole - e*dot(pole, e));
  vec3 E = S + e*a + pole*h;

  // ---- the hand: forearm twist, then the wrist
  vec3 af = normalize(W - E), au = normalize(E - S);
  vec3 ref = vec3(0.0,1.0,0.0) - 0.35*au;                                 // the room's up, leaning on the bend plane when the forearm stands up
  vec3 up = normalize(ref - af*dot(ref, af));
  vec3 nn = cross(af, up);                                                // palm facing in, thumb up: a forearm half way
  float phi0 = mix(0.62, 1.10, lift);
  phi0 = mix(phi0, 1.57, max(thumb, point));                             // a fist with the thumb up, a pointing hand: palm in
  float phi = phi0 + 1.44*turn + (1.44 - phi0)*turn*turn;                 // 0 palm down .. pi palm up
  vec3 n0 = -cos(phi)*up + sin(phi)*nn;
  float w = mix(0.40, 0.14, lift) - 0.46*open - 0.10*max(turn, 0.0);      // the wrist flexes at rest, extends as the hand opens
  vec3 f = normalize(cos(w)*af + sin(w)*n0);
  vec3 n = normalize(cos(w)*n0 - sin(w)*af);
  // a talking hand presents itself: the wrist bends so the fingers lead out and toward the listener, never up at the face
  f = normalize(mix(f, normalize(vec3(0.62,0.14,0.78)), 0.62*lift*(1.0 - 0.8*max(thumb, point))));
  // a hand up by the head stands up (a wave, a hello): the fingers point up, a little out
  float high = smoothstep(-0.60, 0.05, W.y)*(1.0 - max(thumb, point));
  f = normalize(mix(f, normalize(vec3(0.22, 1.0, 0.10)), 0.80*high));
  // ... and its palm follows the contract: turn +1 to the camera, 0 in toward the midline, -1 away
  vec3 nCam = normalize(vec3(0.0,0.0,1.0) - f*f.z), nIn = vec3(-1.0,0.0,0.0) - f*(-f.x);
  nIn = normalize(nIn + 1e-4);
  vec3 nWant = normalize(turn >= 0.0 ? mix(nIn, nCam, turn) : mix(nIn, -nCam, -turn));
  nWant = normalize(mix(nWant, -nCam, cover));                             // over the eyes: the palm on the face
  n = normalize(mix(n, nWant, high));
  // the fingers feel their way round the belly, the thigh and the head
  f = steer(f, W, vec3(0.0,-1.50,-0.24), vec3(0.78,0.88,0.70), 1.2);
  f = steer(f, W, vec3(0.56,-1.84,0.34), vec3(0.40,0.34,0.66), 1.8);
  f = steer(f, W, vec3(-sx*0.14,-0.10,0.10), vec3(0.92,0.92,0.96), 1.0);
  f = normalize(f + vec3(1.2,0.0,0.0)*(1.0 - smoothstep(0.06, 0.24, W.x + 0.54*f.x)));   // and stay off the other hand's side
  n = normalize(n - f*dot(n, f));
  vec3 t = -cross(f, n);

  // ---- the four fingers, each on its own curve so the hand never moves as a block
  int base = i*20;
  // point: the index (digit 0, next to the thumb) straight, the others curled into a fist
  float oc = open*(1.0 - point);
  fingerPose(0, mix(open, 1.0, point), sx, W, t, f, n, gDJ[base+0], gDJ[base+1], gDJ[base+2], gDJ[base+3]);
  fingerPose(1, oc, sx, W, t, f, n, gDJ[base+4], gDJ[base+5], gDJ[base+6], gDJ[base+7]);
  fingerPose(2, oc, sx, W, t, f, n, gDJ[base+8], gDJ[base+9], gDJ[base+10], gDJ[base+11]);
  fingerPose(3, oc, sx, W, t, f, n, gDJ[base+12], gDJ[base+13], gDJ[base+14], gDJ[base+15]);
  // ---- the thumb: from low on the palm, swung across it at rest and out to the side when open
  float ot = smoothstep(0.10, 1.0, oc);
  vec3 C = W + t*0.102 + f*0.062 + n*0.030;
  vec3 dt = normalize(t*mix(0.52, 0.88, ot) + f*mix(0.78, 0.46, ot) + n*mix(0.34, 0.10, ot));
  dt = normalize(mix(dt, normalize(t*0.96 + f*0.20 - n*0.05), thumb));     // thumb_*: out straight along the hand's thumb side
  vec3 tw = -t*0.72 + f*0.34 + n*0.60;                                    // it bends across the palm, toward the fingers
  tw = normalize(tw - dt*dot(tw, dt));
  float b1 = mix(mix(0.34, 0.04, ot), -0.05, thumb), b2 = mix(mix(0.52, 0.10, ot), 0.0, thumb);
  gDJ[base + 16] = vec3(sx*C.x, C.yz);
  C += dt*0.128;                                     gDJ[base + 17] = vec3(sx*C.x, C.yz);
  C += (cos(b1)*dt + sin(b1)*tw)*0.102;              gDJ[base + 18] = vec3(sx*C.x, C.yz);
  b1 += b2;  C += (cos(b1)*dt + sin(b1)*tw)*0.084;   gDJ[base + 19] = vec3(sx*C.x, C.yz);
  gTP[i] = vec3(sx*tw.x, tw.yz);

  // ---- out to the screen
  gAS[i] = vec3(sx*S.x, S.yz); gAE[i] = vec3(sx*E.x, E.yz); gAW[i] = vec3(sx*W.x, W.yz);
  gAU[i] = vec3(sx*au.x, au.yz); gAF[i] = vec3(sx*af.x, af.yz);
  vec3 hg = normalize(cross(au, af));
  gAH[i] = vec3(sx*hg.x, hg.yz);
  gHB[i] = mat3(vec3(sx*t.x, t.yz), vec3(sx*f.x, f.yz), vec3(sx*n.x, n.yz));
  // a tight box around the digits, in hand space, from the joints' extents over the whole of open (tools: a python port of this rig)
  float ob = max(open, max(point, thumb));
  vec3 blo = vec3(-0.17 - 0.04*ob, -0.12, -0.075), bhi = vec3(0.25 + 0.175*ob + 0.10*thumb, min(0.52 + 0.32*ob, 0.69), 0.31 - 0.19*open*(1.0 - point));
  gHBc[i] = 0.5*(bhi + blo); gHBh[i] = 0.5*(bhi - blo);
  gHBr[i] = mix(0.112, 0.048, smoothstep(0.0, 0.85, oc*(1.0 - thumb)));
}

void charInit(){
  // body columns: yaw ~0.35 rad (+ turns toward screen right), pitch ~0.15 (+ toward the camera), lean ~0.12 (+ toward
  // screen right), bounce 0.10 units.  gBodyM maps world -> the body's rest frame (inverse of the body's rotation)
  float by = 0.35*clamp(uP[P_BODY_YAW], -1.0, 1.0), bp = 0.15*clamp(uP[P_BODY_PITCH], -1.0, 1.0), bl = 0.12*clamp(uP[P_BODY_LEAN], -1.0, 1.0);
  gBodyM = transpose(rotY(by)*rotX(bp)*rotZ(-bl));
  gBodyT = vec3(0.0, 0.10*clamp(uP[P_BODY_BOUNCE], -1.0, 1.0), 0.0);
  gShrug[0] = clamp(uP[P_SHOULDER_L], 0.0, 1.0); gShrug[1] = clamp(uP[P_SHOULDER_R], 0.0, 1.0);
  gTgOut = clamp(uP[P_TONGUE_OUT], 0.0, 1.0);
  // yaw>0 toward screen right, pitch>0 nose down, roll>0 crown toward screen right (docs/AUDIO.md)
  mat3 F = rotY(YAW0 + uP[P_HEAD_YAW])*rotX(uP[P_HEAD_PITCH])*rotZ(-uP[P_HEAD_ROLL]);
  gHR = transpose(F);
  gBodyR = transpose(rotY(YAW0));
  gHO = NECK + vec3(uP[P_HEAD_X], uP[P_HEAD_Y], uP[P_HEAD_Z]);
  gSmile  = clamp(uP[P_SMILE], 0.0, 1.0);
  gCheekP = clamp(uP[P_CHEEK], 0.0, 1.0);
  gNostril= clamp(uP[P_NOSTRIL], 0.0, 1.0);
  // the eleven mouth columns.  press overrides every opening
  float mo = clamp(uP[P_MOUTH_OPEN], 0.0, 1.0);
  gPress = clamp(uP[P_LIP_PRESS], 0.0, 1.0)*(1.0 - max(mo, gTgOut));
  float keep = 1.0 - gPress;
  gLipO = max(clamp(uP[P_LIP_OPEN], 0.0, 1.0)*keep, max(mo, 0.45*gTgOut));     // commanded opening; a tongue parts the lips
  gUp   = clamp(uP[P_LIP_UPPER], 0.0, 1.0)*keep;
  gLo   = clamp(uP[P_LIP_LOWER], 0.0, 1.0)*keep;
  float wid = clamp(uP[P_LIP_WIDTH], -1.0, 1.0);
  gPk   = max(-wid, 0.0);
  gSp   = max(wid, 0.0);
  gPro  = clamp(uP[P_LIP_PROTRUDE], 0.0, 1.0);
  gTuck = clamp(uP[P_LIP_TUCK], 0.0, 1.0)*keep;
  gGap  = max(clamp(uP[P_TEETH_GAP], 0.0, 1.0), max(mo, 0.40*gTgOut));
  gTgTip  = pow(clamp(uP[P_TONGUE_TIP], 0.0, 1.0), 0.70);    // most of a low raise is hidden by the teeth: a half-raised tip should look half raised
  gTgDent = clamp(uP[P_TONGUE_DENTAL], 0.0, 1.0);
  gTgBack = clamp(uP[P_TONGUE_BACK], 0.0, 1.0);
  gJawA = MAXJAW*gGap;
  gOpen = max(gGap, gLipO);
  gRound = clamp(0.55*gPk + 0.65*gPro, 0.0, 1.0);
  gSquint = clamp(uP[P_SQUINT], 0.0, 1.0);
  gSmirk  = clamp(uP[P_SMIRK], -1.0, 1.0);
  gBlinkL = clamp(uP[P_BLINK_L], 0.0, 1.0); gBlinkR = clamp(uP[P_BLINK_R], 0.0, 1.0);
  gGaze = vec2(uP[P_GAZE_X], uP[P_GAZE_Y]);
  gBrowL = uP[P_BROW_L]; gBrowR = uP[P_BROW_R]; gKnit = clamp(uP[P_BROW_KNIT], 0.0, 1.0);
  gBrowL += 0.5*clamp(uP[P_EYE_WIDE], 0.0, 1.0); gBrowR += 0.5*clamp(uP[P_EYE_WIDE], 0.0, 1.0);      // surprise: brows go with the lids
  // ears: pose + flick about the base.  _l = screen-left.
  // pitch (ear_*: -1 back .. 1 forward) about the base, yaw (ear_*_turn: -1 the cup faces forward/in .. 1 out), a flick a
  // quick tilt.  ranges widened so a still shows them; the disc is mirrored, so one sign serves both sides
  float aL = 0.60*uP[P_EAR_L] + 0.34*uP[P_EAR_FLICK_L], aR = 0.60*uP[P_EAR_R] + 0.34*uP[P_EAR_FLICK_R];
  gEarRL = rotY(-0.80*uP[P_EAR_L_TURN])*rotX(-aL)*rotZ(0.30*uP[P_EAR_FLICK_L]);
  gEarRR = rotY(-0.80*uP[P_EAR_R_TURN])*rotX(-aR)*rotZ(0.30*uP[P_EAR_FLICK_R]);
  gBreath = uP[P_BREATH];
  armPose(0); armPose(1);
  tailPose();

  // ---- the skin around the mouth: pucker squeezes it in and brings it forward, spread stretches it round the muzzle
  gKx = 0.56*gPk + 0.16*gPro - 0.27*gSp;
  gKz = 0.055*gPk + 0.100*gPro;

  // ---- the corners, in x
  float spk = gSp*(1.0 - 0.50*gPress);     // a closure draws the corners back in from a grin
  gW = W0*(1.0 + 0.26*spk - 0.80*gPk + 0.06*(gSmile - 0.29))*(1.0 - 0.08*gLipO)*(1.0 - 0.03*gPress - 0.12*gTuck);

  // ---- the opening: two arcs, each lip its own.  The long upper lip lifts off the teeth on its own column.
  float au = 0.072*gLipO + 0.094*gUp;
  float al = 0.255*gLipO + 0.060*gLo;
  // a round O needs height as well as narrowness: the pucker pulls both heights toward the width
  float rnd = gPk*sqrt(gPk)*smoothstep(0.0, 0.16, gLipO + 0.5*(gUp + gLo));
  float rh = 0.012 + 0.40*gLipO;           // in scale with lip_open: a pucker does not gape
  au = mix(au, min(0.92*gW*(1.0 + 0.3*gLipO), rh), rnd);
  al = mix(al, min(1.00*gW*(1.0 + 0.5*gLipO), 1.1*rh), rnd);
  au += 0.034*gTuck;                       // /f/: the upper lip clears the teeth ...
  al *= 1.0 - 0.95*gTuck;                  // ... and the lower lip comes up to them
  float amt = au + al;
  float drop = mix(0.20*al, 0.5*(al - au), rnd);   // the corners ride down with the lower lip; an O keeps them at its middle
  gYC = Y0 - drop;
  gAu = au + drop;
  gAl = al - drop;
  gHa = 0.5*amt;
  gNoseY = 0.020*gUp + 0.006*gSp - 0.014*gPro + 0.008*gTuck - 0.009*gGap + 0.050*clamp(uP[P_NOSE_WIGGLE], -1.0, 1.0);   // the long upper lip pulls the nose with it, a wide jaw draws it down
  gNoseW = 1.0 + 0.09*gSp + 0.06*gUp - 0.10*gPk;              // and its wings spread with a grin, draw in with a pucker
  gRc = max(0.95*rnd*min(gAu, gAl), min(0.40*min(gAu, gAl), 0.020));   // soft corners at any opening, all the way to a circle
  // pressed lips stick: they peel apart from the middle, the seam stays shut out to the corners until the press lets go
  gStick = smoothstep(0.04, 0.85, gPress);
  gWi = max(gW*(1.0 - 0.62*gStick) - gRc, 0.004);
  float hu = clamp(gAu - gRc, 0.0009, 0.98*gWi), hl = clamp(gAl - gRc, 0.0009, 0.98*gWi);   // never past a semicircle: the corner branch below stays valid
  gRu = 0.5*(gWi*gWi + hu*hu)/hu; gCu = gRu - hu;
  gRl = 0.5*(gWi*gWi + hl*hl)/hl; gCl = gRl - hl;
  float o = smoothstep(0.0, 0.6, gLipO);
  gBend = (0.012 + 0.105*gSmile + 0.03*gCheekP + 0.070*spk)*(1.0 - 0.50*o)*(1.0 - 0.45*gPress)*(1.0 - 0.90*gPk);

  gSeal = 1.0 - smoothstep(0.0, 0.010, amt);
  gTub  = 0.0050*(1.0 - smoothstep(0.0, 0.10, amt))*(1.0 - gPk);
  gOpenK = smoothstep(0.0, 0.07, amt);

  // ---- the two lips
  const float HU = 0.0540, HL = 0.0720, TU = 0.0230, TL = 0.0300;
  gCornT = smoothstep(0.15, 0.85, gPk);
  gCornMin = mix(0.26 + 0.36*smoothstep(0.0, 0.20, amt) - 0.07*gSp, 1.0, gCornT);
  gCapU = 0.5*(HU + HL)*gCornMin*0.80/gW;
  gPad = sq(1.0 - gCornT)*(1.0 - 0.72*smoothstep(0.05, 0.40, gLipO));
  float thin = 1.0 - 0.26*spk - 0.10*gLipO;
  float hm = 0.5*(HU + HL), tm = 0.5*(TU + TL);
  float bulge = 1.0 + 0.16*bell((gPress - 0.62)/0.38);                  // the pressure builds: the lips fill just before they part
  gHU = mix(HU, hm*0.90, gPk)*(thin + 0.26*gPk)*(1.0 - 0.30*gPress)*bulge;
  gHL = mix(HL, hm, gPk)*(thin + 0.26*gPk)*(1.0 - 0.30*gPress)*(1.0 - 0.10*gTuck)*bulge;
  gTU = mix(TU, tm, gPk)*(1.0 + 0.30*gPress + 0.34*gPk + 0.46*gPro)*bulge;     // pressed flesh thickens: it has to go somewhere
  gTL = mix(TL, tm, gPk)*(1.0 + 0.30*gPress + 0.34*gPk + 0.46*gPro)*(1.0 - 0.15*gTuck)*bulge;
  gDU = -0.004*keep - 0.003*gPress + 0.018*gPro;                         // sunk a little, so the lip rises out of the skin
  gDL = gDU - 0.006*gTuck;

  // ---- the carve, closed at rest; dark first, colour and teeth with room
  gHoleM = mix(-0.024, 0.030, smoothstep(0.0, 0.020, amt));
  gDark  = 1.0 - smoothstep(0.012, 0.050, amt);
  gShow  = smoothstep(0.012, 0.032, amt + 0.03*gTuck);
  gTunnel = smoothstep(0.10, 0.70, gPk)*smoothstep(0.10, 0.80, gPro);

  // ---- teeth wait behind the lip until there is room
  gTipUY = TIP_U + 0.030*(1.0 - gShow);
  gTipLY = TIP_L - 0.030*(1.0 - gShow);

  // ---- tongue targets, in the jaw frame: behind the upper teeth, and the middle of the bite
  const float TG_Y = Y0 - 0.095;
  float c = cos(gJawA), s = sin(gJawA);
  float tipJ = HINGE.y + c*(TIP_U - HINGE.y) + s*(0.66 - HINGE.z);
  gToTip = tipJ - 0.014 - TG_Y;
  gToGap = tipJ - 0.026 - TG_Y;
  gTgFwd = 0.085*gTgDent + 0.30*gTgOut;
  gTgDim = 1.0 - smoothstep(0.035, 0.120, amt);
}


// outputs of monkeyEval (parts, for fur + materials)
float gFur, gBare, gMask, gFurLen, gBowl, gWBody, gWJaw, gHeart, gM, gRing, gHole, gMV, gMX, gNoseM, gNoseD, gBrowD;
float gE, gU, gUpperW, gLipH, gDepthS, gTgX, gToothL, gSect, gArX, gArZ;
vec4  gToothSt;
vec3  gRest;            // rest-pose coordinate (fur + colour stick to it)
float gEye, gTeeth, gTongue, gEarD, gBodyD, gHeadD, gEarS, gGrooveT, gEarM, gBrowP, gSkinD, gBuS;
vec3  gQ, gQJ, gQE, gQTJ;

float sdEarShape(vec3 e){
  // e: disc in xy, z out of the ear's face.
  float d = sdEllipsoid(e, vec3(0.292,0.325,0.090));
  // a true sphere for the dish: sdEllipsoid overestimates ~3x INSIDE a flat ellipsoid, and subtracting
  // one makes the march tunnel straight through this thin ear
  float bowl = length((e - vec3(0.0,0.0,0.425))*vec3(1.0,0.90,1.0)) - 0.420;
  gBowl = bowl;
  d = smax(d, -bowl, 0.04);
  // the soft inner fold of a real ear
  d = smin(d, sdEllipsoid(e - vec3(-0.075,-0.04,0.012), vec3(0.058,0.135,0.040)), 0.05);
  return d;
}

// the heart-shaped bare face, in rest space; roughly world units, negative inside
float heart(vec3 r){
  vec2 p = vec2(abs(r.x), r.y);
  float l = length((p - vec2(0.238,0.150))/vec2(0.272,0.292)) - 1.0;   // the two lobes around the eyes
  l = smax(l, ((p.x - 0.46) + 0.9*(p.y - 0.22))*3.3, 0.20);             // no bare corner past the brow's outer end
  float m = length((p - vec2(0.0,-0.240))/vec2(0.405,0.415)) - 1.0;    // the muzzle and chin
  return smin(l, m, 0.42)*0.30;
}

// A row of soft sculpted teeth on a dental arch.  w = height above the biting edge, into the gum.  x, z in plan.
// Each tooth is convex across its face, rounded at the biting edge, and meets its neighbours in a soft valley, so
// the row has no hairline gaps to alias.  The gum over the roots is the same surface; paint separates them by the
// scalloped gum line, never by comparing two coincident distances.
// st = (arc length from the middle, across the tooth 0 centre .. 1 edge, height above the gum line, radius from the arch)
float teethRow(float x, float z, float w, vec3 arch, float tw, float len, float th, float cut, out vec4 st){
  float gum;
  vec2 e = vec2(x/arch.x, (z - arch.z)/arch.y);
  float R = 0.5*(arch.x + arch.y);
  float rad = (length(e) - 1.0)*R;
  float a = atan(abs(e.x), e.y)*R/tw;              // tooth number from the middle, running
  float c = 2.0*(fract(a) - 0.5);                  // -1 .. 1 across one tooth
  float cc = c*c;
  float L = len*(1.0 - 0.050*a) + 0.008*bell((a - 2.5)/0.9);          // shortening backward; the small canine reaches down
  float tip = 0.013*cc*cc + 0.005*bell((a - 2.5)/0.9)*abs(c) - w;     // the biting edge, scalloped; the canine comes to a soft point
  float band = abs(rad + 0.0050*cc) - th;
  float d = smax(band, tip, 0.008);
  float top = L*(0.74 + 0.26*(1.0 - cc));          // the gum line arches over each tooth
  gum = smax(rad + 0.0050*cc - th - 0.0025, smax(top - w, -rad - 0.040, 0.010), 0.012);
  gum = max(gum, w - 0.165);
  d = smin(max(d, w - L*1.25), gum, 0.005);
  // the body of the teeth behind their faces: the notches between tips show shaded tooth, never black specks where two rows meet
  d = min(d, max(max(abs(rad + 0.024) - 0.009, -w), w - L));
  float arc = a*tw;                                // the row ends where the dark has already taken it
  st = vec4(arc, abs(c), w - top, rad);
  return smax(d, arc - cut, 0.012);
}


// ---- the arms: fur-covered limbs.  r(h) is a line plus a muscle belly
float sdLimb(vec3 p, vec3 a, vec3 b, float r1, float r2, float belly, out float h){
  vec3 pa = p - a, ba = b - a;
  h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  return length(pa - ba*h) - mix(r1, r2, h) - belly*4.0*h*(1.0 - h);
}
float sdPhal(vec3 p, vec3 a, vec3 b, float r1, float r2){       // a phalanx is waisted: its ends are the knuckles
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  return length(pa - ba*h) - mix(r1, r2, h) + 0.0030*4.0*h*(1.0 - h);
}
float gArmU, gArmDS;     // along the straight arm, distance from the shoulder
float gHandD, gHandI, gWArm, gLegM;
vec3  gArmRest, gArmDir;
float armEval(vec3 b){
  float dL, dR;
  float hu, hf;
  float d0 = smin(sdLimb(b, gAS[0], gAE[0], 0.135, 0.086, 0.014, hu), sdLimb(b, gAE[0], gAW[0], 0.092, 0.064, 0.018, hf), 0.05);
  float d1 = smin(sdLimb(b, gAS[1], gAE[1], 0.135, 0.086, 0.014, hu), sdLimb(b, gAE[1], gAW[1], 0.092, 0.064, 0.018, hf), 0.05);
  float dA = min(d0, d1);
  gArmU = 0.0; gArmDS = 9.0; gArmDir = vec3(0.0,-1.0,0.0); gArmRest = vec3(0.0);
  for(int once=0; once<1 && dA < 0.24; once++){
    float m = d0 < d1 ? 0.0 : 1.0;
    vec3 S = mix(gAS[0], gAS[1], m), E = mix(gAE[0], gAE[1], m), u = mix(gAU[0], gAU[1], m), f = mix(gAF[0], gAF[1], m), g = mix(gAH[0], gAH[1], m);
    // the straight arm: unbend the elbow, keep the distance along the bones.  Fur and paint live on this
    float bis = dot(b - E, normalize(f - u));
    float wF = smoothstep(-0.05, 0.05, bis);
    vec3 cu = vec3(dot(b - S, u), dot(b - S, g), dot(b - S, cross(g, u)));
    vec3 cf = vec3(BONE_U + dot(b - E, f), dot(b - E, g), dot(b - E, cross(g, f)));
    vec3 c = mix(cu, cf, wF);
    gArmU = c.x;
    gArmDS = length(b - S);
    gArmDir = normalize(mix(u, f, wF));
    float sx = 2.0*m - 1.0;
    // furCoord reads only the xy angle round its pole, so the arm's section is laid flat in xy, by arc length: every side of the
    // arm gets the same strand density whichever way the arm has turned.  The front and back share coordinates, unseen
    float rho = max(length(c.yz), 1e-3);
    vec3 r0 = vec3(sx*SHLD.x, SHLD.yz);
    r0.xz = rot(-BYAW)*r0.xz;
    r0 = gBodyR*(r0 - NECK) + NECK;
    vec3 d0r = vec3(normalize(r0.xy - vec2(0.0,-0.25)), 0.0);        // straight out from the body's fur pole: strands run down the arm
    gArmRest = r0 + d0r*c.x + vec3(-d0r.y, d0r.x, 0.0)*rho*asin(clamp(c.z/rho, -1.0, 1.0));
  }
  return dA;
}
// ---- the hands: bare skin.  q is hand space, x toward the thumb, y toward the fingertips, z out of the palm
float palmD(vec3 q){
  float w = mix(0.070, 0.112, smoothstep(0.015, 0.20, q.y));
  vec3 c = q - vec3(0.0,0.158,0.0);
  vec3 bx = abs(c) - vec3(w, 0.125, 0.020);
  float d = length(max(bx, 0.0)) + min(max(bx.x, max(bx.y, bx.z)), 0.0) - 0.030;
  d = smin(d, sdEllipsoid(q - vec3(0.062,0.090,0.040), vec3(0.058,0.092,0.044)), 0.030);     // thenar
  d = smin(d, sdEllipsoid(q - vec3(-0.078,0.112,0.030), vec3(0.042,0.100,0.036)), 0.030);    // hypothenar
  // the knuckles: one sphere, the nearest of the four, on the arc of the metacarpal heads
  float kc = clamp(floor((q.x + 0.135)/0.0695), 0.0, 3.0)*0.0695 - 0.100;
  d = smin(d, length((q - vec3(kc, 0.290 - 2.2*kc*kc, -0.022))*vec3(1.0,0.82,1.0)) - 0.0325, 0.022);
  return d;
}
// llvmpipe: an array read with a run-time index goes lane by lane, and one such read anywhere keeps the whole array in memory.  So
// every read of the rig below is at a constant index: the macros write the fifteen phalanges out, once per hand
// Inside the hand's box each digit is first a capsule round its chord, a bound only, and its three phalanges run only near it; the
// same for the palm and its box.  Every switch sits farther out than the 0.03 over which the coat lies down, so that has no seam
#define PHAL(j,r1,r2) dd = min(dd, sdPhal(b, gDJ[(j)], gDJ[(j)+1], r1, r2));
#define DIGIT(j,r,R) ch = sdCapsule(b, gDJ[j], gDJ[(j)+3], R); dd = ch + 0.012; \
  for(int o=0; o<1 && ch < 0.04; o++){ dd = 1e3; PHAL(j,r,r*0.93) PHAL((j)+1,r*0.93,r*0.84) PHAL((j)+2,r*0.84,r*0.72) } \
  dg = min(dg, dd);
#define HANDD(name,I,B) float name(vec3 b, vec3 q){ \
  vec3 pe = abs(q - vec3(0.0,0.135,0.015)) - vec3(0.165,0.205,0.090); \
  float pd = length(max(pe, 0.0)) + min(max(pe.x, max(pe.y, pe.z)), 0.0); \
  float dg = 1e3, dd, ch, R = gHBr[I]; \
  for(int o=0; o<1 && pd < 0.05; o++) pd = smin(palmD(q), sdTaper(b, gAW[I] - gAF[I]*0.05, gAW[I] + gHB[I][1]*0.065, 0.058, 0.066), 0.04); /* the wrist: its root is inside the cuff */ \
  DIGIT(B,DIG_R[0],R) DIGIT(B+4,DIG_R[1],R) DIGIT(B+8,DIG_R[2],R) DIGIT(B+12,DIG_R[3],R) DIGIT(B+16,DIG_R[4],R) \
  return smin(pd, dg, 0.024); }
HANDD(handL,0,0)
HANDD(handR,1,20)
void handsEval(vec3 b){
  const float G = 0.04;
  vec3 q = (b - gAW[0])*gHB[0];
  vec3 e = abs(q - gHBc[0]) - gHBh[0];
  float d = length(max(e, 0.0)) + min(max(e.x, max(e.y, e.z)), 0.0);     // never a constant: a flat field zeroes the engine's normal
  for(int k=0; k<1 && d < G; k++) d = handL(b, q);
  gHandD = d;
  q = (b - gAW[1])*gHB[1];
  e = abs(q - gHBc[1]) - gHBh[1];
  d = length(max(e, 0.0)) + min(max(e.x, max(e.y, e.z)), 0.0);
  for(int k=0; k<1 && d < G; k++) d = handR(b, q);
  gHandI = d < gHandD ? 1.0 : 0.0;
  gHandD = min(gHandD, d);
}

void headEval(vec3 pw){
  vec3 q = gHR*(pw - gHO) + NECK;
  gQ = q;
  float s = (q.x < 0.0) ? -1.0 : 1.0;
  vec3 qa = vec3(abs(q.x), q.yz);

  // ---- the skin around the mouth moves first: a pucker draws it in and forward, a grin out and back
  vec3 md = vec3(q.x/0.33, (q.y - gYC)/(0.21 + gHa), (q.z - 0.78)/0.40);
  float wm = exp(-dot(md, md));
  vec3 qm = q;
  qm.x *= 1.0 + wm*gKx;
  qm.z += wm*(-gKz + 0.075*gSp*sq(q.x/0.30));
  vec3 pa = vec3(abs(qm.x), qm.yz);
  vec3 qj = jawSpace(qm, HINGE, -gJawA);
  gQJ = qj;

  // ---- skull
  float skull = sdEllipsoid(q - vec3(0.0,0.075,-0.05), vec3(0.625,0.590,0.590));
  float lift = gCheekP*0.040 + gSmile*0.040 + 0.030*gSp + 0.012*gUp + 0.012*gSquint - 0.050*gGap + 0.024*max(gSmirk*s, 0.0);
  float cheek = sdEllipsoid(qa - vec3(0.375 - 0.018*gGap,-0.150 + lift,0.300), vec3(0.235,0.215 + 0.012*gGap,0.235 - 0.020*gPk + 0.008*gPress));
  skull = smin(skull, cheek, 0.11);
  // soft side tufts along the jaw and a cap on top: a bit of mischief in the silhouette
  float fluff = sdTaper(qa, vec3(0.50,-0.10,-0.02), vec3(0.76,-0.21,-0.10), 0.17, 0.012);
  fluff = smin(fluff, sdTaper(qa, vec3(0.45,-0.26,0.00), vec3(0.64,-0.46,-0.08), 0.15, 0.012), 0.07);
  fluff = smin(fluff, sdTaper(qa, vec3(0.32,-0.42,0.02), vec3(0.42,-0.64,-0.04), 0.13, 0.012), 0.07);
  skull = smin(skull, fluff, 0.08);
  float cap = sdTaper(q, vec3(-0.02,0.48,0.12), vec3(-0.075,0.735,0.215), 0.19, 0.022);
  cap = smin(cap, sdTaper(q, vec3(0.16,0.50,0.02), vec3(0.235,0.680,0.03), 0.17, 0.030), 0.10);
  cap = smin(cap, sdTaper(q, vec3(-0.20,0.47,0.02), vec3(-0.300,0.640,0.02), 0.15, 0.030), 0.10);
  cap = smin(cap, sdTaper(q, vec3(0.02,0.48,-0.22), vec3(0.03,0.690,-0.30), 0.17, 0.030), 0.10);
  skull = smin(skull, cap, 0.10);

  skull = smin(skull, sdEllipsoid(qa - vec3(0.250,-0.085 + lift*0.5,0.500), vec3(0.190,0.075,0.105)), 0.08);

  // ---- the big brows.  This animal acts with them.
  float bu = (q.x < 0.0) ? gBrowL : gBrowR;
  vec3 bq = qa - vec3(0.255, 0.378 + 0.150*bu - 0.040*gKnit, 0.440 - 0.075*max(bu, 0.0));
  bq.xy = rot(0.20 - 0.70*gKnit - 0.16*bu)*bq.xy;   // rest: inner ends a little high.  Knit: down and in
  bq.x += 0.060*gKnit;
  bq.y += 0.60*bq.x*bq.x;                          // arched
  float btap = 1.0 - 0.40*smoothstep(-0.12, 0.20, bq.x);     // fat at the inner end, thin outside
  float brow = sdEllipsoid(bq, vec3(0.190,0.050,0.112));
  gBrowP = (length(bq.xy/vec2(0.195,0.050*btap)) - 1.0)*0.050;
  gBrowD = brow;
  skull = smin(skull, brow, 0.075);

  // ---- muzzle: upper part in head space, lower in the jaw's rest frame, blended into ONE closed form
  float mU = sdEllipsoid(qm - vec3(0.0,-0.165,0.490), vec3(0.335,0.205,0.395));
  mU = smin(mU, sdEllipsoid(pa - vec3(0.130,-0.190,0.680), vec3(0.170,0.125,0.195)), 0.06);
  float mL = sdEllipsoid(qj - vec3(0.0,-0.348,0.490), vec3(0.280,0.188,0.315));
  mL = smin(mL, sdEllipsoid(qj - vec3(0.0,-0.438 + 0.014*gPress + 0.010*gPk,0.600 + 0.008*gPress), vec3(0.128,0.072,0.105)), 0.09);
  float M = smin(mU, mL, 0.055);
  // the cheek web: a third form between the two, half way round the hinge, that stretches as the jaw drops.  Well
  // inside the muzzle at rest.  Without it the opening runs off the muzzle at the corners and the lips have nothing to ride
  float wG = smoothstep(0.0, 0.25, gGap);
  M = smin(M, sdEllipsoid(0.5*(qm + qj) - vec3(0.0,-0.262,0.490), vec3(0.240 + 0.052*wG, 0.130 + 0.200*gGap, 0.270 + 0.040*wG)), 0.010 + 0.060*wG);
  M /= 1.0 + 0.55*max(gKx, 0.0);                    // the pucker warp squeezes the field
  gM = M;
  gWJaw = smoothstep(-0.03, 0.03, mU - mL);

  // ---- the aperture: two arcs through the corners.  e < 0 inside, 0 on the inner lip edge, > 0 out through the lips
  float x = q.x - 0.030*gSmirk, ax = abs(x), y = q.y;           // smirk: the whole line slides toward one corner ...
  float u = ax/gW, uu = u*u;
  float v = y - gYC - gBend*(u < 1.0 ? uu : 2.0*u - 1.0) - 0.026*gSmirk*x/gW + gTub*exp(-x*x/0.0048);   // ... and that corner climbs.  The tubercle dips the line at the middle
  gMV = v; gMX = x; gU = u;
  float eU = length(vec2(ax, v + gCu)) - gRu;
  float eL = length(vec2(ax, v - gCl)) - gRl;
  float e = max(eU, eL);
  float cc = v > 0.0 ? gCu : gCl;
  // past a corner the corner itself is nearest, for an arc shallower than a semicircle only
  if(cc > 0.0 && gWi*(abs(v) + cc) < cc*ax) e = length(vec2(ax - gWi, v));
  e -= gRc;
  e = min(e, length(vec2(max(ax - gW, 0.0), v)));    // the seam, out to the corners: a no-op unless the opening is held in by the stick
  gE = e;

  // ---- the skin the lips ride on
  float lipTop = gYC + gAu, lipBot = gYC - gAl;
  float D = M;
  // the funnel: the whole mouth comes forward, the long upper lip leading
  D -= 0.030*gPro*bell(length(vec2(x/0.27, (y - lipTop - 0.01)/0.14)));
  // a lifted upper lip bunches under the nose; a drawn lower lip bunches above the chin
  D -= 0.016*gUp*bell(length(vec2(x/0.24, (y - lipTop - 0.070)/0.052)));
  // /f/: the skin under the mouth line folds in, and the lower lip goes with it
  D += 0.018*gTuck*bell(length(vec2(x/0.30, (y - Y0 + 0.085)/0.095)));
  // press: the skin bunches softly above and below the sealed lips, the chin pad comes up
  D -= (0.012*gPress + 0.006*gPro)*bell(length(vec2(x/0.20, (y - lipBot + gHL + 0.060)/0.040)));
  D -= 0.009*gPress*bell(length(vec2(x/0.20, (y - lipTop - 0.070)/0.040)));
  // the crease under the lower lip rides the lip, deeper when it is drawn, pressed or pushed
  D += (0.0030 + 0.0060*gLo + 0.0050*gPress + 0.0040*gPro)*bell(length(vec2(x/0.17, (y - lipBot + gHL + 0.016)/0.026)));
  // the fold from the nose wing round the mouth: there at rest, cut deeper by a grin, a lifted lip, a smirk on its side
  // it swings out early and passes just outside the corner, wherever a grin has lifted the corner to
  float tf = clamp((-0.125 - y)/(-0.065 - gYC - gBend), 0.0, 1.0);
  D += (0.0016 + 0.0040*gSp + 0.0026*gUp + 0.0040*max(gSmirk*s, 0.0))*(1.0 - gPk)*bell((ax - mix(0.135, gW + 0.058, tf*(2.0 - tf)))/0.052)*bell(2.0*tf - 1.0);
  // corners: pulled or pressed they dimple
  D += (0.0040 + 0.0055*gSp + 0.0060*gPress)*(1.0 - 0.80*gOpenK)*(1.0 - smoothstep(0.0, 0.5, gPk))*bell(length(vec2((ax - gW - 0.012)/0.040, v/0.052)));

  // ---- each lip: a soft section around the curve, overlapping the other a little at the line
  float T = 1.0 - (1.0 - gCornMin)*smoothstep(0.40, 1.02, u);   // tapers to the corner and stops at a soft end, or not at all in a ring
  float capT = clamp((u - 1.0)/(gCapU + 1e-3), 0.0, 1.0);
  T *= mix(sq(max(1.0 - capT*capT, 0.0)), 1.0, gCornT);
  float ws = mix(0.0045, 0.80*(gW + 0.056), gCornT); // a ring blends upper into lower around its sides
  float wu = smoothstep(-ws, ws, v);
  gUpperW = wu;
  float rx = x*W0/gW;
  float bow = 1.0 + (1.0 - gPk)*(0.12*bell((abs(rx) - 0.056)/0.076) - 0.13*bell(rx/0.030));   // cupid's bow, soft dip
  float hAvg = 0.5*(gHU + gHL);                      // both lips arrive at the corner equal: one soft end
  float kc = smoothstep(0.55, 0.98, u);
  float h = mix(mix(gHL, hAvg, kc), mix(gHU, hAvg, kc)*bow, wu)*T;
  gLipH = h;
  float dc = mix(gDL, gDU, wu);
  float t = mix(mix(gTL, gTU, wu), 0.5*(gTL + gTU), kc)*sqrt(T + 1e-3);
  const float OV = 0.26;
  float hh = max(0.5*h, 0.0030), tt = max(t, 0.0030);
  gSect = (e - hh*(1.0 - OV))/hh;                    // -1 inner edge .. 0 the crown .. 1 outer edge
  vec2 tp = vec2(e - hh*(1.0 - OV), D - dc);
  vec2 tr = vec2(hh, tt);
  float k0 = length(tp/tr), k1 = length(tp/(tr*tr));
  float ring = k0*(k0 - 1.0)/max(k1, 1e-4);
  // the corner: a small soft pad where the lips meet, never a knife point
  vec3 pr = vec3(0.026, 0.014, 0.013)*gPad + 0.001;
  ring = smin(ring, sdEllipsoid(vec3(ax - gW + 0.006 - 1.3*gRc, v, D - 0.5*(gDU + gDL)), pr) + 0.03*(1.0 - smoothstep(0.0, 0.3, gPad)), 0.013);
  ring = max(ring, 0.54 - q.z);                      // the far side of the muzzle carries no lips
  gRing = ring;
  gDepthS = (D - dc)/tt;

  // ---- the carve: closed at rest, wider under the lips and behind them, a pocket with the jaw
  float ce = e - min(gHoleM*T, 0.62*h) - 0.42*clamp(-D - 0.016, 0.0, 0.17)*smoothstep(0.0, 0.012, gHoleM);
  float hole = smax(ce, 0.13 - q.z, 0.05);
  // the mouth itself: a cavity inside the dental arches, roof on the skull, floor on the jaw.  It is always there, sealed
  // inside the muzzle; only the aperture opens it, so nothing has to scale in and the 3/4 view sees past no wall
  vec3 qjr = jawSpace(q, HINGE, -gJawA);
  float cx = max(abs(qm.x), abs(q.x));               // squeezed by a pucker, never stretched toward the cheek valley by a grin
  float cavU = (length(vec2(cx/0.262, (qm.z - 0.460)/0.312)) - 1.0)*0.262 + 0.030*gRound;
  float cavL = (length(vec2(cx/0.240, (qj.z + 0.040*gTuck - 0.460)/0.302)) - 1.0)*0.240 + 0.030*gRound;
  cavU = smax(cavU, max(q.y - (Y0 + 0.125), (Y0 - 0.040) - q.y), 0.03);
  cavL = smax(cavL, max(qjr.y - (Y0 + 0.030 + 0.30*gGap), (Y0 - 0.145) - qjr.y), 0.03);
  hole = smin(hole, smax(min(cavU, cavL), 0.13 - q.z, 0.05), 0.03);
  gHole = hole;

  // ---- nose: a small soft button, skin not rubber
  vec3 nq = qm - NOSEP - vec3(0.0, gNoseY, 0.0);
  float nose = sdEllipsoid(nq, vec3(0.098*gNoseW,0.058,0.062));
  vec3 nna = vec3(abs(nq.x), nq.yz);
  float nostril = length((nna - vec3(0.046*gNoseW,-0.022,0.048))*vec3(1.0,1.5,1.0)) - (0.0165 + 0.009*gNostril);
  gNoseD = nose;
  float philt = (0.009*exp(-qm.x*qm.x/0.0018) - 0.0016*exp(-sq((abs(qm.x) - 0.052)/0.026)))*smoothstep(-0.10,-0.13,q.y)*smoothstep(Y0 - 0.02, Y0 + 0.06, q.y);   // groove and its two columns
  float muz = smin(D + philt, nose, 0.040);

  float headRaw = smin(skull, muz, 0.070);
  // sockets
  vec3 ea = qa - EYEC;
  headRaw = smax(headRaw, -(length(ea) - (EYER + 0.010)), 0.060);
  headRaw = smax(headRaw, -nostril, 0.016);
  // the opening through everything, then the lips on top of it
  float skin = smin(smax(headRaw, -hole, 0.010), ring, 0.022 + 0.010*gPro);

  // ---- teeth, rigid on their arches: upper row on the skull, lower row on the jaw.  Only inside the carve
  vec4 stU, stL;
  // the arches lean a little toward the camera, about their front: the far limb comes round and fills the far corner
  const float AC = 0.9872, AS = 0.1593;
  float axU = AC*q.x - AS*(q.z - 0.74),   azU = AS*q.x + AC*(q.z - 0.74) + 0.74;
  float axL = AC*q.x - AS*(qjr.z - 0.74), azL = AS*q.x + AC*(qjr.z - 0.74) + 0.74;
  float tU = teethRow(axU, azU, q.y - gTipUY, ARCH_U, 0.0590, 0.069, 0.016, 0.46, stU);
  float tL = teethRow(axL, azL, gTipLY - qjr.y, ARCH_L, 0.0480, 0.056, 0.014, 0.16, stL);
  gToothL = tL < tU ? 1.0 : 0.0;
  gToothSt = tL < tU ? stL : stU;
  gArX = tL < tU ? axL : axU; gArZ = tL < tU ? azL : azU;
  float clipT = hole + 0.007;
  gTeeth = max(min(tU, tL), clipT);
  gQTJ = qjr;

  // ---- tongue: lies low in the jaw; tip up behind the upper teeth, tip out between the teeth, or its back raised
  vec3 tq = qjr - vec3(-0.22*(0.74 - qjr.z), Y0 - 0.095, 0.465 + gTgFwd);   // sheared along the view axis: mid-opening in the 3/4 view
  float tz = clamp((tq.z + 0.265)/0.530, 0.0, 1.0);  // 0 back .. 1 tip
  tq.y += 0.13*(tq.z < 0.0 ? tq.z : tq.z*(1.0 - max(gTgTip, gTgDent)));   // the back rises into the throat, the idle tip dips
  tq.y -= gTgTip*gToTip*smoothstep(0.42, 1.0, tz) + gTgDent*gToGap*smoothstep(0.30, 0.95, tz);
  tq.y -= gTgBack*0.135*bell((tz - 0.30)/0.36);
  tq.x *= 1.0 + 0.60*gTgTip*smoothstep(0.45, 1.0, tz) - 0.22*gTgBack;   // a raised tip is narrower, a raised back spreads
  tq.y -= gTgOut*(0.050 - 0.10*smoothstep(0.55, 1.0, tz));                // out: up onto the lower lip, the tip dipping over it
  gTgX = tq.x/0.155;
  float tongue = sdEllipsoid(tq, vec3(0.155,0.050,0.265))*0.72;
  tongue += 0.0018*smoothstep(0.022, 0.0, abs(tq.x)); // centre groove
  gTongue = max(tongue, clipT - 0.40*gTgOut*smoothstep(0.68, 0.78, qjr.z)*smoothstep(Y0 - 0.20, Y0 - 0.10, qjr.y));     // out: only the front passes the lips

  // ---- eyeballs
  gEye = length(ea) - EYER;

  // ---- eyelids
  float blink = (q.x < 0.0) ? gBlinkL : gBlinkR;
  vec3 el = ea;
  el.xy = rot(-EYETILT)*el.xy;
  float squint = gSmile*0.13 + gCheekP*0.14 + 0.26*gSquint;
  float wide = clamp(uP[P_EYE_WIDE], 0.0, 1.0);
  float aU = mix(0.56 + 0.30*wide, -0.30, blink) + 0.07*bu;
  float aL = mix(-0.70 + squint - 0.10*wide, -0.34, blink);
  float lids = sdEyelids(el, EYER + 0.008, 0.013, aU, aL);
  skin = smin(skin, lids, 0.030);

  // ---- ears: big, round, standing out
  vec3 e2 = q - vec3(s*EARB.x, EARB.y, EARB.z);
  e2.x *= s;
  e2.xz = rot(0.52)*e2.xz;           // the disc faces forward and out
  e2.yz = rot(-0.10)*e2.yz;
  e2 = ((q.x < 0.0) ? gEarRL : gEarRR)*e2;
  gQE = e2; gEarS = s;
  float ear = sdEarShape(e2);
  gEarD = ear;
  float head = smin(skin, ear, 0.05);
  gHeadD = head;
  gSkinD = skin; gBuS = bu;
}

// Below the head only the body and arms are needed for the coat's rest frame and length: the tail of monkeyEval skips the head's
// paint fields there.  The head itself is always evaluated: behind a loop guard its many globals leave the registers (measured 1.3x)
const vec3  HB_C = vec3(0.0,0.03,0.06), HB_R = vec3(1.12,0.90,1.02);
const float HB_G = 0.27;

void monkeyEval(vec3 pw){
  float dH = (length((gHR*(pw - gHO) + NECK - HB_C)/HB_R) - 1.0)*HB_R.y;
  headEval(pw);      // always: behind a loop guard the head's many globals cannot stay in registers, and that costs more than it saves
  vec3 q = gQ, qj = gQJ, qjr = gQTJ, e2 = gQE, ea = vec3(abs(q.x), q.yz) - EYEC;
  float s = gEarS, skin = gSkinD, ear = gEarD, head = gHeadD, hole = gHole, bu = gBuS;

  // ---- body, world space (does not follow the head except a touch of breath)
  vec3 b = pw;
  float br = 1.0 + 0.018*gBreath;
  b.y += 0.012*gBreath*smoothstep(-2.0, -1.2, b.y);       // the seat stays on the log
  b.xz = rot(BYAW)*b.xz;
  // seated: a short torso over the hips, thighs forward, shins hanging down the front of the log
  float torso = smin(sdEllipsoid(b - vec3(0.0,-1.30,-0.30), vec3(0.70,0.58,0.52)*br),
                     sdEllipsoid(b - vec3(0.0,-1.70,-0.20), vec3(0.68,0.48,0.60)*br), 0.22);
  float neck  = sdTaper(b, vec3(0.0,-0.38,-0.22), vec3(0.0,-1.14,-0.34), 0.30, 0.54);
  vec3 ba = vec3(abs(b.x), b.yz);
  float sh = sdEllipsoid(ba - vec3(0.56,-1.06 + 0.13*gShrug[b.x < 0.0 ? 0 : 1],-0.28), vec3(0.34,0.27,0.31));
  float bodyD = smin(smin(torso, neck, 0.25), sh, 0.20);
  bodyD = smin(bodyD, sdEllipsoid(b - vec3(0.0,-0.74,0.06), vec3(0.31,0.32,0.29)), 0.16);
  // a chest ruff
  bodyD = smin(bodyD, sdTaper(ba, vec3(0.20,-0.86,0.10), vec3(0.27,-1.04,0.02), 0.15, 0.08), 0.14);
  bodyD = smin(bodyD, sdTaper(ba, vec3(0.05,-0.92,0.14), vec3(0.08,-1.14,0.08), 0.15, 0.08), 0.14);
  // hips and legs
  float legs = sdEllipsoid(ba - vec3(0.42,-1.92,-0.30), vec3(0.42,0.30,0.42));
  legs = smin(legs, sdTaper(ba, vec3(0.40,-1.88,-0.14), KNEE, 0.30, 0.200), 0.14);
  legs = smin(legs, sdTaper(ba, KNEE, vec3(0.58,-2.78,0.52), 0.185, 0.12), 0.10);
  gLegM = smoothstep(-0.05, 0.05, bodyD - legs);
  bodyD = smin(bodyD, legs, 0.34);       // a wide blend: no dark crease between thigh and belly under the resting hands
  float tl = tailD(b);
  gTailW = smoothstep(-0.04, 0.04, bodyD - tl);
  bodyD = smin(bodyD, tl, 0.08);
  gBodyD = bodyD;

  float fz = smoothstep(0.0, 0.6, gOpen)*smoothstep(0.0, 0.3, q.z);
  gFur = smin(head, smax(bodyD, -hole, 0.010), 0.16 - 0.13*fz);
  gBare = min(gEye, min(gTeeth, gTongue));

  // ---- arms: they join the coat at the shoulder only.  Down the arm the blend closes, so a forearm lying on the thigh
  // stays a forearm, and a hand near the face never melts into it
  float furNA = gFur;
  float armD = armEval(b);
  float kA = 0.012 + 0.24*(1.0 - smoothstep(0.12, 0.66, gArmDS));
  gFur = smin(gFur, armD, kA);
  gWArm = smoothstep(-0.012 - 0.45*kA, 0.012 + 0.45*kA, furNA - armD);
  // ---- hands: bare skin of their own, see charEval
  handsEval(b);

  for(int k=0; k<1 && dH >= HB_G; k++){ gWBody = 1.0; gEarM = 0.0; gHeart = 1.0; gMask = 1.0; gFurLen = 0.100; gRest = gBodyR*(pw - NECK) + NECK; }
  for(int k=0; k<1 && dH < HB_G; k++){
  // rest coordinate for fur + paint
  float wB = 0.10 - 0.07*fz;
  float wBody = smoothstep(-wB, wB, head - bodyD);
  gWBody = wBody;
  float wEar  = smoothstep(-0.02, 0.02, skin - ear);
  gEarM = smoothstep(0.015, 0.11, skin - ear);
  float wJ = gWJaw*(1.0 - smoothstep(-0.30, -0.08, head - bodyD));
  vec3 rq = mix(q, qjr, wJ);                  // the skull does not take the pucker warp, so neither does the coat: no smear on the cheeks
  rq = mix(rq, vec3(s*(e2.x + EARB.x), e2.y + EARB.y, e2.z + EARB.z), wEar);
  gRest = mix(rq, gBodyR*(pw - NECK) + NECK, wBody);

  // where fur grows, how long: the heart-shaped face and the ears are bare skin
  float hd = heart(vec3(q.x, mix(q.y, qj.y, gWJaw) - 0.125*max(bu, 0.0)*smoothstep(0.10, 0.34, q.y), 0.0));
  hd = max(hd, 0.20 - q.z);
  // the brows carry their own short coat and take the top of the mask with them
  gHeart = hd;
  float m = smoothstep(-0.005, 0.034, hd);
  m = max(m, 0.26*smoothstep(0.006, -0.012, gBrowP)*step(0.15, q.z));
  m *= 1.0 - gEarM;
  m *= smoothstep(0.005, 0.05, length(ea) - EYER - 0.022);
  m *= smoothstep(0.0, 0.05, hole);
  gMask = mix(m, 1.0, smoothstep(-0.035, 0.035, head - bodyD))*smoothstep(0.0, 0.05, hole);
  gFurLen = mix(0.058 + 0.026*smoothstep(0.26, 0.62, q.y) + 0.012*smoothstep(0.35, 0.65, abs(q.x)), 0.100, wBody);
  }

  // ---- arms and hands take over everything the head and body wrote here
  float uA = gArmU/(BONE_U + BONE_F);
  float lenA = mix(0.092, 0.060, smoothstep(0.25, 0.95, uA))*(1.0 - smoothstep(0.955, 1.035, uA));
  gRest = mix(gRest, gArmRest, gWArm);
  gFurLen = mix(gFurLen, lenA, gWArm);
  gFurLen = mix(gFurLen, 0.045, gTailW*(1.0 - gWArm));     // the tail's coat is short
  gMask = mix(gMask, 1.0, gWArm)*smoothstep(0.0, 0.03, gHandD);      // the coat lies flat under and around a hand
  gWBody = max(gWBody, gWArm);
  float wAH = max(gWArm, 1.0 - smoothstep(0.0, 0.01, gHandD));
  gHole = mix(gHole, 1.0, wAH); gE = mix(gE, 1.0, wAH); gRing = mix(gRing, 1.0, wAH);
}

// fur flows away from the face on the head, away from the neck on the body
vec3 furPole(){ return mix(POLE, vec3(0.0,-0.25,0.15), gWBody); }

void charEval(vec3 p){
  p = bodyXf(p);
  monkeyEval(p);
  // Smooth skin must not overshoot.  The engine's coarse march exits at h<0.0015 but its solid test is
  // <0.0009, so a ray landing between them takes a 0.0045 step into the surface; fur hides that, skin
  // shows it as contour lines.  A tiny negative length on bare skin, and the skin field kept just under
  // the bare parts, make a coarse exit always a solid hit.
  gcBase = min(gFur, gBare - 0.0007); gcBare = gBare; gcRest = gRest;
  gcLen = gFurLen*gMask - 0.0006;
  // A hand is bare skin floating free of the coat.  Outside the coat's tips it owns the point, with the same two tiny offsets;
  // inside the coat the fur keeps it and the hand is only a solid to hit, so no shell of fur grows around the fingers
  float own = step(gHandD, gFur - gFurLen*gMask);
  gcBase = mix(gcBase, min(gHandD, gBare) - 0.0007, own);
  gcLen = mix(gcLen, -0.0006, own);
  gcBare = min(gBare, gHandD);
}
float charTuft(){
  float h = 0.5;
  for(int k=0; k<1 && gcLen > 0.008; k++) h = furHeight(gcRest, furPole(), mix(F_K, F_K*1.7, gWBody), F_L);
  return mix(0.86, h, smoothstep(0.008, 0.034, gcLen));   // where fur tapers out it is a closed nap
}

//@shadow-begin
// cheap proxy for shadow rays: 9 primitives, no fur, no mouth
float charShadow(vec3 pw){
  pw = bodyXf(pw);
  vec3 q = gHR*(pw - gHO) + NECK;
  vec3 qa = vec3(abs(q.x), q.yz);
  float d = sdEllipsoid(q - vec3(0.0,0.07,-0.05), vec3(0.60,0.56,0.56));
  d = min(d, sdEllipsoid(qa - vec3(0.36,-0.15,0.28), vec3(0.22,0.20,0.22)));
  d = smin(d, sdEllipsoid(q - vec3(0.0,-0.25,0.46), vec3(0.25*(1.0 - 0.32*gRound),0.21,0.30)), 0.08);
  vec3 e = qa - EARB;
  e.xz = rot(0.52)*e.xz;
  e.yz = rot(-0.10)*e.yz;
  e = ((q.x < 0.0) ? gEarRL : gEarRR)*e;
  d = min(d, sdEllipsoid(e, vec3(0.24,0.27,0.035)));
  // the body must sit INSIDE the real coat: a proxy larger than the body shadows strand tips at random
  vec3 b = pw; b.xz = rot(BYAW)*b.xz;
  d = min(d, sdEllipsoid(b - vec3(0.0,-1.48,-0.26), vec3(0.56,0.70,0.44)));
  d = min(d, sdTaper(b, vec3(0.0,-0.42,-0.22), vec3(0.0,-1.14,-0.34), 0.20, 0.42));
  vec3 ba = vec3(abs(b.x), b.yz);
  d = min(d, sdEllipsoid(ba - vec3(0.56,-1.06,-0.28), vec3(0.24,0.18,0.22)));
  d = min(d, sdTaper(ba, vec3(0.40,-1.88,-0.14), KNEE, 0.22, 0.13));
#ifndef PASS_BG
  // the arms move, so they shadow the character only: the baked set knows nothing of them.  Thin, inside the coat
  for(int i=0; i<2; i++){
    d = min(d, sdTaper(b, gAS[i], gAE[i], 0.11, 0.075));
    d = min(d, sdTaper(b, gAE[i], gAW[i], 0.080, 0.050));
    d = min(d, sdTaper(b, gAW[i] + gHB[i][1]*0.10, gAW[i] + gHB[i][1]*0.26, 0.038, 0.038));
  }
  d = min(d, min(sdTaper(b, gTl[1], gTl[2], 0.05, 0.045), sdTaper(b, gTl[2], gTl[4], 0.045, 0.03)));
#endif
  return d;
}
//@shadow-end

const vec3 C_BROWN = vec3(0.112,0.043,0.017);
const vec3 C_WARM  = vec3(0.245,0.108,0.044);
const vec3 C_DEEP  = vec3(0.040,0.018,0.011);
const vec3 C_TAN   = vec3(0.640,0.420,0.250);
const vec3 C_SKIN  = vec3(0.700,0.350,0.205);
const vec3 C_PALE  = vec3(0.800,0.500,0.345);
const vec3 C_BLUSH = vec3(0.760,0.205,0.130);
const vec3 C_LIPU  = vec3(0.520,0.170,0.145);
const vec3 C_LIPL  = vec3(0.585,0.195,0.160);
const vec3 C_LINE  = vec3(0.210,0.062,0.052);
const vec3 C_CAVE  = vec3(0.260,0.062,0.058);   // cavity near the lips
const vec3 C_CDEEP = vec3(0.060,0.016,0.019);   // cavity, deep
const vec3 C_TOOTH = vec3(0.720,0.660,0.550);
const vec3 C_GUM   = vec3(0.640,0.230,0.220);
const vec3 C_TONGUE= vec3(0.760,0.270,0.270);


// ---- hand skin: a little darker than the face on the back, pale and pink in the palm.  Which digit, which phalanx and which
// side are found here, once per sample, from the same joints handD uses
const vec3 C_HBACK = vec3(0.520,0.262,0.160);
const vec3 C_HPALM = vec3(0.770,0.440,0.310);
const vec3 C_HLINE = vec3(0.235,0.075,0.060);
const vec3 C_NAIL  = vec3(0.640,0.430,0.340);
void nearSeg(vec3 b, vec3 a, vec3 e, float r, float k, float j, inout float best, inout float bh, inout float bhu, inout float br, inout float bk, inout float bj, inout vec3 bp, inout vec3 bd, inout float bl){
  vec3 pa = b - a, ba = e - a;
  float hu = dot(pa, ba)/dot(ba, ba);
  float h = clamp(hu, 0.0, 1.0);
  vec3 c = pa - ba*h;
  float rr = r*(1.0 - 0.09*(j + h));
  float d = length(c) - rr;
  if(d < best){ best = d; bh = h; bhu = hu; br = rr; bk = k; bj = j; bp = c; bd = ba; bl = length(ba); }
}
void handSurface(vec3 pw, vec3 n, inout Surf s){
  vec3 b = pw;
  b.y += 0.012*gBreath*smoothstep(-2.0, -1.2, b.y);
  b.xz = rot(BYAW)*b.xz;
  float m = gHandI, sx = 2.0*m - 1.0;
  mat3 H = mat3(mix(gHB[0][0], gHB[1][0], m), mix(gHB[0][1], gHB[1][1], m), mix(gHB[0][2], gHB[1][2], m));
  vec3 q = (b - mix(gAW[0], gAW[1], m))*H;
  float pd = palmD(q);
  float best = 1e3, bh = 0.0, bhu = 0.0, br = 0.03, bk = 0.0, bj = 0.0;
  vec3 bp = vec3(0.0), bd = vec3(0.0,1.0,0.0); float bl = 0.1;
#define SEG(J,K,JJ,R0) nearSeg(b, mix(gDJ[J], gDJ[(J)+20], m), mix(gDJ[(J)+1], gDJ[(J)+21], m), R0, float(K), float(JJ), best, bh, bhu, br, bk, bj, bp, bd, bl);
#define DIG3(J,K) SEG(J,K,0,DIG_R[K]) SEG((J)+1,K,1,DIG_R[K]) SEG((J)+2,K,2,DIG_R[K])
  DIG3(0,0) DIG3(4,1) DIG3(8,2) DIG3(12,3) DIG3(16,4)
  bd /= bl;
  float digit = smoothstep(-0.006, 0.006, pd - best);                  // 1 on a digit, 0 on the palm
  vec3 pal = bk > 3.5 ? mix(gTP[0], gTP[1], m) : -sx*cross(H[0], bd);  // the pad side of this phalanx
  pal = normalize(pal - bd*dot(pal, bd));
  float cp = dot(bp, pal)/br;                                          // +1 pad .. -1 back of the digit
  float cl = dot(bp, cross(bd, pal))/br;                               // across it
  float yj = bh*bl;                                                    // along the phalanx, world units
  float back = smoothstep(0.30, -0.30, cp);

  // ---- a digit
  vec3 cD = mix(C_HPALM, C_HBACK, back);
  float nk = vnoise3(q*vec3(46.0,20.0,46.0));
  // creases at the joints: pads get one or two clean lines, backs a small fan of wrinkles
  float toJ = min(yj, bl - yj);                                        // distance to the nearer joint
  float isTip = step(1.5, bj)*step(bl*0.5, yj);
  float isBase = step(bj, 0.5)*step(yj, bl*0.5);
  float nearJ = (1.0 - isTip)*(1.0 - 0.0*isBase);
  float wr = 0.5 + 0.5*cos(toJ/0.0066*PI);
  float mid = smoothstep(0.95, 0.40, abs(cl));                         // creases fade before the side of the digit
  float padLine = smoothstep(0.0052, 0.0014, toJ)*(1.0 - back)*mid;
  float fan = (1.0 - smoothstep(0.008, 0.026, toJ))*wr*back*(1.0 - isBase)*mid;
  cD = mix(cD, C_HLINE, nearJ*(0.55*padLine + 0.24*fan)*(0.75 + 0.5*nk));
  cD = mix(cD, cD*vec3(1.12,0.98,0.94), (1.0 - back)*(1.0 - smoothstep(0.0, 0.05, abs(yj - 0.5*bl)))*0.45);   // the pads between
  // the nail: on the back of the last phalanx
  float tipL = bhu;
  float nailR = length(vec2(cl/0.66, (tipL - 0.82)/0.36));
  float dside = cp;
  float nail = step(1.5, bj)*smoothstep(1.0, 0.86, nailR)*smoothstep(-0.20, -0.42, dside);
  float nailEdge = step(1.5, bj)*smoothstep(0.80, 0.97, nailR)*smoothstep(1.10, 1.0, nailR)*smoothstep(-0.08, -0.36, dside);
  cD = mix(cD, cD*vec3(0.74,0.60,0.56), 0.60*nailEdge);
  vec3 cN = mix(C_NAIL, vec3(0.700,0.540,0.440), smoothstep(0.98, 1.10, tipL + 0.16*cl*cl));
  cD = mix(cD, cN, nail);

  // ---- the palm and the back of the hand, in hand space: z > 0 is the palm
  float side = smoothstep(-0.018, 0.018, q.z);
  vec3 cP = mix(C_HBACK, C_HPALM, side);
  // the three lines of a palm
  float u = q.x, v = q.y;
  float l1 = abs(v - (0.205 + 0.018*u/0.1 - 0.62*(u - 0.01)*(u - 0.01)));                                // distal crease
  float l2 = abs(v - (0.150 + 0.050*u/0.1 - 0.30*(u - 0.04)*(u - 0.04)));                                // proximal crease
  float l3 = abs(u - (0.016 + 0.052*smoothstep(0.03, 0.17, v) - 0.20*(v - 0.10)*(v - 0.10)));            // around the thenar
  float crease = max(max(smoothstep(0.0062, 0.0016, l1), smoothstep(0.0062, 0.0016, l2)*(1.0 - smoothstep(0.05,0.10,-u))), smoothstep(0.0066, 0.0016, l3)*smoothstep(0.215,0.16,v)*smoothstep(0.015,0.05,v));
  cP = mix(cP, C_HLINE, 0.66*crease*side*(0.70 + 0.5*nk));
  cP = mix(cP, cP*vec3(1.08,0.95,0.90), side*(1.0 - smoothstep(0.0, 0.07, length(q.xy - vec2(0.062,0.09))))*0.55);   // the pads flush
  // the back: tendons fan to the knuckles, knuckle wrinkles
  float kn = 1.0;
  for(int k=0; k<4; k++){ kn = min(kn, length((q.xy - vec2(DIG_L[k], DIG_D[k]))/vec2(0.030,0.030))); }
  float wrk = (1.0 - side)*(1.0 - smoothstep(0.55, 1.15, kn))*(0.5 + 0.5*cos(q.y/0.0085*PI));
  cP = mix(cP, C_HLINE, 0.34*wrk*(0.7 + 0.6*nk));
  cP = mix(cP, cP*vec3(1.16,1.10,1.02), (1.0 - side)*(1.0 - smoothstep(0.35, 1.0, kn))*0.45);

  vec3 c = mix(cP, cD, digit);
  c *= 0.965 + 0.07*vnoise3(q*31.0);
  float nailM = nail*digit;
  s.alb = c;
  s.fur = 0.0; s.n = n; s.sheen = 0.22; s.wrap = 0.46; s.spec = mix(0.20, 0.55, nailM); s.shin = mix(24.0, 70.0, nailM);
  s.trans = 0.0; s.mirror = 0.0; s.occ = 1.0;
  s.emit = s.alb*vec3(0.105,0.050,0.026);
}

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  pw = bodyXf(pw);
  monkeyEval(pw);
  vec3 q = gQ, r = gRest;
  vec3 n = s.n;
  bool isEye = gEye < 0.002 && gEye <= gFur + 0.002;
  bool isTeeth = gTeeth < 0.003 && gTeeth <= min(gFur, gTongue) + 0.003;
  bool isTongue = gTongue < 0.003 && gTongue <= min(gFur, gTeeth) + 0.003;

  // ---- fur paint
  vec3 c1 = furCoord(r, furPole(), 140.0, 15.0);
  float st = furStrand(c1);
  float tuft = furTuft(furCoord(r, furPole(), F_K, F_L));
  float ax = abs(r.x);
  vec3 alb = mix(C_BROWN, C_WARM, 0.30*smoothstep(0.35,0.80,st) + 0.30*smoothstep(0.35,0.75,vnoise3(r*3.1)));
  // dark cap, gold around the face: the classic capuchin read
  float capM = smoothstep(0.34, 0.62, r.y + 0.10*(tuft - 0.5))*(1.0 - gWBody);
  alb = mix(alb, C_DEEP*1.6, 0.74*capM);
  float halo = smoothstep(0.11, 0.0, gHeart + 0.06*(tuft - 0.5))*(1.0 - capM)*(1.0 - gWBody);
  alb = mix(alb, C_TAN*0.72, 0.62*halo);
  // lighter chest
  vec3 rb = mix(gBodyR*(pw - NECK) + NECK, gRest, gWArm);
  float bib = smoothstep(-0.05, 0.05, 0.27 + 0.12*(rb.y + 1.0) - abs(rb.x) + 0.16*(tuft - 0.5))*smoothstep(-0.30, -0.60, rb.y)*(1.0 - gLegM);
  alb = mix(alb, C_TAN*0.86, 0.80*bib);
  // brows: short dark coat
  float browM = smoothstep(0.012, -0.010, gBrowP + 0.012*(st - 0.5))*step(0.0, q.z - 0.15);
  browM *= 1.0 - gWBody;
  alb = mix(alb, C_BROWN*0.92, browM);
  alb *= 0.94 + 0.12*st + gWBody*0.30*(st - 0.5);

  // ---- skin paint
  float skinM = (1.0 - smoothstep(-0.022, 0.018, gHeart + 0.018*(vnoise3(r*19.0) - 0.5) + 0.026*(st - 0.5)))*(1.0 - gWBody)*(1.0 - browM);
  vec3 sk = mix(C_SKIN, C_PALE, 0.70*smoothstep(0.40, -0.10, length((q - vec3(0.0,-0.27,0.80))/vec3(0.40,0.32,0.5)) - 0.5));
  float blushB = exp(-sq(length((vec3(abs(q.x), q.yz) - vec3(0.37,-0.11,0.48))/vec3(0.17,0.12,0.25))));
  sk = mix(sk, C_BLUSH, 0.42*blushB);
  vec3 ea = vec3(abs(q.x), q.yz) - EYEC;
  float lidr = length(ea);
  sk = mix(sk, sk*vec3(0.94,0.88,0.84), smoothstep(0.30, 0.215, lidr));         // warm around the eyes
  sk *= 0.965 + 0.07*vnoise3(q*26.0);
  // nose: the same skin, deepened
  float noseM = smoothstep(0.018, -0.014, gNoseD);
  sk = mix(sk, vec3(0.50,0.215,0.150), 0.82*noseM);
  vec3 nna = vec3(abs(q.x), q.yz) - NOSEP - vec3(0.0, gNoseY, 0.0);
  float nDark = smoothstep(0.026, 0.012, length((nna - vec3(0.046*gNoseW,-0.022,0.048))*vec3(1.0,1.5,1.0)) - 0.009*gNostril);
  sk = mix(sk, vec3(0.05,0.016,0.012), nDark);
  // ears: peach, deeper in the bowl
  float earW;
  earW = gEarM*(1.0 - gWBody)*(1.0 - smoothstep(0.08, 0.34, gMask));   // fur tips near the root stay fur
  vec3 earC = mix(C_SKIN*vec3(0.95,0.88,0.86), mix(C_SKIN, C_BLUSH, 0.55)*0.80, smoothstep(0.01,-0.06,gBowl));
  sk = mix(sk, earC, earW);
  skinM = max(skinM, earW);

  // eyeliner: thin, warm
  float liner = smoothstep(EYER + 0.030, EYER + 0.018, lidr)*step(0.0, ea.z + 0.05);
  sk = mix(sk, vec3(0.20,0.085,0.055), liner);

  // ---- lips
  float lipM = smoothstep(0.0120, -0.0020, gRing)*(1.0 - earW);
  // flesh with a volume, not a band of paint.  sc runs across the lip, -1 at the mouth .. 1 at the skin
  float sc = clamp(gSect, -1.2, 1.2);
  float crown = max(1.0 - sc*sc, 0.0);
  vec3 lipC = mix(C_LIPL, C_LIPU, gUpperW);
  lipC = mix(lipC, lipC*vec3(0.86,0.62,0.62), 0.80*smoothstep(0.30, -0.90, sc));     // deepening into the mouth
  lipC = mix(lipC, sk, 0.46*smoothstep(0.25, 1.10, sc));                             // the border pales into the skin
  lipC = mix(lipC, sk*vec3(1.0,0.92,0.88), 0.36*gPress*smoothstep(1.0, -0.6, sc));    // pressed flesh pales a little
  // fine creases across the lip: they stretch with it, run radial on a pucker, close up when it is pressed
  float crC = mix(gMX*W0/gW*135.0, atan(gMV, abs(gMX) + 0.002)*15.0, gCornT);
  float cr = vnoise2(vec2(crC, 2.0*gUpperW + 0.7*sc));
  float crA = crown*(0.40 + 0.45*gPress + 0.7*gPk)*(1.0 - 0.45*gSp);
  lipC *= 1.0 - 0.13*crA*smoothstep(0.52, 0.30, cr);
  float under = smoothstep(0.0, -0.9, gDepthS)*smoothstep(0.30, -0.10, gE/max(gLipH, 1e-3));
  lipC = mix(lipC, C_CAVE, 0.85*under);
  // closed, the lips shade each other at the seam; open, that is gone and only a thin wet edge is left
  lipC *= 1.0 - 0.26*(1.0 - gOpenK)*smoothstep(0.034, 0.0, gE);
  float line = smoothstep(0.0042, 0.0006, gE)*max(smoothstep(mix(1.10, 1.01, gSp), 0.96, gU), gCornT);
  float dimple = exp(-sq(length(vec2((abs(gMX) - gW*1.05)/0.030, gMV/0.032))))*(1.0 - smoothstep(0.0, 0.5, gPk));
  // the lower lip's shadow on the chin; the soft raised edge of the upper lip
  float below = smoothstep(0.010, -0.030, gMV)*(1.0 - smoothstep(0.0, 0.062, gE - gLipH))*smoothstep(1.10, 0.25, gU);
  float roll = smoothstep(0.018, 0.002, abs(gE - gLipH))*smoothstep(-0.01, 0.02, gMV)*smoothstep(1.05, 0.5, gU);
  vec3 skinC = sk*(1.0 + 0.055*roll);
  skinC = mix(skinC, skinC*vec3(0.79,0.66,0.62), 0.72*below);
  skinC = mix(skinC, lipC, lipM);
  skinC = mix(skinC, vec3(0.260,0.070,0.065), mix(0.82, 0.34, gOpenK)*line);
  dimple *= 1.0 - 0.80*gOpenK;                                    // a stretched corner has no dimple
  skinC = mix(skinC, skinC*vec3(0.80,0.68,0.64), (0.55 + 0.20*gPress + 0.20*gSp)*dimple);

  s.alb = mix(alb, skinC, skinM);
  s.alb = mix(s.alb, skinC, lipM);
  float bareSkin = max(skinM, lipM);
  float furAmt = 1.0 - bareSkin;

  vec3 v = gRest - furPole();
  s.T = furFlow(v, n, vec3(0.0,-0.25,0.0));
  if(gWArm > 0.5){ vec3 dw = gArmDir; dw.xz = rot(-BYAW)*dw.xz; s.T = furFlow(dw, n, vec3(0.0,-0.10,0.0)); }
  float nb = (st - 0.5);
  vec3 bt = normalize(cross(s.T, n));
  s.n = normalize(n + (bt*nb*(0.10 + 0.22*gWBody) + s.T*nb*0.03)*furAmt);
  s.fur = furAmt;
  s.wrap = mix(0.50, 0.42, furAmt);
  s.sheen = mix(0.22, 0.9, furAmt);
  s.spec = mix(0.20, 0.0, furAmt);
  s.shin = 26.0;
  s.occ = mix(1.0, (0.86 + 0.14*smoothstep(0.1,0.7,tuft))*(0.93 + 0.12*st), furAmt);
  s.alb *= mix(vec3(1.0), mix(vec3(0.90,0.82,0.76), vec3(1.0), smoothstep(0.05,0.6,tuft)), furAmt);
  s.trans = earW*0.85;
  s.transCol = vec3(1.0,0.40,0.24);
  s.occ *= 1.0 - 0.34*dimple;
  // the coat under a resting hand: the engine's 4-step AO reads the fingers over the fur as a cave and blackens it; fur there is
  // lit through the gaps between the fingers, so the painted occlusion gives most of it back
  s.occ *= 1.0 + 1.5*(1.0 - smoothstep(0.004, 0.11, gHandD))*s.fur;
  // a wet sheen on the crowns, the lower lip most, broken by the creases
  vec3 lt = normalize(cross(n, vec3(0.0,1.0,0.0)));
  s.n = normalize(s.n + lipM*lt*(cr - 0.5)*0.42*crA);
  float gloss = (0.42 + 0.58*bell(gMX/(0.70*gW + 0.05)))*(0.50 + 0.70*cr)*(1.0 - 0.75*(1.0 - gOpenK)*smoothstep(0.030, 0.006, gE));
  s.spec = mix(s.spec, (0.46 + 0.50*(1.0 - gUpperW))*(0.30 + 0.70*crown)*gloss, lipM);
  s.shin = mix(s.shin, 30.0, lipM);
  s.sheen = mix(s.sheen, 0.25, lipM);
  s.mirror = mix(s.mirror, 0.05*crown*gloss, lipM);
  // light scattered through the skin: keeps shadows peach in a green world
  s.emit = s.alb*vec3(0.105,0.050,0.026)*bareSkin;

  // ---- the cavity
  float wall = smoothstep(0.008, 0.0, gHole)*(1.0 - lipM);
  if(wall > 0.0 || gE < 0.0){
    float cm = max(wall, smoothstep(0.0, -0.002, gE)*(1.0 - lipM));
    float deep = max(smoothstep(ZF - 0.03, ZF - 0.20, q.z), 0.92*gTunnel);
    vec3 c = mix(C_CAVE, C_CDEEP, deep);
    c = mix(c, C_CAVE*1.25, 0.34*smoothstep(-0.120, -0.004, gE)*(1.0 - gTunnel));   // daylight comes in past the lips and dies in the throat
    c = mix(c, C_CDEEP*0.7, max(gDark, gTgDim*0.8));
    s.alb = mix(s.alb, c*0.25, cm);     // lit by emit: the sun has no way in, and AO would draw hard edges here
    s.fur *= 1.0 - cm;
    s.occ = mix(s.occ, 1.0, cm);
    s.spec *= 1.0 - cm; s.mirror *= 1.0 - cm; s.sheen *= 1.0 - cm;
    s.emit = mix(s.emit, c*0.62*(1.0 - 0.6*gDark), cm);
    s.wrap = mix(s.wrap, 0.3, cm);
    s.n = normalize(mix(s.n, n, cm));
  }
  if(isTongue){
    vec3 c = C_TONGUE*(0.92 + 0.14*vnoise3(gQTJ*34.0));
    float deep = smoothstep(ZF - 0.14, ZF - 0.42, gQTJ.z);
    c = mix(c, c*0.42, max(deep, 0.8*gTunnel));
    c = mix(c, C_CDEEP, smoothstep(0.55, 1.0, abs(gTgX)));     // its edges sink into the dark
    c *= 1.0 - 0.15*smoothstep(0.14, 0.0, abs(gTgX));          // the soft groove down its middle
    c *= 0.78 + 0.22*smoothstep(-0.13, -0.01, gE);             // and the light falls off behind the lips, as on the walls
    float tgDark = max(gDark*(1.0 - 0.80*max(gTgTip, gTgDent)), gTgDim*(1.0 - max(gTgTip, max(gTgDent, gTgBack))));
    c = mix(c, C_CDEEP, tgDark);
    s.alb = c*0.24;                     // lit mostly by emit: the engine's 4-step AO draws hard edges in here
    s.fur = 0.0; s.n = n; s.spec = 0.14; s.shin = 28.0; s.wrap = 0.35; s.sheen = 0.0; s.mirror = 0.0; s.occ = 1.0; s.trans = 0.0;
    s.emit = c*0.56*(1.0 - 0.7*tgDark);
  }
  if(isTeeth){
    float dk = max(max(max(1.0 - gShow, 0.90*gTunnel), 0.86*gDark), smoothstep(-0.011, -0.0065, gHole));
    // the lip's shade, in scale with the opening: strongest at the gum, gone by the biting edge
    float yEdge = gYC + gAu + gBend*(gU < 1.0 ? gU*gU : 2.0*gU - 1.0);
    float sw = max(0.70*gAu, 0.048);                           // in a small slit all that shows is in the lip's shadow
    float shade = mix(0.38, 1.0, smoothstep(0.0, sw, yEdge - q.y));
    // falling off to the sides: the dark corridors at the mouth corners, wherever the head has turned.  Measured where the
    // ray crosses the lip plane, so the far limb of the arch is in the dark too.  The lower row goes sooner
    vec3 rh = gHR*rd;
    float xp = abs(q.x + (ZF - q.z)*rh.x/min(rh.z, -0.3))/gW;
    float side = max(smoothstep(0.30, 1.02, xp), smoothstep(mix(0.12, 0.04, gToothL), mix(0.50, 0.15, gToothL), gToothSt.x));
    float groove = max(smoothstep(0.70, 1.0, gToothSt.y), 3.2*smoothstep(-0.015, -0.019, gToothSt.w));
    float rowK = mix(1.0, 0.70, gToothL);
    vec3 c = C_TOOTH*rowK*(1.0 - 0.13*groove);
    c = mix(c, c*vec3(1.0,0.92,0.82), 0.5*smoothstep(-0.040, -0.004, gToothSt.z));     // warmer toward the root
    c = mix(c, c*vec3(0.90,0.93,0.97), 0.55*smoothstep(-0.052, -0.066, gToothSt.z));   // a little translucent at the biting edge
    float gumK = smoothstep(-0.004, 0.004, gToothSt.z);
    c = mix(c, mix(C_GUM, C_CAVE*0.7, 0.3*gToothL), gumK);
    // and the arch rolls off where it turns from the camera, by its plan normal so the teeth's own relief does not break it up
    vec3 ar = mix(ARCH_U, ARCH_L, gToothL);
    float f = dot(normalize(vec2(gArX/(ar.x*ar.x), (gArZ - ar.z)/(ar.y*ar.y))), -normalize(vec2(0.9872*rh.x - 0.1593*rh.z, 0.1593*rh.x + 0.9872*rh.z)));
    side = max(side, 1.0 - smoothstep(0.05, 0.90, f));
    c *= shade;
    c = mix(c, C_CAVE*0.34, 0.92*side);
    c = mix(c, C_CDEEP, dk);
    s.alb = c*0.24;
    s.fur = 0.0; s.n = n; s.shin = 40.0; s.wrap = 0.3; s.sheen = 0.0; s.mirror = 0.0; s.occ = 1.0; s.trans = 0.0;
    s.spec = mix(0.30, 0.10, gumK)*(1.0 - dk);
    s.emit = c*vec3(0.68,0.65,0.62)*(1.0 - 0.7*dk);
  }
  for(int k=0; k<1 && gHandD < 0.003 && gHandD <= gFur + 0.003; k++) handSurface(pw, n, s);
  if(isEye){
    // gaze space
    vec3 toCam = rootGaze(gHR*(uCamPos - gHO)) + NECK - vec3(sign(q.x)*EYEC.x, EYEC.y, EYEC.z);
    vec3 eyeF = normalize(normalize(toCam) + vec3(gGaze.x*0.30, gGaze.y*0.24, 0.0));
    vec3 ex = normalize(cross(vec3(0,1,0), eyeF)), ey = cross(eyeF, ex);
    mat3 toE = transpose(mat3(ex, ey, eyeF));
    vec3 ee = vec3(q.x - sign(q.x)*EYEC.x, q.y - EYEC.y, q.z - EYEC.z)/EYER;
    vec3 e = toE*ee;
    vec3 vv = toE*(gHR*rd);
    float irisM;
    vec3 c = eyeColour(e, vv, vec3(0.235,0.085,0.026), vec3(0.085,0.030,0.012), vec3(0.84,0.77,0.68), 0.93, 0.53, irisM);
    s.alb = c;
    // shadow cast by the upper lid
    s.occ = mix(0.68, 1.0, smoothstep(0.75, 0.15, ee.y));
    s.fur = 0.0; s.sheen = 0.0; s.spec = 1.5*smoothstep(0.74, 0.42, ee.y); s.shin = 900.0; s.wrap = 0.0; s.trans = 0.0;
    s.mirror = 0.9*smoothstep(0.74, 0.40, ee.y);
    s.n = n;
    // painted catch-lights, locked to the light rig: huge
    vec3 r2 = reflect(rd, n);
    float k1 = smoothstep(0.955, 0.970, dot(r2, normalize(uSunDir + vec3(0.05,0.2,0.3))));
    float k2 = smoothstep(0.984, 0.993, dot(r2, normalize(vec3(0.5,-0.05,0.85))));
    s.emit = vec3(1.0,0.97,0.90)*(4.2*k1 + 1.5*k2);
  }
}

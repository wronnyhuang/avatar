// ---------------------------------------------------------------------------
//  animals/fox.glsl — the fox.  Head space: origin at the skull centre, +x screen right,
//  +y up, +z toward the camera (the nose).  Units: skull radius ~0.6.
//  See docs/ANIMAL_GUIDE.md for the API.
// ---------------------------------------------------------------------------
const vec3  NECK   = vec3(0.0,-0.62,-0.18);    // head rotates about this
const vec3  HINGE  = vec3(0.0,-0.13, 0.12);    // jaw hinge
const vec3  NOSEP  = vec3(0.0,-0.085, 0.975);  // nose ball centre
const vec3  POLE   = vec3(0.0,-0.10, 1.00);    // fur flows away from here
const vec3  EYEC   = vec3(0.262,0.100,0.385);  // mirrored
const float EYER   = 0.192;
const float EYETILT= 0.05;                     // outer corners up
const float YAW0   = 0.36;                     // resting 3/4 turn
const vec3  EARB   = vec3(0.40,0.40,-0.10);   // ear base (mirrored)
const float EARL   = 0.60;
const float MAXJAW = 0.62;                     // radians at teeth_gap=1
const float ZF     = 0.965;                    // the mouth line's front, at the middle of the muzzle
const float DROPL  = 0.84;                     // jaw length: hinge to the lower lip's front, sets how far lip_open drops the lip
const vec3  JAW_A = vec3(0.0,-0.205,0.16), JAW_B = vec3(0.0,-0.262,0.850);
const float JAW_RA = 0.285, JAW_RB = 0.092;
const float F_K = 13.0, F_L = 3.6;             // tuft density across / along the flow
const float WHISK_R0 = 0.0046, WHISK_R1 = 0.0023;   // whisker radius root, tip: 1.5 px at the tip keeps 3 spp stable, at the wide camera

// ---- the mouth rig, on the eleven lip_* columns (docs/V2.md)
//  jaw     a hinge, teeth_gap only.  Teeth and tongue ride it.
//  lens    the opening is a lens between two edges that meet at the corners: the upper edge lives in the head frame, the
//          lower edge in the jaw frame, and each column moves one of them or the corner.  lensT: 0 at the front .. 1 on the
//          corner, an ellipse in plan so both lips share one outline.  gLU, gLF lift the upper edge; gLD, gLO drop the lower.
//  corner  gCorner slides along the muzzle: forward in a pucker, back in a spread.  Behind it the lips are shut skin.
//  cheek   where the lens holds the lower edge above the jaw's bone, the jaw's rim outline is extruded up to it: the skin
//          stretched across the gape.  So the lens may close while the jaw is open, and the corner stays put.
//  lips    a rolled bead on each edge, thin to nothing at the corner, absent in the sealed rest pose.
//  teeth   hang from the rest line, not from the lips, and wait above it (gShow) until the lens has room: slit first.
//  rounds  an open round (gRO) stops following the hinge: its depth is a half ellipse and its corners sink to mid height,
//          so it reads round from the 3/4 view.  The tongue rises to its floor and the lower lip covers the lower row.
//  sealed  all columns zero is the sealed line of the shipped fox, with no gap and no interior.

// ---- per-fragment pose (filled by charInit)
mat3  gHR;          // world -> head
mat3  gBodyR;       // world -> head at rest (body fur + paint frame)
vec3  gHO;          // head-space origin offset
float gJawA, gRound, gSpread, gPress, gLab, gSmile, gCheekP, gNostril;
float gPk, gPro;                // pucker 0..1, lips forward 0..1
float gTgTip, gTgDent, gTgBack; // tongue
float gSinA, gCosA;             // of the jaw angle
float gLU, gLF, gLD, gLO, gAmt;      // lens at the front: upper edge rise, lower edge fall (hinge part, lip_lower part), total height
float gRnd, gProfR, gLensX;     // 1 = a round O | lens profile: 0 a wedge from the front to the corner .. 1 full to the corner
float gTight;                   // 1 = a tight round
float gSmirk, gSquint, gPadUp, gPull;   // corners to one side | lower lids | the whisker pad rides the upper lip | the cheek draws back along the lip line in a spread
float gLipO, gLipF, gPeel;      // 0 sealed .. 1 the lips have shown their thickness | how full the lips are: a press or a pucker gathers them, stretch thins them | sqrt(press): the lips stick, the opening peels out from the middle
float gRO, gDip;                // 1 = an open round: its depth no longer follows the hinge | how far the corners sink around it
float gShow;                    // teeth wait behind the lips until there is room: 0 hidden .. 1 out
float gBlinkL, gBlinkR;
vec2  gGaze;
float gBrowL, gBrowR, gKnit;
mat3  gEarRL, gEarRR;
vec3  gNoseP; float gNoseW, gTgOut, gEyeWide, gBounce, gShrL, gShrR;
mat3  gTailR;
mat3  gBodyPose;                // the torso's pose change from its rest (yaw, lean, pitch about the hip): carries the head
vec3  gEarTip[2], gJawTip;
float gBreath, gCorner, gSeal;   // gSeal 1 = mouth closed: the interior cannot show
float gDark;                     // 1 closed .. 0 with some height in the lens: a small opening is a dark slit, its colour comes up as it opens

mat3 rotX(float a){ float c=cos(a),s=sin(a); return mat3(1,0,0, 0,c,s, 0,-s,c); }
mat3 rotY(float a){ float c=cos(a),s=sin(a); return mat3(c,0,-s, 0,1,0, s,0,c); }
mat3 rotZ(float a){ float c=cos(a),s=sin(a); return mat3(c,s,0, -s,c,0, 0,0,1); }

// ---- the arms, on the hand_* columns (docs/V3.md).  A pure function of this frame's columns: nothing here remembers the last
//  frame, so the weight and the lag are read off the pose.  The shoulder rides up and forward with the hand; the elbow's swing
//  grows with the square of the height, so it stays down while the hand starts to rise and catches up late; a slack paw falls
//  from the wrist toward the ground and straightens as it opens; the digits open one after the other.  Any column value stays
//  anatomical: the target stays on its own side, outside the body and the head, and inside the reach.
//  Side index 0 = screen left, 1 = screen right.  World space, as the body is.
const float ARM_U = 0.60, ARM_F = 0.58;        // bones: shoulder to elbow, elbow to wrist (+11%, so a paw reaches the eyes and the lens)
const float BODY_YAW = 0.24;                   // the torso follows the head's turn most of the way
const vec3  SHO_P = vec3(0.56,-1.02,-0.20);    // shoulder, body frame, mirrored
const vec3  HAND_REST_L = vec3(-0.372,-1.37,0.42), HAND_REST_R = vec3(0.347,-1.41,0.30);    // round 3's resting wrists as they landed    // wrist at rest: a loose pair, not a mirror
const vec3  TORSO_C = vec3(0.0,-1.50,-0.08), TORSO_R = vec3(0.80,0.92,0.60);              // what a wrist keeps out of, body frame
const vec3  THIGH_C = vec3(0.50,-2.02,0.14), THIGH_R = vec3(0.42,0.42,0.62);              // mirrored, in the torso's rest turn
const float HAND_FLOOR = -1.60, PAW_FLOOR = -1.80;                                        // wrist and digit tips stop here, the seat
const vec3  HEAD_C = vec3(0.08,-0.06,0.12), HEAD_R = vec3(1.18,0.88,1.12);                // the same for the head, ruff included, head frame

vec3  gSh[2], gEl[2], gWr[2], gPawC[2];        // joints as posed
mat3  gUF[2], gFF[2], gPF[2];                  // bone frames as posed.  Paw: x toward the thumb, y to the digits, z out of the palm
vec3  gSh0[2]; mat3 gUF0[2];                   // the upper arm at the rest pose
vec3  gDg0[8], gDg1[8], gDg2[8], gDg3[8], gDgP[8];      // digits, paw frame: root, knuckle, last joint, tip, pad
vec3  gTh0[2], gTh1[2], gTh2[2];               // thumb, the same
mat3  gBodyW;                                  // world -> body frame, turned and bent by the hands
float gTwist, gLean, gFistNow, gPointNow, gThumbNow;
float gThK[2];                                 // thumb size: an extended thumb is a touch longer and thicker, so a thumbs-up reads

const vec3 HIP = vec3(0.0,-2.05,-0.12);          // the torso turns and leans about this
vec3 toBody(vec3 p){ return gBodyW*(p - HIP) + HIP; }
vec3 fromBody(vec3 q){ return (q - HIP)*gBodyW + HIP; }
vec3 fromBody0(vec3 q){ return rotY(BODY_YAW)*(q - HIP) + HIP; }    // the torso as it sits at rest: the coat stays on it
float softp(float u){ return 0.5*(sqrt(u*u + 0.020) + u) - 0.0707; }       // a soft ramp: 0 below zero, u above it
float lohi(float u, float lo, float hi){ return lo*u + (hi - lo)*softp(u); }    // the columns reach further one way than the other
vec3  keepOut(vec3 p, vec3 c, vec3 r){       // a soft ellipsoid wall, for joints
  vec3 q = (p - c)/r;
  float l = max(length(q), 1e-3);
  return c + r*q*(1.0 + 0.5*(sqrt((1.0 - l)*(1.0 - l) + 0.012) + (1.0 - l))/l);
}

// the contract's hand target (CONTRACT.md): an offset from the rest wrist in units of one arm length, x outward on that side, y up,
// z toward the camera; per-rig scales 0.62 / 1.12 / 0.74..0.40 (forward reach shrinks as the hand climbs past the face), then shaped
vec3 handTarget(float s, vec3 h, vec3 rest){
  h = clamp(h, vec3(-1.0), vec3(1.2));
  float AL = ARM_U + ARM_F;
  float wide = smoothstep(0.25, 0.60, h.x);
  vec3 T = rest + AL*vec3(s*h.x*(0.62 + 0.75*wide), h.y*(1.12 - 0.30*wide), h.z*mix(0.74, 0.40, smoothstep(0.3, 1.0, h.y)));
  float up = clamp((T.y + 1.45)/1.0, 0.0, 1.2);
  T.x += s*0.30*up*up;                                                      // a rising hand opens outward, past the ruff
  T.z += 0.10*up - 0.62*softp(s*T.x - 0.52);                                // in front of the chest it is forward; out wide it is in the body's plane, staged for the eye
  T.y -= 0.05*max(s*T.x - 0.8, 0.0)*max(s*T.x - 0.8, 0.0);                // an arm held out sags
  float dx = s*T.x + 0.16;
  T.x = s*(-0.16 + 0.5*(sqrt(dx*dx + 0.02) + dx));                          // across the chest a little, no further (the paws meet at the midline: armsInit)
  return T;
}
// over the eyes: both paws high and across = palms on the eyes.  The paw centre goes to the eye's front, the digits up and in
float coverW(vec3 h){ return smoothstep(0.82, 0.97, h.y)*smoothstep(-0.08, -0.28, h.x)*smoothstep(0.15, 0.35, h.z); }
vec3 eyeFront(float s, out vec3 n){
  vec3 eq = vec3(s*EYEC.x, EYEC.y + 0.02, EYEC.z);
  n = transpose(gHR)*normalize(vec3(s*0.45, 0.10, 1.0));
  return transpose(gHR)*(eq - NECK) + gHO;
}

void armSolve(float s, vec3 h, vec3 T, float open, float turn, vec3 rest, out vec3 S, out vec3 E, out vec3 W, out mat3 U, out mat3 F, out mat3 P){
  h = clamp(h, -1.0, 1.2);
  open = clamp(open, 0.0, 1.0);
  turn = clamp(turn, -1.0, 1.0);
  float cw = coverW(h);
  vec3 en; vec3 ef = eyeFront(s, en);
  vec3 wdir = normalize(vec3(-s*0.30, 1.0, 0.0) - 0.15*en);
  T = mix(T, ef + en*(EYER + 0.075) - wdir*0.21, cw);
  // the thinking pose: one paw high and across, below the face = knuckles under the chin
  float chw = smoothstep(0.55, 0.70, h.y)*(1.0 - smoothstep(0.84, 0.95, h.y))*smoothstep(-0.12, -0.28, h.x)*(1.0 - cw);
  vec3 chin = transpose(gHR)*(vec3(0.0,-0.47,0.70) - NECK) + gHO;
  vec3 cdir = normalize(transpose(gHR)*vec3(0.0, 1.0, 0.55));
  T = mix(T, chin - cdir*0.27, chw);
  wdir = normalize(mix(wdir, cdir, chw));
  cw = max(cw, chw);
  float up = clamp((T.y + 1.45)/1.0, 0.0, 1.2);
  T = fromBody(keepOut(toBody(T), TORSO_C, TORSO_R));
  vec3 Th = gHR*(T - gHO) + NECK;
  T = mix(transpose(gHR)*(keepOut(Th, HEAD_C, HEAD_R) - NECK) + gHO, T, cw);      // a paw placed on the face is not kept off it
  T.y = HAND_FLOOR + softp(T.y - HAND_FLOOR);                               // it rests on the thigh, it does not sink past it
  // the shoulder rides with the hand
  float shr = s < 0.0 ? gShrL : gShrR;
  S = fromBody(vec3(s*SHO_P.x, SHO_P.y + 0.014*gBreath + 0.15*shr, SHO_P.z + 0.02*shr));
  S += vec3(s*0.050*softp(h.x), 0.100*up*up + 0.018*h.y + 0.05*cw, 0.075*h.z + 0.02*up + 0.06*cw);
  vec3 v = T - S;
  float dm = 0.975*(ARM_U + ARM_F);
  float d0 = length(v);
  float d = max(smin(d0, dm, 0.16), 0.34);
  vec3 dir = v/max(d0, 1e-3);
  W = S + dir*d;
  // elbow: down and back and out; it swings out late as the hand climbs, and further if the ribs are in the way.  It stays on the
  // circle the two bones allow, so neither bone ever changes length
  vec3 pole = vec3(s*(0.42 + 0.34*up*up), -0.90 + 0.16*up, -0.34 + 0.34*up);
  vec3 n = normalize(pole - dir*dot(pole, dir));
  float a = (ARM_U*ARM_U - ARM_F*ARM_F + d*d)/(2.0*d);
  float hh = sqrt(max(ARM_U*ARM_U - a*a, 0.0));
  E = S + dir*a + n*hh;
  vec3 eb = toBody(E) - TORSO_C;
  float ribs = smoothstep(1.12, 0.84, length(eb/(TORSO_R*vec3(0.92,1.0,0.92))));
  vec3 aw = fromBody(vec3(eb.x, 0.0, 0.6*eb.z) + TORSO_C) - fromBody(TORSO_C);
  n = normalize(n + (aw - dir*dot(aw, dir))*2.4*ribs);
  E = S + dir*a + n*hh;
  vec3 du = normalize(E - S), df = normalize(W - E), hx = normalize(cross(dir, n));
  U = mat3(du, hx, cross(du, hx));
  F = mat3(df, hx, cross(df, hx));
  // the forearm turns the palm: toward the elbow's inside is supine, palm up on a bent arm; away is prone.  An open palm is shown, not kept
  float ang = -0.45 + 1.05*turn + 0.75*softp(turn);        // -1 palm down .. 0 to the midline, a touch down .. +1 palm up
  vec3 pn = cos(ang)*(-s*hx) + sin(ang)*F[2] + vec3(0.0, 0.0, 0.62*softp(turn));
  pn = normalize(pn - df*dot(pn, df));
  // a raised forearm has no up to show a palm to: it shows it to whoever is listening
  vec3 cf = vec3(0.0, 0.0, 1.0) - df*df.z;
  float sh = 0.80*smoothstep(0.10, 0.60, df.y);
  pn = normalize(mix(pn, cf/max(length(cf), 0.05), sh) + (-s*hx)*3.2*sh*(1.0 - sh));      // always round by the inside, never through the back of the hand
  // the paw hangs from the wrist: slack it falls and flexes toward the palm, open it straightens
  float slack = (1.0 - smoothstep(0.0, 0.85, open))*(1.0 - 0.85*gFistNow);
  vec3 w = normalize(df + vec3(0.0, -0.20 - 0.42*slack, 0.0) + pn*(0.10 + 0.20*slack));
  // the wrist stages the paw across the view, as a forearm pointing at the lens would only show its fingertips.  The palm turns with it
  vec3 w2 = normalize(vec3(w.xy, mix(0.62, 1.0, gPointNow)*w.z));
  pn += 2.0*dot(w, pn)*w2 - (w + w2)*(dot(w, pn) + dot(w2, pn))/(1.0 + dot(w, w2));
  w = w2;
  // the paw never goes through the belly or below the seat: the wrist gives way
  vec3 tip = W + w*0.42;
  vec3 tb = (toBody(tip) - TORSO_C)/(TORSO_R*vec3(0.95,0.98,0.94));
  float pen = smoothstep(1.10, 0.86, length(tb));
  vec3 tn = (normalize(tb/(TORSO_R*vec3(0.95,0.98,0.94))))*gBodyW;
  w += tn*max(-dot(w, tn), 0.0)*1.05*pen + vec3(-s*0.12, -0.10, 0.14)*pen;     // the inward part goes, it never turns back through the wrist
  // ... and lies along the top of the thigh, so a resting paw is held up by something
  vec3 tt = (rotY(-BODY_YAW)*tip - vec3(s*THIGH_C.x, THIGH_C.yz))/(THIGH_R*1.17);
  vec3 tnT = rotY(BODY_YAW)*normalize(tt/(THIGH_R*1.17));
  w += tnT*max(-dot(w, tnT), 0.0)*1.05*smoothstep(1.08, 0.90, length(tt));
  float seat = smoothstep(PAW_FLOOR + 0.16, PAW_FLOOR - 0.10, tip.y);
  w.y *= 1.0 - 0.92*seat;
  w = normalize(w + vec3(-s*0.22, 0.0, 0.34)*seat);
  w = normalize(mix(w, wdir, cw));
  // under the chin the palm faces the chest (the back of the paw to the lens): across the digits' line, never along it
  vec3 cpn = normalize(transpose(gHR)*vec3(0.0, 0.55, -1.0));
  pn = normalize(mix(pn, mix(-en, cpn, chw), cw));
  float tu = gThumbNow*(1.0 - open)*(1.0 - cw);
  vec3 fwd = normalize(vec3(-s*0.25, 0.10, 1.0));
  w = normalize(mix(w, fwd, tu));
  pn = normalize(mix(pn, normalize(vec3(-s, 0.0, 0.25)), tu));
  // a shrug: shoulders up, the turned paws lie out flat, palms up, to the sides
  float shw = (s < 0.0 ? gShrL : gShrR)*smoothstep(0.3, 0.9, turn)*smoothstep(0.10, 0.30, h.x)*open*(1.0 - cw);
  w = normalize(mix(w, normalize(vec3(s*1.0, 0.20, 0.45)), shw));
  pn = normalize(mix(pn, normalize(vec3(-s*0.15, 1.0, 0.25)), shw));
  // pointing: the paw (its index) points where the arm reaches, from the shoulder to the target: out to the side, or at the lens
  w = normalize(mix(w, normalize(dir + vec3(0.0, 0.10, 0.0)), 0.85*gPointNow*(1.0 - cw)));
  pn = pn - w*dot(pn, w);
  vec3 alt = cross(w, vec3(s, 0.0, 0.0));                                    // a palm normal along the digits has no frame: fall back, never a collapsed paw
  pn = normalize(mix(alt - w*dot(alt, w), pn, smoothstep(0.02, 0.15, length(pn))) + vec3(0.0, 0.0, 1e-4));
  pn = normalize(pn - w*dot(pn, w));
  P = mat3(-s*cross(w, pn), w, pn);
}

// digits, in the paw's own frame, thumb side first: they open in a wave and fan out.  Relaxed they are a soft curl, not a fist
void pawSolve(int side, float open, float point, float thumb){
  open = clamp(open, 0.0, 1.0); point = clamp(point, 0.0, 1.0); thumb = clamp(thumb, 0.0, 1.0);
  float fist = max(point, thumb)*(1.0 - open);       // pointing or a thumb up closes the other digits into a fist
  for(int i=0;i<4;i++){
    float fi = float(i), k = 1.5 - fi;
    float o = clamp(open*1.14 - 0.045*fi, 0.0, 1.0);
    float c1 = mix(0.46, 0.02, o) + 0.05*fi*(1.0 - o), c2 = mix(0.46, 0.08, o);
    float sp = k/1.5*(0.07 + 0.33*o);
    float fz = i == 0 ? fist*(1.0 - point) : fist;  // the index (thumb side, i = 0) stays out to point
    c1 = mix(c1, 1.30, fz); c2 = mix(c2, 1.45, fz); sp *= 1.0 - 0.7*fz;
    if(i == 0){ c1 = mix(c1, 0.0, point); c2 = mix(c2, 0.04, point); sp *= 1.0 - point; }
    vec3 d0_ = vec3(sin(sp), cos(sp), 0.0);
    vec3 r0 = vec3(0.068*k, 0.262 - 0.016*k*k, 0.0);
    float l1 = 0.112 - 0.012*abs(k - 0.5), l2 = 0.082 - 0.008*abs(k - 0.5);
    vec3 a1 = d0_*cos(c1) + vec3(0.0,0.0,sin(c1));
    vec3 a2 = d0_*cos(c1 + c2) + vec3(0.0,0.0,sin(c1 + c2));
    vec3 a3 = d0_*cos(c1 + c2 + 0.14) + vec3(0.0,0.0,sin(c1 + c2 + 0.14));
    int o4 = side*4 + i;
    gDg0[o4] = r0; gDg1[o4] = r0 + a1*l1; gDg2[o4] = gDg1[o4] + a2*l2; gDg3[o4] = gDg2[o4] + a3*0.062;
    gDgP[o4] = gDg1[o4] + a2*l2*0.52 + vec3(-d0_.x*sin(c1+c2), -d0_.y*sin(c1+c2), cos(c1+c2))*0.026;
  }
  float ot = clamp(open*1.1, 0.0, 1.0);
  gTh0[side] = vec3(0.112, 0.070, 0.022);
  vec3 t1 = normalize(vec3(mix(0.55, 0.95, ot), mix(0.72, 0.42, ot), mix(0.42, 0.10, ot)));
  vec3 t2 = normalize(vec3(mix(0.20, 0.72, ot), mix(0.72, 0.68, ot), mix(0.66, 0.12, ot)));
  vec3 ts = normalize(vec3(1.0, 0.16, 0.10));        // thumb straight out along the paw's thumb side: up, in a thumbs-up fist
  t1 = normalize(mix(t1, ts, thumb)); t2 = normalize(mix(t2, ts, thumb));
  t1 = normalize(mix(t1, vec3(0.30, 0.55, 0.78), fist*(1.0 - thumb)));      // a pointing fist tucks the thumb over the curled digits
  gThK[side] = 1.0 + 0.18*thumb*(1.0 - open);
  gTh1[side] = gTh0[side] + t1*0.112*gThK[side];
  gTh2[side] = gTh1[side] + t2*0.080*(1.0 + 0.35*thumb*(1.0 - open));
}

void armsInit(){
  vec3 hl = vec3(uP[P_HAND_L_X], uP[P_HAND_L_Y], uP[P_HAND_L_Z]);
  vec3 hr = vec3(uP[P_HAND_R_X], uP[P_HAND_R_Y], uP[P_HAND_R_Z]);
  // the torso goes with the hands a little: it turns toward the hand that reaches, and leans into two
  gTwist = 0.055*(hl.z - hr.z) + 0.030*(hr.x - hl.x);
  gLean = 0.022*(hl.z + hr.z) - 0.012*(hl.y + hr.y);
  gTwist = clamp(gTwist, -0.10, 0.10); gLean = clamp(gLean, -0.05, 0.05);
  gBodyW = rotX(-gLean)*rotY(-BODY_YAW - gTwist)*transpose(gBodyPose);
  // the upper arm's coat rides on it from its rest pose
  vec3 E, W; mat3 F, P;
  mat3 keep = gBodyW;
  gBodyW = rotY(-BODY_YAW);
  gFistNow = 0.0; gPointNow = 0.0; gThumbNow = 0.0;
  armSolve(-1.0, vec3(0.0), handTarget(-1.0, vec3(0.0), HAND_REST_L), 0.0, 0.0, HAND_REST_L, gSh0[0], E, W, gUF0[0], F, P);
  armSolve( 1.0, vec3(0.0), handTarget( 1.0, vec3(0.0), HAND_REST_R), 0.0, 0.0, HAND_REST_R, gSh0[1], E, W, gUF0[1], F, P);
  gBodyW = keep;
  // targets, carried by the torso's pose; two paws at the midline stop at contact, palm to palm (half a paw's thickness each side)
  vec3 TL = HIP + gBodyPose*(handTarget(-1.0, hl, HAND_REST_L) - HIP), TR = HIP + gBodyPose*(handTarget(1.0, hr, HAND_REST_R) - HIP);
  vec3 bl = toBody(TL), br = toBody(TR);
  float gapX = br.x - bl.x, near = smoothstep(0.40, 0.15, length(br.yz - bl.yz));
  float push = 0.5*max(0.17 - gapX, 0.0)*near;
  TL = fromBody(bl - vec3(push, 0.0, 0.0)); TR = fromBody(br + vec3(push, 0.0, 0.0));
  gFistNow = max(uP[P_POINT_L], uP[P_THUMB_L])*(1.0 - uP[P_HAND_L_OPEN]); gPointNow = clamp(uP[P_POINT_L], 0.0, 1.0); gThumbNow = clamp(uP[P_THUMB_L], 0.0, 1.0);
  armSolve(-1.0, hl, TL, uP[P_HAND_L_OPEN], uP[P_HAND_L_TURN], HAND_REST_L, gSh[0], gEl[0], gWr[0], gUF[0], gFF[0], gPF[0]);
  gFistNow = max(uP[P_POINT_R], uP[P_THUMB_R])*(1.0 - uP[P_HAND_R_OPEN]); gPointNow = clamp(uP[P_POINT_R], 0.0, 1.0); gThumbNow = clamp(uP[P_THUMB_R], 0.0, 1.0);
  armSolve( 1.0, hr, TR, uP[P_HAND_R_OPEN], uP[P_HAND_R_TURN], HAND_REST_R, gSh[1], gEl[1], gWr[1], gUF[1], gFF[1], gPF[1]);
  pawSolve(0, uP[P_HAND_L_OPEN], uP[P_POINT_L], uP[P_THUMB_L]); pawSolve(1, uP[P_HAND_R_OPEN], uP[P_POINT_R], uP[P_THUMB_R]);
  gPawC[0] = gWr[0] + gPF[0][1]*0.21;
  gPawC[1] = gWr[1] + gPF[1][1]*0.21;
}

void charInit(){
  // yaw>0 toward screen right, pitch>0 nose down, roll>0 crown toward screen right (docs/AUDIO.md)
  // the torso's own pose about the hip (body_yaw +-0.35 rad toward screen right, body_lean +-0.12 rad toward screen right,
  // body_pitch +-0.15 rad toward the camera) carries the neck, the head and the shoulders; body_bounce lifts the whole fox +-0.10
  float bYaw = 0.35*clamp(uP[P_BODY_YAW], -1.0, 1.0), bLean = 0.12*clamp(uP[P_BODY_LEAN], -1.0, 1.0), bPitch = 0.15*clamp(uP[P_BODY_PITCH], -1.0, 1.0);
  gBounce = 0.10*clamp(uP[P_BODY_BOUNCE], -1.0, 1.0);
  gBodyPose = rotY(bYaw)*rotZ(-bLean)*rotX(bPitch);
  mat3 F = gBodyPose*rotY(YAW0 + uP[P_HEAD_YAW])*rotX(uP[P_HEAD_PITCH])*rotZ(-uP[P_HEAD_ROLL]);
  gHR = transpose(F);
  gBodyR = transpose(rotY(YAW0));
  gHO = HIP + gBodyPose*(NECK - HIP) + gBodyPose*vec3(uP[P_HEAD_X], uP[P_HEAD_Y], uP[P_HEAD_Z]);
  // the mouth reads the eleven lip columns (docs/V2.md).  The jaw hinge is teeth_gap and nothing else; the lips are a
  // lens of their own between two edges, so a column can move one lip without the other.
  gPress  = clamp(uP[P_LIP_PRESS], 0.0, 1.0);
  float keep = 1.0 - gPress;
  gTgOut = clamp(uP[P_TONGUE_OUT], 0.0, 1.0);
  float mOpen = clamp(uP[P_MOUTH_OPEN], 0.0, 1.0);
  float open = max(clamp(uP[P_LIP_OPEN], 0.0, 1.0), max(mOpen, 0.34*gTgOut))*keep;
  float up   = clamp(uP[P_LIP_UPPER], 0.0, 1.0)*keep;
  float lo   = clamp(uP[P_LIP_LOWER], 0.0, 1.0)*keep;
  float wid  = clamp(uP[P_LIP_WIDTH], -1.0, 1.0);
  gPro    = clamp(uP[P_LIP_PROTRUDE], 0.0, 1.0);
  gLab    = clamp(uP[P_LIP_TUCK], 0.0, 1.0)*keep;
  float gap = max(clamp(uP[P_TEETH_GAP], 0.0, 1.0), max(mOpen, 0.30*gTgOut));
  gTgTip  = clamp(uP[P_TONGUE_TIP], 0.0, 1.0);
  gTgDent = clamp(uP[P_TONGUE_DENTAL], 0.0, 1.0);
  gTgBack = clamp(uP[P_TONGUE_BACK], 0.0, 1.0);
  gPk = max(-wid, 0.0); gSpread = max(wid, 0.0);
  gRound = clamp(0.55*gPk + 0.65*gPro, 0.0, 1.0);    // the muzzle pad and the whiskers come in with this
  gSmile  = clamp(uP[P_SMILE], 0.0, 1.0);
  gCheekP = clamp(uP[P_CHEEK], 0.0, 1.0);
  gNostril= clamp(uP[P_NOSTRIL] + 0.30*up, 0.0, 1.0);    // a lifted lip draws the nostril
  gSmirk  = clamp(uP[P_SMIRK], -1.0, 1.0);
  gSquint = clamp(uP[P_SQUINT], 0.0, 1.0);
  // pressed lips or a lip tucked under the teeth hold the jaw nearly shut, whatever teeth_gap says on the way there
  gJawA = MAXJAW*(gap + 0.22*max(open - gap, 0.0))*keep*(1.0 - 0.75*gLab) - 0.022*gPress;    // lips that open pull the chin a little after them
  gSinA = sin(gJawA); gCosA = cos(gJawA);
  // the lens, at the middle of the front
  float o = open*(1.0 - gLab);
  gRnd = gPk*sqrt(gPk)*smoothstep(0.0, 0.10, open + 0.5*(up + lo));
  float lu = 0.034*open + 0.082*up;
  gLF = 0.030*gLab;
  float ld = DROPL*sin(MAXJAW*o) - 0.010*gLab;
  float h = 0.092*(1.0 + 1.6*open);                  // a round O is as tall as it is wide
  gLU = mix(lu, 0.26*h, gRnd);
  gLD = mix(ld, 0.74*h, gRnd);
  gLO = 0.070*lo*(1.0 - gRnd);
  gAmt = gLU + gLF + gLD + gLO;
  gRO = gRnd*smoothstep(0.10, 0.22, open);
  gDip = 0.40*(gLU + gLD)*gRO;                       // the corners sit at mid height: a circle's widest point, not a triangle's top edge
  gProfR = mix(gPk, 1.0, gRO);
  gLensX = 1.0/(0.667 + 9.8*gPk*gPk);                // half width of the opening: no limit at rest, 0.095 in a full pucker
  gSeal = 1.0 - smoothstep(0.0, 0.020, gAmt);
  float room = gAmt + 0.055*gLab + 0.90*(0.082*up + 0.070*lo);     // lips drawn off the teeth show them: the rows meet as the jaw says, never a frame with a gap in it
  gDark = 1.0 - smoothstep(0.022, 0.078, room);
  gLipO = smoothstep(0.0, 0.075, gAmt);
  gLipF = (1.0 + 0.40*gPk + 0.42*gPress)*(1.0 - 0.30*smoothstep(0.10, 0.55, gAmt) - 0.20*gSpread);
  gPeel = max(sqrt(gPress), 0.72*(1.0 - smoothstep(0.012, 0.115, gAmt)));   // the lips stick: any opening starts at the middle and runs out to the corners, with or without a press to release
  gTight = smoothstep(0.74, 0.96, gPk)*smoothstep(0.40, 0.14, open);       // a tight round covers the teeth
  gShow = smoothstep(0.034, 0.072, room)*(1.0 - gTight);
  gBlinkL = clamp(uP[P_BLINK_L], 0.0, 1.0); gBlinkR = clamp(uP[P_BLINK_R], 0.0, 1.0);
  gGaze = vec2(uP[P_GAZE_X], uP[P_GAZE_Y]);
  gBrowL = uP[P_BROW_L]; gBrowR = uP[P_BROW_R]; gKnit = clamp(uP[P_BROW_KNIT], 0.0, 1.0);
  // ears: pose + flick about the base.  _l = screen-left.
  //  pitch ear_l/ear_r -1 back/droop .. 1 up/forward, 0.60 rad; turn ear_*_turn -1 the opening forward/in .. 1 out, 0.80 rad about the
  //  ear's own long axis; a flick (0..1) snaps it back and out, a twitch you see.
  float fL = clamp(uP[P_EAR_FLICK_L], 0.0, 1.0), fR = clamp(uP[P_EAR_FLICK_R], 0.0, 1.0);
  float aL = 0.60*clamp(uP[P_EAR_L], -1.0, 1.0) + 0.40*fL, aR = 0.60*clamp(uP[P_EAR_R], -1.0, 1.0) + 0.40*fR;
  float tL = 0.80*clamp(uP[P_EAR_L_TURN], -1.0, 1.0) + 0.30*fL, tR = 0.80*clamp(uP[P_EAR_R_TURN], -1.0, 1.0) + 0.30*fR;
  gEarRL = rotY(-tL)*rotX(-aL)*rotZ(0.30*fL);
  gEarRR = rotY(-tR)*rotX(-aR)*rotZ(0.30*fR);
  // the ear tips as posed, head space: the head guard's capsules
  for(int i=0;i<2;i++){
    float sd = i == 0 ? -1.0 : 1.0;
    vec3 v = transpose(i == 0 ? gEarRL : gEarRR)*vec3(0.0, EARL, 0.0);
    v.yz = rot(0.10)*v.yz;
    v.xy = rot(-0.26)*v.xy;
    v.x *= sd;
    gEarTip[i] = vec3(sd*EARB.x, EARB.yz) + v;
  }
  gJawTip = jawSpace(JAW_B + vec3(0.0,-0.05,0.0), HINGE, gJawA);
  gNoseW = 0.036*clamp(uP[P_NOSE_WIGGLE], -1.0, 1.0);
  gNoseP = NOSEP + vec3(0.0, gNoseW, -0.25*abs(gNoseW));
  gEyeWide = clamp(uP[P_EYE_WIDE], 0.0, 1.0);
  gShrL = clamp(uP[P_SHOULDER_L], 0.0, 1.0); gShrR = clamp(uP[P_SHOULDER_R], 0.0, 1.0);
  float wg = clamp(uP[P_TAIL_WAG], -1.0, 1.0);
  gTailR = rotY(wg > 0.0 ? 3.0*wg : 0.6*wg);
  gBreath = uP[P_BREATH];
  armsInit();
  gCorner = 0.40 + 0.43*gPk*gPk*(3.0 - 2.0*gPk)*0.93 - 0.11*gSpread + 0.035*gPress;      // mouth corner, z: forward along the muzzle in a pucker, back in a spread
  // an opening lens closes short of the skull, so the corner is a real point in the cheek's skin and not where the opening
  // runs into bone; sealed it is the old line's end
  gCorner += 0.105*gLipO*(1.0 - gPk*gPk*(3.0 - 2.0*gPk)) - 0.02*gSpread;
  gPull = 0.105*gSpread;
  gPadUp = 0.022*up + 0.010*gSpread;
}

float sq(float x){ return x*x; }

// which side of the muzzle a point is on, -1 .. 1, smooth through the middle: a kink here tears the sealed line
float sideOf(float x){ return x/sqrt(x*x + 0.010); }

// the lip line at rest, head space (also evaluated in jaw space).  notch: the philtrum's dip, on the upper lip
float mouthY(vec3 q, float notch){
  float s = clamp((q.z - 0.34)/0.60, 0.0, 1.0);
  float y = -0.236 + 0.020*s;
  y += sq(1.0 - s)*(0.055 + gSmile*0.085 + gSpread*0.055 + gSmirk*0.10*sideOf(q.x));     // a smirk lifts one corner, drops the other
  y += notch*0.016*exp(-q.x*q.x*500.0)*s*s;
  return y;
}

// the lens: how much of the front's opening a point of the mouth line gets.  1 at the front, 0 at the corner.  The
// corner is where both edges come back to the line, so it is a real point that moves, and the mouth ends there
// t: 0 at the front .. 1 on the corner.  In plan the opening is an ellipse, so both lips share one outline even though
// the snout is wider than the jaw; a pucker draws its half width in x, and seen from the front it is a round O
float lensT(float z, float x){
  float a = max(ZF - z, 0.0)/(ZF - gCorner + 0.07*gSmirk*sideOf(x)), b = abs(x)/gLensX;
  return sqrt(a*a + b*b);
}
float lensProf(float t){
  t = min(t, 1.0);
  float t2 = t*t;
  return mix(1.0 - t2, sqrt(1.0 - t2), gProfR);
}

// 2D ellipse, soft distance: the lip's section and the dental arch
float sdEll2(vec2 p, vec2 r){ float k0 = length(p/r), k1 = length(p/(r*r)); return k0*(k0 - 1.0)/max(k1, 1e-5); }

// sealed lip line: half thickness at height LINE_H above the lip edge, gone at the corner
const float LINE_H = 0.0043;    // 1.35x since the camera pulled back (r3): the sealed line keeps its weight on a phone
float lineTaper(float z, float x){ float c = gCorner - 0.07*gSmirk*sideOf(x); return smoothstep(c - 0.03, c + 0.17, z); }

// outputs of foxEval (parts, for fur + materials)
float gFur, gBare, gMask, gFurLen, gBowl, gWBody, gToothL, gToothW, gToothF, gRestL, gBehind, gProfU, gTgW, gRoundE, gLipHU, gLipHL, gCornB, gCavO, gTL, gTgU;
vec3  gRest;            // rest-pose coordinate (fur + colour stick to it)
float gSnoutP, gLowerRaw, gUp, gLo, gCavU, gCavL, gEye, gNose, gTeeth, gTongue, gEarD, gBodyD, gHeadD, gMYu, gMYl, gWhisk;
vec3  gQ, gQJ, gQE; float gEarS;
float gBodyNoArm, gBodyNoTail, gTailD, gTailT, gWTail, gWThigh; vec3 gRestW, gTailDir, gTailC;
const vec3 TAIL0 = vec3(-0.30,-2.28,-0.60), TAIL1 = vec3(-0.98,-2.14,-0.72), TAIL2 = vec3(-1.40,-1.68,-0.72), TAIL3 = vec3(-1.60,-1.12,-0.66);

float sdEarShape(vec3 e){
  // e: y along the ear, z out of the ear's face.  A flattened round cone with a bowl on the front.
  vec3 s = vec3(1.0, 1.0, 2.3);
  float d = sdTaper(e*s, vec3(0.0,0.0,0.0), vec3(0.0,EARL,0.0), 0.285, 0.022)/2.3;
  vec3 b = e - vec3(0.0,0.10,0.075);
  float bowl = sdTaper(b*s, vec3(0.0,0.0,0.0), vec3(0.0,EARL*0.8,0.0), 0.205, 0.008)/2.3;
  gBowl = bowl;
  return smax(d, -bowl, 0.03);
}

// ---- the paw, in its own frame.  Built only near a hand: see the guards in foxEval
float gPadD, gClawD, gWArm, gArmT;                     // pads and claws are bare; 1 on the arm; along the arm 0 shoulder .. 1 wrist, past 1 on the paw
vec3  gPawL, gArmDir, gLimb; float gWLimb, gSleek;             // paw-frame point; the bone here; the limb's coat coordinates and how much of it shows
float pawSDF(int side, vec3 pw){
  vec3 l = (pw - gWr[side])*gPF[side];
  float back = sdEllipsoid(l - vec3(0.0,0.140,-0.012), vec3(0.132,0.175,0.080));       // the metacarpus, domed on the back
  float d = smin(back, sdTaper(l, vec3(0.0,-0.06,-0.005), vec3(0.0,0.06,-0.005), 0.090, 0.104), 0.05);
  // the big pad under the digits: three lobes run together
  vec3 pa = vec3(abs(l.x), l.yz);
  float pad = sdEllipsoid(l - vec3(0.0,0.118,0.066), vec3(0.074,0.060,0.030));
  pad = smin(pad, sdEllipsoid(pa - vec3(0.060,0.158,0.056), vec3(0.046,0.046,0.028)), 0.030);
  pad = smin(pad, sdEllipsoid(l - vec3(0.0,0.178,0.058), vec3(0.042,0.040,0.028)), 0.030);
  float dg = 1e3, claw = 1e3;
  int o = side*4;
  for(int i=0;i<4;i++){
    int j = o + i;
    float a = smin(sdTaper(l, gDg0[j], gDg1[j], 0.050, 0.044), sdTaper(l, gDg1[j], gDg2[j], 0.044, 0.038), 0.012);
    dg = smin(dg, a, 0.014);
    claw = min(claw, sdTaper(l, gDg2[j], gDg3[j], 0.019, 0.0035));
    pad = min(pad, length(l - gDgP[j]) - 0.033);
  }
  float tk = gThK[side];
  float th = smin(sdTaper(l, gTh0[side], gTh1[side], 0.052*tk, 0.046*tk), sdTaper(l, gTh1[side], gTh2[side], 0.046*tk, 0.038*tk), 0.012);
  d = smin(d, th, 0.030);
  d = smin(d, dg, 0.030);
  gPawL = l;
  gPadD = min(gPadD, max(pad, -d - 0.012));                                           // a pad only proud of the skin
  gClawD = min(gClawD, claw);
  return d;
}

// after the head's rest and coat are done: the arms' coat.  The shoulder is the same coat as the chest, so it keeps the body's strands and
// rides on the bone; further down a limb's own combed coat (gLimb: across, across, along the arm) cross-fades in, with no pole to swirl around
void armsFinish(vec3 pw, float dUL, float dUR, float dFL, float dFR, float dPL, float dPR){
  float dL = min(min(dUL, dFL), dPL), dR = min(min(dUR, dFR), dPR);
  float wA = 0.0;
  gArmDir = vec3(0.0, 1.0, 0.0); gArmT = 0.0; gLimb = vec3(0.0); gWArm = 0.0; gWLimb = 0.0; gSleek = 0.0;
  // the arm's coat only where an arm is the nearest thing (its weight is exactly 0 past nearBody + 0.06)
  for(int i=0;i<1 && min(dL, dR) < min(gBodyNoArm, gHeadD) + 0.06;i++){
  float isR = dR < dL ? 1.0 : 0.0;
  float dU = mix(dUL, dUR, isR), dF = mix(dFL, dFR, isR), dP = mix(dPL, dPR, isR);
  vec3 S = isR > 0.5 ? gSh[1] : gSh[0], E = isR > 0.5 ? gEl[1] : gEl[0];
  mat3 U = isR > 0.5 ? gUF[1] : gUF[0], F = isR > 0.5 ? gFF[1] : gFF[0], P = isR > 0.5 ? gPF[1] : gPF[0];
  vec3 lU = (pw - S)*U, lF = (pw - E)*F;
  vec3 rU = (isR > 0.5 ? gSh0[1] : gSh0[0]) + (isR > 0.5 ? gUF0[1] : gUF0[0])*lU;
  float wE = smoothstep(-0.05, 0.05, dU - dF);
  float wW = smoothstep(-0.03, 0.03, min(dF, dU) - dP);
  gArmDir = wW > 0.5 ? P[1] : (wE > 0.5 ? F[0] : U[0]);
  float tU = clamp(lU.x/ARM_U, 0.0, 1.0), tF = clamp(lF.x/ARM_F, 0.0, 1.0);
  gArmT = mix(mix(0.5*tU, 0.5 + 0.5*tF, wE), 1.0 + 0.5*clamp(gPawL.y*2.0, 0.0, 1.0), wW);
  gLimb = mix(mix(vec3(lU.y, lU.z, lU.x), vec3(lF.y, lF.z, lF.x + ARM_U), wE), vec3(gPawL.x, gPawL.z, gPawL.y + ARM_U + ARM_F), wW) + 5.3*isR;
  float nearBody = min(gBodyNoArm, gHeadD);
  wA = smoothstep(-0.06, 0.05, nearBody - min(min(dU, dF), dP))*smoothstep(0.02, 0.40, length(pw - S));
  wA = max(wA, smoothstep(-0.03, 0.03, nearBody - min(dF, dP)));
  gWArm = wA;
  gWLimb = wA*smoothstep(0.07, 0.28, gArmT);
  gSleek = wA*smoothstep(0.58, 0.82, gArmT);
  gRest = mix(gRest, gBodyR*(rU - NECK) + NECK, wA);
  float len = mix(0.088, 0.030, smoothstep(0.05, 0.86, gArmT));       // long on the shoulder, sleek in the stocking
  len = mix(len, 0.013 + 0.006*smoothstep(0.30, 0.0, gPawL.y), wW);
  len *= smoothstep(0.0, 0.03, gPadD)*smoothstep(0.0, 0.02, gClawD);
  gFurLen = mix(gFurLen, len, wA);
  gWBody = max(gWBody, wA);
  }
  // the brush is a limb too: combed along its length, the longest coat on the fox
  float wT = smoothstep(-0.05, 0.05, gBodyNoTail - gTailD)*(1.0 - wA);
  gWTail = wT;
  gSleek *= 1.0 - wT;
  gLimb = mix(gLimb, gTailC, wT);
  gWLimb = mix(gWLimb, 1.0, wT);
  gArmDir = mix(gArmDir, gTailDir, wT);
  gFurLen = mix(gFurLen, 0.100*smoothstep(1.02, 0.86, gTailT), wT);
}

// a limb's combed coat, the same recipe as furHeight, with no pole
float limbHeight(vec3 c){
  float clump = vnoise3(c*vec3(16.0,16.0,3.0));
  float st = vnoise3(c*vec3(58.0,58.0,6.4));
  float st2 = vnoise3(c*vec3(133.0,133.0,7.2) + vec3(4.1,1.7,8.3));
  return clamp(clump*0.60 + st*0.34 + st2*0.26 - 0.06, 0.0, 1.0);
}

// ---- the head, in head space.  Only built inside the head guard (foxEval): the body, the paws and the brush do not pay for it
vec3 gQM, gQJM;
float foxHead(vec3 q){
  // lips: a pucker gathers the muzzle around the mouth and draws it forward, a spread widens it
  float lipZ = smoothstep(0.34, 0.76, q.z);
  float lipY = exp(-sq((q.y + 0.215)/0.125))*smoothstep(-0.125, -0.205, q.y);        // never up at the nose
  float pinch = 0.74*gPk*(0.55 + 0.45*gPro) - 0.14*gSpread;
  vec3 qm = q;
  qm.x *= 1.0 + pinch*lipZ*lipY;
  qm.z -= 0.062*gPro*smoothstep(0.52, 0.92, q.z)*lipY;
  float yLine = mouthY(qm, 1.0);                      // the rest line: the teeth hang from it
  float tU = lensT(qm.z, q.x);
  gProfU = lensProf(tU)*(1.0 - gPeel*smoothstep(0.08, 0.80, tU));
  float yL = yLine - gDip*exp(-sq((tU - 1.0)/0.55)) + gLU*gProfU + gLF*smoothstep(0.66, 0.88, qm.z);
  gMYu = yL;

  // ---- skull
  // a spread or a smirk draws the skin back along the lip line, so the corner can travel past where the skull stood
  float sideS = sideOf(q.x);
  float pullZ = (gPull + 0.085*max(gSmirk*sideS, 0.0))*exp(-sq((q.y - yLine)/0.070))*smoothstep(0.05, 0.32, q.z);
  float skull = sdEllipsoid(q + vec3(0.0,-0.05,0.02 + pullZ), vec3(0.635,0.555,0.585));
  skull = smax(skull, q.y - 0.50, 0.24);
  // cheeks: broad, then pointed ruff tufts
  vec3 qa = vec3(abs(q.x), q.yz);
  float side = sign(q.x);
  float cheek = sdEllipsoid(qa - vec3(0.42,-0.20 - 0.075*gSinA,0.10 - pullZ), vec3(0.34,0.26,0.30));      // the cheek's mass goes with the jaw
  skull = smin(skull, cheek, 0.14);
  float lift = gCheekP*0.02 + gSmile*0.02 + 0.016*gSquint + 0.012*gSpread - 0.060*gSinA;    // the ruff rides the cheek: down with the jaw
  float ruff = sdTaper(qa, vec3(0.56,-0.10+lift,0.02), vec3(0.96,-0.25+lift,-0.08), 0.20, 0.012);
  ruff = smin(ruff, sdTaper(qa, vec3(0.52,-0.20+lift,0.04), vec3(0.86,-0.42+lift,-0.05), 0.19, 0.012), 0.06);
  ruff = smin(ruff, sdTaper(qa, vec3(0.44,-0.28+lift,0.08), vec3(0.70,-0.57+lift,-0.01), 0.17, 0.012), 0.06);
  ruff = smin(ruff, sdTaper(qa, vec3(0.32,-0.33+lift,0.14), vec3(0.47,-0.64+lift,0.06), 0.14, 0.012), 0.06);
  skull = smin(skull, ruff, 0.07);
  // brows
  float bu = (q.x < 0.0) ? gBrowL : gBrowR;
  vec3 bq = qa - vec3(0.27, 0.395 + 0.05*bu - 0.03*gKnit, 0.34);
  bq.xy = rot(0.16 + 0.34*gKnit)*bq.xy;     // rest: inner ends high, a soft open look
  bq.x += 0.045*gKnit;
  skull = smin(skull, sdEllipsoid(bq, vec3(0.19,0.042,0.12)), 0.10);

  // ---- snout (upper)
  vec3 qn = qm - vec3(0.0, gNoseW*smoothstep(0.62, 0.95, q.z)*smoothstep(-0.26, -0.12, q.y), 0.0);       // the nose's wiggle drags the tip of the snout
  float snoutRaw = sdTaperSq(qn, vec3(0.0,-0.02,0.26), vec3(0.0,-0.112,0.925), 0.325, 0.110, 1.10);
  vec3 pa = vec3(abs(qm.x), qm.yz);
  float pad = sdEllipsoid(pa - vec3(0.125,-0.150 + gCheekP*0.018 + gSmile*0.012 + gPadUp + 0.016*gSmirk*sideS - 0.006*gPress,0.70 - 0.030*gSpread), vec3(0.135,0.102,0.25));
  snoutRaw = smin(snoutRaw, pad, 0.05);
  gSnoutP = snoutRaw;
  float snoutCut = smax(snoutRaw, yL - q.y, 0.012);
  // the upper lip: a rolled bead along the edge, thin to nothing at the corner.  Sealed at rest it is not there;
  // pressed it comes back, so a closed m is not the same as a closed resting mouth
  float lipV = max(gLipO, 0.85*gPress)*gLipF;
  float lipC = sqrt(max(1.0 - min(tU, 1.0)*(1.0 - 0.30*gPress), 0.0));
  vec2 lipUr = vec2(0.0120*(1.0 - 0.55*gPress), 0.0125)*lipV*lipC + 0.0009;
  gLipHU = 2.2*0.0125*lipV*max(lipC, 0.62*smoothstep(1.12, 0.98, tU))*(1.0 - 0.62*gPress);     // the paint keeps going to the corner, past where the bead has thinned out: the two lips meet there
  snoutCut = smin(snoutCut, sdEll2(vec2(snoutRaw + 0.0015, q.y - yL - 0.8*lipUr.y), lipUr), 0.006 + 0.012*gPress);
  float headRaw = smin(skull, snoutCut, 0.16);
  // sockets
  vec3 ea = qa - EYEC;
  headRaw = smax(headRaw, -(length(ea) - (EYER + 0.012)), 0.035);
  // the skull keeps a pocket where the jaw sleeps; the side walls stay as cheeks
  float pocketW = mix(0.145, 0.075, smoothstep(0.20, 0.85, q.z));   // always inside the jaw, even pinched by round
  float pocketY = min(mix(yL, -0.46, smoothstep(0.0, 0.30, gJawA)), yL - 1.3*gAmt);   // no height at rest: the sealed pose stays solid
  float pocket = max(max(abs(qm.x) - pocketW, q.y - yL), max(0.14 - q.z, pocketY - q.y));
  float upper = smax(headRaw, -pocket, 0.02);
  // roof of the mouth
  float behind = smoothstep(0.90, 1.14, tU);       // behind the corner the lips are shut: skin, no rim
  gBehind = behind;
  float cavU = mouthCavity(snoutRaw, q.y, yL, 0.024 + (0.060 - 0.046*gPk)*behind, max(yLine + 0.075 - yL, 0.03), 1.0);
  upper = smax(upper, -cavU, 0.012);
  gCavO = cavU;
  gUp = upper; gCavU = min(cavU, pocket + 0.012*gDark);    // inMouth tests < 0.012: at rest the pocket has no height, and its slab must not paint the lips

  // ---- lower jaw, built in its rest frame
  vec3 qj = jawSpace(q, HINGE, -gJawA);
  gQJ = qj;
  vec3 qjm = qj;
  // only near the jaw (a capsule around it as posed): the cranium, the brow and the ears do not pay for it
  float jawG = sdCapsule(q, vec3(0.0,-0.205,0.16), gJawTip, 0.31);
  float yRest = mouthY(qj, 0.0), tL = 9.0, lowerRaw = 1e3, yLj = 1e3, lower = jawG, cavL = 1e3;
  gRestL = yRest; gTL = tL; gMYl = yLj; gRoundE = 0.0; gLipHL = 0.0; gLowerRaw = lowerRaw;
  for(int i=0;i<1 && jawG < max(0.34, upper + 0.30);i++){      // skipped only where the head's blend with the jaw (k <= 0.27) cannot see it
  // the jaw is narrower than the snout, so it gathers less for the two rims to land on one outline; the lower lip's rim
  // carries the pinch down with the jaw
  float yBell = clamp(-0.215, q.y, qj.y);
  qjm.x *= 1.0 + (pinch > 0.0 ? 0.50*pinch : pinch)*max(lipZ, smoothstep(0.34, 0.76, qj.z))*exp(-sq((yBell + 0.215)/0.125));
  qjm.z -= 0.040*gPro*smoothstep(0.50, 0.86, qj.z)*exp(-sq((qj.y + 0.235)/0.10));
  // The lower lip's edge, in the jaw's rest frame.  Rigid with the jaw at the front when lip_open equals teeth_gap;
  // it rises in this frame where the lens says the lips are closer than the jaw, up to the rest line at the corner,
  // whatever the jaw does: the skin between the jaws stretches, the corner stays put.
  yRest = mouthY(qj, 0.0);
  gRestL = yRest;
  float zHead = qjm.z - 0.14*gSinA;                  // a jaw point sits this far back in the head frame
  tL = lensT(zHead, q.x);
  gTL = tL;
  gRoundE = gRnd*max(smoothstep(0.24, 0.09, abs(tL - 1.0)), smoothstep(0.0, 0.03, q.x)*smoothstep(0.50, 0.16, tL - 1.0));    // the steep side wall of a round opening, which a band in y cannot reach; wider on the far side, seen edge on
  float prL = lensProf(tL)*(1.0 - gPeel*smoothstep(0.08, 0.80, tL));
  float dzH = max(qjm.z - HINGE.z, 0.0);
  yLj = mouthY(qjm, gSeal) + (dzH*gSinA - prL*gLD*mix(dzH/(ZF - HINGE.z), 1.0, gRO))/gCosA - prL*gLO
            - gDip*exp(-sq((tL - 1.0)/0.55)) + 0.012*gSeal - 0.002;    // closed: the lips overlap, no seam to see through
  gMYl = yLj;
  gRoundE *= smoothstep(0.028, 0.004, q.y - yL)*smoothstep(0.028, 0.004, yLj - qj.y);      // only between the two edges
  gCornB = gLipO*exp(-sq((tU - 1.0 + 0.17*smoothstep(0.03, 0.18, gAmt - 0.12*gSpread))/0.11) - sq((q.y - yLine + gDip)/(0.022 + 0.06*gAmt)));      // the corner: a soft bald pad that travels with it, forward of t = 1 as the jaw opens and the cheek's web hides the lens's end
  // labiodental: the lower lip rolls back under the top teeth, the chin stays
  // Above its rim the jaw is its own rim outline, extruded: where the lens holds the edge higher than the bone, this
  // is the cheek skin stretched across the gape.  Walls only, the top stays open.  At rest it does nothing
  vec3 ql = qjm;
  ql.z += gLab*0.072*smoothstep(0.55,0.86,qj.z)*exp(-sq((qj.y - yRest + 0.012)/0.070));
  ql.y = min(ql.y, yRest);
  lowerRaw = sdTaperSq(ql, JAW_A, JAW_B, JAW_RA, JAW_RB, 1.15);
  lowerRaw = smin(lowerRaw, sdEllipsoid(ql - vec3(0.0,-0.300 - 0.016*gLO/0.070 + 0.010*gPress,0.815), vec3(0.105,0.075,0.10)), 0.05);    // the chin goes with the lower lip: down as it draws down, up as the lips press
  lowerRaw -= 0.011*gPress*exp(-sq((qj.y - yRest + 0.056)/0.040))*smoothstep(0.55,0.80,qj.z);    // pressed lips: the skin below bunches
  gLowerRaw = lowerRaw;
  lower = smax(lowerRaw, qj.y - yLj, 0.012);
  float lipCL = sqrt(max(1.0 - min(tL, 1.0)*(1.0 - 0.30*gPress), 0.0));
  vec2 lipLr = vec2(0.0140*(1.0 - 0.55*gPress), 0.0165)*lipV*lipCL + 0.0009;
  gLipHL = 2.2*0.0165*lipV*max(lipCL, 0.62*smoothstep(1.14, 1.00, tL))*(1.0 - 0.62*gPress);
  lower = smin(lower, sdEll2(vec2(lowerRaw + 0.0015, qj.y - yLj + 0.8*lipLr.y), lipLr), 0.006 + 0.012*gPress);
  cavL = mouthCavity(lowerRaw, qj.y, yLj, 0.022 + (0.060 - 0.046*gPk)*smoothstep(0.90, 1.14, tL), max(yLj - yRest + 0.112, 0.03), -1.0);       // the floor stays with the bone when the lip stretches up
  lower = smax(lower, -cavL, 0.012);
  }
  gLo = lower; gCavL = cavL;

  // ---- teeth + tongue (bare), only inside the mouth's box: the skull, the cheeks and the ears do not pay for them
  // two tight boxes: the upper row in the head frame, the lower row and the tongue (out too) in the jaw's.  Their distance is
  // also the value outside them, so it must not be small where only the coat is (it steers the tile pre-pass and the march)
  vec3 mgd = abs(q - vec3(0.0,-0.21,0.79)) - vec3(0.18,0.145,0.16);
  float mouthU = length(max(mgd, 0.0)) + min(max(mgd.x, max(mgd.y, mgd.z)), 0.0);
  mgd = abs(qj - vec3(0.0,-0.235,0.72)) - vec3(0.17,0.24,0.58);
  float mouthJ = length(max(mgd, 0.0)) + min(max(mgd.x, max(mgd.y, mgd.z)), 0.0);
  float mouthG = min(mouthU, mouthJ);
  gTeeth = mouthG; gTongue = mouthJ; gToothL = 0.0; gToothW = 0.0; gToothF = 0.0; gTgU = 0.0; gTgW = 0.0;
  for(int i=0;i<1 && mouthG < 0.03;i++){
  // A small neat fox set, hung from the rest line, not from the lips.  One soft band around the dental arch, scalloped;
  // a slim canine each side.  The tips wait above the line until the lens has room, so a small opening is a dark slit
  // and the teeth grow out of it instead of arriving as blocks.
  float hide = 1.0 - gShow;
  float yTipU = yLine + 0.040*hide + 0.045*gTight;       // a tight round is seen from below too: the row goes up behind the lip altogether
  vec2 archP = vec2(qm.x, q.z - 0.46);
  float archU = sdEll2(archP, vec2(0.150, 0.455));
  float arcU = abs(qm.x) + (0.915 - q.z);
  float fU = abs(fract(arcU*23.0) - 0.5)*2.0;           // 0 middle of a tooth, 1 between two
  float sideU = smoothstep(0.80, 0.63, q.z);            // incisors at the front, then the row sinks into the gum: no cut end
  float teethU = smax(abs(archU) - 0.0115, yTipU + (0.018 + 0.070*sideU)*sideU + 0.0075*fU*fU - q.y, 0.006);
  teethU = max(max(teethU, q.y - yLine - 0.075), 0.63 - q.z);
  vec3 fa = vec3(abs(qm.x), q.y - yTipU, q.z);
  float fang = sdTaper(fa, vec3(0.103,0.060,0.800), vec3(0.106,-0.025,0.792), 0.0175, 0.0042);
  teethU = smin(teethU, fang, 0.006);
  float bite = 0.020*gShow*smoothstep(0.11, 0.0, gJawA);                  // shut, the lower row stands behind the upper one: an overbite, no gap at the sides in a 3/4 view
  float yTipL = yRest - 0.002 + bite - 0.040*hide - 0.075*max(gTight, gRO);      // a round lower lip stands above its rim and covers the row
  vec2 archQ = vec2(qjm.x, qj.z - 0.46);
  float archL = sdEll2(archQ, vec2(0.128, 0.425));
  float arcL = abs(qjm.x) + (0.885 - qj.z);
  float fL = abs(fract(arcL*26.0) - 0.5)*2.0;
  float sideL = smoothstep(0.81, 0.70, qj.z);
  float teethL = smax(abs(archL) - 0.0105, qj.y - yTipL + (0.024 + 0.040*sideL)*sideL + 0.0065*fL*fL, 0.006);
  teethL = max(max(teethL, yTipL - 0.046 - qj.y), 0.70 - qj.z);
  vec3 fb = vec3(abs(qjm.x), qj.y - yTipL, qj.z);
  float fangL = sdTaper(fb, vec3(0.083,-0.050,0.800), vec3(0.081,0.026,0.804), 0.0150, 0.0040);
  teethL = smin(teethL, fangL, 0.006);
  teethL = max(teethL, lowerRaw + 0.007);                // inside the skin, whatever the tuck does to it
  gTeeth = min(teethU, teethL);
  gToothL = step(teethL, teethU);
  gToothW = gToothL > 0.5 ? yTipL - qj.y : q.y - yTipU;
  gToothF = gToothL > 0.5 ? fL : fU;

  // the tongue lies low in the jaw; its tip goes up behind the upper teeth or out between them, its back humps
  float dzT = 0.80 - HINGE.z;
  // tongue_out slides it forward past the lips (tip ~0.17 beyond the front at 1), a little down, and fattens it
  vec3 tq = qj - vec3(0.0, yRest - 0.086 + 0.090*gTgOut, 0.49 + 0.075*gTgDent + 0.36*gTgOut);     // up over the lower row, then out
  tq.y += 0.55*gTgOut*sq(max(tq.z - 0.08, 0.0));       // past the lips the tip hangs a little
  tq.y += (1.0 - gTgBack)*0.34*sq(max(0.60 - qj.z, 0.0));     // at rest its back falls away into the throat
  tq.y -= gRO*(0.040 + max(DROPL*gSinA - gLD, 0.0));          // an open round's floor is the stretched lip: the tongue rises to it
  float tz = clamp((tq.z - 0.08)/0.24, 0.0, 1.0);        // 0 the blade .. 1 the tip: only the last third curls
  float tgUp = gTgTip*(0.096 + dzT*gSinA) + gTgDent*(0.084 + 0.50*dzT*gSinA);
  tq.y -= tgUp*tz*tz*(3.0 - 2.0*tz);
  float hump = clamp((tq.z + 0.27)/0.46, 0.0, 1.0);
  tq.y -= gTgBack*(0.090 + 0.42*gSinA)*sq(hump*(1.0 - hump)*4.0);
  tq.x = qjm.x*(1.0 + (0.34 + 0.46*gTgDent + 0.75*gTgTip*tz)*smoothstep(-0.08, 0.30, tq.z));
  gTgU = tq.y;
  gTgW = abs(tq.x)/0.092;
  float tongue = sdEllipsoid(tq, vec3(0.092 + 0.012*gTgOut,0.033 + 0.012*max(max(gTgTip, gTgBack), gTgOut),0.32))*0.70;
  tongue += 0.005*smoothstep(0.02,0.0,abs(tq.x));     // centre groove
  gTongue = tongue;
  }

  // ---- eyeballs
  gEye = length(ea) - EYER;

  // ---- nose
  vec3 nq = q - gNoseP;
  float nose = length(nq) - 0.12;          // a bound, away from the nose
  for(int i=0;i<1 && nose < 0.10;i++){
    nose = sdEllipsoid(nq, vec3(0.096,0.072,0.078));
    nose = smin(nose, sdEllipsoid(nq - vec3(0.0,-0.035,-0.012), vec3(0.04,0.05,0.05)), 0.03);
    vec3 nna = vec3(abs(nq.x), nq.yz);
    float nostril = length((nna - vec3(0.044,-0.012,0.058))*vec3(1.0,1.6,1.0)) - (0.016 + 0.009*gNostril);
    nose = smax(nose, -nostril, 0.012);
  }
  gNose = nose;

  // ---- whiskers: one mirrored fan out of the muzzle pads: six whiskers for the price of three capsules.
  // Thin, so they live in gcBare (never get fur).  They ride the pad: up with smile/cheek, in with round.
  vec3 wgd = abs(qa - vec3(0.41,-0.11,0.69)) - vec3(0.31,0.19,0.13);     // the fan, every pad lift and droop: x .10..72, y -.30..08, z .56..82
  float wh = length(max(wgd, 0.0)) + min(max(wgd.x, max(wgd.y, wgd.z)), 0.0);
  for(int i=0;i<1 && wh < 0.10;i++){
    vec3 wq = qa + vec3(0.21*(1.0 - 1.0/(1.0 + 0.50*gRound - 0.13*gSpread)), -(gCheekP*0.018 + gSmile*0.012 + gPadUp + 0.016*gSmirk*sideS), 0.0);
    wq.y += 0.24*sq(max(wq.x - 0.20, 0.0));                  // the droop is one shared bend
    wh = sdTaper(wq, vec3(0.185,-0.125,0.790), vec3(0.610,-0.005,0.640), WHISK_R0, WHISK_R1);
    wh = min(wh, sdTaper(wq, vec3(0.190,-0.158,0.770), vec3(0.675,-0.110,0.610), WHISK_R0, WHISK_R1));
    wh = min(wh, sdTaper(wq, vec3(0.180,-0.190,0.750), vec3(0.620,-0.225,0.590), WHISK_R0, WHISK_R1));
  }
  gWhisk = wh;

  // ---- eyelids
  float blink = (q.x < 0.0) ? gBlinkL : gBlinkR;
  vec3 el = ea;
  el.xy = rot(-EYETILT)*el.xy;
  float squint = gSmile*0.10 + gCheekP*0.10 + 0.40*gSquint;
  float aU = mix(0.47 + 0.30*gEyeWide, -0.32, blink);      // rest: 10% less opening than wide-eyed 0.60; eye_wide lifts it (a blink still closes)
  float aL = mix(-0.74 + squint - 0.10*gEyeWide, -0.36, blink);
  float bru = (q.x < 0.0) ? gBrowL : gBrowR;
  aU += 0.06*bru*(1.0 - blink);            // no raised brow may hold a blink open
  float lids = length(ea) - EYER - 0.032;         // a bound off the lids' shell: they are only built near the eye
  for(int i=0;i<1 && lids < 0.10;i++) lids = sdEyelids(el, EYER, 0.016, aU, aL);
  upper = smin(upper, lids, 0.030);
  gUp = upper;

  // ---- ears
  float s = (q.x < 0.0) ? -1.0 : 1.0;
  vec3 e = q - vec3(s*EARB.x, EARB.y, EARB.z);
  e.x *= s;
  // tilt outward + a little back, then the pose rotation
  e.xy = rot(0.26)*e.xy;
  e.yz = rot(-0.10)*e.yz;
  e = ((q.x < 0.0) ? gEarRL : gEarRR)*e;
  gQE = e; gEarS = s;
  float ear = sdCapsule(q, vec3(s*EARB.x, EARB.yz), q.x < 0.0 ? gEarTip[0] : gEarTip[1], 0.31);     // a bound, away from the ear
  gBowl = 1e3;
  for(int i=0;i<1 && ear < 0.25;i++) ear = sdEarShape(e);
  gEarD = ear;

  // sealed, the cheek skirt melts into the jaw: no second edge below the lip line
  float head = smin(upper, lower, 0.012 + 0.10*smoothstep(gCorner + 0.05, gCorner - 0.12, qj.z) + 0.05*gSeal*smoothstep(0.74, 0.56, qj.z)
                                  + 0.10*smoothstep(0.08, 0.40, gJawA)*smoothstep(0.62, 0.34, qj.z)) - 0.006*gCornB;
  head = smin(head, ear, 0.05);
  gQM = qm; gQJM = qjm; gMYu = yL; gMYl = yLj; gSnoutP = snoutRaw; gTL = tL;
  return head;
}

// far from the head: every head part is far, and its weights hand the point to the body
void foxHeadFar(float hg){
  gQJ = gQ; gQM = gQ; gQJM = gQ; gQE = vec3(0.0, -1.0, 0.0); gEarS = 1.0;
  gMYu = 1e3; gMYl = 1e3; gSnoutP = 1e3; gLowerRaw = 1e3; gCavU = 1e3; gCavL = 1e3; gCavO = 1e3; gTL = 9.0;
  gEye = 1e3; gNose = 1e3; gTeeth = 1e3; gTongue = 1e3; gWhisk = 1e3; gEarD = 1e3; gBowl = 1e3;
  gUp = hg; gLo = hg; gRoundE = 0.0; gCornB = 0.0; gLipHU = 0.0; gLipHL = 0.0; gBehind = 0.0; gProfU = 0.0;
  gTgW = 0.0; gTgU = 0.0; gToothL = 0.0; gToothW = 0.0; gToothF = 0.0; gRestL = 0.0;
}

// a conservative bound of everything foxHead builds (ruff, open jaw, tongue out, whiskers; the ears as posed): a box and two capsules
const vec3 HG_C = vec3(0.0,-0.04,0.30), HG_R = vec3(0.99,0.68,0.93);      // y -0.72 .. 0.64, z -0.63 .. 1.23 (tongue out)
const float HG_M = 0.31;            // skip the head past this: the coat (0.105) and the head/body blend (0.20, at the surface) never reach out further
float headGuard(vec3 q){
  vec3 d = abs(q - HG_C) - HG_R;
  float box = length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
  box = min(box, sdCapsule(q, vec3(0.0,-0.205,0.16), gJawTip, 0.31));      // the open jaw and its chin
  float s = q.x < 0.0 ? -1.0 : 1.0;
  float ear = sdCapsule(q, vec3(s*EARB.x, EARB.yz), q.x < 0.0 ? gEarTip[0] : gEarTip[1], 0.31);
  return min(box, ear);
}

void foxHeadRest(vec3 q, float wBody);
// Round 3's coat length away from the head: there the ear's distance (squashed, so short) beat the skull's, the ear weight won and
// the coat was the ear's (x0.40): short on the lower chest, belly and haunches.  Kept with a cheap skull (cranium, cheeks, ruff)
float coatEarFar(vec3 q){
  vec3 qa = vec3(abs(q.x), q.yz);
  float skull = smax(sdEllipsoid(q + vec3(0.0,-0.05,0.02), vec3(0.635,0.555,0.585)), q.y - 0.50, 0.24);
  skull = smin(skull, sdEllipsoid(qa - vec3(0.42,-0.20 - 0.075*gSinA,0.10), vec3(0.34,0.26,0.30)), 0.14);
  float lift = gCheekP*0.02 + gSmile*0.02 + 0.016*gSquint + 0.012*gSpread - 0.060*gSinA;
  float ruff = sdTaper(qa, vec3(0.56,-0.10+lift,0.02), vec3(0.96,-0.25+lift,-0.08), 0.20, 0.012);
  ruff = min(ruff, sdTaper(qa, vec3(0.52,-0.20+lift,0.04), vec3(0.86,-0.42+lift,-0.05), 0.19, 0.012));
  ruff = min(ruff, sdTaper(qa, vec3(0.44,-0.28+lift,0.08), vec3(0.70,-0.57+lift,-0.01), 0.17, 0.012));
  ruff = min(ruff, sdTaper(qa, vec3(0.32,-0.33+lift,0.14), vec3(0.47,-0.64+lift,0.06), 0.14, 0.012));
  skull = min(skull, ruff);
  float s = (q.x < 0.0) ? -1.0 : 1.0;
  vec3 e = q - vec3(s*EARB.x, EARB.y, EARB.z);
  e.x *= s;
  e.xy = rot(0.26)*e.xy;
  e.yz = rot(-0.10)*e.yz;
  e = ((q.x < 0.0) ? gEarRL : gEarRR)*e;
  float bowl = gBowl;
  float ear = sdEarShape(e);
  gBowl = bowl;
  return smoothstep(-0.02, 0.02, skull - ear);
}
void foxEval(vec3 pw){
  pw.y -= gBounce;
  vec3 q = gHR*(pw - gHO) + NECK;
  gQ = q;
  float hg = headGuard(q);
  float head = hg;
  foxHeadFar(hg);
  for(int i=0;i<1 && hg < HG_M;i++) head = foxHead(q);
  gHeadD = head;

  // ---- body: sitting up on its haunches, turned a little after the head.  The torso takes breath and a touch of the hands;
  // the haunches stay on the ground
  vec3 b = toBody(pw);
  gRestW = fromBody0(b);
  float br = 1.0 + 0.016*gBreath;
  b.y += 0.012*gBreath;
  vec3 ba = vec3(abs(b.x), b.yz);
  // nothing of the torso reaches above the neck's top (body frame y 0.05): up at the ears and the crown it is that far away
  float bodyD = b.y - 0.05;
  gWThigh = 0.0;
  for(int i=0;i<1 && bodyD < max(0.32, head + 0.24);i++){      // skipped only where the head/body blend (0.20) cannot see it
  float neck  = sdTaper(b, vec3(0.0,-0.35,-0.28), vec3(0.0,-0.95,-0.24), 0.40, 0.50);
  float chest = sdEllipsoid(b - vec3(0.0,-1.18,-0.10), vec3(0.60,0.50,0.40)*br);
  float belly = sdEllipsoid(b - vec3(0.0,-1.70,-0.12), vec3(0.66,0.66,0.46)*br);
  float shrB = b.x < 0.0 ? gShrL : gShrR;          // a shrug lifts that shoulder's mass with the joint: the neck reads shorter
  float sh = sdEllipsoid(ba - vec3(0.48,-1.02 + 0.14*shrB,-0.20 + 0.02*shrB), vec3(0.26,0.20 + 0.02*shrB,0.25));
  bodyD = smin(smin(smin(neck, chest, 0.25), belly, 0.25), sh, 0.20);
  vec3 h0 = rotY(-BODY_YAW)*pw;
  vec3 ha = vec3(abs(h0.x), h0.yz);
  float thigh = sdEllipsoid(ha - THIGH_C, THIGH_R);
  gWThigh = smoothstep(-0.08, 0.05, bodyD - thigh);
  bodyD = smin(bodyD, thigh, 0.14);
  // chest ruff tufts
  bodyD = smin(bodyD, sdTaper(ba, vec3(0.30,-0.90,0.16), vec3(0.46,-1.20,0.24), 0.15, 0.01), 0.10);
  bodyD = smin(bodyD, sdTaper(ba, vec3(0.06,-0.98,0.24), vec3(0.14,-1.30,0.30), 0.15, 0.03), 0.18);
  }
  // the brush: up out of the haunch behind the fox, on the side it turned from.  It only sways with what the body does
  // tail_wag -1..1 = a POSITION (the controller oscillates it): a swing about the upright through the root, behind the fox: +1 the
  // brush lies out past the screen-right haunch (3.0 rad), -1 it comes round the screen-left haunch toward the lens (0.6 rad).
  // tail_raise lifts the far half (tip +0.40); tail_curl turns the far half forward round the haunch (up to 1.1 rad)
  float wag = clamp(uP[P_TAIL_WAG], -1.0, 1.0);
  // the brush never rises above y -0.68 (raised and swept): above that it is at least this far
  float tlDist = pw.y + 0.68;
  gTailT = 0.0; gTailC = vec3(0.0); gTailDir = vec3(0.0, 1.0, 0.0);
  for(int i=0;i<1 && tlDist < max(0.32, bodyD + 0.14);i++){
  float tlSw = 0.05*gBreath + 1.6*gTwist + 0.45*uP[P_HEAD_YAW];
  vec3 tlq = gTailR*(pw - TAIL0 - vec3(0.45*max(wag, 0.0), 0.0, 0.0));     // the root slides to the middle of the seat
  float tll = length(tlq);
  tlq.xy = rot(tlSw*smoothstep(0.1, 1.5, tll))*tlq.xy;
  float traise = clamp(uP[P_TAIL_RAISE], 0.0, 1.0), tcurl = clamp(uP[P_TAIL_CURL], 0.0, 1.0);
  tlq.y -= 0.40*traise*smoothstep(0.2, 1.6, tll);
  if(tcurl > 0.001) tlq = rotY(-1.1*tcurl*smoothstep(0.5, 1.9, tll))*tlq;
  tlq += TAIL0;
  float tl1 = sdTaper(tlq, TAIL0, TAIL1, 0.15, 0.23), tl2 = sdTaper(tlq, TAIL1, TAIL2, 0.23, 0.25), tl3 = sdTaper(tlq, TAIL2, TAIL3, 0.25, 0.040);
  tlDist = smin(smin(tl1, tl2, 0.06), tl3, 0.06);
  float tlW3 = smoothstep(-0.05, 0.05, min(tl1, tl2) - tl3), tlW2 = smoothstep(-0.05, 0.05, tl1 - tl2);
  vec3 tlA = mix(TAIL0, TAIL1, tlW2), tlB = mix(TAIL1, TAIL2, tlW2);
  tlA = mix(tlA, TAIL2, tlW3); tlB = mix(tlB, TAIL3, tlW3);
  vec3 tlD = normalize(tlB - tlA);
  float tlAl = dot(tlq - tlA, tlD);
  gTailDir = tlD;
  gTailT = mix(mix(tlAl/0.78, 1.0 + tlAl/0.79, tlW2), 2.0 + tlAl/0.60, tlW3)/3.0;
  gTailC = vec3(dot(tlq - tlA, vec3(-tlD.y, tlD.x, 0.0)), tlq.z - tlA.z, gTailT*2.2 + 9.0);
  }
  gTailD = tlDist;
  gBodyNoTail = bodyD;
  bodyD = smin(bodyD, tlDist, 0.10);
  gBodyNoArm = bodyD;
  // ---- arms.  The upper arm is the shoulder's: it melts into the coat at the top and comes free toward the elbow
  float aUL = sdTaper(pw, gSh[0], gEl[0], 0.200, 0.138), aUR = sdTaper(pw, gSh[1], gEl[1], 0.200, 0.138);
  float aFL = sdTaper(pw, gEl[0], gWr[0], 0.138, 0.090), aFR = sdTaper(pw, gEl[1], gWr[1], 0.138, 0.090);
  float aKL = 0.04 + 0.26*smoothstep(0.62, 0.06, length(pw - gSh[0])), aKR = 0.04 + 0.26*smoothstep(0.62, 0.06, length(pw - gSh[1]));
  bodyD = smin(smin(bodyD, aUL, aKL), aUR, aKR);
  gBodyD = bodyD;
  // the paws: every digit and pad, only inside a guard, so the rest of the fox does not pay for them
  const float PG = 0.09;
  float bP0 = length(pw - gPawC[0]) - 0.38, bP1 = length(pw - gPawC[1]) - 0.38;
  float aPL = bP0, aPR = bP1;      // a live bound outside the guard: a flat constant has no normal, for a ray that ends off the surface
  gPadD = PG + 0.5*abs(min(bP0, bP1) - PG); gClawD = gPadD; gPawL = vec3(0.0);
  for(int i=0;i<1 && bP0 < PG;i++) aPL = pawSDF(0, pw);
  for(int i=0;i<1 && bP1 < PG;i++) aPR = pawSDF(1, pw);
  float arms = min(smin(smin(aUL, aFL, 0.05), aPL, 0.04), smin(smin(aUR, aFR, 0.05), aPR, 0.04));
  gFur = smin(smin(head, bodyD, 0.20), arms, 0.025);

  gWhisk = max(gWhisk, 0.003 - gFur);       // cut flush with the skin: a tube buried just under it dirties AO and normals
  gBare = min(min(min(gEye, gNose), min(gTeeth, gTongue)), gWhisk);
  gBare = min(gBare, min(gPadD, gClawD));

  // rest coordinate for fur + paint
  float wBody = smoothstep(-0.10, 0.10, head - bodyD);
  gWBody = wBody;
  gRest = gBodyR*(gRestW - NECK) + NECK; gMask = 1.0;
  float muzzleShort = mix(0.30, 1.0, smoothstep(0.82, 0.30, q.z));
  // round 3's coat was short (x0.40) away from the head, long near it (its ear's squashed distance decided): a fade past the guard
  gFurLen = mix(0.050*muzzleShort, 0.105, wBody)*mix(1.0, 0.40, smoothstep(HG_M, HG_M + 0.30, hg));
  for(int i=0;i<1 && hg < HG_M;i++) foxHeadRest(q, wBody);
  armsFinish(pw, aUL, aUR, aFL, aFR, aPL, aPR);
}

// the head's part of the rest coordinate, the coat mask and the coat length (inside the guard only)
void foxHeadRest(vec3 q, float wBody){
  float upper = gUp, lower = gLo, ear = gEarD, s = gEarS, snoutRaw = gSnoutP, lowerRaw = gLowerRaw, yL = gMYu, yLj = gMYl;
  float cavU = gCavO, cavL = gCavL, tL = gTL;
  vec3 qm = gQM, qj = gQJ, qjm = gQJM, e = gQE;
  vec3 ea = vec3(abs(q.x), q.yz) - EYEC;
  float wJaw  = smoothstep(-0.02, 0.02, upper - lower);
  float wEar  = smoothstep(-0.02, 0.02, min(upper,lower) - ear);
  // the skin before the lips moved it: paint and fur go with it.  A stretched cheek keeps the colour of the jaw's edge.
  // Skin weights: the head's frame at the hinge, the jaw's toward the chin, so coat and colour run on across the cheek
  float wSkin = smoothstep(0.34, 0.74, qj.z);
  float wRest = smoothstep(-0.02 - 0.07*gBehind, 0.02 + 0.07*gBehind, upper - lower);     // a wide seam behind the corner: no swirl in the coat
  vec3 rq = mix(qm, vec3(qjm.x, min(qjm.y, gRestL + 0.12*(qjm.y - gRestL)), qjm.z), wRest*wSkin);
  rq = mix(rq, vec3(s*(e.x + EARB.x), e.y + EARB.y, e.z + EARB.z), wEar);
  gRest = mix(rq, gBodyR*(gRestW - NECK) + NECK, wBody);

  // where fur grows, how long
  float m = 1.0;
  m *= smoothstep(0.02, 0.09, length(q - gNoseP) - 0.10);
  float under = gSeal*smoothstep(0.0, 0.02, yL - q.y);             // sealed, the skin just below the lip line: the coat comes back slowly there, or it steps out as a lit shelf
  m *= smoothstep(0.015, 0.05 + 0.13*under, max(abs(q.y - yL) - 0.012 - 0.9*gLipHU + wJaw*1.0, snoutRaw - 0.03));       // upper lip band
  m *= smoothstep(0.015, 0.05 + 0.13*gSeal, max(abs(qj.y - yLj) - 0.012 - 0.9*gLipHL + (1.0-wJaw)*1.0, lowerRaw - 0.03 - 0.10*gLipO*smoothstep(0.68, 0.88, tL)));// lower lip band: no shelf of coat under the closed rim
  m *= 1.0 - gSeal*lineTaper(q.z, q.x)*smoothstep(0.034, 0.006, max(q.y - yL, 0.3*(yL - q.y)));     // the sealed line runs clear of the coat to its corner
  m *= 1.0 - gDark*smoothstep(0.02, 0.0, max(max(q.y - yL, yLj - qj.y), min(snoutRaw, lowerRaw)));  // no coat bridges a thin slit between the lips
  m *= smoothstep(0.005, 0.05, length(ea) - EYER - 0.022);
  m *= smoothstep(-0.015, 0.03, min(cavU, cavL) - 0.0);
  m *= smoothstep(-0.030, -0.006, max(q.y - yL, lowerRaw + 0.010 + 2.0*gSeal));      // nothing grows inside the mouth
  m *= (1.0 - gRoundE)*(1.0 - 0.92*gCornB);
  gMask = m;
  float muzzleShort = mix(0.30, 1.0, smoothstep(0.82, 0.30, q.z));
  gFurLen = mix(0.050*muzzleShort, 0.105, wBody)*mix(1.0, 0.40, wEar);
  gFurLen *= 1.0 - 0.30*wJaw*smoothstep(0.0, 0.05, qj.y - gRestL)*(1.0 - wBody);     // the stretched cheek's coat thins a little
}

// fur flows away from the nose on the head, away from the neck on the body
vec3 furPole(){ return mix(POLE, vec3(0.0,-0.25,0.15), gWBody); }

void charEval(vec3 p){
  foxEval(p);
  gcBase = gFur; gcBare = gBare; gcLen = gFurLen*gMask; gcRest = gRest;
}
float charTuft(){
  float h = 0.0, l = 0.0;
  for(int i=0;i<1 && gWLimb < 0.999;i++) h = furHeight(gcRest, furPole(), mix(F_K, F_K*1.7, gWBody), F_L);
  for(int i=0;i<1 && gWLimb > 0.001;i++) l = limbHeight(gLimb);
  return mix(h, mix(l, 0.62 + 0.42*(l - 0.62), gSleek), gWLimb);       // the stocking lies flat
}

//@shadow-begin
// cheap proxy for shadow rays: 10 ellipsoids, no fur, no mouth
float charShadow(vec3 pw){
  pw.y -= gBounce;
  vec3 q = gHR*(pw - gHO) + NECK;
  vec3 qa = vec3(abs(q.x), q.yz);
  float d = sdEllipsoid(q - vec3(0.0,0.03,0.0), vec3(0.64,0.56,0.58));
  d = min(d, sdEllipsoid(qa - vec3(0.55,-0.25,0.05), vec3(0.45,0.26,0.28)));
  d = smin(d, sdTaper(q, vec3(0.0,-0.05,0.3), vec3(0.0,-0.13,1.0), 0.28 - 0.05*gRound, 0.10 - 0.05*gRound), 0.12);      // gathers with the muzzle in a pucker: skin drawn in must not sit inside the proxy
  d = min(d, sdTaper(jawSpace(q, HINGE, -gJawA), vec3(0.0,-0.24,0.2), vec3(0.0,-0.30,0.85), 0.22, 0.08));     // the jaw's proxy goes with the jaw, or it shadows it
  vec3 e = qa - EARB;
  e.xy = rot(0.26)*e.xy;
  e.z += 0.075;      // a thin sheet behind the bowl: a thick proxy swallows the bowl and paints a hard dome of shadow in it
  d = min(d, sdTaper(e*vec3(1.,1.,4.), vec3(0.0), vec3(0.0,EARL,0.0), 0.24, 0.02)/4.0);
  // the body must sit INSIDE the real coat: a proxy larger than the body shadows strand tips at random
  vec3 b = toBody(pw);
  d = min(d, sdEllipsoid(b - vec3(0.0,-1.18,-0.10), vec3(0.48,0.40,0.31)));
  d = min(d, sdEllipsoid(b - vec3(0.0,-1.70,-0.12), vec3(0.54,0.54,0.36)));
  d = min(d, sdTaper(b, vec3(0.0,-0.4,-0.28), vec3(0.0,-0.9,-0.26), 0.26, 0.36));
  // the arms cast with the hand: the same bones, thinner
  for(int i=0;i<2;i++){
    d = min(d, sdTaper(pw, gSh[i], gEl[i], 0.135, 0.100));
    d = min(d, sdTaper(pw, gEl[i], gWr[i], 0.100, 0.070));
    d = min(d, length(pw - gPawC[i]) - 0.115);
  }
  return d;
}
//@shadow-end

const vec3 C_ORANGE = vec3(0.72,0.185,0.022);
const vec3 C_DEEP   = vec3(0.46,0.105,0.018);
const vec3 C_GOLD   = vec3(0.92,0.50,0.12);
const vec3 C_CREAM  = vec3(0.90,0.84,0.73);
const vec3 C_DARK   = vec3(0.022,0.016,0.014);
const vec3 WHISK_COL = vec3(0.60,0.55,0.48);
const vec3 C_SOCK = vec3(0.030,0.018,0.015), C_SOCK2 = vec3(0.082,0.048,0.034);
const vec3 C_PAD  = vec3(0.230,0.105,0.105);
const vec3 C_CLAW = vec3(0.300,0.245,0.190);

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  vec3 n = s.n;
  foxEval(pw);
  vec3 q = gQ, r = gRest;
  float dist = min(min(gFur, gBare), 1.0);
  bool isEye = gEye < 0.002 && gEye <= gFur + 0.002;
  bool isNose = gNose < 0.004 && gNose <= gFur + 0.004;
  bool isTeeth = gTeeth < 0.003 && gTeeth <= min(gFur, gTongue) + 0.003;
  bool isTongue = gTongue < 0.003 && gTongue <= gFur + 0.003;
  float mTol = 0.01*(1.0 - gDark);       // nearly closed: the jaw skin under the upper rim is skin, not cavity
  // the upper solid below the lip edge and inside the jaw's outline: the skull's face and its plug, seen through the mouth
  bool inLens = gUp < gLo + 0.012 && q.y < gMYu - 0.002 && gLowerRaw < -0.010 - 0.070*gBehind && gSeal < 0.99;
  bool inMouth = (gCavU < 0.012 && gUp < gLo + mTol && min(gLowerRaw, gSnoutP) < -0.008) || (gCavL < 0.012 && gLo < gUp + mTol) || inLens;

  // ---- fur paint
  vec3 c1 = furCoord(r, furPole(), 140.0, 15.0);
  float st = furStrand(c1);
  float tuft = furTuft(furCoord(r, furPole(), F_K, F_L));
  float lst = 0.6*vnoise3(gLimb*vec3(120.0,120.0,10.0)) + 0.4*vnoise3(gLimb*vec3(240.0,240.0,16.0));      // colour takes the strands, not the clumps
  st = mix(st, lst, gWLimb); tuft = mix(tuft, 0.5 + 0.5*lst, gWLimb);
  float ax = abs(r.x);
  // cream muzzle/cheek mask, in rest space
  float wob = 0.05*(tuft - 0.5) + 0.03*(st - 0.5);
  float yb = -0.165 + 0.115*smoothstep(0.0,0.30,ax) - 0.25*smoothstep(0.32,1.0,ax);
  float cream = smoothstep(-0.015, 0.015, max(yb, -0.135*smoothstep(0.62, 0.74, r.z) - 2.0*(1.0 - smoothstep(0.62, 0.74, r.z))) - r.y + wob)*(1.0 - gWBody);
  // chest bib
  float bib = smoothstep(-0.04, 0.04, 0.34 + 0.20*smoothstep(-1.5, -0.6, r.y) + 0.06*(r.y + 1.0) - ax + 0.14*(tuft - 0.5))*smoothstep(0.1,-0.25, r.y + 0.35);
  cream = max(cream, bib*(1.0 - gWArm)*(1.0 - gWThigh*smoothstep(-0.18, 0.02, ax - 0.20 + 0.20*(tuft - 0.5))));
  // ears: orange, black backs + tips, cream bowl
  float earW = smoothstep(-0.02, 0.02, min(gUp, gLo) - gEarD)*smoothstep(-0.02, 0.02, gBodyD - gEarD);
  vec3 alb = mix(C_ORANGE, C_GOLD, 0.22*smoothstep(0.35,0.80,st) + 0.20*smoothstep(0.35,0.75,vnoise3(r*3.1)));
  alb = mix(alb, C_DEEP, 0.28*smoothstep(0.1,0.5, r.y)*smoothstep(0.75,0.15,tuft));
  alb = mix(alb, C_CREAM, cream);
  float eY = gQE.y/EARL;
  float earTip = smoothstep(0.50, 0.70, eY + 0.14*(tuft-0.5));
  vec3 earC = mix(mix(C_ORANGE, C_DARK, earTip), C_DARK, 0.92*smoothstep(0.03,-0.04, gQE.z)*smoothstep(0.02,0.2,eY));
  float bowlM = smoothstep(0.03,0.0,gBowl);
  // inner ear: pale rim -> peach -> warm shadow at the root, with fine pale hairs combed in from the rim
  float hair = vnoise2(vec2(gQE.x*150.0 + eY*14.0, eY*5.0))*0.65 + vnoise2(vec2(gQE.x*310.0 - eY*9.0, eY*9.0 + 4.0))*0.35;
  float bowlW = mix(0.205, 0.008, clamp((gQE.y - 0.10)/(EARL*0.8), 0.0, 1.0));   // bowl half width at this height
  float inn = 1.0 - clamp(abs(gQE.x)/bowlW + 0.20*(hair - 0.5), 0.0, 1.0);        // 0 at the rim, 1 on the ear's axis
  vec3 earIn = mix(C_CREAM*0.97, vec3(0.80,0.46,0.27), smoothstep(0.10,0.58,inn));
  earIn = mix(earIn, vec3(0.30,0.095,0.045), smoothstep(0.42,0.98,inn)*smoothstep(0.66,0.06,eY));
  earIn = mix(earIn, C_CREAM, smoothstep(0.48,0.70,hair)*(1.0 - smoothstep(0.18,0.72,inn))*0.9);
  earC = mix(earC, earIn, bowlM);
  alb = mix(alb, earC, earW);
  // the arm: the body's orange, deepening down the forearm into a fox's black stockings, feathered into the coat
  float sock = smoothstep(0.58, 0.82, gArmT + 0.10*(tuft - 0.5) + 0.05*(st - 0.5));
  vec3 armC = mix(mix(C_ORANGE, C_GOLD, 0.16 + 0.20*smoothstep(0.35,0.80,st)), C_DEEP, 0.34*smoothstep(0.30, 0.60, gArmT));
  armC = mix(armC, mix(C_SOCK, C_SOCK2, 0.25 + 0.5*smoothstep(0.30, 0.74, st)), sock);
  alb = mix(alb, armC, gWArm);
  float tip = smoothstep(0.70, 0.80, gTailT + 0.05*(tuft - 0.5) + 0.16*(vnoise3(gTailC*vec3(5.0,5.0,4.0)) - 0.5));
  vec3 tailC = mix(mix(C_ORANGE, C_GOLD, 0.20 + 0.25*smoothstep(0.3,0.8,st)), C_DEEP, 0.30*smoothstep(0.2,0.7,st));
  alb = mix(alb, mix(tailC, C_CREAM, tip), gWTail);
  // forehead streaks + brow marks
  vec3 bq = vec3(ax, r.yz) - vec3(0.26, 0.385, 0.0);
  float browMark = smoothstep(0.02,-0.03, length(bq.xy*vec2(1.0,2.4)) - 0.095);
  alb = mix(alb, mix(alb, C_CREAM, 0.55), browMark*(1.0 - earW));
  // strand contrast
  alb *= 0.94 + 0.12*st + (1.0 - 0.8*gSleek)*gWBody*0.30*(st - 0.5);
  float fur = 1.0;

  // eyeliner + dark lid margin
  vec3 ea = vec3(abs(q.x), q.yz) - EYEC;
  float lidr = length(ea);
  float liner = smoothstep(EYER + 0.026, EYER + 0.016, lidr)*step(0.0, ea.z + 0.05);
  alb = mix(alb, vec3(0.17,0.075,0.04), liner);

  // lip bands
  // the broad dark margins of the open lips narrow away as they seal, into one fine line on the upper lip's edge
  // a soft fleshy edge: the band is the lip's own section, its height from the bead
  float lipW = max(max(gLipO, 0.34*gPress), 0.001);
  float bandU = gLipHU + 0.006*lipW, bandL = (gLipHL + 0.006*lipW)*(1.0 + 3.5*gAmt*sq(min(gTL, 1.0)));     // measured in y: the lower edge climbs steeply to the corner
  float lipU = smoothstep(bandU*1.18, bandU*0.62, abs(q.y - gMYu))*smoothstep(0.035, 0.015, gSnoutP);
  float lipL = smoothstep(bandL*1.18, bandL*0.62, abs(gQJ.y - gMYl))*max(smoothstep(0.035, 0.015, gLowerRaw), gLipO*smoothstep(0.68, 0.88, gTL));    // by the corner the skin is the blended cheek, well out from the jaw's own
  float isUpper = step(gUp, gLo + 0.002);
  float lip = max(lipU*isUpper, lipL*(1.0 - isUpper))*smoothstep(0.20,0.32,gQ.z)*(1.0 - 0.75*gBehind*(1.0 - gSeal));
  lip = max(lip, gRoundE*smoothstep(0.034, 0.014, min(gSnoutP, gLowerRaw)));
  lip = max(lip, smoothstep(0.30, 0.75, gCornB)*max(gRnd, smoothstep(0.03, 0.12, gAmt - 0.10*gSpread)));    // a thin spread corner closes by itself
  lip = max(lip, gLipO*smoothstep(-0.003, -0.010, gLowerRaw)*step(gLo, gUp + 0.002)*smoothstep(1.30, 1.12, gTL)*smoothstep(0.50, 0.75, gTL)*smoothstep(0.0, 0.05, q.x));   // the far rim's top, seen edge on, is lip too
  float lineT = lineTaper(q.z, q.x);
  float lineR = LINE_H*lineT*(1.0 + 1.3*gPress);      // pressed, the two lips roll in: the line goes heavier and shorter
  lip = max(lip, smoothstep(0.0, 0.5, gSeal)*lineT*smoothstep(0.0013, -0.0013, abs(q.y - gMYu - lineR) - lineR));
  // dark skin outside, wet rose on the inner edge, running on into the cavity's paint
  float wetLip = gLipO*smoothstep(0.022, 0.012, min(gCavO, gCavL));
  vec3 lipCol = mix(vec3(0.030,0.018,0.018), mix(vec3(0.052,0.030,0.029), vec3(0.20,0.060,0.066), wetLip), gLipO);
  alb = mix(alb, lipCol, lip);
  alb = mix(alb, vec3(0.055,0.032,0.030), 0.55*gCornB*max(gRnd, smoothstep(0.03, 0.12, gAmt - 0.10*gSpread))*smoothstep(0.016, 0.003, abs(q.y - gMYu)));     // the crease where the two lips meet
  float bareSkin = lip;

  // philtrum: a dark seam from the nose down to the lip
  float phil = smoothstep(0.012,0.003,abs(q.x))*smoothstep(0.30,0.8,q.z)*step(gMYu, q.y)*step(q.y, -0.11);
  alb = mix(alb, vec3(0.03,0.02,0.02), phil);

  s.alb = alb;
  vec3 v = gRest - furPole();
  v = mix(v, gArmDir*length(v), gWLimb);
  s.T = furFlow(v, n, vec3(0.0,-0.25,0.0)*(1.0 - gWLimb));
  float nb = (st - 0.5);
  vec3 b = normalize(cross(s.T, n));
  s.n = normalize(n + b*nb*(0.10 + 0.22*gWBody)*(1.0 - 0.72*gSleek) + s.T*nb*0.03);
  s.fur = 1.0 - bareSkin;
  s.wrap = 0.42;
  s.sheen = 0.9;
  s.occ = (0.86 + 0.14*smoothstep(0.1,0.7,tuft))*(0.93 + 0.12*st);
  s.alb *= mix(vec3(0.90,0.82,0.76), vec3(1.0), smoothstep(0.05,0.6,tuft));
  s.trans = earW*mix(0.75, 1.0, bowlM);
  s.transCol = vec3(1.0,0.42,0.14);
  // light through the thin of the ear: the glow lives where the albedo gradient is peach, not in the root
  s.emit += earW*bowlM*vec3(0.17,0.052,0.016)*smoothstep(0.02,0.40,eY)*smoothstep(1.0,0.45,inn);
  s.n = normalize(mix(s.n, n, bareSkin*gLipO));      // an open lip has no strands to break its normal
  if(bareSkin > 0.5){ s.spec = mix(0.25, 0.62, gLipO); s.shin = mix(40.0, 52.0, gLipO); s.sheen = 0.2; s.wrap = mix(0.42, 0.5, gLipO); }

  if(inMouth){
    float dz = smoothstep(0.30, 0.88, gQ.z);
    s.alb = mix(vec3(0.070,0.013,0.017), vec3(0.34,0.075,0.085), dz*dz);
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.1; s.shin = 30.0; s.wrap = 0.3; s.occ = 1.0; s.trans = 0.0;
    s.n = n;
    s.emit = s.alb*0.20;                // a little fill so the cavity stays red, not black
    // the arched upper lip's underside faces forward in a round mouth: it is dark lip, not cavity
    float underU = gRnd*smoothstep(0.034, 0.012, q.y < gMYu + 0.03 ? abs(q.y - gMYu) : 1.0)*step(gUp, gLo + 0.002);
    s.alb = mix(s.alb, vec3(0.030,0.018,0.018), underU);
    s.emit *= 1.0 - underU;
  }
  if(isTongue){
    vec3 c = vec3(0.80,0.27,0.29)*(0.90 + 0.18*vnoise3(gQJ*40.0));
    c *= mix(0.42, 1.0, smoothstep(0.40, 0.72, gQJ.z));      // its root falls into the throat
    c *= mix(1.0, 0.52, smoothstep(0.45, 1.0, gTgW));        // its edges turn away
    c *= 1.0 - 0.16*smoothstep(0.16, 0.03, gTgW);            // the groove
    c *= mix(1.0, 0.62, gTgTip*smoothstep(0.008, -0.022, gTgU));   // a raised tip shows its darker underside
    s.alb = c*0.62;
    s.fur = 0.0; s.sheen = 0.2; s.spec = 0.3; s.shin = 30.0; s.wrap = 0.35; s.occ = 1.0; s.trans = 0.0; s.n = n;
    s.emit = c*0.26;
  }
  if(isTeeth){
    float gum = smoothstep(0.036, 0.046, gToothW);
    vec3 c = mix(vec3(0.93,0.89,0.79)*mix(1.0, 0.80, gToothL), vec3(0.50,0.15,0.16), gum);
    c *= mix(0.22, 1.0, smoothstep(0.65, 0.82, mix(gQ.z, gQJ.z, gToothL)));     // the row falls into the shade toward the back
    c *= 1.0 - 0.45*smoothstep(0.70, 1.0, gToothF)*(1.0 - gum);                  // a fine seam between two teeth
    c *= mix(0.70, 1.0, smoothstep(0.0, 0.030, mix(gMYu - q.y, gQJ.y - gMYl, gToothL)));      // the lip's shade on the row
    c *= mix(1.0, 0.62, smoothstep(0.03, 0.11, q.x));                            // the far side is out of the light
    s.alb = c*0.55;                                // lit mostly by emit: the engine's 4-step AO draws hard edges in here
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.40*(1.0 - gum); s.shin = 60.0; s.wrap = 0.25; s.occ = 1.0; s.trans = 0.0; s.n = n;
    s.emit = c*0.42;
  }
  s.emit += s.alb*gWBody*vec3(0.17,0.105,0.055);        // the coat scatters: a shadow on it stays warm
  if(gPadD < 0.003){
    float grain = 0.86 + 0.28*vnoise3(gPawL*90.0);
    s.alb = C_PAD*grain;
    s.fur = 0.0; s.sheen = 0.15; s.spec = 0.30; s.shin = 28.0; s.wrap = 0.35; s.occ = 1.0; s.trans = 0.0; s.n = n;
    s.emit = vec3(0.0);
  }
  if(gClawD < 0.003){
    s.alb = C_CLAW;
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.45; s.shin = 50.0; s.wrap = 0.3; s.occ = 1.0; s.trans = 0.0; s.n = n;
    s.emit = C_CLAW*0.10;
  }
  if(gWhisk < 0.0025){
    s.alb = WHISK_COL;
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.25; s.shin = 40.0; s.wrap = 0.5; s.occ = 1.0; s.trans = 0.0; s.n = n;
    s.emit = s.alb*0.42;                // mostly self-lit: the proxy shadow and AO must not break a thin line into dashes
  }
  float noseZone = length((q - gNoseP)/vec3(0.112,0.090,0.095));
  // while the mouth is closed nothing inside may read: teeth, tongue and cavity are all lip-dark
  // ... and while it is barely open, everything inside the muzzle between the two lip lines is the dark slit
  bool inSlit = q.y < gMYu + 0.003 && gQJ.y > gMYl - 0.010 && min(gSnoutP, gLowerRaw) < 0.014;
  if((inMouth || isTeeth || isTongue || inSlit) && gDark > 0.0){
    s.alb = mix(s.alb, vec3(0.030,0.018,0.018), gDark);
    s.emit *= 1.0 - gDark;
    s.spec *= 1.0 - gDark;
  }
  if(isNose || noseZone < 1.0){
    float fine = vnoise3(q*140.0);
    s.alb = vec3(0.014,0.012,0.012)*(0.85 + 0.3*fine);
    s.n = normalize(n + 0.018*(vec3(vnoise3(q*90.0), vnoise3(q*90.0+3.0), vnoise3(q*90.0+7.0)) - 0.5));
    s.fur = 0.0; s.sheen = 0.25; s.spec = 0.9; s.shin = 90.0; s.wrap = 0.1; s.occ = 1.0; s.trans = 0.0; s.mirror = 0.55;
  }
  if(isEye){
    // gaze space
    vec3 toCam = rootGaze(gHR*(uCamPos - vec3(0.0, gBounce, 0.0) - gHO)) + NECK - vec3(sign(q.x)*EYEC.x, EYEC.y, EYEC.z);
    vec3 eyeF = normalize(normalize(toCam) + vec3(gGaze.x*0.30, gGaze.y*0.24, 0.0));
    vec3 ex = normalize(cross(vec3(0,1,0), eyeF)), ey = cross(eyeF, ex);
    mat3 toE = transpose(mat3(ex, ey, eyeF));
    vec3 ee = vec3(q.x - sign(q.x)*EYEC.x, q.y - EYEC.y, q.z - EYEC.z)/EYER;
    vec3 e = toE*ee;
    vec3 v = toE*(gHR*rd);
    float irisM;
    vec3 c = eyeColour(e, v, vec3(0.80,0.44,0.10), vec3(0.38,0.15,0.035), vec3(0.84,0.78,0.68), 0.88, 0.52, irisM);
    s.alb = c;
    // shadow cast by the upper lid
    s.occ = mix(0.70, 1.0, smoothstep(0.62, 0.02, ee.y));
    s.fur = 0.0; s.sheen = 0.0; s.spec = 1.5; s.shin = 900.0; s.wrap = 0.0; s.trans = 0.0; s.mirror = 0.9;
    s.n = n;
    // painted catch-lights, locked to the light rig
    vec3 r2 = reflect(rd, n);
    float k1 = smoothstep(0.975, 0.985, dot(r2, normalize(uSunDir + vec3(-0.05,0.2,0.3))));
    float k2 = smoothstep(0.990, 0.996, dot(r2, normalize(vec3(-0.5,0.1,0.85))));
    s.emit = vec3(1.0,0.97,0.90)*(4.0*k1 + 1.4*k2);
  }
}

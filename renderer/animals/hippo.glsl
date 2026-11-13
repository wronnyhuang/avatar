// ---------------------------------------------------------------------------
//  animals/hippo.glsl — a round, jolly hippo rising from the river.  Hairless: gcLen = 0.
//  Head space: origin at the skull centre, +x screen right, +y up, +z toward the camera.
//  The muzzle is much wider than the skull and the jaw hinge sits far back, so the mouth is the giant
//  one of the cast.  See docs/ANIMAL_GUIDE.md for the API.
//
//  THE MOUTH RIG reads the eleven lip_* columns (docs/V2.md).  A huge mouth still makes small shapes: most speech
//  happens at the front of the muzzle and the big gape is saved for the open vowels.  Two things carry that:
//    The APERTURE lives on the mouth line, not on the jaw.  (ss, vv) are coordinates on the muzzle: ss along the line
//    from the front centre, around the wrap of the lobes, about x across the front; vv height above the resting line.
//    Two circular arcs through the mouth corners at ss = gLW bound the opening, as in engine/lips.glsl.  Past a corner
//    the rims stay on the resting line, sealed, however far the jaw has dropped.
//    The RIMS are cut in HEAD space, so the lips never ride the jaw.  The lower body hinges beneath them and stands up
//    to its rim as a straight wall.  teeth_gap moves the jaw and the teeth; the lips go where the lip columns say.
//
//  The body ends at the water plane (charEval); its reflection lives in the set's bake.
//
//  THE FORELIMBS read the hand_* columns.  The hippo sits up in the shallows with its belly at the waterline and its forefeet
//  resting on the belly: all zero is that rest.  Shoulder, elbow and wrist are a two-bone IK from a hand target, worked out once a
//  fragment in armsInit, a pure function of the columns.  See "the forelimbs" below; the water around them is charWater.
// ---------------------------------------------------------------------------
const vec3  NECK   = vec3(0.0,-0.22,-0.42);    // head rotates about this
const vec3  HINGE  = vec3(0.0,-0.08,-0.16);    // jaw hinge, far back: this is what makes the gape huge
const vec3  EYEC   = vec3(0.372,0.490,0.335);  // mirrored
const float EYER   = 0.150;
const float EYETILT= 0.03;
const float YAW0   = 0.40;                     // resting 3/4 turn
const float PITCH0 = 0.15;                     // resting nod, the muzzle must not hide the eyes
const vec3  EARB   = vec3(0.395,0.615,-0.24);  // ear base (mirrored)
const float MAXJAW = 0.84;                     // radians at teeth_gap = 1
const float ARC    = 0.86;                     // ss = ARC*(angle around the muzzle): about x across the front
const float ARCZ   = 0.16;                     // ... the axis of that angle
const float AIMX   = 0.42;                     // head-space slope of the line of sight into the mouth, x
const float AIMY   = 0.20;                     // ... and y

// short bristles on the muzzle lobes, x > 0 side
const int NWH = 8;
const vec3 WH_A[NWH] = vec3[NWH](vec3(0.586,0.082,1.009), vec3(0.708,0.019,0.981), vec3(0.499,-0.021,1.075), vec3(0.646,-0.084,1.042), vec3(0.538,-0.179,1.068), vec3(0.782,-0.076,0.947), vec3(0.842,0.031,0.825), vec3(0.724,-0.187,0.982));
const vec3 WH_B[NWH] = vec3[NWH](vec3(0.607,0.105,1.093), vec3(0.776,0.024,1.058), vec3(0.502,-0.030,1.191), vec3(0.691,-0.105,1.116), vec3(0.548,-0.230,1.157), vec3(0.869,-0.101,1.021), vec3(0.917,0.040,0.873), vec3(0.785,-0.237,1.049));

// ---- per-fragment pose (filled by charInit)
mat3  gHR;          // world -> head
mat3  gBodyR;       // world -> head at rest
vec3  gHO;
float gJawA, gRound, gSmile, gCheekP, gNostril;
float gPress, gPk, gSp, gPro, gTuck, gTip, gDent, gBack;
float gLW, gLAu, gLAl, gLRu, gLCu, gLRl, gLCl, gLKC, gLOv, gLBand;    // the aperture
float gHU, gHL, gTU, gTL, gKU, gKL, gPush, gPushL, gPushX, gPushY, gTuckIn;      // the lips
float gBunch, gSquint, gSmirk, gGather, gLX, gJawFwd, gJowlDrop, gDarkI, gCornD, gAim, gLScale, gLH, gDark, gShow, gShowL, gShowT, gTipY, gDentY, gTunnel, gRing, gRingH, gRingY, gRingB;
float gBlinkL, gBlinkR;
vec2  gGaze;
float gBrowL, gBrowR, gKnit;
mat3  gEarRL, gEarRR;
float gBreath, gCorner;

float gTgOut, gNoseW, gEyeW;         // tongue out 0..1, nostril twitch (head units), eyes wide 0..1
mat3  gBM; float gBounce;            // world -> the frame the body carries (yaw, lean, pitch about the hips), the bounce
const vec3 BPIV = vec3(0.0,-1.55,-0.90);   // the body turns and leans about its middle, under the water
vec3 bodyIn(vec3 p){ return gBM*(p - BPIV - vec3(0.0, gBounce, 0.0)) + BPIV; }
vec3 bodyOut(vec3 p){ return transpose(gBM)*(p - BPIV) + BPIV + vec3(0.0, gBounce, 0.0); }
mat3 rotX(float a){ float c=cos(a),s=sin(a); return mat3(1,0,0, 0,c,s, 0,-s,c); }
mat3 rotY(float a){ float c=cos(a),s=sin(a); return mat3(c,0,-s, 0,1,0, s,0,c); }
mat3 rotZ(float a){ float c=cos(a),s=sin(a); return mat3(c,s,0, -s,c,0, 0,0,1); }

// ---- the forelimbs ---------------------------------------------------------------------------------------------------------
// Index 0 is screen left, 1 screen right.  World space.  The body is turned by BODY_A; rest poses are given in its frame.
const float BODY_A  = 0.30;
const float ARM_U   = 0.58;                    // shoulder to elbow
const float ARM_F   = 0.52;                    // elbow to wrist
const float ARM_MID = 0.0;                     // the plane between the arms: a point only ever measures its own side.  Rays from the camera stay on theirs
const vec3  ARM_S   = vec3(1.02,-1.00,-0.46);  // shoulder, body frame, mirrored in x
const vec3  ARM_REST= vec3(0.54,-1.45,-0.02);  // wrist at rest: in the lap, the toes in the water.  Body frame, mirrored in x
vec3  gArmS[2], gArmE[2], gArmW[2], gArmB[2];  // shoulder, elbow, wrist, the axis both bones turn about
mat3  gArmF[2];                                // world -> foot: x across the toes, y out of the sole, z along the toes
vec3  gToeA[2], gToeB[2], gToeC[2], gToeD[2];  // inner toe root and tip, outer toe root and tip.  Foot space, x >= 0
vec3  gArmC[2];                                // the middle of the foot, world
float gChestUp, gLean;                         // what the arms do to the torso: the chest rises with them, the body counters a lifted arm
vec3  gArmT[2];                                // the tips of the middle toes, world
vec4  gToeK[2];                                // x: the index toe straight (point), y: the thumb toe out (thumb), z: the others in a fist

vec3 bodyToWorld(vec3 v){ v.xz = rot(BODY_A)*v.xz; return v; }
vec3 worldToBody(vec3 v){ v.xz = rot(-BODY_A)*v.xz; return v; }

// Any column value lands inside the body's range: a hand never leaves the reach of the leg, and never enters belly or head.
vec3 keepOut(vec3 p, vec3 c, vec3 r, float k){
  vec3 e = (p - c)/r;
  float l = max(length(e), 1e-3);
  return c + e*(smax(l, 1.0, k)/l)*r;
}

void armRig(int i, float sd, float cx, float cy, float cz, float open, float turn, vec3 dg){
  float x = clamp(cx, -1.2, 1.3), y = clamp(cy, -1.2, 1.3), z = clamp(cz, -1.2, 1.3);
  float up = max(y, 0.0), lift = smoothstep(0.0, 0.8, up);
  gChestUp += 0.5*lift; gLean -= sd*(0.5*lift + 0.5*max(x, 0.0));
  vec3 sb = ARM_S*vec3(sd, 1.0, 1.0), rb = ARM_REST*vec3(sd, 1.0, 1.0);
  // a raised arm carries the shoulder with it: the small rise and roll that make it an arm, not a crane
  float wFront = smoothstep(0.10, -0.40, x)*smoothstep(0.05, 0.35, z);
  float wEye = smoothstep(0.80, 0.97, y)*wFront;
  float wChin = smoothstep(0.55, 0.70, y)*(1.0 - smoothstep(0.82, 0.95, y))*wFront;
  // reaching for the face the shoulder comes up and forward (a hippo's leg is short)
  vec3 S = bodyToWorld(sb) + vec3(sd*0.050*max(x, 0.0), 0.070*up + 0.012*gBreath + 0.15*dg.z, 0.035*z) + vec3(-sd*0.10, 0.20, 0.30)*wEye;   // shoulder_* shrugs it
  // the hand target (the contract): an offset from the rest wrist in leg lengths, x outward on this side (negative across the
  // belly), y up, z toward the camera.  A raised hand that is not crossing comes up beside the head: screen right has the muzzle to clear
  const float AL = ARM_U + ARM_F;
  vec3 T = bodyToWorld(rb);
  T += AL*vec3(sd*x*(x > 0.0 ? 0.84 : 0.80), y*(y > 0.0 ? 1.12 : 0.24), z*(z > 0.0 ? 0.82 : 0.26));
  float outw = smoothstep(0.0, 0.6, up)*smoothstep(-0.35, 0.10, x);
  T.x += sd*(sd > 0.0 ? 0.80 : 0.66)*outw; T.z += 0.20*lift;
  // a hand brought up across in front of the face reaches for the eye on its side (cover eyes: as high as a hippo's leg goes,
  // the paw pressed to the face below it), or for the underside of the chin
  vec3 eyw = transpose(gHR)*(vec3(sd*(EYEC.x - 0.05), EYEC.y - 0.24, EYEC.z + 0.28) - NECK) + gHO;
  vec3 chw = transpose(gHR)*(vec3(sd*0.30, -0.80, 0.62) - NECK) + gHO;
  T = mix(T, eyw, wEye);
  T = mix(T, chw, wChin);
  T.y += 0.012*gBreath;
  // one foot laid across the chest stays on its own side of the midline (each side measures only its own leg); two feet
  // crossing meet there, palm to palm (clap)
  float oc = smoothstep(0.0, -0.30, sd > 0.0 ? uP[P_HAND_L_X] : uP[P_HAND_R_X]);
  T.x = sd*mix(smax(sd*T.x, 0.42, 0.10), sd*T.x, oc);
  // keep out of the chest and of the head.  The head moves, so its keep-out moves with it.
  vec3 tb = worldToBody(T);
  tb = keepOut(tb, vec3(0.0,-1.38,-0.98), vec3(1.33,1.00,1.03 + 0.14*smoothstep(0.0, -0.30, x)), 0.10);   // a foot laid across the chest lies on it, not in it
  T = bodyToWorld(tb);
  vec3 qh = gHR*(T - gHO) + NECK;
  qh = keepOut(qh, vec3(0.0,-0.14,0.36), mix(vec3(1.28,0.98,1.02), vec3(1.0,0.80,0.92), wEye), 0.12);   // at the eyes the keep-out is the face itself
  T = transpose(gHR)*(qh - NECK) + gHO;
  // two bones.  The reach is cut softly, or the elbow snaps straight at the end of it
  vec3 v = T - S;
  float l = max(length(v), 1e-3);
  vec3 d = v/l;
  float rc = smin(max(l, 0.46), (ARM_U + ARM_F)*0.965, 0.24);
  vec3 W = S + d*rc;
  float a = (rc*rc + ARM_U*ARM_U - ARM_F*ARM_F)/(2.0*rc);
  float h = sqrt(max(ARM_U*ARM_U - a*a, 0.0));
  vec3 pole = normalize(vec3(sd*0.62, -1.0, -0.42));
  vec3 b = normalize(pole - d*dot(pole, d) + vec3(0.0,0.0,-1e-3));
  vec3 E = S + d*a + b*h;
  vec3 fw = normalize(W - E);
  // The sole.  Up is the forearm's own "up", swung so that it stays defined with the forearm vertical: raised, palm up is toward
  // the hippo; hanging, palm up is toward the camera.  turn -1 sole down, 0 sole toward the midline, +1 sole up.
  vec3 ref = vec3(0.0, 1.0 - fw.y*fw.y, 0.0) + vec3(0.0, 0.0, -fw.y);
  vec3 u = normalize(ref - fw*dot(ref, fw) + vec3(0.0,1e-3,0.0));
  vec3 m = sd*cross(fw, u);
  float ph = clamp(turn, -1.1, 1.1)*1.42;
  // the contract: on a raised forearm turn +1 is the sole to the camera, -1 away from it
  float tn = clamp(turn, -1.0, 1.0);
  vec3 gc = vec3(0.0, 0.0, sign(tn)) - fw*fw.z*sign(tn);
  ph += abs(tn)*smoothstep(0.30, 0.75, fw.y)*smoothstep(0.05, 0.25, length(gc))*(mod(atan(dot(gc, u), dot(gc, m)) - ph + PI, 2.0*PI) - PI);
  float op = clamp(open, 0.0, 1.0);
  // a fist with the thumb toe out rolls that side of the foot up: thumbs up.  A pointing foot rolls its index side up, so it reads
  float thU = max(dg.x, 0.8*dg.y)*(1.0 - op);
  vec3 xm = cross(m, fw), xu = cross(u, fw);      // where the foot's +x side (the thumb toe) points for ph = 0, pi/2
  ph += thU*(mod(atan(xu.y, xm.y) - ph + PI, 2.0*PI) - PI);
  vec3 n0 = cos(ph)*m + sin(ph)*u;
  float straight = max(dg.x, dg.y)*(1.0 - op);       // a thumb up or a pointing toe holds the wrist straight
  // the wrist: a relaxed foot hangs and curls toward the sole, an open one lifts in line with the forearm
  vec3 ax = normalize(fw + (n0*mix(0.42, -0.10, op) + vec3(0.0,-0.14*(1.0 - op),0.0))*(1.0 - straight) + vec3(0.0,0.0,0.20)*max(fw.y, 0.0));     // a raised hand leans toward the listener
  // a pointing foot lines up with the leg's reach, held nearly level
  vec3 pd = W - S; pd.y *= 0.25;
  ax = normalize(mix(ax, normalize(pd), 0.70*dg.y*(1.0 - op)));
  vec3 n = normalize(n0 - ax*dot(n0, ax));
  vec3 s = cross(n, ax);
  gArmS[i] = S; gArmE[i] = E; gArmW[i] = W; gArmB[i] = normalize(cross(d, b));
  mat3 Fw = mat3(s, n, ax);
  gArmF[i] = transpose(Fw);
  gArmC[i] = W + ax*0.11;
  float c1 = mix(0.82, 0.10, op), c2 = mix(0.74, 0.16, op);          // curl toward the sole
  float s1 = mix(0.05, 0.22, op), s2 = mix(0.22, 0.66, op);          // fan
  gToeA[i] = vec3(0.074,-0.012,0.205);
  gToeB[i] = gToeA[i] + 0.112*vec3(sin(s1)*cos(c1), sin(c1), cos(s1)*cos(c1));
  gToeC[i] = vec3(0.166,-0.018,0.145);
  gToeD[i] = gToeC[i] + 0.096*vec3(sin(s2)*cos(c2), sin(c2), cos(s2)*cos(c2));
  gArmT[i] = W + Fw*vec3(0.0, gToeB[i].y, gToeB[i].z + 0.06);
  gToeK[i] = vec4(dg.y*(1.0 - 0.5*op), dg.x, max(dg.x, dg.y)*(1.0 - op), 0.0);
}
void armsInit(){
  gChestUp = 0.0; gLean = 0.0;
  armRig(0, -1.0, uP[P_HAND_L_X], uP[P_HAND_L_Y], uP[P_HAND_L_Z], uP[P_HAND_L_OPEN], uP[P_HAND_L_TURN],
         clamp(vec3(uP[P_THUMB_L], uP[P_POINT_L], uP[P_SHOULDER_L]), 0.0, 1.0));
  armRig(1,  1.0, uP[P_HAND_R_X], uP[P_HAND_R_Y], uP[P_HAND_R_Z], uP[P_HAND_R_OPEN], uP[P_HAND_R_TURN],
         clamp(vec3(uP[P_THUMB_R], uP[P_POINT_R], uP[P_SHOULDER_R]), 0.0, 1.0));
}

void charInit(){
  // the body carries the head and the legs: yaw about the vertical, lean and pitch about its middle, bounce = bob in the water
  gBM = rotX(-0.15*clamp(uP[P_BODY_PITCH], -1.0, 1.0))*rotZ(0.12*clamp(uP[P_BODY_LEAN], -1.0, 1.0))*rotY(-0.35*clamp(uP[P_BODY_YAW], -1.0, 1.0));
  gBounce = 0.10*clamp(uP[P_BODY_BOUNCE], -1.0, 1.0);
  // a commanded mouth, and a tongue that is out, open the mouth themselves
  float mo = clamp(uP[P_MOUTH_OPEN], 0.0, 1.0);
  gTgOut = clamp(uP[P_TONGUE_OUT], 0.0, 1.0);
  gNoseW = 0.036*clamp(uP[P_NOSE_WIGGLE], -1.0, 1.0);
  gEyeW = clamp(uP[P_EYE_WIDE], 0.0, 1.0);
  float press = clamp(uP[P_LIP_PRESS], 0.0, 1.0)*(1.0 - gTgOut), keep = 1.0 - press;
  float open = max(clamp(uP[P_LIP_OPEN], 0.0, 1.0), max(mo, 0.30*gTgOut))*keep;
  float up   = clamp(uP[P_LIP_UPPER], 0.0, 1.0)*keep;
  float lo   = clamp(uP[P_LIP_LOWER], 0.0, 1.0)*keep;
  float wid  = clamp(uP[P_LIP_WIDTH], -1.0, 1.0);
  float gap  = max(clamp(uP[P_TEETH_GAP], 0.0, 1.2), max(mo, 0.26*gTgOut))*(1.0 - 0.85*press);   // sealed lips hold the jaw: it leaves with them
  gPress = press; gPk = max(-wid, 0.0); gSp = max(wid, 0.0);
  gPro   = clamp(uP[P_LIP_PROTRUDE], 0.0, 1.0);
  gTuck  = clamp(uP[P_LIP_TUCK], 0.0, 1.0)*keep;
  gTip   = clamp(uP[P_TONGUE_TIP], 0.0, 1.0);
  gDent  = clamp(uP[P_TONGUE_DENTAL], 0.0, 1.0);
  gBack  = clamp(uP[P_TONGUE_BACK], 0.0, 1.0);
  gRound = clamp(0.55*gPk + 0.65*gPro, 0.0, 1.0);
  gSmile  = clamp(uP[P_SMILE], 0.0, 1.0);
  gCheekP = clamp(uP[P_CHEEK], 0.0, 1.0);
  gNostril= clamp(uP[P_NOSTRIL], 0.0, 1.0);
  gSquint = clamp(uP[P_SQUINT], 0.0, 1.0);
  gSmirk  = clamp(uP[P_SMIRK], -1.0, 1.0);
  // nothing moves alone: the jaw follows the lips a little even when the teeth barely part, and a pressed mouth clenches.
  // Rounded lips bring the jaw forward with them, spread lips draw it back; the jowls hang from it
  float jc = gap + (0.16*open + 0.08*lo)*(1.0 - gap);
  gJawA = MAXJAW*jc - 0.055*press;
  gJawFwd = 0.030*gPro - 0.012*gSp;
  gJowlDrop = 0.095*sin(max(gJawA, 0.0));
  // the head tips back as the jaw drops: it shows the palate and keeps the chin out of the river
  float lift = 0.15*jc*jc;                            // only a big gape: a heavy head does not bob on every syllable
  // both feet coming up to the eyes: the head ducks to meet them and turns a little to the front (a hippo's legs are short)
  // feet brought up in front of the chest: the big head lifts a little off them, so they are seen under the muzzle
  float wE = 0.0, wU = 0.0;
  for(int k=0;k<2;k++){
    vec3 h = k == 0 ? vec3(uP[P_HAND_L_X], uP[P_HAND_L_Y], uP[P_HAND_L_Z]) : vec3(uP[P_HAND_R_X], uP[P_HAND_R_Y], uP[P_HAND_R_Z]);
    wE += 0.5*smoothstep(0.80, 0.97, h.y)*smoothstep(0.10, -0.40, h.x)*smoothstep(0.05, 0.35, h.z);
    wU = max(wU, smoothstep(0.15, 0.45, h.y)*(1.0 - smoothstep(0.80, 0.95, h.y))*smoothstep(0.10, 0.40, h.z));
  }
  mat3 F = rotY(YAW0 + uP[P_HEAD_YAW] - 0.24*wE)*rotX(PITCH0 + uP[P_HEAD_PITCH] - lift + 0.36*wE - 0.20*wU)*rotZ(-uP[P_HEAD_ROLL]);
  gHR = transpose(F);
  gBodyR = transpose(rotY(YAW0)*rotX(PITCH0));
  gHO = NECK + vec3(uP[P_HEAD_X], uP[P_HEAD_Y], uP[P_HEAD_Z]) + vec3(0.0,-0.05,0.10)*wE + vec3(0.0,0.05,-0.03)*wU;
  gBlinkL = clamp(uP[P_BLINK_L], 0.0, 1.0); gBlinkR = clamp(uP[P_BLINK_R], 0.0, 1.0);
  gGaze = vec2(uP[P_GAZE_X], uP[P_GAZE_Y]);
  gBrowL = uP[P_BROW_L]; gBrowR = uP[P_BROW_R]; gKnit = clamp(uP[P_BROW_KNIT], 0.0, 1.0);
  // tiny ears, big flicks
  // pitch -1 back/droop .. 1 up/forward, turn -1 opening forward/in .. 1 out (about the ear's own long axis), flick a twitch.  About the base
  float aL = 0.60*clamp(uP[P_EAR_L], -1.0, 1.0) + 0.68*uP[P_EAR_FLICK_L], aR = 0.60*clamp(uP[P_EAR_R], -1.0, 1.0) + 0.68*uP[P_EAR_FLICK_R];
  // seen from the front a droop is the ear tipping out and down, up is the ear standing straight
  gEarRL = rotY(-0.80*clamp(uP[P_EAR_L_TURN], -1.0, 1.0))*rotX(-aL)*rotZ(0.55*uP[P_EAR_FLICK_L] - 0.50*clamp(uP[P_EAR_L], -1.0, 1.0));
  gEarRR = rotY(-0.80*clamp(uP[P_EAR_R_TURN], -1.0, 1.0))*rotX(-aR)*rotZ(0.55*uP[P_EAR_FLICK_R] - 0.50*clamp(uP[P_EAR_R], -1.0, 1.0));
  gBreath = uP[P_BREATH];
  armsInit();
  gCorner = 0.17 + 0.14*gRound - 0.10*gSp;      // the far end of the mouth line, z

  // ---- the aperture: half width along the line, rise of the upper arc, fall of the lower one
  float w = 0.40*(1.0 + 0.55*gSp - 0.72*gPk);
  w += 0.86*smoothstep(0.02, 0.95, open)*(1.0 - 0.55*gPk);
  float wRest = w;
  w *= 1.0 - 0.62*press;                             // pressed lips let go from the middle: the opening peels outward
  // A round mouth gathers on the middle of what the viewer sees, not on the far end of a 3/4 mouth line: the lip mass
  // slides toward the camera as it purses, the cleft going with it.  A smirk slides it too
  gGather = smoothstep(0.02, 0.85, gPk);
  gLX = -0.21*gGather + 0.13*gSmirk;
  // lip_open = 1 lands the lower lip on the jaw at teeth_gap = 1.  Small openings part both lips; a big one is the jaw's
  float upShare = mix(0.36, 0.15, smoothstep(0.08, 0.55, open));
  float au = 0.88*upShare*open + 0.17*up;
  float al = 0.88*(1.0 - upShare)*open + 0.15*lo;
  float rnd = smoothstep(0.28, 1.10, gPk)*smoothstep(0.0, 0.05, open + 0.5*(up + lo));   // a round O needs height as well as narrowness
  au = mix(au, 0.92*w, rnd);
  al = mix(al, 1.00*w, rnd);
  au += 0.095*gTuck;                                 // FF: the upper lip clears the teeth ...
  al *= 1.0 - 0.85*gTuck;                            // ... and the lower lip comes up to them
  gLW = w;
  gLAu = max(au, 0.0009); gLAl = max(al, 0.0009);
  gLRu = 0.5*(w*w + gLAu*gLAu)/gLAu; gLCu = gLRu - gLAu;
  gLRl = 0.5*(w*w + gLAl*gLAl)/gLAl; gLCl = gLRl - gLAl;
  float h = au + al;
  gLH = h;
  gLKC = min(0.030, 0.40*h + 0.0005);                // a soft corner, in scale with the opening
  gLKC = mix(gLKC, 0.060, smoothstep(0.10, 0.72, gPk));   // the ends of an O are folds, not corners
  gLOv = 0.010 + 0.008*press;                        // sealed, the rims overlap: no seam

  // ---- the lips: a soft band above and below the rims, fuller around the working part of the line
  gLBand = wRest + 0.32 - 0.10*press;
  float thin = 1.0 - 0.32*gSp;
  float hv = smoothstep(0.0, 0.10, h);               // heavy, fleshy lips at every opening
  gHU = (0.105 - 0.022*press + 0.014*hv)*thin;
  gHL = (0.120 - 0.024*press + 0.020*hv)*thin*(1.0 - 0.35*gTuck);
  gTU = 0.008 + 0.074*press + 0.040*gPk + 0.022*gPro + 0.040*hv*thin;
  gTL = 0.008 + 0.086*press + 0.040*gPk + 0.022*gPro + 0.052*hv*thin + 0.012*gTuck;
  gKU = 0.045 + 0.018*hv - 0.012*press; gKL = 0.045 + 0.024*hv - 0.012*press;
  gRing = smoothstep(0.12, 0.90, gPk);               // a pucker is a ring of lip, not two bands
  gRingH = 0.300; gRingY = 0.5*(gLAu - gLAl); gRingB = max(0.5*(gLAu + gLAl), 0.012);
  gAim = smoothstep(0.04, 0.78, gPk);
  // the working corner: a soft dent where the lips part, deeper when pulled or pressed, gone in a ring.  Nothing at rest
  gCornD = (0.014*smoothstep(0.02, 0.12, h) + 0.016*gSp*smoothstep(0.0, 0.05, h) + 0.016*press + 0.012*abs(gSmirk))*(1.0 - gRing);
  gBunch = 0.020*up + 0.012*gTuck;
  gPush  = 0.150*gPro + 0.016*press;                 // pressed, the air behind the lips bulges the middle of the mouth
  gPushL = gPush + 0.065*smoothstep(0.1, 0.7, gPro) + 0.006*press;  // the lower lip sits behind the upper at rest: forward, they meet
  gPushX = mix(0.62, 0.64, gPk); gPushY = mix(0.42, 0.52, gPk);
  gTuckIn = 0.050*gTuck;
  // the displacement steepens the field: march slower there rather than overshoot into the lips
  float slope = 1.54*gPushL/gPushY + 2.4*max(gTU, gTL)/mix(gHL, gRingH, gRing);
  gLScale = 1.0/sqrt(1.0 + slope*slope);             // displacement is across the normal: it adds in quadrature

  // ---- what shows inside: a dark slit first; colour, then teeth, come up with room
  gDark  = 1.0 - smoothstep(0.070, 0.280, h);
  gDarkI = max(gDark, 0.86*max(gTip, gBack));         // a raised tongue stands against a dark mouth
  gTunnel = smoothstep(0.30, 0.95, gPk)*smoothstep(0.1, 0.9, gPro);       // deep behind a pucker it all falls into the dark
  gShow  = smoothstep(0.090, 0.190, h + 0.09*gTuck + 0.10*up)*(1.0 - 0.8*gTunnel);
  gShowL = max(smoothstep(0.200, 0.360, h), smoothstep(0.08, 0.25, lo))*(1.0 - 0.8*gTunnel)*(1.0 - gDent);   // the lower lip covers them unless drawn down
  gShowT = smoothstep(0.260, 0.520, h);
  float dj = 1.10*sin(gJawA);                         // how far the front of the jaw has dropped under the skull
  gTipY  = 0.078 + dj;                                // from the tongue's bed to behind the upper teeth, jaw frame
  gDentY = 0.062 + dj - 0.6*al;                       // or to rest on the lower lip, between the teeth
}

float sq(float x){ return x*x; }
float lipsBell(float x){ x = clamp(1.0 - x*x, 0.0, 1.0); return x*x; }
// a lip's fullness: 1 at the rim, flat to 0.35 of its height, gone at the top of it
float lipBand(float r, float h){ float x = max(r - 0.35*h, 0.0)/(0.65*h); return lipsBell(min(x, 1.0)); }

// lip line height in head space (also evaluated in jaw space).  It only depends on z and the front of the
// barrel is a wide arc, so the line wraps the muzzle: level at the front, sagging under the lobes, then
// rising to a long lazy smile under the eye.
float mouthY(vec3 q){
  float s = clamp((q.z - 0.10)/0.92, 0.0, 1.0);
  float i = 1.0 - s;
  float y = -0.345 + 0.012*s*s;
  y += i*i*(0.25 + gSmile*0.30 + gSp*0.07);
  y += 0.050*gSmirk*q.x/sqrt(q.x*q.x + 0.16);                // one corner up, the other down
  y += 0.018*exp(-q.x*q.x*70.0)*smoothstep(0.7, 1.0, s);     // notch at the cleft
  return y;
}

// the waterline laps a little
float waterHeight(vec3 p){
  return WL + 0.0060*sin(p.x*3.1 + uTime*1.3) + 0.0045*sin(p.z*4.3 - uTime*1.1 + p.x*1.4);
}

// outputs of hippoEval (parts, for materials)
float gFur, gBare, gBowl, gWBody, gWJaw, gWEar, gWArm, gArmD;
vec3  gRest;
float gSnout, gLowerRaw, gUp, gLo, gCavU, gCavL, gEye, gTeeth, gTongue, gWhisk, gEarD, gBodyD, gHeadD, gNostr;
vec3 gRestLip;
float gTZ, gLipF, gML, gXA, gVA, gRJ, gPocket, gSS, gSSJ, gVV, gRimU, gRimL, gMY, gMYj, gTgX, gER;
vec3  gQ, gQJ, gQE; float gEarS;

float sdEar(vec3 e){
  // e: y along the ear, z out of the ear's face
  float d = sdEllipsoid(e - vec3(0.0,0.085,0.0), vec3(0.108,0.138,0.058));
  vec3 b = e - vec3(0.0,0.105,0.045);
  float bowl = sdEllipsoid(b, vec3(0.072,0.100,0.042));
  gBowl = bowl;
  return smax(d, -bowl, 0.022);
}


// the leg of this side
float gArmK;      // 1 at the shoulder: the leg grows out of the body there and only lies on it everywhere else
int gArmI;        // which leg armEval measured
float armEvalI(vec3 p, int i){
  vec3 S = gArmS[i], E = gArmE[i], W = gArmW[i];
  float d = smin(sdTaper(p, S, E, 0.272, 0.218), sdTaper(p, E, W, 0.214, 0.168), 0.11);
  vec3 f = gArmF[i]*(p - W);
  vec3 fa = vec3(abs(f.x), f.yz);
  float foot = sdEllipsoid(f - vec3(0.0,0.0,0.095), vec3(0.212,0.122,0.192));
  float toes = min(sdTaper(fa, gToeA[i], gToeB[i], 0.090, 0.078), sdTaper(fa, gToeC[i], gToeD[i], 0.082, 0.070));
  vec4 k = gToeK[i];
  for(int j_=0; j_<1 && k.z > 0.0 && length(f - vec3(0.0,0.0,0.15)) < 0.75; j_++){
    // a fist curls the toes under; on the +x side the inner toe can point straight on and the outer one stand out as a thumb
    vec3 cB = gToeA[i] + 0.104*normalize(vec3(0.05,0.85,0.52)), cD = gToeC[i] + 0.090*normalize(vec3(0.10,0.85,0.50));
    float fist = min(sdTaper(fa, gToeA[i], cB, 0.090, 0.078), sdTaper(fa, gToeC[i], cD, 0.082, 0.070));
    toes = mix(toes, fist, k.z);
    float pt = sdTaper(f, gToeA[i], gToeA[i] + vec3(0.02,0.0,0.25), 0.090, 0.068);
    float th = sdTaper(f, gToeC[i] - vec3(0.010,0.0,0.040), gToeC[i] + vec3(0.150,0.020,-0.010), 0.082, 0.066);
    float plus = min(mix(sdTaper(f, gToeA[i], cB, 0.090, 0.078), pt, k.x), mix(sdTaper(f, gToeC[i], cD, 0.082, 0.070), th, k.y));
    toes = f.x > 0.0 ? mix(toes, plus, max(k.x, k.y)) : toes;
  }
  foot = smin(foot, toes, 0.070);
  d = smin(d, foot, 0.075);
  gArmK = smoothstep(0.80, 0.30, length(p - S));
  return d;
}
// a point measures the leg on its side of the midline (measuring both near it cost ~7 ms on the K620: register pressure)
float armEval(vec3 p){ if(p.x < ARM_MID){ gArmI = 0; return armEvalI(p, 0); } gArmI = 1; return armEvalI(p, 1); }
// Where paint sticks to the leg: each bone's own frame, laid end to end down rest-space y, from the point of the body it grows from,
// so that the skin neither swims nor tears as the elbow folds.  Only charSurface asks.  The masks: sole, nails, folds
float gArmSole, gArmNail, gArmFold;
vec3 armRest(vec3 p){
  int i = gArmI;      // the leg armEval found nearest (charSurface ran the eval at this point just before)
  vec3 S = gArmS[i], E = gArmE[i], W = gArmW[i], B = gArmB[i];
  vec3 du = normalize(E - S), df = normalize(W - E);
  vec3 pu = p - S, pf = p - E;
  float au = dot(pu, du), af = dot(pf, df);
  vec3 ru = vec3(dot(pu, B), -au, dot(pu, cross(B, du)));
  vec3 rf = vec3(dot(pf, B), -ARM_U - af, dot(pf, cross(B, df)));
  vec3 r = mix(ru, rf, smoothstep(-0.10, 0.10, dot(pf, du + df)));
  vec3 f = gArmF[i]*(p - W);
  vec3 fa = vec3(abs(f.x), f.yz);
  vec3 rh = vec3(f.x, -ARM_U - ARM_F - f.z, f.y);
  r = mix(r, rh, smoothstep(-0.05, 0.05, f.z));
  gArmSole = smoothstep(0.012, 0.070, f.y)*smoothstep(-0.02, 0.05, f.z);
  vec3 e1 = gToeB[i] + 0.072*normalize(gToeB[i] - gToeA[i]) - vec3(0.0,0.052,0.0);
  vec3 e2 = gToeD[i] + 0.064*normalize(gToeD[i] - gToeC[i]) - vec3(0.0,0.048,0.0);
  gArmNail = smoothstep(0.060, 0.040, min(length(fa - e1), length(fa - e2)));    // blunt, on the back of the toe tip
  // the wrist and the inside of the elbow fold
  float inner = smoothstep(0.02, 0.13, dot(p - E, normalize(du - df)));
  gArmFold = 0.75*exp(-sq((f.z + 0.035)/0.018))*smoothstep(0.30, 0.12, length(f.xy)) + inner*exp(-sq(dot(pf, normalize(du + df))/0.024));
  return vec3(0.0,-0.46,1.30) + r;
}

void hippoEval(vec3 pw){
  vec3 q = gHR*(pw - gHO) + NECK;
  gQ = q;
  // the pucker gathers the lips in from the sides, the spread stretches them.  Only the lips: a heavy muzzle keeps its
  // outline from sound to sound
  float lipF = smoothstep(0.40, 0.95, q.z)*exp(-sq((q.y + 0.33)/0.26))*lipsBell((q.x - gLX)/0.95);
  gLipF = lipF;
  vec3 qm = q;
  qm.x += ((q.x - gLX)*(0.22*gRound - 0.08*gSp) - gLX)*lipF;
  qm.z -= 0.035*gRound*lipF;

  // ---- the mouth line and its aperture.  Head space: the lips do not ride the jaw
  float yM = mouthY(qm);
  float vv = q.y - yM;
  // A pucker's hole is a tunnel through thick lips.  Along head z it would close to nothing in this 3/4 view, so its
  // axis swings onto the camera as the ring forms (the same trap as LIP_TILT in engine/lips.glsl).
  float dpt = gAim*max(1.06 - q.z, 0.0);
  float xa = q.x - gLX - AIMX*dpt;
  float ss = ARC*atan(abs(xa), q.z - ARCZ + 1e-4);
  float va = vv + AIMY*dpt;
  float eU = length(vec2(ss, va + gLCu)) - gLRu;        // > 0 above the upper arc
  float rimU = smin(vv, eU, gLKC);                      // > 0 on the upper lip
  gSS = ss; gVV = vv; gMY = yM; gXA = xa; gVA = va;
  float mL = lipsBell(ss/gLBand);
  gML = mL;
  float pz = lipsBell(length(vec2(ss/gPushX, (vv + 0.01)/gPushY)));
  float er = (length(vec2(ss/gLW, (va - gRingY)/gRingB)) - 1.0)*min(gLW, gRingB);   // distance out from the O
  gER = er;
  float ringB = lipBand(er, gRingH);
  float cornD = gCornD*exp(-sq((ss - gLW)/0.070) - sq(vv/0.075));

  // ---- skull: small, high, set back
  vec3 qa = vec3(abs(q.x), q.yz);
  float skull = sdEllipsoid(q - vec3(0.0,0.20,-0.20), vec3(0.60,0.50,0.68));
  float bu = (q.x < 0.0) ? gBrowL : gBrowR;
  float bump = sdEllipsoid(qa - vec3(0.365,0.490 + 0.055*bu - 0.02*gKnit,0.235), vec3(0.235,0.215,0.235));
  skull = smin(skull, bump, 0.13);
  // jowls behind the mouth corner: the side walls of the mouth when it opens
  float lift = gCheekP*0.030 + gSmile*0.030 + 0.030*gSp + 0.022*gSquint + 0.030*abs(gSmirk)*step(0.0, gSmirk*q.x);
  vec3 qaj = qa; qaj.x *= 1.0 + 0.05*gRound - 0.03*gSp;
  float jowl = sdEllipsoid(qaj - vec3(0.60,-0.25 + lift - gJowlDrop,0.02), vec3(0.34,0.36 + 0.5*gJowlDrop,0.46));
  skull = smin(skull, jowl, 0.22);

  // ---- muzzle: two fat lobes + a centre, a bridge up to the forehead, nostril bumps on top
  vec3 pa = vec3(abs(qm.x), qm.yz);
  float lobe = sdEllipsoid(pa - vec3(0.44,-0.10 + lift*0.4,0.60), vec3(0.50,0.41,0.50));
  float snoutRaw = smin(lobe, sdEllipsoid(qm - vec3(0.0,-0.12,0.66), vec3(0.60,0.37,0.49)), 0.14);
  snoutRaw = smin(snoutRaw, sdTaper(qm, vec3(0.0,0.24,-0.02), vec3(0.0,0.02,0.70), 0.40, 0.33), 0.22);
  float fl = 1.0 + 0.12*gNostril;
  snoutRaw = smin(snoutRaw, sdEllipsoid(pa - vec3(0.300,0.205 + gNoseW,0.885 - 0.3*gNoseW), vec3(0.215,0.150,0.180)*fl), 0.085);
  // nostril slits, tilted on the bumps
  vec3 nq = pa - vec3(0.298,0.302 + gNoseW,0.985 - 0.3*gNoseW);
  nq.xy = rot(0.48)*nq.xy;
  float nostr = sdEllipsoid(nq, vec3(0.088 + 0.016*gNostril,0.030 + 0.018*gNostril,0.060));
  gNostr = nostr;
  gRimU = rimU;
  float bandU = lipBand(rimU, gHU);
  float snoutOut = smax(snoutRaw, -nostr, 0.018) - gPush*pz - gTU*mix(mL*bandU, ringB, gRing) + cornD
                 - gBunch*exp(-sq((vv - 0.17)/0.075) - sq(ss/0.62));
  gSnout = snoutOut;
  float snoutCut = smax(snoutOut, -rimU, gKU);
  float headRaw = smin(skull, snoutCut, 0.15);
  // sockets
  vec3 ea = qa - EYEC;
  headRaw = smax(headRaw, -(length(ea) - (EYER + 0.012)), 0.030);
  // the skull keeps a pocket where the jaw sleeps, with no height at rest.  It also opens the neck behind.
  float pocketW = mix(0.50, 0.46, smoothstep(0.0, 0.5, q.z));
  float pocketY = mix(yM, -0.80, smoothstep(0.0, 0.32, gJawA));
  float pocket = max(max(abs(q.x) - pocketW, q.y - yM), max(-0.24 - q.z, pocketY - q.y));
  float upper = smax(headRaw, -pocket, 0.07);
  // roof of the mouth
  float roof = mix(0.20, 0.075, smoothstep(0.60, 0.84, q.z));              // the gum comes down at the front
  float cavU = mouthCavity(snoutRaw, q.y, yM, 0.085, roof, 1.0);
  upper = smax(upper, -cavU, 0.03);
  // The aperture belongs to the lip.  It has cut straight through; the rigid gum and palate behind go back in as a solid,
  // so a lifted lip shows gum, and a peg stays a peg.
  upper = min(upper, max(snoutRaw + 0.085, yM + roof - q.y));
  gUp = upper; gCavU = min(cavU, pocket); gPocket = pocket;

  // ---- lower jaw, built in its rest frame.  Above its resting rim it is a straight wall, so the lip can stay up
  //      when the jaw has dropped; the cut that makes the rim is the head-space one above.
  vec3 qj = jawSpace(q, HINGE, -gJawA);
  qj.z -= gJawFwd;
  gQJ = qj;
  float lipFj = smoothstep(0.40, 0.95, qj.z)*exp(-sq((qj.y + 0.33)/0.26))*lipsBell((qj.x - gLX)/0.95);
  vec3 qjm = qj;
  qjm.x += ((qj.x - gLX)*(0.22*gRound - 0.08*gSp) - gLX)*lipFj;
  qjm.z -= 0.035*gRound*lipFj;
  float yMj = mouthY(qjm);
  gMYj = yMj;
  // Its place along the line comes from the jaw frame: at a big gape the front of the jaw has swung back under the skull.
  // Its height is still head space.  The rim is the lip's: deep inside, and behind the bowl, the jaw's own rim stands.
  float xj = qj.x - gLX - AIMX*gAim*max(1.06 - qj.z, 0.0);
  float ssj = ARC*atan(abs(xj), qj.z - ARCZ + 1e-4);
  gSSJ = ssj;
  float eL = length(vec2(ssj, va - gLCl)) - gLRl;        // > 0 below the lower arc
  float rimL = smin(gLOv - vv, eL, gLKC);                // > 0 on the lower lip
  float rj = length(vec2(xj, qj.z - ARCZ));
  gRJ = rj;
  vec3 ql = qjm;
  ql.y = smin(ql.y, yMj, 0.05);
  ql.z += gTuckIn*mL*lipBand(rimL, gHL);                // labiodental: the lip rolls in under the top teeth
  vec3 ja = vec3(abs(ql.x), ql.yz);
  float lowerRaw = sdEllipsoid(ql - vec3(0.0,-0.46,0.47), vec3(0.72,0.33,0.54));
  lowerRaw = smin(lowerRaw, sdEllipsoid(ja - vec3(0.34,-0.43,0.60), vec3(0.41,0.30,0.355)), 0.14);
  lowerRaw = smin(lowerRaw, sdTaper(ja, vec3(0.50,-0.30,-0.06), vec3(0.54,-0.38,0.50), 0.24, 0.24), 0.12);
  gRimL = rimL;
  float bandL = lipBand(rimL, gHL);
  float lowerOut = lowerRaw - gPushL*pz - gTL*mix(mL*bandL, ringB, gRing) + cornD;
  gLowerRaw = lowerOut;
  float lower = smax(lowerOut, -rimL, gKL);
  // the aperture is about the muzzle's rim: the back of the bowl keeps the jaw's own rim, or the throat wall would go
  lower = min(lower, smax(smax(lowerOut, qj.y - yMj - gLOv, gKL), rj - 0.50, 0.06));
  float depthL = mix(0.24, 0.085, smoothstep(0.62, 0.86, qj.z));           // the lower gum
  float cavL = max(lowerRaw + 0.075, yMj - qj.y - depthL);
  lower = smax(lower, -cavL, 0.03);
  lower = min(lower, max(lowerRaw + 0.075, qj.y - yMj + depthL));          // floor and gum, rigid on the jaw
  gLo = lower; gCavL = max(cavL, vv - 0.03);

  // ---- teeth: widely spaced round pegs, two stubby lower tusks.  Rigid on the skull and on the jaw, set so the two
  //      rows interleave.  They wait behind the lips until there is room, then slide out once: no flicker in a slit.
  vec3 fa = vec3(abs(qm.x), q.y - yM, q.z);
  float tipU = mix(0.060, -0.022, gShow);
  float teethU = sdTaper(fa, vec3(0.115,0.24,0.925), vec3(0.115,tipU,0.925), 0.040, 0.031);
  teethU = min(teethU, sdTaper(fa, vec3(0.325,0.24,0.885), vec3(0.325,tipU + 0.006,0.885), 0.037, 0.029));
  teethU = min(teethU, sdTaper(fa, vec3(0.560,0.24,0.725), vec3(0.560,tipU + 0.012,0.725), 0.034, 0.026));
  vec3 fb = vec3(abs(qjm.x), qj.y - yMj, qj.z);
  float tipL = mix(-0.060, 0.030, gShowL);
  float tusk = sdTaper(fb, vec3(0.435,-0.20,0.775), vec3(0.458,mix(-0.185, 0.125, gShowT*(1.0 - 0.8*gTunnel)),0.788), 0.064, 0.041);
  float teethL = min(tusk, sdTaper(fb, vec3(0.045,-0.16,0.895), vec3(0.045,tipL,0.905), 0.040, 0.030));
  teethL = min(teethL, sdTaper(fb, vec3(0.225,-0.16,0.865), vec3(0.225,tipL,0.875), 0.038, 0.029));
  gTeeth = min(teethU, teethL);

  // ---- the big tongue: lies low; its tip up behind the upper teeth, or between the teeth; or its back raised
  // tongue_out: it reaches on out past the lips, narrowing, its tip drooping over the lower lip
  float tLen = 0.47 + 0.22*gTgOut;
  vec3 tq = qj - vec3(0.0,-0.452 + 0.030*gTgOut,0.40 + 0.22*gTgOut);
  float tz = clamp((tq.z + 0.10)/0.52, 0.0, 1.0);
  tq.y += 0.10*gTgOut*sq(smoothstep(0.25, 0.69, tq.z));
  tq.x *= 1.0 + 0.9*gTgOut*smoothstep(0.0, 0.69, tq.z);
  tq.z *= 0.47/tLen;
  float nar = 1.0 + (3.20*gTip + 0.80*gDent)*tz*tz + 0.42*gBack;      // the working tip is narrow, and so is a dome
  tq.x -= 0.10*gTip*tz + 0.15*gBack;                     // seen from the 3/4 view: this puts the tip in the gap between the two middle pegs
  tq.x *= nar;
  gTZ = tz;
  tq.y -= gTip*gTipY*smoothstep(0.30, 1.0, tz) + gDent*gDentY*smoothstep(0.20, 0.95, tz);
  tq.z -= 0.120*gDent*tz;
  tq.y -= gBack*(0.070 + 0.050*lipsBell((tz - 0.30)/0.62));       // a low dome: anything taller is a wall behind the upper lip
  gTgX = tq.x/0.425;
  float tongue = sdEllipsoid(tq, vec3(0.425 - 0.08*tz*tz*gRound,0.092 + 0.012*gTip,0.47))/nar;
  tongue += 0.012*(1.0 - gDent)*(1.0 - gBack)*smoothstep(0.035, 0.0, abs(tq.x));
  gTongue = tongue*(1.0 - 0.34*gBack)/(1.0 + 0.9*gTgOut);

  // ---- eyeballs + lids
  gEye = length(ea) - EYER;
  float blink = (q.x < 0.0) ? gBlinkL : gBlinkR;
  vec3 el = ea;
  el.xy = rot(-EYETILT)*el.xy;
  float squint = gSmile*0.10 + gCheekP*0.10 + gSquint*0.28;
  float aU = mix(0.60 + 0.34*gEyeW, -0.38, blink) + 0.06*bu;
  float aL = mix(-0.72 + squint - 0.08*gEyeW, -0.42, blink);
  float lids = sdEyelids(el, EYER, 0.017, aU, aL);
  upper = smin(upper, lids, 0.030);
  gUp = upper;

  // ---- ears
  float s = (q.x < 0.0) ? -1.0 : 1.0;
  vec3 e = q - vec3(s*EARB.x, EARB.y, EARB.z);
  e.x *= s;
  e.xy = rot(0.46)*e.xy;
  e.yz = rot(-0.14)*e.yz;
  e = ((q.x < 0.0) ? gEarRL : gEarRR)*e;
  gQE = e; gEarS = s;
  float ear = sdEar(e);
  gEarD = ear;

  // ---- whiskers
  float wh = 1e3;
  for(int i=0;i<NWH;i++) wh = min(wh, sdTaper(pa, WH_A[i], mix(WH_A[i], WH_B[i], 0.72), 0.0034, 0.0010));
  gWhisk = wh;

  float head = smin(upper, lower, 0.012 + 0.10*smoothstep(gCorner + 0.05, gCorner - 0.14, qj.z) + 0.14*gRing*smoothstep(0.0, 0.10, er));
  // after the blend: a scaled k makes a crease.  Near the lips only: a scale reaching the chin would mark the neck under it
  head *= mix(1.0, gLScale, smoothstep(0.0, 0.10, max(pz*gPro, max(ringB*gRing, mL*max(bandU, bandL)*smoothstep(0.50, 0.12, abs(vv))))));
  head = smin(head, ear, 0.035);
  gHeadD = head;

  // ---- body, world space: it does not follow the head except a touch of breath
  vec3 b = pw;
  float br = 1.0 + 0.012*gBreath;
  b.y -= 0.012*gBreath + 0.022*gChestUp;
  b.x -= 0.024*gLean;
  b.xz = rot(-0.30)*b.xz;
  float torso = sdEllipsoid(b - vec3(-0.15,-1.42,-2.05), vec3(1.78,1.16,2.25)*br);
  float neck  = sdTaper(b, vec3(0.0,-0.24,-0.55), vec3(0.0,-1.02,-1.28), 0.68, 1.20);
  float chest = sdEllipsoid(b - vec3(0.0,-1.38,-0.98), vec3(1.30,0.86,0.94)*br);
  float bodyD = smin(smin(torso, neck, 0.34), chest, 0.34);
  // fat rolls: shells around the chin, so the creases read as arcs under the jaw
  float u = length((b - vec3(0.10,-0.20,0.10))*vec3(0.72,1.0,1.0)) + 0.03*sin(b.x*3.1 + 0.6);
  float roll = 1.0 - abs(sin(u*9.2 + 0.4));
  bodyD += 0.030*roll*roll*smoothstep(0.52, 0.68, u)*smoothstep(1.60, 1.25, u);
  bodyD = smax(bodyD, -pocket, 0.07);
  float arm = armEval(pw);
  gArmD = arm;
  bodyD += 0.030*smoothstep(0.09, -0.02, arm)*(1.0 - gArmK);   // soft: the belly gives a little under a leg lying on it.  Contact only: wider, a hovering arm draws its outline
  gBodyD = bodyD;

  gFur = smin(head, bodyD, 0.14);
  gFur = smin(gFur, arm, 0.022 + 0.21*gArmK);
  gBare = min(min(gEye, wh), min(gTeeth, gTongue));

  // rest coordinate for paint
  float wArm = smoothstep(-0.05, 0.05, min(head, bodyD) - arm);
  gWArm = wArm;
  float wBody = max(smoothstep(-0.10, 0.10, head - bodyD), wArm);
  gWBody = wBody;
  float wJaw = smoothstep(-0.02, 0.02, upper - lower);
  gWJaw = wJaw;
  gRestLip = mix(q + (qm - q)*0.8, qj + (qjm - qj)*0.8, wJaw);
  float wEar = smoothstep(-0.02, 0.02, min(upper,lower) - ear);
  gWEar = wEar;
  vec3 rq = gRestLip;
  rq = mix(rq, vec3(s*(e.x + EARB.x), e.y + EARB.y, e.z + EARB.z), wEar);
  gRest = mix(rq, gBodyR*(pw - NECK) + NECK, wBody);
}

void charEval(vec3 p){
  hippoEval(bodyIn(p));
  float wet = waterHeight(p) - p.y;
  gcBase = smax(gFur, wet, 0.012); gcBare = max(gBare, wet); gcLen = 0.0; gcRest = gRest;
}
float charTuft(){ return 0.0; }

//@shadow-begin
// cheap proxy for shadow rays, inside the real skin
float charShadow(vec3 pw){
  pw = bodyIn(pw);
  vec3 q = gHR*(pw - gHO) + NECK;
  vec3 qa = vec3(abs(q.x), q.yz);
  float d = sdEllipsoid(q - vec3(0.0,0.20,-0.20), vec3(0.57,0.47,0.65));
  d = min(d, sdEllipsoid(qa - vec3(0.365,0.49,0.235), vec3(0.20,0.18,0.20)));
  d = min(d, sdEllipsoid(qa - vec3(0.60,-0.24,0.02), vec3(0.30,0.32,0.42)));
  d = min(d, sdEllipsoid(qa - vec3(0.42,-0.10,0.58), vec3(0.46,0.37,0.46)));
  d = min(d, sdEllipsoid(jawSpace(q, HINGE, -gJawA) - vec3(0.0,-0.44,0.50), vec3(0.70,0.26,0.50)));   // it must leave with the jaw
  vec3 b = pw;
  b.xz = rot(-0.30)*b.xz;
  d = min(d, sdEllipsoid(b - vec3(-0.15,-1.42,-2.05), vec3(1.70,1.10,2.17)));
  d = min(d, sdTaper(b, vec3(0.0,-0.24,-0.55), vec3(0.0,-1.02,-1.28), 0.64, 1.14));
  d = min(d, sdEllipsoid(b - vec3(0.0,-1.38,-0.98), vec3(1.24,0.80,0.88)));
#ifndef PASS_BG
  // the legs move, so they stay out of the bake: charWater draws their reflection live.  The small lift keeps the shadow of a leg on
  // the belly a partial one: full dark on this skin reads as a bruise.
  int i = (pw.x < ARM_MID) ? 0 : 1;
  float dl = min(sdTaper(pw, gArmS[i], gArmE[i], 0.235, 0.19), sdTaper(pw, gArmE[i], gArmW[i], 0.185, 0.14));
  d = min(d, min(dl, length(pw - gArmC[i]) - 0.15) + 0.05);
#endif
  return d;
}
//@shadow-end

const vec3 C_SLATE = vec3(0.095,0.088,0.150);    // the wet grey-purple back
const vec3 C_MAUVE = vec3(0.185,0.150,0.235);
const vec3 C_PINK  = vec3(0.640,0.255,0.285);
const vec3 C_ROSE  = vec3(0.420,0.185,0.240);
const vec3 C_BELLY = vec3(0.440,0.265,0.300);

// skin height for shading: pebbled, with long soft creases around the neck and the corner of the mouth
float skinH(vec3 r){
  float h = 0.16*vnoise3(r*vec3(13.0,17.0,13.0)) + 0.20*vnoise3(r*37.0);
  vec3 ra = vec3(abs(r.x), r.yz);
  // creases fanning from the mouth corner over the jowl
  vec2 v = ra.zy - vec2(0.14,-0.20);
  float fan = sin(atan(v.y, v.x)*11.0 + 1.3*vnoise3(r*3.0));
  h += 0.95*(1.0 - abs(fan))*smoothstep(0.46, 0.14, length(v))*smoothstep(0.07, 0.16, length(v))*smoothstep(0.30, 0.55, ra.x);
  // forehead lines over the eyes
  float fl = sin(r.y*38.0 + 3.0*sin(r.x*3.0) + 2.0*vnoise3(r*3.0));
  h += 0.32*(1.0 - abs(fl))*smoothstep(0.42, 0.62, r.y)*smoothstep(0.55, 0.15, r.z);
  // fine lines running round the neck
  float nw = sin(r.y*64.0 + 6.0*vnoise3(r*2.3));
  h += 0.34*(1.0 - abs(nw))*smoothstep(-0.34, -0.55, r.y)*smoothstep(-1.40, -1.05, r.y)*(1.0 - gWArm);   // not on a leg: there they ring it as bright stripes
  return h;
}

// a bead of water: xyz dome slope, w coverage.  3D cells sliced by the skin give every size.
vec4 bead(vec3 r, float f, float seed, float amount){
  vec3 g = r*f*vec3(1.0,0.72,1.0);
  vec3 c = floor(g);
  vec3 h = hash33(c + seed);
  vec3 o = fract(g) - (0.34 + 0.32*hash33(c + seed + 7.0));
  float rad = 0.15 + 0.15*h.y;
  float d = length(o)/rad;
  float m = smoothstep(1.0, 0.86, d)*step(h.x, amount);
  return vec4(o/rad, m);
}

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  vec3 pb = bodyIn(pw);
  hippoEval(pb);
  if(gWArm > 0.0) gRest = mix(gRest, armRest(pb), gWArm);
  float wl = max(pw.y - waterHeight(pw), 0.0);
  vec3 q = gQ, r = gRest;
  vec3 n = s.n;
  bool isEye = gEye < 0.002 && gEye <= gFur + 0.002;
  bool isTeeth = gTeeth < 0.003 && gTeeth <= min(gFur, gTongue) + 0.003;
  bool isTongue = gTongue < 0.003 && gTongue <= gFur + 0.003;
  bool isWhisk = gWhisk < 0.003 && gWhisk <= gFur + 0.003;
  // The lower hollow is a closed bowl, so once the mouth is open whatever lies in it is seen through the mouth,
  // whoever owns the surface: the jaw swings down around the jowl and the neck, and their skin ends up in there.
  // sealed here?  The line closes point by point: past a mouth corner it is still the resting line
  float upW = step(gUp, gLo + 0.002);
  float gapS = gLH*max(1.0 - sq(mix(gSSJ, gSS, upW)/gLW), 0.0);
  float rimR = length(mix(gQJ, q, upW).xz - vec2(0.0, ARCZ));     // only the rim seals: the inside of an open mouth is open
  float seal = max(1.0 - smoothstep(0.0, 0.020, gLH), (1.0 - smoothstep(0.0, 0.016, gapS))*smoothstep(0.52, 0.66, rimR)*smoothstep(0.075, 0.035, abs(gVV))
                   *smoothstep(-0.10, -0.055, mix(gLowerRaw, gSnout, upW)));
  bool anyOpen = gLH > 0.003;
  bool inMouth = (gCavU < 0.014 && gUp < gLo + 0.01) || (gCavL < 0.014 && (gLo < gUp + 0.01 || anyOpen));
  float isUpper = step(gUp, gLo + 0.002);
  float rimD = mix(gRimL, gRimU, isUpper);

  // ---- skin colour.  Rest space, so it sticks.
  float ax = abs(r.x);
  vec3 ra = vec3(ax, r.yz);
  float mot = fbm3(r*1.5 + 2.0);
  float mot2 = vnoise3(r*5.3);
  // slate on the crown and the back; the face stays mauve, so low noise there is no bruise
  float crown = max(smoothstep(0.34, 0.66, r.y), gWBody);
  vec3 alb = mix(mix(mix(C_SLATE, C_MAUVE, 0.72), C_SLATE, crown), C_MAUVE, smoothstep(0.20, 0.80, mot));
  alb *= 0.94 + 0.12*mot2;
  float earW = gWEar;
  // pale along the underside of the face and the chest
  float low = smoothstep(-0.30, -0.62, r.y)*(1.0 - gWBody) + gWBody*smoothstep(-0.70, -1.15, r.y)*smoothstep(-0.2, 0.35, r.z);
  alb = mix(alb, C_BELLY*(0.96 + 0.08*mot2), 0.72*low);
  // pink flushes: eyes, ears, mouth, muzzle tip, cheeks
  float eyeF = 1.0 - smoothstep(0.17, 0.40, length(ra - EYEC));
  float lipF = exp(-sq(rimD/0.085))*smoothstep(0.02, 0.28, r.z);
  float tipF = smoothstep(0.82, 1.08, r.z)*smoothstep(0.02, -0.20, r.y);
  float cheekF = exp(-dot(ra - vec3(0.74,-0.06,0.36), ra - vec3(0.74,-0.06,0.36))/0.040);
  float nosF = 1.0 - smoothstep(0.05, 0.17, length(ra - vec3(0.298,0.300,0.985)));
  float ringF = gRing*exp(-sq(max(gER, 0.0)/0.20));                   // a gathered mouth is all lip: the blood goes with it
  float flush = max(max(eyeF*0.42, max(lipF*0.62, ringF*0.56)), max(tipF*0.25, max(cheekF*0.42, nosF*0.60)));
  flush *= (1.0 - gWBody)*(0.92 + 0.16*mot2);
  alb = mix(alb, mix(C_ROSE, C_PINK, flush), flush);
  // ears: flushed, a deep pink bowl
  float bowlM = smoothstep(0.015, -0.01, gBowl);
  vec3 earC = mix(mix(C_MAUVE, C_ROSE, 0.55), vec3(0.74,0.30,0.32), bowlM);
  alb = mix(alb, earC, earW);
  // follicles on the muzzle
  mat3 BR = mat3(0.80,0.36,-0.48, -0.60,0.48,-0.64, 0.0,0.80,0.60);    // any rotation: no lattice along the axes
  vec3 fg = BR*r*23.0; vec3 fc = floor(fg);
  vec3 fo = fract(fg) - 0.5 - 0.62*(hash33(fc + 3.0) - 0.5);
  float pore = smoothstep(0.14, 0.07, length(fo))*step(0.72, hash13(fc))*smoothstep(0.78, 0.96, r.z)*smoothstep(0.18, 0.05, abs(r.y + 0.08))*(1.0 - gWBody);
  alb *= 1.0 - 0.32*pore;
  // the leg: the same wet slate as the back, pink flushing the sole and the folds, pale blunt nails
  if(gWArm > 0.0){
    vec3 leg = mix(mix(C_SLATE, C_MAUVE, 0.80), C_MAUVE, smoothstep(0.15, 0.85, mot))*(0.97 + 0.06*mot2);
    leg = mix(leg, C_BELLY*(0.96 + 0.08*mot2), 0.58*smoothstep(0.25, -0.55, n.y));
    leg = mix(leg, mix(C_ROSE, C_PINK, 0.55*(0.8 + 0.4*mot2)), 0.86*gArmSole);
    leg = mix(leg, C_ROSE*0.42, 0.62*gArmFold);
    leg = mix(leg, vec3(0.31,0.245,0.235)*(0.92 + 0.14*mot2), gArmNail);
    alb = mix(alb, leg, gWArm);
  }

  // ---- wet bump: wrinkles and pebbling
  const float EB = 0.0035;
  float h0 = skinH(r);
  vec3 grad = vec3(skinH(r + vec3(EB,0,0)) - h0, skinH(r + vec3(0,EB,0)) - h0, skinH(r + vec3(0,0,EB)) - h0)/EB;
  grad -= n*dot(grad, n);
  float faceAmt = 0.0026 + 0.0022*gWBody;
  vec3 ns = normalize(n - grad*faceAmt*(1.0 - earW));
  // long run-off streaks decide where the skin is wettest
  float wet = mix(smoothstep(0.34, 0.66, fbm3(r*vec3(3.1,1.1,3.1) + 7.0)), 0.60, max(0.70*gWArm, 0.42*gWBody));     // on a leg the patches read as bruises, and they must not fight the legs on the belly
  float spec = mix(0.10, 1.05, wet);
  float shin = mix(130.0, 460.0, wet);

  // ---- water beads
  float bAmt = 0.20*smoothstep(0.50, 0.64, fbm3(r*2.1 + 11.0))*smoothstep(-0.55, 0.15, n.y);
  vec3 rw = r + 0.045*vec3(vnoise3(r*5.1), vnoise3(r*5.1 + 4.0), vnoise3(r*5.1 + 8.0));
  vec4 b1 = bead(BR*rw, 13.0, 3.0, bAmt);
  vec4 b2 = bead(BR*rw.zxy + 0.37, 25.0, 9.0, bAmt*1.3);
  vec4 bb = (b1.w > b2.w) ? b1 : b2;
  float beadM = bb.w;
  vec3 beadEmit = vec3(0.0);
  if(beadM > 0.0){
    vec3 bn = bb.xyz - n*dot(bb.xyz, n);
    vec3 bt = uSunDir - n*dot(uSunDir, n);
    float lb = length(bn);
    float hi = smoothstep(0.34, 0.10, length(bn - bt*0.55));                 // the sun in the drop
    float cr = smoothstep(0.45, 0.92, lb)*smoothstep(0.0, 0.8, -dot(bn, bt)/max(lb*length(bt), 1e-3));   // light gathered on the far side
    beadEmit = beadM*(vec3(1.0,0.90,0.74)*hi*1.5 + vec3(1.0,0.72,0.56)*cr*0.22);
    alb *= 1.0 + 0.22*beadM;                                                 // a lens: a touch brighter than the skin
    spec *= 1.0 - beadM;
  }

  // subsurface warmth: light that wraps past the terminator comes out red
  float ndl = dot(ns, uSunDir);
  float sss = smoothstep(0.55, -0.10, ndl);
  alb *= mix(vec3(1.0), vec3(1.32,0.86,0.80), sss);

  s.alb = alb;
  s.n = ns;
  s.T = vec3(0.0,1.0,0.0);
  s.fur = 0.0;
  s.wrap = 0.34;
  s.sheen = 0.42;
  s.spec = spec;
  s.shin = shin;
  s.mirror = (mix(0.40, 0.95, wet) + beadM*0.8)*(1.0 - 0.50*gWArm);     // this much mirror on a form as small as a leg reads as glass
  s.emit = beadEmit*smoothstep(-0.15, 0.25, ndl);
  s.occ = 1.0;
  // a leg lying on the belly: the soft dark seam between the two is what makes them two things
  float seam = mix(smoothstep(-0.03, 0.17, gArmD), smoothstep(-0.05, 0.13, gBodyD), gWArm);
  s.occ *= seam*0.60 + 0.40;
  // Under a leg the engine's occlusion falls to zero, and out of the sun that skin has no light at all: a black hole that follows the
  // elbow.  A multiplier cannot lift a zero, so flesh beside flesh keeps a fill of about half the ambient.
  s.emit += uCharKey*alb*vec3(0.30,0.20,0.22)*(1.0 - smoothstep(0.02, 0.30, mix(gArmD, gBodyD, gWArm)));
  s.trans = earW*0.75;
  s.transCol = vec3(1.0,0.34,0.30);

  // the fat rolls of the neck are drawn too: the geometry alone is too soft to hold them at phone size
  vec3 fb = pw; fb.xz = rot(-0.30)*fb.xz;
  float fu = length((fb - vec3(0.10,-0.20,0.10))*vec3(0.72,1.0,1.0)) + 0.03*sin(fb.x*3.1 + 0.6);
  float fc2 = 1.0 - abs(sin(fu*9.2 + 0.4));
  float fold = fc2*fc2*fc2*fc2*smoothstep(0.52, 0.68, fu)*smoothstep(1.60, 1.25, fu)*gWBody*(1.0 - gWArm);    // world space: not on a moving leg
  s.alb *= mix(vec3(1.0), vec3(0.56,0.46,0.56), fold);
  // a soft crease from the mouth corner up into the cheek: this is the smile
  vec3 smq = ra - vec3(0.80,-0.12,0.18);
  float smc = exp(-sq((smq.z + 0.30*smq.y)/0.022))*smoothstep(0.20, 0.04, abs(smq.y))*(1.0 - gWBody);
  s.alb *= 1.0 - 0.34*smc*(0.5 + gSmile*1.2);

  // a closed lid is one soft line, not a blister
  vec3 lq = vec3(abs(q.x), q.yz) - EYEC;
  float blinkP = (q.x < 0.0) ? gBlinkL : gBlinkR;
  float lidA = atan(lq.y, lq.z);
  float lidLine = exp(-sq((lidA + 0.40)/0.055))*smoothstep(0.0, 0.045, lq.z + 0.03)*smoothstep(EYER + 0.05, EYER + 0.02, length(lq));
  s.alb = mix(s.alb, vec3(0.10,0.045,0.055), 0.85*lidLine*smoothstep(0.65, 0.95, blinkP));

  // a thin warm liner around the eye
  vec3 ea = vec3(abs(q.x), q.yz) - EYEC;
  float liner = smoothstep(EYER + 0.030, EYER + 0.018, length(ea))*step(0.0, ea.z + 0.05);
  s.alb = mix(s.alb, vec3(0.13,0.055,0.06), 0.85*liner);

  // lips: a plumper, pinker, wetter band, with a clean dark crease where they meet
  float lipU = smoothstep(0.050, 0.014, abs(gRimU))*smoothstep(0.05, 0.015, gSnout);
  float lipL = smoothstep(0.050, 0.014, abs(gRimL))*smoothstep(0.05, 0.015, gLowerRaw);
  float lip = max(lipU*isUpper, lipL*(1.0 - isUpper))*smoothstep(0.10, 0.26, gQ.z)*(1.0 - gWBody);
  lip = max(lip, gRing*smoothstep(0.125, 0.045, gER)*(1.0 - gWBody));     // a pucker is all lip
  s.alb = mix(s.alb, vec3(0.56,0.25,0.27), 0.55*lip);
  float pl = gPress*gML;                                       // pressed: the line thickens where the lips bear on each other
  float crease = exp(-sq((gVV - 0.5*gLOv)/(0.0105*(1.0 + 2.6*pl))))*smoothstep(0.05, 0.22, r.z)*(1.0 - gWBody)*seal;
  crease *= (1.0 - 0.72*gRing)*(1.0 - 0.94*gRing*smoothstep(0.02, 0.24, gER));      // a pucker gathers the line into soft folds
  // the fat of the pressed lips stands between two soft valleys
  float valley = pl*(1.0 - gWBody)*mix(exp(-sq((gRimL - 0.92*gHL)/0.030)), exp(-sq((gRimU - 0.92*gHU)/0.026)), isUpper);
  s.alb *= 1.0 - 0.60*valley;
  // clamped lips: blood pressed out of the bearing edge, a tight dry surface instead of the wet rest shine
  float clamp0 = pl*(1.0 - gWBody)*smoothstep(0.17, 0.05, abs(rimD))*smoothstep(0.10, 0.26, gQ.z);
  s.alb = mix(s.alb, vec3(0.74,0.40,0.38), 0.55*clamp0);
  s.spec *= 1.0 - 0.80*clamp0; s.mirror *= 1.0 - 0.70*clamp0;
  // the drawstring: folds running in to the O from all round it
  float foldA = atan(gVA - gRingY, (gXA < 0.0 ? -gSS : gSS));
  float pfold = gRing*smoothstep(0.015, 0.075, gER)*smoothstep(0.30, 0.10, gER)*(1.0 - gWBody)*pow(0.5 + 0.5*cos(foldA*11.0 + 2.2*sin(foldA*3.0)), 3.0);
  s.alb *= 1.0 - 0.24*pfold;
  s.alb = mix(s.alb, vec3(0.055,0.022,0.030), 0.90*crease);
  // the cleft between the lobes
  float cleft = smoothstep(0.016, 0.003, abs(r.x - gLX*gLipF))*smoothstep(0.55, 0.85, r.z)*step(gMY, r.y)*smoothstep(0.08, -0.04, r.y)*(1.0 - gWBody)*isUpper;
  s.alb = mix(s.alb, vec3(0.10,0.045,0.055), 0.75*cleft);
  // nostrils: dark soft slits
  float nd = smoothstep(0.012, -0.012, gNostr);
  s.alb = mix(s.alb, vec3(0.040,0.016,0.022), nd);
  s.spec *= 1.0 - nd; s.mirror *= 1.0 - nd;

  float lipFace = smoothstep(0.035, 0.0, rimD)*smoothstep(-0.004, -0.030, (isUpper > 0.5) ? gSnout : gLowerRaw);
  // the opening itself: below the upper lip plane, above the lower one.  Anything in there is inside the mouth,
  // fading to skin across the jowl, which faces out at the mouth corner.
  vec3 nh = gHR*n;
  float wedge = step(gRimU, -0.004)*step(gMYj + 0.004, gQJ.y)*step(gWBody, 0.5);
  wedge *= (isUpper > 0.5) ? 1.0 : smoothstep(-0.004, -0.012, gLowerRaw);    // the jaw's outer wall is skin
  wedge *= 1.0 - smoothstep(0.58, 0.82, abs(q.x))*smoothstep(-0.10, 0.45, nh.x*sign(q.x));
  // skin never lies deep inside the muzzle volume: anything there is interior
  float deep = ((isUpper > 0.5) ? smoothstep(-0.012, -0.040, gSnout) : smoothstep(-0.012, -0.040, gLowerRaw))*(1.0 - nd);
  // The rear of the lower jaw stands behind the opening once the mouth gapes, outside both cavities: it is the throat.
  float throat = (anyOpen ? 1.0 : 0.0)*step(gRimU, -0.004)*step(-0.70, gVV)*step(q.z, 0.2)
               *smoothstep(0.62, 0.50, abs(q.x))*max(step(gWBody, 0.5), step(gPocket, 0.05));   // the neck, seen down the throat
  float bowlBack = (anyOpen ? 1.0 : 0.0)*(1.0 - isUpper)*step(gRJ, 0.55)*step(gMYj - 0.24, gQJ.y);   // the rear of the bowl, seen behind the top teeth
  float mouthM = max(max(max(max(max(inMouth ? 1.0 : 0.0, lipFace), wedge), deep), throat), bowlBack)*(1.0 - gWArm);
  if(mouthM > 0.0){
    float dz = smoothstep(-0.14, 0.80, gQ.z);
    vec3 c = mix(vec3(0.30,0.060,0.105), vec3(0.86,0.31,0.34), dz);
    float dp = (isUpper > 0.5) ? gSnout : gLowerRaw;
    c = mix(c, vec3(0.040,0.012,0.020), max(gDarkI, gTunnel*smoothstep(-0.015, -0.075, dp)));     // a small opening is a dark slit first
    // ridges across the palate
    float ridge = (gUp < gLo) ? 0.88 + 0.12*sin(gQ.z*46.0 + 3.0*abs(gQ.x)) : 1.0;
    s.alb = mix(s.alb, c*ridge, mouthM);
    s.sheen *= 1.0 - mouthM; s.trans *= 1.0 - mouthM;
    s.spec = mix(s.spec, 0.8, mouthM); s.shin = mix(s.shin, 60.0, mouthM);
    s.mirror = mix(s.mirror, 0.3, mouthM);
    s.wrap = mix(s.wrap, 0.3, mouthM);
    s.n = normalize(mix(s.n, n, mouthM));
    s.emit = mix(s.emit, c*ridge*0.42, mouthM);
    s.spec *= 1.0 - gDarkI*mouthM; s.mirror *= 1.0 - gDarkI*mouthM;
  }
  // the inner lip: heavy, wet flesh at the edge of every opening
  float wetL = smoothstep(1.04, 0.42, mix(gSSJ, gSS, upW)/gLW)*smoothstep(0.085, 0.012, abs(rimD))*smoothstep(0.10, 0.26, gQ.z)*(1.0 - gWBody)*smoothstep(0.005, 0.05, gLH)
             *smoothstep(-0.095, -0.025, (isUpper > 0.5) ? gSnout : gLowerRaw)*(1.0 - gTunnel);   // the front edge only, never down a tunnel
  s.alb = mix(s.alb, vec3(0.66,0.205,0.235), 0.72*wetL);
  s.emit = mix(s.emit, vec3(0.66,0.205,0.235)*0.20, wetL*mouthM);
  s.spec = mix(s.spec, 1.9, wetL); s.shin = mix(s.shin, 240.0, wetL); s.mirror = mix(s.mirror, 0.75, wetL);
  if(isTongue){
    vec3 j = gQJ;
    float bumps = vnoise3(j*vec3(46.0,20.0,46.0));
    s.alb = mix(vec3(0.46,0.13,0.18), vec3(0.93,0.40,0.42), smoothstep(-0.05, 0.60, j.z))*(0.84 + 0.30*bumps);
    s.alb = mix(s.alb, vec3(0.040,0.012,0.020), gDark*(1.0 - max(gTip, max(gDent, gBack))));
    s.alb *= mix(1.0, 0.74, gDark);                  // in a slit, a tongue, not a light
    s.alb = mix(s.alb, vec3(0.070,0.022,0.030), 0.90*gTip*(1.0 - smoothstep(0.42, 0.74, gTZ)));   // a raised tip stands alone in the mouth
    s.alb = mix(s.alb, vec3(1.0,0.50,0.50), 0.55*max(gTip, gDent)*smoothstep(0.70, 0.96, gTZ));   // and it catches the light: this is what a phone sees
    s.alb *= mix(1.0, 0.55 + 0.45*smoothstep(0.95, 0.35, abs(gTgX)), gBack);                      // a dome rounds off into the dark
    s.alb = mix(s.alb, vec3(0.040,0.012,0.020), 0.94*gTunnel);
    s.sheen = 0.2; s.spec = mix(2.2, 0.4, gDark); s.shin = 40.0; s.wrap = 0.35; s.trans = 0.0; s.mirror = mix(0.4, 0.05, gDark); s.n = normalize(n + 0.05*(bumps - 0.5));
    s.emit = s.alb*(0.25 + 0.14*max(gTip, gDent)*smoothstep(0.70, 0.96, gTZ));
    s.spec *= 1.0 - gTunnel; s.mirror *= 1.0 - gTunnel;
  }
  if(isTeeth){
    float tdk = max(smoothstep(0.45, 1.0, gTunnel)*0.92, 1.0 - min(gShow + gShowL, 1.0));    // fallen into the dark, warm: never a grey tooth
    s.alb = mix(mix(vec3(0.92,0.87,0.74), vec3(0.46,0.21,0.19), smoothstep(0.0, 0.6, tdk)), vec3(0.070,0.022,0.028), smoothstep(0.5, 1.0, tdk));
    s.alb *= mix(0.84, 1.0, smoothstep(0.0, 0.11, -gVV));              // the upper lip's soft shade
    s.sheen = 0.0; s.spec = 2.0*(1.0 - tdk); s.shin = 70.0; s.wrap = 0.3; s.trans = 0.0; s.mirror = 0.3*(1.0 - tdk); s.n = n;
    s.emit = s.alb*vec3(0.60,0.57,0.49)*mix(0.60, 1.0, smoothstep(0.05, 0.75, dot(n, -rd)));   // a round peg, not a cut-out
  }
  if(isWhisk){
    s.alb = vec3(0.070,0.050,0.065);
    s.sheen = 0.1; s.spec = 0.3; s.shin = 40.0; s.wrap = 0.5; s.trans = 0.0; s.mirror = 0.0; s.n = n;
    s.emit = vec3(0.0);
  }
  // while the mouth is closed nothing inside may read
  // The lip faces near the skin can peek out of the groove, so they close to lip colour, not black.
  if((mouthM > 0.0 || isTeeth || isTongue) && seal > 0.0){
    float shD = (isUpper > 0.5) ? gSnout : gLowerRaw;
    vec3 shut = mix(vec3(0.055,0.022,0.030), vec3(0.50,0.225,0.245), max(smoothstep(-0.085, -0.025, shD), 0.70*gRing));
    s.alb = mix(s.alb, shut, seal);
    s.emit *= 1.0 - seal;
    s.spec *= 1.0 - seal;
    s.mirror *= 1.0 - seal;
  }
  s.alb = mix(s.alb, vec3(0.055,0.022,0.030), 0.90*crease);    // the mouth line itself, always clean
  if(isEye){
    vec3 toCam = rootGaze(gHR*(bodyIn(uCamPos) - gHO)) + NECK - vec3(sign(q.x)*EYEC.x, EYEC.y, EYEC.z);
    vec3 eyeF3 = normalize(normalize(toCam) + vec3(gGaze.x*0.30, gGaze.y*0.24, 0.0));
    vec3 ex = normalize(cross(vec3(0,1,0), eyeF3)), ey = cross(eyeF3, ex);
    mat3 toE = transpose(mat3(ex, ey, eyeF3));
    vec3 ee = vec3(q.x - sign(q.x)*EYEC.x, q.y - EYEC.y, q.z - EYEC.z)/EYER;
    vec3 e = toE*ee;
    vec3 v = toE*(gHR*(gBM*rd));
    float irisM;
    vec3 c = eyeColour(e, v, vec3(0.50,0.235,0.075), vec3(0.20,0.085,0.035), vec3(0.84,0.78,0.72), 0.90, 0.58, irisM);
    s.alb = c;
    s.occ = mix(0.70, 1.0, smoothstep(0.75, 0.15, ee.y));
    s.sheen = 0.0; s.spec = 1.5; s.shin = 900.0; s.wrap = 0.0; s.trans = 0.0; s.mirror = 0.9;
    s.n = n;
    vec3 r2 = reflect(rd, n);
    float k1 = smoothstep(0.970, 0.982, dot(r2, normalize(uSunDir + vec3(-0.05,0.2,0.3))));
    float k2 = smoothstep(0.988, 0.995, dot(r2, normalize(vec3(-0.5,0.1,0.85))));
    s.emit = vec3(1.0,0.97,0.90)*(4.0*k1 + 1.4*k2);
  }

  // Inside the mouth the sun arrives only through the gaps of the shadow proxy, as hard patches and bands: bounce light only
  {
    float sunOff = max(max(mouthM*(1.0 - wetL), isTeeth ? 1.0 : 0.0), isTongue ? 1.0 : 0.0)*(anyOpen ? 1.0 : 0.0);
    s.n = normalize(s.n - uSunDir*sunOff*(max(dot(s.n, uSunDir), 0.0) + 0.34));
  }
  {
    // the waterline: a darker wet band, a thin bright meniscus, light rippling up off the river
    s.alb *= 1.0 - 0.42*exp(-wl/0.085);
    vec2 cg = (pw.xz + n.xz*wl*0.9)*vec2(4.1,5.3);
    float T = uTime;
    float ca = vnoise2(cg + vec2(T*0.61, -T*0.43))*vnoise2(cg*1.73 + vec2(-T*0.37, T*0.71));
    ca = smoothstep(0.12, 0.42, ca);
    float faceDown = clamp(0.55 - 0.85*n.y, 0.0, 1.0)*(1.0 - mouthM);     // the river's light does not reach into the mouth
    s.emit += uCharKey*s.alb*vec3(1.0,0.78,0.48)*ca*faceDown*exp(-wl*0.85)*1.6;
    s.emit += uCharKey*vec3(1.0,0.74,0.56)*0.14*exp(-sq(wl/0.016));
  }
}

// ---- the water around the legs, for the set's live pass (PASS_LIVE).  Everything is the current pose and the clock, so a hand that
// leaves the water takes its rings with it, and the reflection follows the leg.  ------------------------------------------------
#ifdef PASS_LIVE
float raySeg(vec3 ro, vec3 rd, vec3 a, vec3 b, float r1, float r2, float blur){
  vec3 ba = b - a, oa = ro - a;
  float baba = dot(ba, ba), bard = dot(ba, rd), baoa = dot(ba, oa), rdoa = dot(rd, oa);
  float k = clamp((baoa - rdoa*bard)/max(baba - bard*bard, 1e-5), 0.0, 1.0);
  float tt = max(k*bard - rdoa, 0.0);
  vec3 c = oa + tt*rd - k*ba;
  float y = a.y + k*ba.y;
  float wl = WL - gBounce;          // the river, in the frame the body carries
  float rr = mix(r1, r2, k), bl = blur + 0.10*max(y - wl, 0.0);
  return smoothstep(rr + bl, rr - bl, length(c))*smoothstep(wl - 0.02, wl + 0.03, y);
}
// .x coverage of a leg's reflection along the mirrored ray
float charWaterRefl(vec3 X, vec3 m){
  X = bodyIn(X); m = gBM*m;          // the legs live in the frame the body carries
  int i = (X.x < ARM_MID) ? 0 : 1;
  float c = raySeg(X, m, gArmS[i], gArmE[i], 0.262, 0.214, 0.035);
  c = max(c, raySeg(X, m, gArmE[i], gArmW[i], 0.210, 0.165, 0.035));
  c = max(c, raySeg(X, m, gArmW[i], gArmT[i], 0.190, 0.120, 0.035));
  return c;
}
// .x distance over the water to the leg, .y how much the leg is in the water.  The true leg: the rings fit the foot that made them
vec2 charWaterDist(vec3 X){
  X = bodyIn(X);
  int i = (X.x < ARM_MID) ? 0 : 1;
  float low = min(min(bodyOut(gArmE[i]).y - 0.214, bodyOut(gArmW[i]).y - 0.168), bodyOut(gArmT[i]).y - 0.085);
  return vec2(armEval(X), smoothstep(0.05, -0.05, low - WL));
}
#endif

// ---------------------------------------------------------------------------
//  animals/rabbit.glsl — the rabbit.  Head space: origin at the skull centre, +x screen right,
//  +y up, +z toward the camera (the nose).  Units: skull radius ~0.6.
//  See docs/ANIMAL_GUIDE.md for the API, animals/fox.glsl for the worked example.
//
//  What is different from the fox:
//   - the lip line (mouthY) has a CUSP at the centre: the cleft.  Sealed, it is the inverted Y under
//     the nose; lifting the upper lip raises the cusp and parts the two lip pads over the buck teeth.
//   - the jaw is short and small and sits behind the pads (an overbite).
//   - the long ears are bent on a real circular arc (bendEar), driven by ear_l/r.
//
//  The mouth reads the lip_* columns directly (docs/V2.md).  One curve carries it, the APERTURE: two arcs
//  through the two corners, on the resting lip surface.  The upper arc is the edge of the pads and takes the
//  cusp; the lower arc is the bare lower lip (gRing), a soft roll that hugs the chin.  e = signed distance to
//  the aperture: < 0 inside, 0 on the inner lip edge, > 0 out onto the fur.  The carve (gHoleD) runs through
//  the whole face, so the corners are one point on one curve and cannot come apart.  The hinged jaw only
//  carries the chin, the lower teeth and the tongue; the lips move on their own.
//
//  Body and arms (round three, docs/V3.md).  The torso has its own frame (gTR, about the hips): it takes a small
//  share of the head and the hands, all pure functions of the columns.  Its rest frame is the head's rest frame, so
//  the neck blends without a smear.  The arms live in torso space: armSolve maps hand_* to a wrist target, keeps it
//  out of the torso and the head, and solves shoulder, elbow, wrist by two-bone IK.  gRest on a limb is the point
//  carried back to that bone's rest pose, so at all-zero columns it equals the torso's own coordinate and the coat
//  runs over the shoulder unbroken.
// ---------------------------------------------------------------------------
const vec3  NECK   = vec3(0.0,-0.60,-0.16);    // head rotates about this
const vec3  HINGE  = vec3(0.0,-0.14, 0.18);    // jaw hinge
const vec3  NOSEP  = vec3(0.0,-0.034,0.868);   // nose centre
const vec3  POLE   = vec3(0.0,-0.09, 0.91);    // fur flows away from here
const vec3  EYEC   = vec3(0.302,0.125,0.345);  // mirrored
const float EYER   = 0.190;
const float EYETILT= 0.15;                     // outer corners up
const float YAW0   = 0.30;                     // resting 3/4 turn
const vec3  EARB   = vec3(0.225,0.46,-0.16);   // ear base (mirrored)
const float EARL   = 0.90;
const float EARY0  = 0.26;                     // the ear is straight below this, an arc above it
const float MAXJAW = 0.40;                     // radians at teeth_gap=1
const float GAPMAX = 0.235;                    // gap between the lip edges at lip_open=1
const float LIPY   = -0.272;                   // height of the pad bottoms
const vec3  JAW_A = vec3(0.0,-0.205,0.18), JAW_B = vec3(0.0,-0.272,0.640);
const float JAW_RA = 0.235, JAW_RB = 0.100;
const float BARE_EPS = 0.0055;                 // just above the fine march's minimum step
const float F_K = 19.0, F_L = 9.0;             // tuft density across / along the flow
// ---- torso and arms, in torso space
const vec3  BPIV   = vec3(0.0,-2.30,-0.22);    // the torso leans about the hips
const vec3  CHEST  = vec3(0.0,-1.00,-0.22), CHESTR = vec3(0.62,0.52,0.47);
const vec3  BELLY  = vec3(0.0,-1.62,-0.15), BELLYR = vec3(0.86,0.86,0.64);
const vec3  RUFF   = vec3(0.0,-0.68,-0.10), RUFFR  = vec3(0.60,0.21,0.48);
const vec3  HEADKO = vec3(0.0,-0.12,0.16),  HEADKR = vec3(0.86,0.52,0.86);    // head space: the wrist stays out of this
const float ARM_U  = 0.44, ARM_F = 0.44;       // bone lengths (a little longer, so raised paws clear the cheeks)
const float ARM_R0 = 0.100, ARM_R1 = 0.068, ARM_R2 = 0.046;   // shoulder, elbow, wrist, under the coat

// ---- per-fragment pose (filled by charInit)
mat3  gHR;          // world -> head
mat3  gBodyR;       // world -> head at rest (body fur + paint frame)
vec3  gHO;          // head-space origin offset
float gJawA, gRound, gSpread, gPress, gLab, gSmile, gCheekP, gNostril;
float gBlinkL, gBlinkR;
vec2  gGaze;
float gBrowL, gBrowR, gKnit;
mat3  gEarRL, gEarRR;
float gEarKL, gEarKR;              // ear curvature, radians per unit length
float gBreath, gCorner, gSeal;     // gSeal 1 = mouth closed: the interior cannot show
float gPart, gCleft, gCleftW;      // pads parted 0..1, cusp height, cusp width
float gPk, gSqz, gSqzJ, gPush, gPull;            // pucker; muzzle and jaw pinch in x; forward, back travel
float gW, gYC, gRu, gCu, gRl, gCl, gAmt;         // aperture: half width, corner drop, the two arcs, height at the centre
vec3 gCH;
float gFarK, gTgB, gSmirk, gSquint, gJn, gChkY, gChin, gRnd, gRndU, gZp, gRetr, gLowHide, gHoleM, gDark, gShow, gLipH, gLipT, gLipD; // carve margin; 1 dark slit; teeth out; lower lip height, fullness, depth
float gTipT, gDent, gBackT, gToTip, gToGap, gTgY;// tongue
mat3  gTR;          // world -> torso
vec3  gTO;          // the hips, in the world
vec3  gAS[2], gAE[2], gAW[2];      // shoulder, elbow, wrist, torso space.  [0] is screen left
vec3  gAS0[2], gAE0[2], gAW0[2];   // the same at rest
mat3  gAU[2], gAF[2], gAP[2];      // torso space -> the rest pose of the upper arm, the forearm, the paw
mat3  gPL[2];                      // torso space -> paw space: x across, y out of the palm, z along the toes
vec4  gToeP[2];                    // a paw's toes: first joint angle, second joint angle, fan step, unused

// ---- rig (docs/RIG.md): the whole character turns with body_yaw / body_pitch / body_lean about the seat and lifts with
// body_bounce: the entry points map the world point into the body's rest frame.  point / thumb / cover per paw.
const vec3 BODY_PIV = vec3(0.0,-2.20,-0.25);
mat3 gBodyM; vec3 gBodyT;
vec3 bodyXf(vec3 p){ return gBodyM*(p - BODY_PIV - gBodyT) + BODY_PIV; }
float gPointP[2], gThumbP[2], gShrugP[2], gTgOut, gNoseW;

mat3 rotX(float a){ float c=cos(a),s=sin(a); return mat3(1,0,0, 0,c,s, 0,-s,c); }
mat3 rotY(float a){ float c=cos(a),s=sin(a); return mat3(c,0,-s, 0,1,0, s,0,c); }
mat3 rotZ(float a){ float c=cos(a),s=sin(a); return mat3(c,s,0, -s,c,0, 0,0,1); }

// a column, about itself inside -1..1, and never past 1.12 whatever it is handed
float colSat(float x){ return x/pow(1.0 + pow(abs(x)/1.12, 4.0), 0.25); }
float hinge(float x){ return 0.5*(x + sqrt(x*x + 0.03)) - 0.0866; }           // about 0 below zero, about x above it, smooth

// a point moved out of an ellipsoid, softly
vec3 pushOut(vec3 p, vec3 c, vec3 r, float k){
  vec3 v = p - c;
  float e = length(v/r);
  return c + v*smax(e, 1.0, k)/max(e, 1e-3);
}

// mat3 from three axes, as columns
void boneFrame(vec3 f, vec3 h, out mat3 R){
  vec3 y = normalize(h - f*dot(h, f));
  R = mat3(cross(y, f), y, f);
}

// One arm, torso space.  s = -1 screen left, c = hand_x,y,z, open, turn.  All pure functions of the columns.
void armSolve(float s, vec3 c, float open, float turn, float brth, bool useHead, out vec3 S, out vec3 E, out vec3 W, out mat3 RU, out mat3 RF, out mat3 RP){
  float x = colSat(c.x), y = colSat(c.y), z = colSat(c.z);
  float up = hinge(y);
  // the shoulder goes with the hand a little: it lifts, comes forward, slides out
  int si = s < 0.0 ? 0 : 1;
  float shr = useHead ? gShrugP[si] : 0.0;
  S = vec3(s*(0.46 + 0.030*hinge(x)), -0.84 + 0.060*up + 0.012*brth + 0.13*shr, 0.02 + 0.050*z);
  // contract: the wrist's offset from rest (paws together at the chest) in units of one arm length; inward x is
  // scaled so x = -0.45 meets at the midline; a raised hand (not reaching across) opens out past the cheek
  const float L = ARM_U + ARM_F;
  vec3 T = vec3(s*(0.20 + (x >= 0.0 ? 0.95*L*x : 0.30*x) + 0.22*smoothstep(0.3, 1.0, y)*smoothstep(-0.1, 0.15, x)),
                -0.905 - 0.014*s + 1.15*L*y + 0.012*brth,
                0.60 + 0.012*s + 0.75*L*z);
  float cover = useHead ? smoothstep(0.80, 0.98, c.y)*smoothstep(-0.12, -0.30, c.x)*smoothstep(0.05, 0.30, c.z) : 0.0;   // raw columns
  if(useHead){
    vec3 w = transpose(gTR)*(T - BPIV) + gTO;
    vec3 q = pushOut(gHR*(w - gHO) + NECK, HEADKO, HEADKR, 0.06);
    // COVER_EYES: the paw goes over its own side's eye, palm on it, in front of the cheek
    q = mix(q, vec3(s*0.30, EYEC.y - 0.17, EYEC.z + 0.30), cover);
    w = transpose(gHR)*(q - NECK) + gHO;
    T = gTR*(w - gTO) + BPIV;
  }
  // a limb keeps its whole coat clear of the body's: hit points deep in a coat must not fall in the blend between two skins
  T = pushOut(T, RUFF, RUFFR + 0.22, 0.06);
  T = pushOut(T, CHEST, CHESTR + 0.22, 0.06);
  T = pushOut(T, BELLY, BELLYR + 0.19, 0.06);
  T.x = s*max(s*T.x, 0.075);                                     // each hand keeps to its own side
  // two bones, a soft end to the reach
  vec3 v = T - S;
  float D = length(v);
  vec3 d = v/max(D, 1e-4);
  float B = 0.975*(ARM_U + ARM_F), A = 0.84*(ARM_U + ARM_F);
  float Ds = D < A ? D : A + (B - A)*(1.0 - exp(-(D - A)/(B - A)));
  Ds = max(Ds, 0.34);
  W = pushOut(S + d*Ds, RUFF, RUFFR + 0.17, 0.05);               // the reach pulled it back along the arm: still clear of the coat
  W = pushOut(W, CHEST, CHESTR + 0.17, 0.05);
  W = pushOut(W, BELLY, BELLYR + 0.15, 0.05);
  d = normalize(W - S); Ds = length(W - S);
  float a = 0.5*(Ds*Ds + ARM_U*ARM_U - ARM_F*ARM_F)/Ds;
  float hh = sqrt(max(ARM_U*ARM_U - a*a, 1e-5));
  // an elbow tucks in as the palm turns up and lifts away from the side as it turns down
  float tn = clamp(turn, -1.0, 1.0);
  vec3 pole = normalize(vec3(s*(0.30 + (0.80 - 0.45*tn)*hinge(x) + 0.35*max(-tn, 0.0) + 0.20*up), -1.0, -0.30 - 0.55*hinge(x)));
  vec3 n = normalize(pole - d*dot(pole, d));
  E = S + d*a + n*hh;
  vec3 h = normalize(cross(d, n));                                // the elbow's hinge
  E = pushOut(E, RUFF, RUFFR + 0.13, 0.06);
  E = pushOut(E, CHEST, CHESTR + 0.13, 0.06);
  E = pushOut(E, BELLY, BELLYR + 0.13, 0.06);
  vec3 u = normalize(E - S), f = normalize(W - E);
  boneFrame(u, h, RU);
  boneFrame(f, h, RF);
  // the paw.  Palm down at turn -1, up at +1, a little in from down at rest.  Down means down, whatever the elbow is doing;
  // as the forearm comes vertical gravity gives no direction, and the back of the elbow's bend takes over: the palm faces forward
  vec3 vin = cross(f, RF[1]);                                     // the inside of the elbow: the palm of an arm curled up
  vec3 dn = vec3(0.0,-1.0,0.0) - 0.6*vin;
  dn = normalize(dn - f*dot(dn, f));
  vec3 md = -s*cross(f, dn);                                      // toward the midline
  float t = clamp(turn, -1.0, 1.0);
  float beta = PI*(0.15 + 0.50*t + 0.35*t*t);
  vec3 N = normalize(dn*cos(beta) + md*sin(beta));
  // the wrist: a relaxed paw hangs with gravity, so it flexes palm down and lies back palm up; an open hand comes level
  float op = clamp(open, 0.0, 1.0);
  float phi = mix(0.60 - 0.60*N.y, 0.10 - asin(clamp(f.y, -1.0, 1.0))*N.y, op);
  vec3 az = f*cos(phi) + N*sin(phi), ny = N*cos(phi) - f*sin(phi);
  az = normalize(az + vec3(s*0.42*op*hinge(x), 0.0, 0.0));         // an open hand angles out, away from the body
  ny = normalize(ny - az*dot(ny, az));
  // a paw up by the head stands its toes up, palm by the contract (turn +1 to the camera, 0 to the midline,
  // -1 away); over the eyes the palm is on the face; a fist with the thumb up / a pointing paw: palm in, toes along the arm
  float high = useHead ? smoothstep(-0.55, -0.20, W.y) : 0.0;
  float pt = useHead ? gPointP[si] : 0.0, th = useHead ? gThumbP[si] : 0.0;
  vec3 nCam = vec3(0.0,0.0,1.0), nIn = vec3(-s,0.0,0.0);
  vec3 nWant = t >= 0.0 ? mix(nIn, nCam, t) : mix(nIn, -nCam, -t);
  nWant = mix(nWant, -nCam, cover);
  float wUp = high*(1.0 - max(pt, th));
  az = normalize(mix(az, normalize(vec3(s*0.18, 1.0, 0.12)), 0.85*wUp));
  ny = normalize(mix(ny, nWant, wUp));
  float wF = max(pt, th);
  az = normalize(mix(az, normalize(f + vec3(0.0, 0.10, 0.20)), wF));
  ny = normalize(mix(ny, nIn, wF));
  ny = normalize(ny - az*dot(ny, az));
  RP = mat3(cross(ny, az), ny, az);
}

void charInit(){
  // yaw>0 toward screen right, pitch>0 nose down, roll>0 crown toward screen right (docs/AUDIO.md)
  mat3 F = rotY(YAW0 + uP[P_HEAD_YAW])*rotX(uP[P_HEAD_PITCH])*rotZ(-uP[P_HEAD_ROLL]);
  gHR = transpose(F);
  gBodyR = transpose(rotY(YAW0));
  gHO = NECK + vec3(uP[P_HEAD_X], uP[P_HEAD_Y], uP[P_HEAD_Z]);
  float by = 0.35*clamp(uP[P_BODY_YAW], -1.0, 1.0), bp = 0.15*clamp(uP[P_BODY_PITCH], -1.0, 1.0), bl = 0.12*clamp(uP[P_BODY_LEAN], -1.0, 1.0);
  gBodyM = transpose(rotY(by)*rotX(bp)*rotZ(-bl));
  gBodyT = vec3(0.0, 0.10*clamp(uP[P_BODY_BOUNCE], -1.0, 1.0), 0.0);
  gPointP[0] = clamp(uP[P_POINT_L], 0.0, 1.0); gPointP[1] = clamp(uP[P_POINT_R], 0.0, 1.0);
  gThumbP[0] = clamp(uP[P_THUMB_L], 0.0, 1.0); gThumbP[1] = clamp(uP[P_THUMB_R], 0.0, 1.0);
  gShrugP[0] = clamp(uP[P_SHOULDER_L], 0.0, 1.0); gShrugP[1] = clamp(uP[P_SHOULDER_R], 0.0, 1.0);
  gTgOut = clamp(uP[P_TONGUE_OUT], 0.0, 1.0);
  gNoseW = clamp(uP[P_NOSE_WIGGLE], -1.0, 1.0);
  gSmile  = clamp(uP[P_SMILE], 0.0, 1.0);
  gCheekP = clamp(uP[P_CHEEK], 0.0, 1.0);
  gNostril= clamp(uP[P_NOSTRIL], 0.0, 1.0);
  // ---- the lip columns.  lip_press overrides everything that opens
  float mo = clamp(uP[P_MOUTH_OPEN], 0.0, 1.0);
  float press = clamp(uP[P_LIP_PRESS], 0.0, 1.0)*(1.0 - max(mo, gTgOut)), keep = 1.0 - smoothstep(0.15, 1.0, press);   // the lips hold together until the press is nearly gone
  float open = max(clamp(uP[P_LIP_OPEN], 0.0, 1.0)*keep, max(mo, 0.45*gTgOut));
  float up   = clamp(uP[P_LIP_UPPER], 0.0, 1.0)*keep;
  float lo   = clamp(uP[P_LIP_LOWER], 0.0, 1.0)*keep;
  float wid  = clamp(uP[P_LIP_WIDTH], -1.0, 1.0);
  float pro  = clamp(uP[P_LIP_PROTRUDE], 0.0, 1.0);
  float tuck = clamp(uP[P_LIP_TUCK], 0.0, 1.0)*keep;
  float gap  = max(clamp(uP[P_TEETH_GAP], 0.0, 1.0)*keep, max(mo, 0.40*gTgOut));
  float pk = max(-wid, 0.0), sp = max(wid, 0.0);
  gPress = press; gLab = tuck; gSpread = sp; gPk = pk;
  gRound = clamp(0.70*pk + 0.30*pro, 0.0, 1.0);
  gSmirk = clamp(uP[P_SMIRK], -1.0, 1.0); gSquint = clamp(uP[P_SQUINT], 0.0, 1.0);
  gJawA = MAXJAW*(1.0 - pow(1.0 - gap, 1.4)) - 0.040*press;       // a press pushes the chin up into the pads

  // the muzzle around the mouth
  gSqz  = 0.34*pk + 0.06*pro - 0.12*sp;
  gSqzJ = -0.30*sp - 0.30*open*(1.0 - pk);
  gPush = 0.050*pro;
  gPull = 0.050*sp;

  // the aperture.  A round O needs height as well as narrowness: the pucker pulls both arcs to the width
  gW = (0.112 + 0.062*sp - 0.056*pk*sqrt(pk) + 0.026*open*(1.0 - pk))*(1.0 - 0.60*press);    // peel: as the press lets go the lips part from the middle out
  float gt = GAPMAX*pow(open, 0.85);
  float au = 0.10*gt + 0.016*up;
  float al = 0.90*gt + 0.060*lo;
  float rnd = pk*sqrt(pk);                                        // roundness is a shape; its size comes from the opening
  float o2 = clamp((open + 0.5*(up + lo))/0.16, 0.0, 1.0);
  o2 = o2*(1.5 - 0.5*o2);
  au = mix(au, 0.80*gW*(1.0 + 0.3*open)*o2, rnd);
  al = mix(al, 0.92*gW*(1.0 + 0.5*open)*o2, rnd);
  au += 0.022*tuck;                                               // the upper lip clears the teeth ...
  al *= 1.0 - 0.85*tuck;                                          // ... and the lower lip comes up to them
  float drop = 0.10*al;                                           // the corners ride down with the lower lip
  gYC = -drop;
  float aU = max(au + drop, 0.0009), aL = max(al - drop, 0.0009);
  gRu = 0.5*(gW*gW + aU*aU)/aU; gCu = gRu - aU;
  gRl = 0.5*(gW*gW + aL*aL)/aL; gCl = gRl - aL;
  gAmt = au + al;
  gSeal = 1.0 - smoothstep(0.0, 0.020, gAmt + 0.06*up);
  gDark = 1.0 - smoothstep(0.012, 0.055, gAmt + 0.06*tuck + 0.20*up);
  gHoleM = 0.014*smoothstep(0.0, 0.030, gAmt);

  // the split upper lip: the cusp lifts and the pads part with lip_upper, a little with the opening, least in a pucker
  float part = (1.5*up + 0.45*smoothstep(0.0, 0.70, open) + 0.70*tuck + 0.30*sp*smoothstep(0.0, 0.15, open))*(1.0 - 0.75*rnd);
  gPart = min(part, 1.1);
  gCleft  = 0.052 + 0.046*gPart - 0.022*press;
  gCleftW = 0.026 + 0.034*gPart + 0.014*sp*gPart;
  gRnd = rnd;
  gRndU = smoothstep(0.30, 0.75, rnd);                            // the lip comes all the way round
  gZp = mix(0.72, 0.835, rnd) + 0.020*pro*rnd;                     // the plane of the lips; the aperture's column pivots on it
  gRetr = 0.052*rnd;                                               // the pads draw back and gather around a round mouth
  gCH = gHR*(uCamPos - gHO) + NECK;                                 // the camera, in head space
  // a mouth wraps a curved muzzle: as the head turns away the far half foreshortens, or it would hang past the jaw
  float vs = max(abs(gCH.x)/(gCH.z - gZp) - 0.40, 0.0);
  gFarK = 1.0 + vs*(1.6 + 6.0*vs);
  gLowHide = 0.030*(1.0 - smoothstep(0.0, 0.14, lo));            // the lower teeth are lip_lower's to show
  gShow = smoothstep(0.012, 0.070, gAmt + 0.10*gPart)*(1.0 - rnd);    // the teeth wait behind the lip until there is room

  // the lower lip: nothing at rest, it rolls out as it leaves the pads; a press and a tuck roll it back in
  float lm = smoothstep(0.002, 0.036, gAmt + 0.055*tuck)*(1.0 - 0.5*press);
  gLipH = lm*(0.030 - 0.006*tuck + 0.004*pk - 0.008*sp - 0.006*open);          // lips stretch thin as they spread and open
  gLipT = lm*(0.017 + 0.008*pro - 0.006*tuck + 0.026*rnd);
  gLipD = 0.010*pro*lm + 0.002*tuck - 0.008*press;

  // ---- what follows the lips.  jn: how far the lower face has dropped, teeth or no teeth
  gJn = clamp(max(gap, 0.75*open) + 0.25*lo, 0.0, 1.0);
  gChin = clamp(al - sin(gJawA)*0.54, 0.0, 0.05)*0.60;            // lips parted further than the teeth: the chin's skin follows
  gChkY = 0.034*gJn - 0.020*sp - 0.010*gSmile + 0.008*pk - 0.010*gSquint;   // the cheek slides down with the jaw, rides up on a spread

  // ---- tongue, in the jaw's frame: a bed under the lip line, reaches for two targets fixed to the skull
  gTipT = clamp(uP[P_TONGUE_TIP], 0.0, 1.0);
  gDent = clamp(uP[P_TONGUE_DENTAL], 0.0, 1.0);
  gBackT = clamp(uP[P_TONGUE_BACK], 0.0, 1.0);
  gTgB = gTipT*gBackT;                                            // tip up and back up: the tongue is held broad against the roof
  gTgY = LIPY - 0.070;
  gToTip = jawSpace(vec3(0.0, -0.205, 0.70), HINGE, -gJawA).y - gTgY;
  gToGap = jawSpace(vec3(0.0, -0.285 - 0.40*gAmt, 0.76), HINGE, -gJawA).y - gTgY;
  gBlinkL = clamp(uP[P_BLINK_L], 0.0, 1.0); gBlinkR = clamp(uP[P_BLINK_R], 0.0, 1.0);
  gGaze = vec2(uP[P_GAZE_X], uP[P_GAZE_Y]);
  gBrowL = uP[P_BROW_L] + 0.5*clamp(uP[P_EYE_WIDE], 0.0, 1.0); gBrowR = uP[P_BROW_R] + 0.5*clamp(uP[P_EYE_WIDE], 0.0, 1.0); gKnit = clamp(uP[P_BROW_KNIT], 0.0, 1.0);
  // ears: forward/back about the base, plus the soft bend.  _l = screen-left.  The left ear stands
  // tall, the right one flops at rest: that pair of shapes is the rabbit's silhouette.
  float eL = uP[P_EAR_L], eR = uP[P_EAR_R], fL = uP[P_EAR_FLICK_L], fR = uP[P_EAR_FLICK_R];
  // pitch ±0.50 rad, yaw about the ear's own long axis (ear_*_turn ±0.75: -1 the bowl forward/in, +1 out), a
  // flick 0.34 rad + a 0.30 twist
  gEarRL = rotY(-0.75*uP[P_EAR_L_TURN])*rotX(-(0.50*eL + 0.34*fL))*rotZ(0.30*fL);
  gEarRR = rotY(-0.75*uP[P_EAR_R_TURN])*rotX(-(0.50*eR + 0.34*fR))*rotZ(0.30*fR);
  gEarKL = clamp(0.82 - 0.40*eL + 0.14*abs(eL) + 0.55*fL, 0.10, 2.4);     // droops freely, straightens only so far: the tip stays in frame
  gEarKR = clamp(1.30 - 0.70*eR + 0.25*abs(eR) + 0.75*fR, 0.10, 2.4);
  gBreath = uP[P_BREATH];
  gCorner = 0.56 + 0.10*gRound - 0.07*gSpread;      // mouth corner, z

  // ---- the torso takes a share of the head and of the hands.  At rest its frame is the head's rest frame
  vec3 hl = vec3(uP[P_HAND_L_X], uP[P_HAND_L_Y], uP[P_HAND_L_Z]);
  vec3 hr = vec3(uP[P_HAND_R_X], uP[P_HAND_R_Y], uP[P_HAND_R_Z]);
  float tYaw = 0.22*uP[P_HEAD_YAW] + 0.035*(colSat(hl.z) - colSat(hr.z)) + 0.030*(hinge(colSat(hr.x)) - hinge(colSat(hl.x)));
  float tPit = 0.20*uP[P_HEAD_PITCH] + 0.012*(colSat(hl.z) + colSat(hr.z));
  float tRol = 0.22*uP[P_HEAD_ROLL] + 0.018*(hinge(colSat(hr.y)) - hinge(colSat(hl.y)));
  gTR = transpose(rotY(YAW0 + tYaw)*rotX(tPit)*rotZ(-tRol));
  gTO = NECK + rotY(YAW0)*(BPIV - NECK) + 0.30*vec3(uP[P_HEAD_X], uP[P_HEAD_Y], uP[P_HEAD_Z]);
  // ---- the arms, posed and at rest
  mat3 RU, RF, RP, RU0, RF0, RP0;
  float opL = uP[P_HAND_L_OPEN], opR = uP[P_HAND_R_OPEN];
  armSolve(-1.0, hl, opL, uP[P_HAND_L_TURN], gBreath, true, gAS[0], gAE[0], gAW[0], RU, RF, RP);
  armSolve(-1.0, vec3(0.0), 0.0, 0.0, 0.0, false, gAS0[0], gAE0[0], gAW0[0], RU0, RF0, RP0);
  gAU[0] = RU0*transpose(RU); gAF[0] = RF0*transpose(RF); gAP[0] = RP0*transpose(RP); gPL[0] = transpose(RP);
  armSolve(1.0, hr, opR, uP[P_HAND_R_TURN], gBreath, true, gAS[1], gAE[1], gAW[1], RU, RF, RP);
  armSolve(1.0, vec3(0.0), 0.0, 0.0, 0.0, false, gAS0[1], gAE0[1], gAW0[1], RU0, RF0, RP0);
  gAU[1] = RU0*transpose(RU); gAF[1] = RF0*transpose(RF); gAP[1] = RP0*transpose(RP); gPL[1] = transpose(RP);
  // ---- the toes fan out and straighten as the hand opens
  float o0 = clamp(opL, 0.0, 1.0), o1 = clamp(opR, 0.0, 1.0);
  o0 *= 1.0 - max(gPointP[0], gThumbP[0]); o1 *= 1.0 - max(gPointP[1], gThumbP[1]);     // a pointing paw, a thumbs-up: the rest curl in
  gToeP[0] = vec4(mix(0.42, 0.06, o0) + 0.30*max(gPointP[0], gThumbP[0]), mix(1.00, 0.22, o0) + 0.30*max(gPointP[0], gThumbP[0]), 0.20 + 0.20*o0, 0.0);
  gToeP[1] = vec4(mix(0.42, 0.06, o1) + 0.30*max(gPointP[1], gThumbP[1]), mix(1.00, 0.22, o1) + 0.30*max(gPointP[1], gThumbP[1]), 0.20 + 0.20*o1, 0.0);
}

float sq(float x){ return x*x; }
float sdRoundBox(vec3 p, vec3 b, float r){ vec3 d = abs(p) - b + r; return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0) - r; }

// lip surface height in head space (also evaluated in jaw space).  cleft + cw: the central cusp.
float cuspY(vec3 q, float cleft, float cw){
  float s = clamp((q.z - 0.36)/0.46, 0.0, 1.0);
  return cleft*exp(-abs(q.x)/cw)*s*s;                              // the cusp of the Y
}
float lipY0(vec3 q){
  float s = clamp((q.z - 0.36)/0.46, 0.0, 1.0);
  float c = max(abs(q.x) - 0.075, 0.0)/0.20;
  return LIPY + c*c*(0.060 + (gSmile + 0.9*gSmirk*sign(q.x))*0.090 + gSpread*0.040)         // each pad is an arc, up to the corner
              + sq(1.0 - s)*(0.035 + gSmile*0.050);
}
float mouthY(vec3 q, float cleft, float cw){ return lipY0(q) + cuspY(q, cleft, cw); }

// a real circular bend: above EARY0 the ear's axis follows an arc of curvature k toward +x
vec3 bendEar(vec3 e, float k){
  float R = 1.0/k;
  vec2 c = vec2(R - e.x, e.y - EARY0);
  vec3 o = vec3(R - length(c), EARY0 + atan(c.y, c.x)*R, e.z);
  return (e.y > EARY0) ? o : e;
}

// outputs of rabbitEval (parts, for fur + materials)
float gFur, gBare, gMask, gFurLen, gBowl, gWBody, gHoleD, gRing, gE, gEU, gVM, gFace;
vec3  gRest;            // rest-pose coordinate (fur + colour stick to it)
float gSnout, gUp, gLo, gEye, gNose, gTeeth, gTongue, gWhisk, gEarD, gBodyD, gMYu;
vec3  gQ, gQJ, gQE; float gEarS;
float gArmW, gArmT, gPawW, gPadD, gToeG;      // on a limb 0..1, shoulder 0 .. wrist 1, on a paw, on the bare palm pad, in a groove between toes

float sdEarShape(vec3 e){
  // e: y along the ear, x across, z out of the ear's face.  A long leaf: narrow base, widest at 40%,
  // soft round tip, flattened in z, with a long bowl pressed into the front.
  vec3 sc = vec3(1.0, 1.0, 3.0);
  vec3 p = e*sc;
  float d = sdTaper(p, vec3(0.0,0.0,0.0), vec3(0.0,EARL*0.42,0.0), 0.090, 0.178);
  d = smin(d, sdTaper(p, vec3(0.0,EARL*0.42,0.0), vec3(0.0,EARL*0.93,0.0), 0.178, 0.050), 0.03)/3.0;
  vec3 b = (e - vec3(0.0,0.0,0.052))*sc;
  float bowl = sdTaper(b, vec3(0.0,0.03,0.0), vec3(0.0,EARL*0.42,0.0), 0.062, 0.150);
  bowl = smin(bowl, sdTaper(b, vec3(0.0,EARL*0.42,0.0), vec3(0.0,EARL*0.91,0.0), 0.150, 0.026), 0.03)/3.0;
  gBowl = bowl;
  return smax(d, -bowl, 0.03);
}

// a limb of two tapers.  Returns the distance; t 0 shoulder .. 1 wrist; wf 0 upper arm .. 1 forearm
float limbD(vec3 b, vec3 S, vec3 E, vec3 W, out float t, out float wf){
  float tu = clamp(dot(b - S, E - S)/dot(E - S, E - S), 0.0, 1.0);
  float tf = clamp(dot(b - E, W - E)/dot(W - E, W - E), 0.0, 1.0);
  float d1 = length(b - S - (E - S)*tu) - mix(ARM_R0, ARM_R1, tu);
  float d2 = length(b - E - (W - E)*tf) - mix(ARM_R1, ARM_R2, tf);
  wf = smoothstep(-0.05, 0.05, d1 - d2);
  t = mix(0.5*tu, 0.5 + 0.5*tf, wf);
  return smin(d1, d2, 0.050);
}

// one toe, in its own fan sector: x across, y out of the palm, z along.  c = joint angles, cos and sin of each
float toeDL(vec3 p, float id, float fan, vec4 c, float L){
  float th = (id - 1.5)*fan*0.5;
  vec3 q = vec3(cos(th)*p.x - sin(th)*(p.z + 0.11), p.y, sin(th)*p.x + cos(th)*(p.z + 0.11));       // about the middle of the palm
  q.z -= 0.225 - 0.012*sq(id - 1.5);
  vec3 kn = vec3(0.0, c.y, c.x)*0.060*L;
  vec3 tip = kn + vec3(0.0, c.w, c.z)*0.050*L;
  return min(sdTaper(q, vec3(0.0), kn, 0.031, 0.027), sdTaper(q, kn, tip, 0.027, 0.021));
}
float toeD(vec3 p, float id, float fan, vec4 c){ return toeDL(p, id, fan, c, 1.0); }

void rabbitEval(vec3 pw){
  vec3 q = gHR*(pw - gHO) + NECK;
  gQ = q;
  // the soft muzzle: a pucker pinches it and pushes it forward, a spread stretches it and draws it back.
  // Paint and fur read qm, so they travel with the skin
  float ax = abs(q.x);
  float wm = smoothstep(0.30, 0.74, q.z)*exp(-sq((q.y + 0.24)/0.17));
  float mf = 1.0 + gSqz*wm;
  vec3 qm = q;
  qm.x -= 0.022*gSmirk*wm;
  qm.x *= mf;
  qm.z -= wm*(gPush*(1.0 - gRnd) - gPull*smoothstep(0.04, 0.20, ax) - gRetr*exp(-sq(length(vec2(q.x, q.y + 0.27))/0.15)));
  float cuspK = 1.0 - 0.80*sqrt(gRnd);                                    // the pads gather: the Y closes as the mouth rounds
  float cusp0 = cuspY(qm, (0.052 - 0.022*gPress)*cuspK, 0.026 + 0.030*gRnd);   // the resting Y
  float y0 = lipY0(qm) - 0.010*gPress;
  float yL = y0 + cusp0 + gLab*0.016*smoothstep(0.56,0.80,q.z);
  gMYu = yL;

  // ---- skull
  float skull = sdEllipsoid(q - vec3(0.0,0.04,-0.02), vec3(0.600,0.585,0.610));
  vec3 qa = vec3(abs(q.x), q.yz);
  // round puffy cheeks
  float lift = gCheekP*0.024 + gSmile*0.018 + 0.020*max(gSmirk*sign(q.x), 0.0);
  float wc = exp(-sq((ax - 0.34)/0.29) - sq((q.y + 0.22)/0.27));   // the soft cheek: it slides with the jaw, narrows in a pucker, swells behind a press
  vec3 qc = vec3(ax*(1.0 + wc*(0.040*gPk - 0.020*gPress)), q.y + gChkY*wc, q.z);
  float cheek = sdEllipsoid(qc - vec3(0.372,-0.170+lift,0.125), vec3(0.312,0.285,0.300));
  skull = smin(skull, cheek, 0.12);
  // soft brow ridges: expression, not eyebrows
  float bu = (q.x < 0.0) ? gBrowL : gBrowR;
  vec3 bq = qa - vec3(0.175, 0.365 + 0.05*bu - 0.03*gKnit, 0.37);
  bq.xy = rot(0.20 + 0.30*gKnit)*bq.xy;
  bq.x += 0.04*gKnit;
  skull = smin(skull, sdEllipsoid(bq, vec3(0.105,0.030,0.09)), 0.17);
  // a little forelock between the ears
  skull = smin(skull, sdTaper(qa, vec3(0.02,0.55,0.06), vec3(0.085,0.665,0.16), 0.075, 0.016), 0.08);

  // ---- muzzle (upper).  The two pads are one ellipsoid mirrored in |x|, so the crease between them
  // is real geometry: it is the stem of the Y, and deepens as the pads part.
  float snoutRaw = sdTaperSq(qm, vec3(0.0,0.03,0.22), vec3(0.0,-0.052,0.722), 0.30, 0.118, 1.05);
  vec3 pa = vec3(abs(qm.x), qm.yz);
  float padY = -0.168 + 0.022*gPart + 0.012*gPk + gCheekP*0.012 + gSmile*0.010 + gNostril*0.012 - 0.012*gPress - 0.016*gJn;
  float pad = sdEllipsoid(pa - vec3(0.104 + 0.030*gPart, padY, 0.712 + 0.012*gPress), vec3(0.128,0.112,0.150));
  snoutRaw = smin(snoutRaw, pad, 0.030);
  vec3 qr = qm - smoothstep(0.03, -0.02, pad)*vec3(sign(q.x)*0.030*gPart, 0.022*gPart - 0.016*gJn, 0.0);   // fur and paint ride on the pad
  qr.y += gChkY*wc;
  snoutRaw /= max(mf, 1.0);
  gSnout = snoutRaw;
  float snoutCut = smax(snoutRaw, yL - q.y, 0.014);
  float headRaw = smin(skull, snoutCut, 0.13);
  // sockets
  vec3 ea = qa - EYEC;
  headRaw = smax(headRaw, -(length(ea) - (EYER + 0.012)), 0.035);
  float upper = headRaw;
  gUp = upper;

  // ---- lower jaw, built in its rest frame
  vec3 qj = jawSpace(q, HINGE, -gJawA);
  gQJ = qj;
  // the lip end of the jaw follows the corners in x and rides forward with the lips
  float jb = smoothstep(0.30, 0.56, qj.z)*exp(-sq((qj.y + 0.25)/0.15));
  float jf = 1.0 + gSqzJ*jb;
  vec3 qjm = qj;
  qjm.x -= 0.012*gSmirk*jb;
  qjm.x *= jf;
  qjm.z -= (gPush + 0.040*gRnd)*jb;
  float yLj = mouthY(qjm, gCleft*gSeal, gCleftW) + 0.030*gSeal - 0.002 + 0.20*(1.0 - gSeal);    // closed: the lips overlap, the jaw tucks up behind the pads.  Open: the aperture cuts it
  // labiodental: the lower lip rolls back under the buck teeth
  vec3 ql = qjm;
  ql.z += gLab*0.070*smoothstep(0.48,0.76,qj.z);
  ql.y -= gLab*0.006;
  float lowerRaw = sdTaperSq(ql, JAW_A, JAW_B, JAW_RA, JAW_RB, 1.12);
  lowerRaw = smin(lowerRaw, sdEllipsoid(ql - vec3(0.0,-0.306 - gChin,0.655), vec3(0.094 + 0.006*gPress,0.066 + 0.012*gPress,0.094 + 0.016*gPress)), 0.05);   // soft little chin; a press bunches it
  lowerRaw /= max(jf, 1.0);
  float lower = smax(lowerRaw, qj.y - yLj, 0.014);
  gLo = lower;

  // ---- two buck teeth (bare).  Rigid on the skull: they sit behind the lips while sealed and slide
  // down and forward a little when there is room.
  vec3 tp = vec3(ax, q.y, q.z) - vec3(0.0395, -0.186 - 0.026*gShow, 0.748 + 0.026*gShow + 0.012*gLab);
  tp.z += 0.20*tp.x*tp.x + 0.05*tp.x;                    // the pair follows the arch
  float tw = 0.0345 + 0.01*max(-tp.y, 0.0);              // chisel: a touch wider at the tip
  gTeeth = sdRoundBox(tp, vec3(tw, 0.062, 0.014), 0.0135);

  // ---- two small lower incisors (bare), in the jaw, behind the lower lip and behind the uppers
  vec3 tl = vec3(abs(qj.x), qj.y, qj.z) - vec3(0.0245, -0.325 - 0.055*gRnd - gLowHide, 0.706);
  tl.z += 0.24*tl.x*tl.x;
  float teethL = sdRoundBox(tl, vec3(0.0215, 0.040, 0.010), 0.0095);

  // ---- tongue (bare): a broad low body on a bed in the jaw's frame, a tip that reaches, a back that rises.  The tip can
  // be a slim blade or the whole front laid against the roof (gTgB, what a held n does)
  float bed = gTgY - 0.018*gPk;
  vec3 tq = qj - vec3(0.0, bed, 0.500 - 0.090*gPk);
  float tz = clamp((tq.z + 0.20)/0.40, 0.0, 1.0);        // 0 root .. 1 tip
  tq.y += 0.050*sq(1.0 - tz);                            // the root falls away: the throat stays dark
  tq.x = (tq.x - 0.055*(1.0 - tz))*(1.0 + 0.55*tz*tz);   // seen from the 3/4, a dead straight tongue lolls to the near side
  float tongue = sdEllipsoid(tq, vec3(0.118, 0.036, 0.205))*0.90;
  tongue += 0.0016*smoothstep(0.016,0.0,abs(tq.x))*smoothstep(1.0,0.6,tz);      // centre groove, gone at the tip
  tongue = smin(tongue, sdEllipsoid(qj - vec3(0.0, bed - 0.040 + 0.165*gBackT, 0.430), vec3(0.092 + 0.026*gBackT, 0.060 + 0.012*gBackT, 0.125)), 0.050);
  vec3 tB = vec3(0.0, bed + 0.006 + gTipT*gToTip + gDent*gToGap - 0.045*gTgOut, 0.640 + 0.060*gTipT + 0.120*gDent + 0.44*gTgOut);
  float tipR = 0.040 - 0.014*gJn*gTipT + 0.016*gTgB + 0.014*gTgOut;     // the further the reach the slimmer the blade (out: a round tip)
  tongue = smin(tongue, sdTaper(qj*vec3(1.0 - 0.34*gTgB, 1.0, 1.0), vec3(0.0, bed + 0.004, 0.52), tB, 0.058 - 0.014*gJn, tipR), 0.040);
  gTongue = tongue;

  // ---- whiskers (bare): three per side, a single taper in a drooped domain, riding on the pads
  float wa = abs(qm.x), wx = max(wa - 0.17, 0.0);
  vec3 wq = vec3(wa, q.y + 0.24*wx*wx - gCheekP*0.012 - gNostril*0.010 + 0.014*gJn, qm.z);
  float wh = sdTaper(wq, vec3(0.170,-0.128,0.690), vec3(0.820,0.050,0.380), 0.0036, 0.0010);
  wh = min(wh, sdTaper(wq, vec3(0.182,-0.166,0.676), vec3(0.880,-0.120,0.340), 0.0036, 0.0010));
  wh = min(wh, sdTaper(wq, vec3(0.170,-0.204,0.676), vec3(0.820,-0.300,0.380), 0.0036, 0.0010));
  gWhisk = wh;

  // ---- eyeballs
  gEye = length(ea) - EYER;

  // ---- nose: a tiny pink rounded triangle, point down, that twitches
  vec3 nq = q - NOSEP - vec3(0.0, 0.016*gNostril - 0.008*gJn + 0.040*gNoseW, 0.004*gNostril + 0.22*gPush + 0.010*abs(gNoseW));
  vec3 nna = vec3(abs(nq.x), nq.yz);
  float nose = sdTaper(nna, vec3(0.032,0.016,-0.012), vec3(0.0,-0.040,0.006), 0.036, 0.013);
  nose = smin(nose, sdEllipsoid(nq - vec3(0.0,0.020,-0.012), vec3(0.052,0.024,0.032)), 0.018);
  float nostril = length((nna - vec3(0.050,0.004,0.016))*vec3(1.0,1.8,1.0)) - (0.008 + 0.006*gNostril);
  nose = smax(nose, -nostril, 0.010);
  gNose = nose;

  // ---- eyelids
  float blink = (q.x < 0.0) ? gBlinkL : gBlinkR;
  vec3 el = ea;
  el.xy = rot(-EYETILT)*el.xy;
  float squint = gSmile*0.10 + gCheekP*0.12 + 0.26*gSquint;
  float eyeWide = clamp(uP[P_EYE_WIDE], 0.0, 1.0);
  float aU = mix(0.60 + 0.28*eyeWide, -0.32, blink);
  float aL = mix(-0.76 + squint - 0.10*eyeWide, -0.36, blink);
  float bru = (q.x < 0.0) ? gBrowL : gBrowR;
  aU += 0.06*bru;
  float lids = sdEyelids(el, EYER, 0.016, aU, aL);
  lids = smax(lids, -ea.z - 0.04, 0.04);        // the lids are a full shell: its back end must not surface on the skull
  upper = smin(upper, lids, 0.030);
  gUp = upper;

  // ---- ears
  float s = (q.x < 0.0) ? -1.0 : 1.0;
  vec3 e = q - vec3(s*EARB.x, EARB.y, EARB.z);
  e.x *= s;
  e.xy = rot(0.45)*e.xy;       // lean outward
  e.yz = rot(0.18)*e.yz;       // and back
  e.xz = rot(0.35)*e.xz;       // the bowl turns out a little
  e = ((q.x < 0.0) ? gEarRL : gEarRR)*e;
  e = bendEar(e, (q.x < 0.0) ? gEarKL : gEarKR);
  gQE = e; gEarS = s;
  float ear = sdEarShape(e)*0.85;
  gEarD = ear;

  // ---- the aperture.  vm: height over the corners' line, on the resting lip surface
  // The carve is the cone from the camera through the aperture, pivoting on the plane of the lips.  So no wall of it
  // shows from the side, the two corners stay on the lips, and ap < 0 IS what is seen through the mouth.  In front of
  // the plane it does not tip in y: that keeps the sealed line on the pads
  float xs = gCH.x + (q.x - gCH.x)*(gZp - gCH.z)/(q.z - gCH.z) - 0.022*gSmirk;
  float xa = abs(xs)*(xs*gCH.x < 0.0 ? gFarK : 1.0);
  float kb = (gZp - gCH.z)/(min(q.z, gZp) - gCH.z);      // behind the plane, the scale onto it
  float vm = gCH.y + (q.y - gCH.y)*kb - gYC - lipY0(vec3(qm.x + (gCH.x - q.x)*(1.0 - kb), q.y, max(q.z, gZp))) + 0.010*gPress;
  float vU = vm - cuspY(qm, gCleft*cuspK, gCleftW + 0.030*gRnd)*smoothstep(-0.015, 0.015, vm);   // the upper lip takes the cusp
  float vL = vm - cusp0*gSeal;                           // the lower lip gives it up as soon as it leaves
  float eU = length(vec2(xa, vU + gCu)) - gRu;
  float eL = length(vec2(xa, vL - gCl)) - gRl;
  float ap = max(eU, eL);
  float cc = vm > 0.0 ? gCu : gCl;
  if(cc > 0.0 && gW*(abs(vm) + cc) < cc*xa) ap = length(vec2(xa - gW, vm));
  gE = ap; gEU = eU; gVM = vm;
  // open, nothing of the jaw stands above its own lip: under the pads there is mouth, right up to their edge
  float web = smoothstep(gCorner + 0.08, gCorner - 0.08, qj.z)*smoothstep(0.075, 0.16, ax);
  // however far it leans, it stops inside the jaw: you never see out through the far side.  Behind the teeth and below
  // the lip level that is the jaw's own solid, a shell; higher up, a wall on the jaw's widest section
  float wallD = ax + 0.022 - mix(0.235, 0.100, clamp((qj.z - 0.18)/0.46, 0.0, 1.0))/min(jf, 1.0);
  wallD = max(wallD, mix(-0.30, lowerRaw + 0.020, smoothstep(-0.245, -0.285, qj.y)*smoothstep(0.66, 0.60, qj.z)));
  lower = smax(lower, -max(max(max(eL, xa - gW) + 0.06*gSeal, 0.16 - q.z), wallD), 0.012);
  gLo = lower;
  float head = smin(upper, lower, 0.014 + 0.11*web);
  // the cheek's web: skin stretched from the pad down to the jaw, both sides.  Nothing else closes the side of a dropped
  // jaw, and the carve hollows it from inside.  Absent at rest
  float drop = sin(gJawA)*0.54;
  float cw = sdEllipsoid(vec3(ax - 0.142 - 0.05*gSpread, q.y - LIPY + 0.012 + 0.5*drop, q.z - 0.490), vec3(0.054, 0.050 + 0.58*drop, 0.165));
  head = smin(head, cw + 4.0*(1.0 - smoothstep(0.0, 0.10, gJawA)), 0.024);   // shrinks away: a constant here would cap the march's steps
  gFace = head;
  // the carve: wider under the lips and behind them, down to a floor that swings with the jaw, a roof, a back
  float wide = mix(1.20*clamp(-lowerRaw - 0.020, 0.0, 0.20), 0.55*clamp(0.75 + gPush - qm.z, 0.0, 0.20)*smoothstep(0.000, -0.020, snoutRaw), smoothstep(-0.01, 0.02, vm))*smoothstep(0.0, 0.030, gAmt);
  float hole = max(smax(ap - gHoleM - wide, 0.16 - q.z, 0.05), max(gTgY - 0.078 - 0.11*smoothstep(0.56, 0.34, qj.z) + 0.4*qj.x*qj.x - qj.y, q.y - y0 - 0.17));
  hole = max(hole, mix(-0.30, lowerRaw + 0.018, smoothstep(-0.305, -0.335, qj.y)));   // the jaw's underside stays closed
  hole = max(hole, wallD);
  gHoleD = hole;
  float body = smax(head, -hole, 0.020 - 0.002*(1.0 - gSeal));
  // the lower lip: a soft section around the lower arc, thick in the middle, gone at the corners, hugging the chin.
  // A round mouth lifts it off the chin onto its own plane, and it comes all the way round the aperture
  float u = xa/gW;
  float T = mix(pow(max(1.0 - u*u, 0.0) + 1e-4, 0.42), 1.0, gRndU*gRndU);
  float hh = max(0.5*gLipH*T*mix(1.0, smoothstep(0.015, -0.015, vm), 1.0 - gRndU), 0.0020), tt = max(gLipT*sqrt(T), 0.0020);
  vec2 ts = vec2(ap - hh*0.74, mix(lowerRaw, q.z - gZp, gRnd) - gLipD);
  vec2 tr = vec2(hh, tt);
  float k0 = length(ts/tr), k1 = length(ts/(tr*tr));
  float ring = k0*(k0 - 1.0)/max(k1, 1e-4);
  ring = max(ring, vm + 0.004 - 0.030*gRndU/(1.02 - gRndU));   // the lip wraps up around the sides as the mouth rounds, and closes over the top last
  ring = max(ring, 0.004 - gLipH);
  gRing = ring;
  body = smin(body, ring, 0.012);
  head = smin(body, ear, 0.07);
  gTeeth = min(gTeeth, max(teethL, hole + 0.006));
  gTongue = max(gTongue, hole + 0.006);

  // ---- body, in the torso's frame: a small round chest over a round belly, sitting up.  The haunches are below the frame
  vec3 b = gTR*(pw - gTO) + BPIV;
  float br = 1.0 + 0.016*gBreath;
  vec3 bb = b - vec3(0.0, 0.012*gBreath, 0.0);
  float chest = sdEllipsoid(bb - CHEST, CHESTR*br);
  float belly = sdEllipsoid(bb - BELLY, BELLYR*br);
  float hips  = sdEllipsoid(bb - vec3(0.0,-2.42,-0.22), vec3(0.92,0.84,0.86));
  float neck  = sdTaper(bb, vec3(0.0,-0.30,-0.26), vec3(0.0,-0.95,-0.26), 0.40, 0.50);
  float ruff  = sdEllipsoid(bb - RUFF, RUFFR);
  float bodyD = smin(smin(smin(chest, belly, 0.24), hips, 0.30), neck, 0.18);
  bodyD = smin(bodyD, ruff, 0.16);
  gBodyD = bodyD;

  gFur = smin(head, bodyD, 0.14);

  gBare = min(min(min(gEye, gNose), min(gTeeth, gTongue)), gWhisk);

  // rest coordinate for fur + paint
  float wBody = smoothstep(-0.10, 0.10, head - bodyD);
  gWBody = wBody;
  float wJaw  = smoothstep(-0.02, 0.02, upper - lower);
  float wEar  = smoothstep(-0.02, 0.02, min(upper,lower) - ear);
  vec3 rq = mix(qr, qjm, wJaw);
  rq = mix(rq, vec3(s*(e.x + EARB.x), e.y + EARB.y, e.z + EARB.z), wEar);
  gRest = mix(rq, b, wBody);

  // where fur grows, how long
  float m = 1.0;
  m *= smoothstep(0.010, 0.05, length(q - NOSEP) - 0.075);
  m *= smoothstep(0.012, 0.045, max(abs(q.y - yL) - 0.010 + wJaw*1.0, snoutRaw - 0.03));        // upper lip band
  m *= smoothstep(0.012, 0.045, max(abs(qj.y - yLj) - 0.010 + (1.0-wJaw)*1.0, lowerRaw - 0.03) + gSeal); // lower lip band
  m *= smoothstep(0.012, 0.075, length(ea) - EYER - 0.022);
  m *= mix(smoothstep(0.002, 0.030, min(hole, ring)), smoothstep(-0.040, 0.030, hole), gSeal);
  m *= smoothstep(-0.006, 0.004, ap + gSeal);            // nothing that is seen through the mouth grows fur
  m *= 1.0 - smoothstep(0.022, 0.004, abs(q.x))*smoothstep(0.50,0.68,q.z)*step(yL, q.y + 0.01)*step(q.y, -0.05);   // the philtrum stays clean
  m *= mix(1.0, smoothstep(0.0, 0.040, ap - 0.004), 1.0 - gSeal);                // bare around the aperture while open
  gMask = m;
  float muzzleShort = mix(0.34, 1.0, smoothstep(0.80, 0.34, mix(q.z, qj.z, wJaw)));   // rest z: the chin keeps its short nap as the jaw swings
  float earIn = smoothstep(0.03, -0.01, gBowl);
  gFurLen = mix(0.044*muzzleShort, 0.070 + 0.030*smoothstep(-0.95, -0.72, b.y)*smoothstep(-0.40, -0.62, b.y), wBody)*mix(1.0, mix(0.26, 0.08, earIn), wEar);

  // ---- the arms.  No loops or gates in here: llvmpipe's JIT time goes through the roof with a loop at every charEval
  // site.  Both limbs are plain tapers, and only the nearer paw is evaluated, lane by lane.
  float t0, t1, wf0, wf1;
  float a0 = limbD(b, gAS[0], gAE[0], gAW[0], t0, wf0);
  float a1 = limbD(b, gAS[1], gAE[1], gAW[1], t1, wf1);
  float rw0 = length(b - gAW[0]), rw1 = length(b - gAW[1]);
  bool  n0 = rw0 < rw1;
  mat3  PL = n0 ? gPL[0] : gPL[1];
  vec3  WN = n0 ? gAW[0] : gAW[1];
  vec4  tpv = n0 ? gToeP[0] : gToeP[1];
  // the paw: a soft palm, and of the four toes the two either side of this point
  const float PS = 1.22;                                      // paws a fifth bigger, they read at phone size
  vec3 pl = PL*(b - WN)/PS;
  float palm = sdEllipsoid(pl - vec3(0.0, 0.002, 0.070), vec3(0.086, 0.052, 0.098));
  float fanA = clamp((atan(pl.x, pl.z + 0.11)/(0.5*tpv.z)) + 1.5, 0.0, 2.999);
  float tid = floor(fanA);
  vec4 tc = vec4(cos(tpv.x), sin(tpv.x), cos(tpv.y), sin(tpv.y));
  // point: toe 1 straight and long, the others as posed (curled); thumb: a dewclaw toe straight out along paw x
  float ptN = n0 ? gPointP[0] : gPointP[1], thN = n0 ? gThumbP[0] : gThumbP[1];
  vec4 tS = vec4(cos(0.02), sin(0.02), cos(0.05), sin(0.05));
  vec4 tcA = mix(tc, tS, ptN*step(abs(tid - 1.0), 0.1)), tcB = mix(tc, tS, ptN*step(abs(tid + 1.0 - 1.0), 0.1));
  float toes = min(toeDL(pl, tid, tpv.z, tcA, 1.0 + 0.9*ptN*step(abs(tid - 1.0), 0.1)),
                   toeDL(pl, tid + 1.0, tpv.z, tcB, 1.0 + 0.9*ptN*step(abs(tid), 0.1)));
  float sxT = n0 ? -1.0 : 1.0;
  float thumbD = sdTaper(pl, vec3(sxT*0.060, 0.010, 0.030), vec3(sxT*(0.060 + 0.150*thN), 0.012, 0.050 + 0.020*thN), 0.036, 0.029);
  toes = min(toes, mix(thumbD + 0.08, thumbD, thN));
  float paw = smin(palm, toes, 0.014)*PS;
  float pw0 = smoothstep(0.03, -0.01, paw - (n0 ? a0 : a1));
  if(n0) a0 = smin(a0, paw, 0.040); else a1 = smin(a1, paw, 0.040);
  // the shoulder blends wide, everything else tight, so a limb in front of the coat stays a limb
  float ks0 = mix(0.20, 0.028, smoothstep(0.06, 0.36, length(b - gAS[0])));
  float ks1 = mix(0.20, 0.028, smoothstep(0.06, 0.36, length(b - gAS[1])));
  float f0 = gFur;
  bool  m0 = a0 < a1;
  float aN = m0 ? a0 : a1, kN = m0 ? ks0 : ks1;
  gFur = smin(smin(f0, a0, ks0), a1, ks1);
  float w = smoothstep(-0.4*kN - 0.01, 0.4*kN + 0.01, f0 - aN);
  // gRest of the nearer limb: each bone carried back to its rest pose
  float wf = m0 ? wf0 : wf1;
  vec3 rest = mix(m0 ? gAS0[0] + gAU[0]*(b - gAS[0]) : gAS0[1] + gAU[1]*(b - gAS[1]),
                  m0 ? gAE0[0] + gAF[0]*(b - gAE[0]) : gAE0[1] + gAF[1]*(b - gAE[1]), wf);
  float pawW = (m0 == n0) ? pw0 : 0.0;
  rest = mix(rest, m0 ? gAW0[0] + gAP[0]*(b - gAW[0]) : gAW0[1] + gAP[1]*(b - gAW[1]), pawW);
  gRest = mix(gRest, rest, w);
  // the coat's length changes hands tight at the crease: a wide blend tapers the chest's coat around a paw into a grid of dots
  float wL = smoothstep(-0.3*kN, 0.3*kN, f0 - aN);
  gMask = mix(gMask, 1.0, wL);
  float tA = m0 ? t0 : t1;
  gFurLen = mix(gFurLen, mix(mix(0.056, 0.030, smoothstep(0.1, 0.9, tA)), mix(0.018, 0.009, smoothstep(0.09, 0.14, pl.z)), pawW), wL);
  gArmW = wL; gArmT = tA; gPawW = pawW*wL;
  // only on the paw itself: unbounded in y this cylinder lands on the chest behind a resting palm, and shaves it
  float onPaw = pawW*wL*smoothstep(0.100, 0.070, abs(pl.y));
  gPadD = onPaw*smoothstep(0.16, -0.06, length((pl.xz - vec2(0.0, 0.074))/vec2(0.056, 0.062)) - 1.0)*smoothstep(-0.004, 0.020, pl.y);
  gMask *= 1.0 - gPadD;
  gToeG = onPaw*smoothstep(0.17, 0.05, abs(fract(fanA) - 0.5))*smoothstep(0.085, 0.130, pl.z);
  gWBody = max(gWBody, gArmW);
}

// fur flows away from the nose on the head, away from the neck on the body
vec3 furPole(){ return mix(POLE, vec3(0.0,-0.25,0.15), gWBody); }

void charEval(vec3 p){
  p = bodyXf(p);
  rabbitEval(p);
  gcBase = gFur; gcBare = gBare; gcLen = gFurLen*gMask; gcRest = gRest;
}
// the plateau closes the top of the coat: a dense plush nap rather than combed strands
float charTuft(){ return min(furHeight(gcRest, furPole(), mix(F_K, F_K*(1.5 + 1.1*smoothstep(-1.10, -1.60, gcRest.y)), gWBody), F_L)*1.45, 1.0); }

//@shadow-begin
// cheap proxy for shadow rays: 15 primitives, no fur, no mouth
float charShadow(vec3 pw){
  pw = bodyXf(pw);
  vec3 q = gHR*(pw - gHO) + NECK;
  vec3 qa = vec3(abs(q.x), q.yz);
  float d = sdEllipsoid(q - vec3(0.0,0.03,-0.04), vec3(0.54,0.52,0.53));
  d = min(d, sdEllipsoid(qa - vec3(0.36,-0.19,0.10), vec3(0.26,0.23,0.25)));
  d = smin(d, sdTaper(q, vec3(0.0,-0.02,0.25), vec3(0.0,-0.10,0.70), 0.22, 0.09), 0.10);
  d = min(d, sdTaper(q, vec3(0.0,-0.22,0.2), vec3(0.0,-0.31,0.58), 0.19, 0.08));
  vec3 e = qa - EARB;
  e.xy = rot(0.37)*e.xy;
  e.yz = rot(0.18)*e.yz;
  // only the straight stem of the ear: the ear bends, and a straight proxy would poke through it and
  // print a dark blotch on the pink
  d = min(d, sdTaper(e*vec3(1.,1.,2.5), vec3(0.0), vec3(0.0,0.24,0.0), 0.065, 0.085)/2.5);
  // the body must sit INSIDE the real coat: a proxy larger than the body shadows strand tips at random
  vec3 b = gTR*(pw - gTO) + BPIV;
  d = min(d, sdEllipsoid(b - CHEST, CHESTR - 0.04));
  d = min(d, sdEllipsoid(b - BELLY, BELLYR - 0.05));
  d = min(d, sdTaper(b, vec3(0.0,-0.35,-0.26), vec3(0.0,-0.95,-0.26), 0.30, 0.42));
  for(int i=0;i<2;i++){
    d = min(d, sdTaper(b, gAS[i], gAE[i], 0.095, 0.074));
    d = min(d, sdTaper(b, gAE[i], gAW[i], 0.074, 0.052));
    d = min(d, length(b - gAW[i] - transpose(gPL[i])*vec3(0.0,0.0,0.10)) - 0.062);
  }
  return d;
}
//@shadow-end

const vec3 C_WHITE = vec3(0.84,0.81,0.75);
const vec3 C_CREAM = vec3(0.78,0.68,0.53);
const vec3 C_CARA  = vec3(0.47,0.215,0.075);
const vec3 C_UMBER = vec3(0.26,0.115,0.045);
const vec3 C_PINK  = vec3(0.90,0.50,0.50);
const vec3 C_LINE  = vec3(0.095,0.040,0.045);    // the lip + philtrum line: dusky, never black
const vec3 C_LIP   = vec3(0.60,0.27,0.28);       // the bare lower lip
const vec3 C_LIPIN = vec3(0.30,0.075,0.09);      // its wet inner edge
const vec3 C_CAVE  = vec3(0.20,0.045,0.056);     // mouth, near the lips
const vec3 C_DEEP  = vec3(0.045,0.012,0.018);    // mouth, deep

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  pw = bodyXf(pw);
  rabbitEval(pw);
  vec3 q = gQ, r = gRest;
  vec3 n = s.n;
  // The fine march can land up to its 0.0045 minimum step inside a bare part, and near the lids gFur
  // can be negative there (smin with the lid shell), so these compare against max(gFur, 0).
  float fz = max(gFur, 0.0);
  bool isEye = gEye < BARE_EPS && gEye <= fz + BARE_EPS;
  bool isNose = gNose < BARE_EPS && gNose <= fz + BARE_EPS;
  bool isTeeth = gTeeth < BARE_EPS && gTeeth <= min(fz, max(gTongue, 0.0)) + BARE_EPS;
  bool isTongue = gTongue < BARE_EPS && gTongue <= fz + BARE_EPS;
  bool isWhisker = gWhisk < BARE_EPS && gWhisk <= fz + BARE_EPS;
  bool inMouth = (gE < -0.002 || gHoleD < 0.0040*gSeal) && gRing > 0.003;    // seen through the aperture, and the sealed crease, which narrows away to nothing

  // ---- fur paint
  vec3 c1 = furCoord(r, furPole(), 170.0, 62.0);
  float st = furStrand(c1);
  float tuft = furTuft(furCoord(r, furPole(), F_K, F_L));
  float ax = abs(r.x);
  float wob = 0.05*(tuft - 0.5) + 0.04*(st - 0.5) + 0.05*(vnoise3(r*5.0) - 0.5);
  // caramel cap and eye patches, with a white blaze down the middle and white cheeks + muzzle
  float blaze = 0.075 + 0.05*smoothstep(0.30, 0.0, r.y) + 0.20*smoothstep(0.2,-0.04,r.y);
  float capLine = mix(0.32, -0.035, smoothstep(0.08, 0.50, ax));          // from high by the muzzle to low at the side
  float cap = smoothstep(-0.02, 0.02, r.y - capLine + wob)*smoothstep(-0.02, 0.02, ax - blaze + wob);
  cap *= (1.0 - gWBody);
  // a pale ring around each eye
  float ring = smoothstep(0.070, 0.022, length(vec3(ax, r.yz) - EYEC) - EYER - 0.012 + 0.6*wob);
  // shoulders: a soft caramel saddle, white bib
  float saddle = smoothstep(-0.05, 0.05, ax - 0.415 - 0.16*(r.y + 1.0) - 0.26*smoothstep(-1.05, -1.55, r.y) + wob*2.0 + 0.14*(vnoise3(r*2.7) - 0.5))*gWBody;
  vec3 alb = mix(C_WHITE, C_CREAM, max(0.20*smoothstep(0.35,0.75,vnoise3(r*3.1)), 0.46*gWBody*smoothstep(-0.78, -1.20, r.y)));   // the belly is warmer: white paws read on it
  vec3 capC = mix(C_CARA, C_UMBER, 0.26*smoothstep(0.75,0.15,tuft) + 0.18*smoothstep(0.30,0.60,r.y));
  capC = mix(capC, C_CREAM, 0.12*smoothstep(0.35,0.80,st));
  alb = mix(alb, capC, max(cap, saddle));
  alb = mix(alb, mix(C_WHITE, C_CREAM, 0.45), ring*0.92*(1.0 - gWBody));
  // ears: caramel backs with darker tips, pink inside
  float earW = smoothstep(-0.02, 0.02, min(gUp, gLo) - gEarD)*smoothstep(-0.02, 0.02, gBodyD - gEarD);
  float eY = gQE.y/EARL;
  vec3 earC = mix(C_CARA, C_UMBER, 0.75*smoothstep(0.60, 1.0, eY + 0.10*(tuft - 0.5)));
  float bowlM = smoothstep(0.016, -0.004, gBowl);
  // the ear from inside: warm pink skin, glowing, veins painted along it
  float u = gQE.x/max(0.178*(1.0 - sq(eY - 0.42)*1.5), 0.04);
  float vWarp = 0.05*(vnoise2(vec2(u*3.0, eY*9.0)) - 0.5);
  float artery = smoothstep(0.050, 0.008, abs(u + 0.20*sin(eY*5.0) + vWarp*2.0));
  float chev = abs(fract(eY*7.5 - abs(u)*1.15 + vWarp*3.0) - 0.5);
  float branch = smoothstep(0.075, 0.018, chev)*smoothstep(1.0, 0.25, abs(u))*smoothstep(0.02, 0.18, abs(u));
  float veins = max(artery, 0.70*branch)*smoothstep(0.03,0.15,eY)*smoothstep(0.98,0.70,eY);
  vec3 earIn = mix(vec3(0.86,0.26,0.30), vec3(0.98,0.47,0.42), smoothstep(0.05,0.60,eY));
  earIn = mix(earIn, vec3(0.56,0.08,0.13), 0.70*veins);
  earC = mix(earC, mix(C_WHITE, C_PINK, 0.25), smoothstep(0.030, 0.006, gBowl)*(0.55 + 0.45*st));   // pale hairs on the inner rim
  earC = mix(earC, earIn, bowlM);
  alb = mix(alb, earC, earW);
  // the arms: the saddle comes down over the shoulder and the forearm, the paw is a white sock
  float sock = smoothstep(0.88, 1.02, gArmT + 1.4*wob);
  alb = mix(alb, mix(capC, mix(C_WHITE, C_CREAM, 0.50), sock), gArmW);     // cream socks, so a paw reads against the white chest
  alb = mix(alb, mix(C_WHITE, C_CREAM, 0.50), gPawW);
  // strand contrast
  alb *= 0.95 + 0.10*st + gWBody*0.20*(st - 0.5);
  alb *= 1.0 + gWBody*0.16*smoothstep(-1.05, -1.45, r.y)*(tuft - 0.5);      // the belly has no sun to draw its nap: paint a little of it
  alb *= 1.0 - 0.36*gToeG;

  // eyeliner + dark lid margin
  vec3 ea = vec3(abs(q.x), q.yz) - EYEC;
  float lidr = length(ea);
  float liner = smoothstep(EYER + 0.027, EYER + 0.017, lidr)*step(0.0, ea.z + 0.05);
  alb = mix(alb, vec3(0.13,0.065,0.05), liner);

  // lip band: the two arcs of the inverted Y, then the upper edge of the aperture
  float dyU = abs(q.y - gMYu)/(1.0 + 2.4*exp(-abs(q.x)/0.026));
  dyU = mix(dyU, abs(gEU), 1.0 - gSeal);
  float lipU = smoothstep(0.012, 0.004, dyU*(1.0 + 1.4*gPress))*smoothstep(0.040, 0.018, gSnout)*smoothstep(0.235, 0.17, abs(q.x));
  float lip = lipU*step(gUp, gLo + 0.002)*smoothstep(0.40,0.52,gQ.z);
  alb = mix(alb, C_LINE, lip);
  float bareSkin = lip;

  // philtrum: the stem of the Y, from the nose down to the cusp
  float phil = smoothstep(0.0095,0.0030,abs(q.x))*smoothstep(0.50,0.68,q.z)*step(gMYu - 0.006, q.y)*step(q.y, -0.060)*smoothstep(0.004, 0.022, gE + 0.03*(1.0 - gRndU));
  alb = mix(alb, C_LINE, phil);
  bareSkin = max(bareSkin, phil);

  // whisker dots on the pads
  vec2 wp = vec2(abs(r.x) - 0.125, r.y + 0.185)*vec2(26.0,30.0);
  vec2 wc = floor(wp + 0.5);
  float wd = length(wp - wc - 0.26*(hash22(wc) - 0.5));
  float wArea = step(abs(wc.y), 1.0)*step(abs(wc.x), 1.0)*step(0.66, r.z)*(1.0 - gWBody)*(1.0 - lip);
  alb = mix(alb, vec3(0.34,0.24,0.20), 0.65*wArea*smoothstep(0.17,0.07,wd));

  s.alb = alb;
  vec3 v = gRest - furPole();
  s.T = furFlow(v, n, vec3(0.0,-0.25,0.0));
  float nb = (st - 0.5);
  vec3 b = normalize(cross(s.T, n));
  s.n = normalize(n + b*nb*(0.05 + 0.08*gWBody) + s.T*nb*0.02);
  s.fur = 1.0 - max(bareSkin, gPadD);
  s.wrap = 0.30;
  s.sheen = 0.40;
  s.occ = (0.90 + 0.10*smoothstep(0.1,0.7,tuft))*(0.95 + 0.08*st);
  s.alb *= mix(vec3(0.94,0.90,0.86), vec3(1.0), smoothstep(0.05,0.6,tuft));
  s.alb = mix(s.alb, vec3(0.80,0.42,0.43), gPadD);      // the palm pad: the nose's pink, a soft shine
  s.sheen *= 1.0 - 0.7*gPadD; s.spec = mix(s.spec, 0.22, gPadD); s.shin = mix(s.shin, 30.0, gPadD);
  // sunlight through the ear
  s.trans = earW*mix(0.45, 0.95, bowlM);
  s.transCol = vec3(1.0,0.46,0.40);
  s.emit = earW*bowlM*earIn*vec3(0.62,0.30,0.25)*(0.35 + 0.85*smoothstep(0.02,0.55,eY));
  // white fur scatters: the muzzle's shadow and the paws' shadows land soft and warm on the bib, not as gray blots
  float bib = gWBody*(1.0 - gArmW)*(1.0 - saddle);
  float fill = max(0.26*smoothstep(-1.04, -0.72, r.y)*smoothstep(0.55, 0.22, ax), 0.20*smoothstep(-0.80, -1.05, r.y))*bib;
  s.alb *= 1.0 - fill;
  s.emit += s.alb*vec3(0.81,0.69,0.58)*fill;
  if(bareSkin > 0.5){ s.spec = 0.25; s.shin = 40.0; s.sheen = 0.2; }

  // the lip's edge runs on past the roll, all the way into the corner
  float edge = smoothstep(0.016, 0.006, gE)*smoothstep(-0.008, -0.002, gE)*(1.0 - gSeal)*smoothstep(-0.012, -0.002, gFace)*(1.0 - step(gSnout, 0.0));
  edge = max(edge, smoothstep(0.012, 0.004, gHoleD)*smoothstep(-0.001, -0.006, gFace)*(1.0 - gSeal));   // the sliver of carve wall outside the aperture
  edge *= 1.0 - gWBody;                                      // a paw is never mouth paint
  if(edge > 0.0){
    s.alb = mix(s.alb, C_LIPIN, edge); s.fur *= 1.0 - edge; s.sheen *= 1.0 - edge;
  }
  // the upper lip: a fine bare edge along the pads, as thick as the mouth is open
  float lipTop = smoothstep(0.020, 0.008, gEU)*smoothstep(-0.003, 0.002, gEU)*smoothstep(0.045, 0.02, gSnout)*smoothstep(1.06, 0.92, abs(q.x)/gW)*(1.0 - gSeal);
  s.alb = mix(s.alb, C_LIP*0.85, lipTop); s.fur *= 1.0 - lipTop;
  // a blush where the short fur gives out under the lower lip
  s.alb = mix(s.alb, C_PINK, 0.32*(1.0 - gSeal)*smoothstep(0.030, 0.0, gRing)*step(gVM, 0.0)*step(0.0, gE));
  // the bare lower lip
  float lipM = smoothstep(0.0055, 0.0010, gRing)*(1.0 - gSeal);
  if(lipM > 0.0){
    vec3 c = mix(C_LIPIN, C_LIP, smoothstep(0.05, 0.75, gE/max(gLipH, 1e-3)));
    c = mix(c, mix(C_WHITE, C_PINK, 0.20), gRndU*smoothstep(0.20, 1.05, gE/max(gLipH, 1e-3)));    // round, its outer half is the muzzle's skin
    s.alb = mix(s.alb, c, lipM);
    s.fur *= 1.0 - lipM; s.sheen = mix(s.sheen, 0.2, lipM); s.spec = mix(s.spec, 0.32, lipM); s.shin = mix(s.shin, 46.0, lipM);
    s.wrap = mix(s.wrap, 0.45, lipM); s.occ = mix(s.occ, 1.0, lipM); s.n = normalize(mix(s.n, n, lipM));
    s.emit = mix(s.emit, c*0.10, lipM);
  }
  float teethDark = max(gSeal, 0.92*gDark);
  if(inMouth){
    vec3 c = mix(C_DEEP, C_CAVE, smoothstep(-0.075, -0.004, gE));   // by screen place, not depth: the far surfaces are many and creased
    c = mix(c, C_DEEP*0.7, gDark);
    s.alb = c*0.05;
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.0; s.shin = 30.0; s.wrap = 0.3; s.occ = 1.0; s.trans = 0.0;
    s.n = n;
    s.emit = c*0.56;                    // lit by emit: the engine's AO would draw hard edges in here, and the far wall has creases
  }
  if(isTongue){
    vec3 c = vec3(0.74,0.25,0.28)*(0.92 + 0.16*vnoise3(gQJ*40.0));
    c *= mix(0.30, 1.0, smoothstep(0.30, 0.64, gQ.z));              // into the throat
    c *= mix(1.0, 0.45, smoothstep(0.06, 0.13, abs(gQ.x)));         // and off to the sides
    c *= 0.62 + 0.46*clamp(dot(n, normalize(vec3(-0.25,0.78,0.57))), 0.0, 1.0);   // a soft form: lit from above
    c = mix(c, C_DEEP, gDark*0.85);
    s.alb = c*0.35;
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.22; s.shin = 30.0; s.wrap = 0.35; s.occ = 1.0; s.trans = 0.0; s.n = n;
    s.emit = c*0.48;
  }
  if(isTeeth){
    vec3 c = vec3(0.97,0.94,0.86);
    c *= mix(0.70, 1.0, smoothstep(0.0, 0.022, -gEU))*(1.0 - 0.55*gRndU);    // the lip's shade at the root, the shade of a round mouth
    c = mix(c, C_DEEP, teethDark);
    s.alb = c*0.60;
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.5; s.shin = 70.0; s.wrap = 0.30; s.occ = 1.0; s.trans = 0.0; s.n = n;
    s.emit = c*0.50;
  }
  // while the mouth is closed nothing inside may read: teeth, tongue and cavity are all lip-dark
  if((inMouth || isTeeth || isTongue) && gSeal > 0.0){
    s.alb = mix(s.alb, C_LINE*0.40, gSeal);     // the line reads darker than its paint in full sun
    s.emit *= 1.0 - gSeal;
    s.spec *= 1.0 - gSeal;
    s.sheen *= 1.0 - gSeal;
  }
  if(isWhisker){
    s.alb = vec3(0.92,0.88,0.80);
    s.fur = 0.0; s.sheen = 0.1; s.spec = 0.4; s.shin = 60.0; s.wrap = 0.6; s.occ = 1.0; s.trans = 0.0; s.n = n;
    s.emit = s.alb*0.28;
  }
  if(isNose){
    float fine = vnoise3(q*120.0);
    vec3 nc = mix(vec3(0.74,0.26,0.31), vec3(0.93,0.50,0.52), smoothstep(-0.05,0.03,q.y - NOSEP.y));
    nc = mix(nc, vec3(0.20,0.05,0.07), smoothstep(0.018,0.004,length((vec3(abs(q.x),q.yz) - NOSEP - vec3(0.050,0.004 + 0.016*gNostril,0.016))*vec3(1.0,1.8,1.0)) - 0.009));
    s.alb = nc*(0.92 + 0.16*fine);
    s.n = normalize(n + 0.012*(vec3(vnoise3(q*90.0), vnoise3(q*90.0+3.0), vnoise3(q*90.0+7.0)) - 0.5));
    s.fur = 0.0; s.sheen = 0.25; s.spec = 0.6; s.shin = 70.0; s.wrap = 0.35; s.occ = 1.0; s.trans = 0.0; s.mirror = 0.25;
    s.emit = s.alb*0.10;
  }
  if(isEye){
    // gaze space
    vec3 toCam = rootGaze(gHR*(uCamPos - gHO)) + NECK - vec3(sign(q.x)*EYEC.x, EYEC.y, EYEC.z);
    vec3 eyeF = normalize(normalize(toCam) + vec3(gGaze.x*0.30, gGaze.y*0.24, 0.0));
    vec3 ex = normalize(cross(vec3(0,1,0), eyeF)), ey = cross(eyeF, ex);
    mat3 toE = transpose(mat3(ex, ey, eyeF));
    vec3 ee = vec3(q.x - sign(q.x)*EYEC.x, q.y - EYEC.y, q.z - EYEC.z)/EYER;
    vec3 e = toE*ee;
    vec3 v = toE*(gHR*rd);
    float irisM;
    // big, dark, glossy: nearly all iris, a warm brown ember low in it
    vec3 c = eyeColour(e, v, vec3(0.070,0.034,0.016), vec3(0.030,0.015,0.009), vec3(0.16,0.11,0.09), 0.97, 0.62, irisM);
    c += vec3(0.12,0.055,0.022)*smoothstep(0.10,0.95,-e.y*0.9 + e.x*0.15)*irisM*smoothstep(0.0,0.3,e.z);
    s.alb = c;
    // shadow cast by the upper lid
    s.occ = mix(0.66, 1.0, smoothstep(0.75, 0.15, ee.y));
    s.fur = 0.0; s.sheen = 0.0; s.spec = 1.5; s.shin = 900.0; s.wrap = 0.0; s.trans = 0.0; s.mirror = 0.9;
    s.n = n;
    // painted catch-lights, locked to the light rig: one big soft window, one small kicker
    vec3 r2 = reflect(rd, n);
    float k1 = smoothstep(0.950, 0.966, dot(r2, normalize(uSunDir + vec3(-0.05,0.2,0.3))));
    float k2 = smoothstep(0.984, 0.991, dot(r2, normalize(vec3(-0.5,-0.05,0.85))));
    s.emit = vec3(1.0,0.97,0.92)*(4.0*k1 + 1.6*k2);
  }
}

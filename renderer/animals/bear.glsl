// ---------------------------------------------------------------------------
//  animals/bear.glsl — the bear.  Head space: origin at the skull centre, +x screen right,
//  +y up, +z toward the camera (the nose).  Units: skull radius ~0.65.
//  See docs/ANIMAL_GUIDE.md for the API.
//
//  The mouth reads the 11 lip_* columns (docs/V2.md).  It lives on one curve, the APERTURE: a lens over (x, v),
//  where v is the height above the lip line, a wide U across the FRONT of a short round muzzle.  An upper arc
//  rises gAu, a lower arc falls gAl, both through the two corners, so a corner is the end of one curve and cannot
//  come apart.  Upper muzzle and lower jaw are two warped copies of the rest muzzle, glued along the line past the
//  corners: there is no strut, no web and no wedge running back to the hinge.  The skin follows the arcs with fields
//  that fade away from the lip, the hinge carries chin, lower teeth and tongue, and the lens is carved through both.
//     e < 0 inside the opening; e = 0 on the inner edge of the lips; e > 0 out onto the skin.
//  All lip columns at 0: the lens has no height and the carve is closed.
// ---------------------------------------------------------------------------
const vec3  NECK   = vec3(0.0,-0.58,-0.22);    // head rotates about this
const vec3  HINGE  = vec3(0.0,-0.15, 0.08);    // jaw hinge, far back: chin, lower teeth and tongue swing about it
const vec3  NOSEP  = vec3(0.0,-0.030, 0.872);  // nose ball centre
const vec3  POLE   = vec3(0.0,-0.09, 0.98);    // fur flows away from here
const vec3  EYEC   = vec3(0.298,0.122,0.432);  // mirrored
const float EYER   = 0.141;
const float EYETILT= -0.07;                    // outer corners down: kind, a little shy
const float YAW0   = 0.30;                     // resting 3/4 turn
const vec3  EARB   = vec3(0.605,0.505,-0.12);  // ear base (mirrored)
const float MAXJAW = 0.49;                     // radians at teeth_gap=1
const float JAWDROP= 0.055;                    // and the jaw sinks a little: weight
const float MW     = 0.245;                    // half width of the resting mouth, x
const float LW     = 0.30;                     // the U of the lip line is drawn over this half width: the corners stop short of its ends
const float LINEY  = -0.290;                   // lip line at the centre
const float BENDT  = 0.100;                    // the U of the tooth tips: rigid, they do not follow the lips
const float LU     = 0.040;                    // upper tooth length
const float LL     = 0.034;                    // lower tooth length
const vec3  JAW_A = vec3(0.0,-0.25,0.16), JAW_B = vec3(0.0,-0.325,0.560);
const float JAW_RA = 0.345, JAW_RB = 0.235;
const float F_K = 14.0, F_L = 2.8;             // tuft density across / along the flow

// ---- per-fragment pose (filled by charInit)
mat3  gHR;          // world -> head
mat3  gBodyR;       // world -> head at rest (body fur + paint frame)
vec3  gHO;          // head-space origin offset
float gJawA, gDrop, gJF;   // jaw angle, sink, how far the hinge alone carries the lip centre down
float gPress, gTuck, gSp, gPk, gPro, gTgTip, gTgGap, gTgGapZ, gTgBack, gSmile, gCheekP, gNostril;
float gBlinkL, gBlinkR;
vec2  gGaze;
float gBrowL, gBrowR, gKnit;
mat3  gEarRL, gEarRR;
float gBreath;
float gTgOut, gNoseW, gEyeW, gHGR;   // tongue out 0..1, nose twitch (head units), eyes wide 0..1, the muzzle guard's radius
mat3  gBM; float gBounce;
mat2  gJawM, gBrowM;                 // per-frame rotations, made once: the hinge, the brows' knit
vec3 jawQ(vec3 q){ vec3 r = q - HINGE; r.yz = gJawM*r.yz; return r + HINGE; }            // world -> the frame the body carries (yaw, lean, pitch about the hips), the bounce
const vec3 BPIV = vec3(0.0,-2.05,-0.30);   // the body leans and turns about the hips
const vec3 EPIV = vec3(0.0,-0.10,0.0);     // an ear turns and flicks about its base, ear frame
const vec3 EPIVP = vec3(0.0,-0.45,0.0);    // and pitches about a point deep under it: forward/back over the dome
mat3  gEarPL, gEarPR;
const float HG_SKIP = 0.36;                // past this far from the head bound, the head is skipped
vec3 bodyIn(vec3 p){ return gBM*(p - BPIV - vec3(0.0, gBounce, 0.0)) + BPIV; }
// the aperture
float gW, gAu, gAl, gRu, gCu, gRl, gCl;   // half width, rise, fall, the two arcs: radius, centre's distance from the chord
float gWO, gPeel, gJC, gSmX, gSmY;        // width that has parted, 1 while the lips are stuck past it; jaw 0..1 for the skin that follows it; smirk: corners' shift and tilt
float gYC, gBend, gBackA, gRnd;           // chord height (the corners ride down), the U, its rise behind the mouth, 1 = a round O
float gRK, gCMin, gHole, gOpenK;         // corner rounding, thickness left at a corner, carve margin, carve open 0..1
float gHU, gHL, gTU, gTL;                 // lip height (the band on the skin) and fullness (how far it stands off)
float gRing, gRH, gRT, gCone, gPlug;      // a pucker's lips are one soft ring around the opening
float gFill, gGx, gPushU, gPushL, gLS;           // gather in x, forward travel of each lip, step scale of the warped jaw
float gSeal, gDark, gShow, gOpen, gTunnel;// 1 sealed | 1 dark slit | 0 teeth wait behind the lip .. 1 out | 0..1 open | 1 deep behind a pucker
float gTipU, gTipL;                       // tooth tips: over the tooth line (upper), under it in the jaw frame (lower)

mat3 rotX(float a){ float c=cos(a),s=sin(a); return mat3(1,0,0, 0,c,s, 0,-s,c); }
mat3 rotY(float a){ float c=cos(a),s=sin(a); return mat3(c,0,-s, 0,1,0, s,0,c); }
mat3 rotZ(float a){ float c=cos(a),s=sin(a); return mat3(c,s,0, -s,c,0, 0,0,1); }

float sq(float x){ return x*x; }
float sat(float x){ return clamp(x, 0.0, 1.0); }
float lipBell(float x){ x = sat(1.0 - x*x); return x*x; }

// ---- body + arms, round three.  The bear sits.  Everything here is in the BODY frame: world turned by BYAW about y,
// mirrored per side so an arm always lies at x > 0.  The rig is a pure function of the columns: no state, no history.
// One arm per charEval: no camera ray crosses x = 0 and the rig keeps each paw on its own side of that plane, so a
// point only ever needs the arm on its side of it.  Shadow rays do cross, and charShadow has both.
const float BYAW = 0.14;                       // the whole body turns a little the way the head does
const float UARM = 0.72, FARM = 0.70;          // bones
const float PAWS = 0.86;                       // the paw is drawn at 1 and scaled
const vec3  SH0  = vec3(0.86,-1.04,-0.26);     // shoulder, rest
const vec3  WR0  = vec3(0.80,-1.46, 0.72);     // wrist, rest: the paw lies on the knee, digits draped over it
const vec3  BELLY_C = vec3(0.0,-1.78,-0.20), BELLY_R = vec3(1.02,0.92,0.92);
const vec3  CHEST_C = vec3(0.0,-1.22,-0.18), CHEST_R = vec3(0.82,0.68,0.66);
const vec3  THIGH_A = vec3(0.60,-2.26,-0.06), THIGH_B = vec3(0.86,-2.06,0.96);
const float THIGH_RA = 0.46, THIGH_RB = 0.40;
const vec3  HEADC = vec3(0.0,-0.04,0.00), SNOUTC = vec3(0.0,-0.18,0.62);   // the head's mass and its muzzle, head space
const vec4  DIG[5] = vec4[5](                  // the digits, paw frame: knuckle x, y, bone length, radius.  +x is the thumb side
  vec4( 0.212,0.335,0.112,0.060),
  vec4( 0.112,0.408,0.132,0.064),
  vec4( 0.002,0.432,0.140,0.066),
  vec4(-0.110,0.412,0.134,0.064),
  vec4(-0.208,0.345,0.116,0.060));

struct ArmPose {
  vec3 S, E, W;              // shoulder, elbow, wrist
  vec3 U, F;                 // upper arm and forearm directions
  mat3 MU, MF, MP;           // a point on this bone -> where that skin lies at rest (each about its own joint)
  mat3 P;                    // body -> paw: +y along the digits, +z the back of the paw, palm at -z, +x the thumb side
  vec3 D;                    // the paw's axis
  vec3 off;                  // how far the shoulder followed the hand
  vec4 cT, cI, cO;           // curl of the digits' two bones (cos, sin, cos, sin): the thumb, the index, the other three
  float fan;                 // their spread
  float th;                  // the thumb swung out to the side of the paw, 0..1
};
ArmPose gArmL, gArmR, gArmRest;
vec3 gWSL, gWEL, gWWL, gWPL, gWSR, gWER, gWWR, gWPR;   // both arms in the world, for charShadow

// any column value lands on -1.15..1.15, and the working range is left alone
float softLim(float c){ float a = abs(c)/1.15; return sign(c)*1.15*a/pow(1.0 + a*a*a*a*a*a, 1.0/6.0); }
// one column -> an offset: an unequal reach either way, with no kink in the speed through rest
float reach(float c, float up, float dn){ const float E = 0.12; return 0.5*(up + dn)*c + 0.5*(up - dn)*(sqrt(c*c + E*E) - E); }
vec3 pushEll(vec3 p, vec3 c, vec3 r){
  vec3 v = p - c;
  float k = length(v/r);
  return c + v*smax(k, 1.0, 0.08)/max(k, 1e-3);
}
vec3 pushBall(vec3 p, vec3 c, float r){
  vec3 v = p - c;
  float k = length(v);
  return c + v*smax(k, r, 0.08)/max(k, 1e-3);
}
vec3 toWorld(vec3 b){ b.xz = rot(-BYAW)*b.xz; return b; }

// keeps point p, of radius r, out of everything the arm can run into.  hd, sn: the head's skull and muzzle, this frame
vec3 clearOf(vec3 p, float r, vec3 hd, vec3 sn, vec2 hr){
  p = pushEll(p, CHEST_C, CHEST_R + r);
  p = pushEll(p, BELLY_C, BELLY_R + r);
  vec3 tt = THIGH_B - THIGH_A;
  float th = sat(dot(p - THIGH_A, tt)/dot(tt, tt));
  p = pushBall(p, THIGH_A + tt*th, mix(THIGH_RA, THIGH_RB, th) + r);
  p = pushBall(p, hd, hr.x + r);
  p = pushBall(p, sn, hr.y + r);
  return p;
}
// the arm through wrist W: the shoulder follows, two bones, then the paw
float gFrontW;      // the hand being solved is brought across in front of the face (armSolve sets it)
ArmPose armBones(vec3 W, float open, float tn, float tw, vec3 hd, vec3 dg){
  ArmPose a;
  // the shoulder follows the hand: up with it, forward with it, a little out.  dg = (thumb, point, shrug)
  vec3 dw = (W - WR0)/vec3(1.22, 1.16, 0.96);
  a.off = vec3(0.05*dw.x + 0.02*dg.z, 0.20*dw.y - 0.05*dw.y*dw.y + 0.15*dg.z, 0.12*dw.z + tw);
  a.S = SH0 + a.off;

  // the arm's reach, soft at the end so the elbow never snaps straight
  vec3 d = W - a.S;
  float L = length(d), Lm = 0.965*(UARM + FARM), L0 = 0.80*Lm;
  float Ls = L > L0 ? L0 + (Lm - L0)*(1.0 - exp(-(L - L0)/(Lm - L0))) : L;
  Ls = max(Ls, 0.50);
  vec3 dir = d/max(L, 1e-3);
  W = a.S + dir*Ls;
  a.W = W;
  // two bones.  The elbow hangs: down and out and a little back, further out as the hand comes in across the body
  float along = (UARM*UARM - FARM*FARM + Ls*Ls)/(2.0*Ls);
  float h = sqrt(max(UARM*UARM - along*along, 1e-4));
  float in_ = smoothstep(0.95, 0.55, W.x);
  vec3 pole = normalize(vec3(0.60 + 0.45*in_, -0.72, -0.30));
  vec3 pp = normalize(pole - dir*dot(pole, dir));
  a.E = a.S + dir*along + pp*h;
  a.U = (a.E - a.S)/UARM; a.F = (W - a.E)/FARM;

  // the paw.  Neutral: thumb toward the biceps, palm in.  (F, T, P) right handed
  vec3 T = -a.U + a.F*dot(a.U, a.F);
  T = normalize(T + 0.25*(vec3(0.0,1.0,0.0) - a.F*a.F.y));
  vec3 P = cross(a.F, T);
  // relaxed it rolls palm down on the thigh and comes up palm in; the column goes all the way to flat either way
  float base = mix(1.30, 0.55, smoothstep(-1.50, -0.90, W.y));
  float ph = mix(base, -sign(tn)*0.5*PI, abs(tn));
  // on a raised forearm, turn +1 is the palm to the camera, -1 away from it
  vec3 gc = vec3(0.0, 0.0, sign(tn)) - a.F*a.F.z*sign(tn);
  ph += abs(tn)*smoothstep(0.30, 0.75, a.F.y)*smoothstep(0.05, 0.25, length(gc))*(mod(atan(-dot(gc, T), dot(gc, P)) - ph + PI, 2.0*PI) - PI);
  // near the face or across the chest the palm turns to the surface it touches
  vec3 fc = W + 0.30*a.F - hd;
  float wFace = smoothstep(1.30, 0.95, length(fc))*smoothstep(-0.20, 0.10, fc.z)*gFrontW;   // only a hand brought in front of the face: a wave beside the head keeps its turn
  vec3 cc = W - CHEST_C;
  float wChest = smoothstep(0.62, 0.40, W.x)*smoothstep(-1.45, -1.20, W.y)*smoothstep(-0.60, -0.80, W.y)*smoothstep(1.45, 1.05, length(cc/CHEST_R))*(1.0 - wFace);
  float kT = 1.0 - abs(tn);
  vec3 gq = -normalize(fc);
  gq = gq - a.F*dot(gq, a.F);
  ph += wFace*kT*smoothstep(0.05, 0.25, length(gq))*(mod(atan(-dot(gq, T), dot(gq, P)) - ph + PI, 2.0*PI) - PI);
  gq = -normalize(cc/(CHEST_R*CHEST_R));
  gq = gq - a.F*dot(gq, a.F);
  ph += wChest*kT*smoothstep(0.05, 0.25, length(gq))*(mod(atan(-dot(gq, T), dot(gq, P)) - ph + PI, 2.0*PI) - PI);
  // a fist with the thumb out rolls the thumb side up: thumbs up
  float thU = dg.x*(1.0 - open);
  float phU = atan(P.y, T.y);
  ph += thU*(mod(phU - ph + PI, 2.0*PI) - PI);
  float cp = cos(ph), sp = sin(ph);
  vec3 Pn = P*cp - T*sp;
  // the wrist: a relaxed paw hangs off it, an open one is carried
  float straight = max(dg.x, dg.y)*(1.0 - open);       // a thumb up or a pointing digit holds the wrist straight
  float fl = mix(mix(0.34, -0.10, open), 0.0, straight);
  vec3 D = a.F*cos(fl) + Pn*sin(fl);
  vec3 n = Pn*cos(fl) - a.F*sin(fl);
  D = normalize(D + vec3(0.0,-0.30,0.0)*(1.0 - 0.75*open)*(1.0 - straight));     // and it is heavy
  // a pointing paw lines up with the arm's reach, held nearly level: the digit points where the arm goes
  vec3 pd = W - a.S; pd.y *= 0.25;
  D = normalize(mix(D, normalize(pd), 0.75*dg.y*(1.0 - open)));
  n = normalize(n - D*dot(n, D));
  // an open paw swings out off the forearm, to the little-finger side, as a hand does when it offers something
  vec3 X0 = cross(D, -n);
  float dev = 0.32*open;
  D = normalize(D*cos(dev) - X0*sin(dev)*sign(dot(X0, vec3(-1.0,0.0,0.0))));
  n = normalize(n - D*dot(n, D));
  a.D = D;
  vec3 B = -n, X = cross(D, B);
  a.P = transpose(mat3(X, D, B));
  return a;
}
// digits: curled in a relaxed loose fist, flat and fanned open.  A thumb or a pointing digit closes the rest into a fist
void armDigits(inout ArmPose a, float open, vec3 dg){
  float c = 1.0 - open;
  float fist = max(dg.x, dg.y)*c;
  float a1 = mix(mix(0.06, 0.46, c), 1.20, fist), a2 = mix(mix(0.12, 1.02, c), 1.50, fist);
  a.cO = vec4(cos(a1), sin(a1), cos(a2), sin(a2));
  float i1 = mix(a1, 0.02, dg.y), i2 = mix(a2, 0.04, dg.y);
  a.cI = vec4(cos(i1), sin(i1), cos(i2), sin(i2));
  float t1 = mix(a1, -0.10, dg.x), t2 = mix(a2, 0.02, dg.x);
  a.cT = vec4(cos(t1), sin(t1), cos(t2), sin(t2));
  a.fan = mix(0.30, 1.75, open)*(1.0 - 0.6*fist);
  a.th = dg.x;
}
// hc = the three hand columns, side = -1 screen left, +1 screen right, hd and sn = skull and muzzle in this mirrored frame
// The contract target: an offset from the rest wrist in arm lengths (UARM + FARM), x outward, y up, z toward the camera.
// The rest wrist lies forward on the knee, so as the hand rises its forward baseline comes back toward the shoulder
ArmPose armSolve(float side, vec3 hc, float open, float turn, vec3 hd, vec3 sn, float tw, vec3 dg, vec3 ey, vec3 ch){
  const float AL = UARM + FARM;
  float cx = clamp(hc.x, -1.2, 1.4), cy = clamp(hc.y, -1.0, 1.3), cz = clamp(hc.z, -1.0, 1.4);
  float tn = clamp(turn, -1.0, 1.0);
  open = sat(open);
  float ry = sat(cy);
  vec3 W = WR0 + AL*vec3(cx*(cx > 0.0 ? 0.80 : 0.74), cy*(cy > 0.0 ? 0.96 : 0.30), 0.62*cz);
  W.z -= 0.56*ry*(1.0 - 0.35*sat(cz)) + 0.50*sat(cx)*(1.0 - 0.5*sat(cz));
  W.x += 0.16*ry*sat(cx + 0.35)*sat(1.0 - cz);   // a raised hand comes up beside the head, not in front of it
  // a hand brought up across in front of the face goes to the eye on its side (cover eyes), or under the chin; there the
  // head's clearance shrinks to the face itself, so the paw touches it
  float wFront = smoothstep(0.10, -0.40, cx)*smoothstep(0.05, 0.35, cz);
  gFrontW = wFront;
  float wEye = smoothstep(0.80, 0.97, cy)*wFront;
  float wChin = smoothstep(0.55, 0.70, cy)*(1.0 - smoothstep(0.82, 0.95, cy))*wFront;
  W = mix(W, ey + 0.10*normalize(ey - hd) + vec3(0.0,-0.25,0.06), wEye);
  W = mix(W, ch + vec3(0.0,-0.36,0.02), wChin);
  W += vec3(0.32, 0.0, 0.12)*4.0*wEye*wChin;     // between the chin and the eye the paw goes round the muzzle, not through it
  vec2 hr = vec2(mix(0.92, 0.66, smoothstep(0.55, 0.80, cy)*wFront), mix(0.47, 0.30, wChin));
  W = clearOf(W, 0.13, hd, sn, hr);
  // pose the arm, see where the rest of it landed, move the wrist by what that needs, pose it again
  float cb = cos(BYAW), sb = sin(BYAW);
  ArmPose a;
  for(int k=0;k<3;k++){
    a = armBones(W, open, tn, tw, hd, dg);
    if(k == 2){ armDigits(a, open, dg); break; }
    vec3 dW = vec3(0.0);
    for(int i=0;i<4;i++){
      vec3 Q = i == 0 ? a.W + 0.26*a.D : (i == 1 ? a.W + 0.50*a.D : (i == 2 ? a.W + 0.60*a.D : 0.5*(a.E + a.W)));
      float r = i == 0 ? 0.18 : (i == 1 ? 0.15 : (i == 2 ? 0.12 : 0.20));
      Q += dW;
      vec3 Q2 = clearOf(Q, r, hd, sn, hr);
      // the paw stays on its own side of x = 0 in the world: two paws meet palm to palm there
      if(i < 3) Q2.x = smax(Q2.x, (0.135 - side*sb*Q2.z)/cb, 0.04);
      dW += Q2 - Q;
    }
    W = a.W + dW;
  }
  return a;
}
// a bone's frame: its axis, and the hinge the two bones share
mat3 boneFrame(vec3 A, vec3 N){ return mat3(N, cross(A, N), A); }
void armFinish(inout ArmPose a, ArmPose r){
  vec3 N  = normalize(cross(a.U, a.F));
  vec3 N0 = normalize(cross(r.U, r.F));
  a.MU = boneFrame(r.U, N0)*transpose(boneFrame(a.U, N));
  a.MF = boneFrame(r.F, N0)*transpose(boneFrame(a.F, N));
  a.MP = transpose(r.P)*a.P;
}
void armsInit(){
  vec3 hw = transpose(gHR)*(HEADC - NECK) + gHO;     // the head's centre, world -> body frame
  hw.xz = rot(BYAW)*hw.xz;
  vec3 sw = transpose(gHR)*(SNOUTC - NECK) + gHO;
  sw.xz = rot(BYAW)*sw.xz;
  vec3 eyR = transpose(gHR)*(EYEC - NECK) + gHO;                       // the screen-right eye, and the screen-left one
  eyR.xz = rot(BYAW)*eyR.xz;
  vec3 eyL = transpose(gHR)*(vec3(-EYEC.x, EYEC.yz) - NECK) + gHO;
  eyL.xz = rot(BYAW)*eyL.xz;
  vec3 chn = transpose(gHR)*(vec3(0.0,-0.50,0.56) - NECK) + gHO;          // under the chin
  chn.xz = rot(BYAW)*chn.xz;
  // a hand going forward turns the chest with it, the other shoulder going back a little
  float tw = 0.055*(softLim(uP[P_HAND_R_Z]) - softLim(uP[P_HAND_L_Z])) + 0.025*(softLim(uP[P_HAND_R_X]) - softLim(uP[P_HAND_L_X]));
  vec3 dgL = vec3(sat(uP[P_THUMB_L]), sat(uP[P_POINT_L]), sat(uP[P_SHOULDER_L]));
  vec3 dgR = vec3(sat(uP[P_THUMB_R]), sat(uP[P_POINT_R]), sat(uP[P_SHOULDER_R]));
  gArmL = armSolve(-1.0, vec3(uP[P_HAND_L_X], uP[P_HAND_L_Y], uP[P_HAND_L_Z]), uP[P_HAND_L_OPEN], uP[P_HAND_L_TURN], vec3(-hw.x, hw.yz), vec3(-sw.x, sw.yz), -tw, dgL, vec3(-eyL.x, eyL.yz), vec3(-chn.x, chn.yz));
  gArmR = armSolve( 1.0, vec3(uP[P_HAND_R_X], uP[P_HAND_R_Y], uP[P_HAND_R_Z]), uP[P_HAND_R_OPEN], uP[P_HAND_R_TURN], hw, sw, tw, dgR, eyR, chn);
  ArmPose r = armSolve(1.0, vec3(0.0), 0.0, 0.0, vec3(0.0,0.0,-9.0), vec3(0.0,0.0,-9.0), 0.0, vec3(0.0), vec3(0.0), vec3(0.0));
  armFinish(gArmL, r); armFinish(gArmR, r);
  gArmRest = r;
  vec3 m = vec3(-1.0,1.0,1.0);
  gWSL = toWorld(m*gArmL.S); gWEL = toWorld(m*gArmL.E); gWWL = toWorld(m*gArmL.W); gWPL = toWorld(m*(gArmL.W + 0.20*gArmL.D));
  gWSR = toWorld(gArmR.S);   gWER = toWorld(gArmR.E);   gWWR = toWorld(gArmR.W);   gWPR = toWorld(gArmR.W + 0.20*gArmR.D);
}

void charInit(){
  // yaw>0 toward screen right, pitch>0 nose down, roll>0 crown toward screen right (docs/AUDIO.md)
  mat3 F = rotY(YAW0 + uP[P_HEAD_YAW])*rotX(uP[P_HEAD_PITCH])*rotZ(-uP[P_HEAD_ROLL]);
  gHR = transpose(F);
  gBodyR = transpose(rotY(YAW0));
  gHO = NECK + vec3(uP[P_HEAD_X], uP[P_HEAD_Y], uP[P_HEAD_Z]);
  // the body carries the head and the arms: yaw about the vertical, lean and pitch about the hips, bounce up/down
  gBM = rotX(-0.15*clamp(uP[P_BODY_PITCH], -1.0, 1.0))*rotZ(0.12*clamp(uP[P_BODY_LEAN], -1.0, 1.0))*rotY(-0.35*clamp(uP[P_BODY_YAW], -1.0, 1.0));
  gBounce = 0.10*clamp(uP[P_BODY_BOUNCE], -1.0, 1.0);
  // a commanded mouth, and a tongue that is out, open the mouth themselves
  float mo = sat(uP[P_MOUTH_OPEN]);
  gTgOut = sat(uP[P_TONGUE_OUT]);
  gNoseW = 0.036*clamp(uP[P_NOSE_WIGGLE], -1.0, 1.0);
  gEyeW = sat(uP[P_EYE_WIDE]);
  gPress  = sat(uP[P_LIP_PRESS])*(1.0 - gTgOut);
  float keep = 1.0 - gPress;
  float open = max(sat(uP[P_LIP_OPEN]), max(mo, 0.34*gTgOut))*keep;
  float up   = sat(uP[P_LIP_UPPER])*keep;
  float lo   = sat(uP[P_LIP_LOWER])*keep;
  float wid  = clamp(uP[P_LIP_WIDTH], -1.0, 1.0);
  gPro       = sat(uP[P_LIP_PROTRUDE]);
  gTuck      = sat(uP[P_LIP_TUCK])*keep;
  float gap  = max(clamp(uP[P_TEETH_GAP], 0.0, 1.2), max(mo, 0.30*gTgOut));
  float tip  = sat(uP[P_TONGUE_TIP]);
  tip = tip*(2.0 - tip);                // a tongue gets there early and waits: short d t n l still touch the ridge
  float dent = sat(uP[P_TONGUE_DENTAL]);
  gTgBack    = sat(uP[P_TONGUE_BACK]);
  gSmile  = sat(uP[P_SMILE]);
  gCheekP = sat(uP[P_CHEEK]);
  gNostril= sat(uP[P_NOSTRIL]);
  gPk = max(-wid, 0.0); gSp = max(wid, 0.0);
  float smirk = clamp(uP[P_SMIRK], -1.0, 1.0);
  gSmX = 0.050*smirk;         // the corners slide toward that side of the muzzle
  gSmY = 0.030*smirk;         // and that corner rides up, the other down

  // the hinge: chin, lower teeth, tongue
  gJawA = MAXJAW*gap;
  gJawM = rot(-gJawA);
  gDrop = JAWDROP*gap;
  vec3 c = vec3(0.0, LINEY, 0.84) - HINGE;
  c.yz = rot(gJawA)*c.yz;
  gJF = LINEY - (c.y + HINGE.y - gDrop);
  gLS = 1.0/(1.0 + 1.6*gJF);
  gJC = sat(gJF/0.40);

  // the corners in x
  gW = MW*(1.0 + 0.20*gSp - 0.62*gPk + 0.04*gSmile)*(1.0 - 0.10*open);
  // a hard pucker closes in on a small round whose size is the air let through: the lips fill the rest of the mouth
  gW = mix(gW, 0.012 + 0.13*(1.0 - exp(-2.2*open)), smoothstep(0.40, 0.90, gPk));

  // the opening.  A round O needs height as well as narrowness, so the pucker pulls both heights to the width
  float au = 0.030*open + 0.085*up;
  float al = 0.395*open + 0.095*lo;
  gRnd = gPk*sqrt(gPk)*smoothstep(0.0, 0.10, open + 0.5*(up + lo));
  au = mix(au, 0.62*gW*(1.0 + 0.3*open), gRnd);
  al = mix(al, 1.08*gW*(1.0 + 0.5*open), gRnd);
  au += 0.030*gTuck;
  al *= 1.0 - 0.90*gTuck;
  float amt = au + al;
  // the corners ride down with the lower lip, about a quarter of its travel
  float drop = 0.22*al*(1.0 - gRnd);
  gYC = -drop;
  gAu = max(au + drop, 0.0009);
  gAl = max(al - drop, 0.0009);
  // lips let go of each other from the middle out: while a press lets go the corners are still stuck
  gPeel = smoothstep(0.0, 0.25, gPress);
  gWO = gW*(1.0 - 0.62*smoothstep(0.0, 0.75, gPress));
  gRu = 0.5*(gWO*gWO + gAu*gAu)/gAu; gCu = gRu - gAu;
  gRl = 0.5*(gWO*gWO + gAl*gAl)/gAl; gCl = gRl - gAl;
  gRK = min(mix(0.030, 0.070, gRnd), 0.5*amt);

  gOpen = smoothstep(0.0, 0.24, amt);
  gFill = smoothstep(0.0, 0.11, amt);        // at rest the lip sits back under the pad, as it always has
  gBend = (0.085 + gSmile*0.075 + gSp*0.050)*(1.0 - 0.45*gPress)*(1.0 - 0.70*gPk)*(1.0 - 0.45*gOpen);
  gBackA = 0.030 + gSmile*0.05;
  gSeal = 1.0 - smoothstep(0.0, 0.020, amt);
  gDark = 1.0 - smoothstep(0.030, 0.120, amt);
  gShow = smoothstep(0.020 + 0.05*gPk, 0.095 + 0.10*gPk, amt + 0.05*gTuck);
  gOpenK = smoothstep(0.0, 0.020, amt);
  gHole = mix(-0.024, 0.003, gOpenK);
  float mid = smoothstep(0.15, 0.85, gPk);
  gRing = smoothstep(0.35, 0.80, gPk);
  float tt = 1.0 - smoothstep(0.10, 0.45, open);      // 1 = a tight round
  gRH = mix(0.050, 0.080, tt);                        // the ring of lips: height along e
  gRT = mix(0.030, 0.054, tt)*(1.0 - 0.5*gPress);     // and how far it stands off the muzzle
  gCone = 0.016 + 0.030*gPro;                         // the skin rises toward it, so the lips gather forward
  gPlug = gRnd*tt*smoothstep(0.50, 0.90, gPk);        // a tight round ends in the mouth just behind the lips, not a long tube
  gCMin = mix(mix(0.24, 0.62, gOpen) - 0.07*gSp, 0.92, mid);
  // lips have volume: a soft heavy roll at every opening.  Pressed they roll in, so less of them shows, and they
  // stand out further: what was the lip is now bulging behind the seal
  float thin = 1.0 - 0.25*gSp - 0.38*gPress;
  gHU = 0.018*thin*(1.0 + 1.2*gPk)*(1.0 + 0.90*gFill);
  gHL = 0.036*thin*(1.0 + 0.3*gPk)*(1.0 + 0.20*gFill);
  gTU = 0.015*gFill + 0.004*gOpen + 0.020*gPk + 0.008*gPro;
  gTL = 0.017*gFill + 0.004*gOpen + 0.024*gPk + 0.010*gPro;
  gGx = (0.40 - 0.10*gRing)*gPk - 0.13*gSp;
  gPushU = 0.075*gPro + 0.020*gPk;
  gPushL = 0.105*gPro + 0.050*gPk;
  gTunnel = gPk*smoothstep(0.1, 0.9, gPro);
  gHGR = 0.44 + 0.30*gap + 0.55*gAl + 0.12*gTgOut + 0.6*gPushL;

  // teeth are rigid; while the opening is a slit they wait just behind the lip's edge and come out with it, so a
  // small opening is clean and dark and the row never jumps
  vec3 ct = vec3(0.0, LINEY, 0.745) - HINGE;
  ct.yz = rot(gJawA)*ct.yz;
  float jfT = LINEY - (ct.y + HINGE.y - gDrop);           // the hinge's travel at the tooth tips
  gTipU = mix(max(au + 0.012, -0.006), -0.006, gShow);
  gTipL = mix(max(al - jfT + 0.012, 0.012), 0.012, gShow);

  // the tongue's targets, from head space into the jaw frame
  vec3 t1 = vec3(0.0, LINEY + gTipU + 0.040, 0.72) - HINGE + vec3(0.0, gDrop, 0.0);
  t1.yz = rot(-gJawA)*t1.yz;
  gTgTip = tip*(t1.y + HINGE.y + 0.365);
  vec3 t2 = vec3(0.0, LINEY + 0.5*(gTipU - gJF - gTipL), 0.86) - HINGE + vec3(0.0, gDrop, 0.0);
  t2.yz = rot(-gJawA)*t2.yz;
  gTgGap = dent*(t2.y + HINGE.y + 0.365);
  gTgGapZ = dent*max(t2.z + HINGE.z - 0.695, 0.0);

  gBrowM = rot(-0.30 + 0.30*clamp(uP[P_BROW_KNIT], 0.0, 1.0));
  gBlinkL = clamp(uP[P_BLINK_L], 0.0, 1.0); gBlinkR = clamp(uP[P_BLINK_R], 0.0, 1.0);
  gGaze = vec2(uP[P_GAZE_X], uP[P_GAZE_Y]);
  gBrowL = uP[P_BROW_L]; gBrowR = uP[P_BROW_R]; gKnit = clamp(uP[P_BROW_KNIT], 0.0, 1.0);
  // ears: pose + flick about the base.  _l = screen-left.
  // pitch -1 back/droop .. 1 up/forward, turn -1 opening forward/in .. 1 out, flick a quick twitch back.  About the base
  float fL = sat(uP[P_EAR_FLICK_L]), fR = sat(uP[P_EAR_FLICK_R]);
  float eL = clamp(uP[P_EAR_L], -1.0, 1.0), eR = clamp(uP[P_EAR_R], -1.0, 1.0);
  float aL = (eL > 0.0 ? 0.55 : 0.38)*eL - 0.12*fL, aR = (eR > 0.0 ? 0.55 : 0.38)*eR - 0.12*fR;
  gEarRL = rotY(-0.80*clamp(uP[P_EAR_L_TURN], -1.0, 1.0))*rotX(0.30*fL)*rotZ(0.30*fL);
  gEarRR = rotY(-0.80*clamp(uP[P_EAR_R_TURN], -1.0, 1.0))*rotX(0.30*fR)*rotZ(0.30*fR);
  gEarPL = rotX(-aL); gEarPR = rotX(-aR);
  gBreath = uP[P_BREATH];
  armsInit();
}

// Blunt rounded teeth on a dental arch of their own: a half ellipse in plan, half axes arch.xy, centre z arch.z.
// Not a level set of the muzzle: that set folds back behind the fat lip pads, so a strip cut from it comes
// apart into shards.  Each tooth is a rounded block in arch coordinates, whole teeth only, so the row stops on
// a clean end.  x = |x|, w = height above the gum line, st = (0 seam .. 1 end of the row, -1..1 across a tooth, w).
float archRow(float x, float z, float w, vec3 arch, float tw, float n, float thick, float len, float tip, float rc, out vec3 st){
  vec2 e = vec2(x/arch.x, (z - arch.z)/arch.y);
  float R = 0.5*(arch.x + arch.y);
  float a = atan(e.x, e.y)*R/tw;
  float cell = min(floor(a), n - 1.0);
  float c = a - cell - 0.5;
  st = vec3(a/n, 2.0*c, w);
  float l = len*(1.0 - 0.30*cell/n);
  vec3 p = vec3(c*tw, w + l - tip*st.y*st.y - 0.05, (length(e) - 1.0)*R);    // the tip edge arcs up to both sides
  vec3 d = abs(p) - (vec3(0.5*tw, 0.05, thick) - rc);
  return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0) - rc;
}

// outputs of bearEval (parts, for fur + materials)
float gFur, gBare, gMask, gFurLen, gBowl, gWBody;
vec3  gBq, gBodyOff;     // body frame point, and how far the shoulder carried its skin
float gCrease;
float gSelfArm = 0.0;    // charSurface leaves here how much the shaded point is on an arm
vec3 gBodyWorld(vec3 b){ b.xz = rot(-BYAW)*b.xz; return b; }
vec3  gRest;            // rest-pose coordinate (fur + colour stick to it)
vec3  gPaintR;          // and the same without the lip warps: big colour areas must not pinch where the skin gathers
float gE, gEU, gEL, gV, gHoleD, gDD, gLipBand;      // the aperture at this point, the carve, the host, the lip band's height
float gTgX, gTgL;       // across the tongue, -1..1; along it, 0 root .. 1 tip
float gUp, gLo, gEye, gNose, gTeeth, gTongue, gEarD, gBodyD, gHeadD;
vec3  gToothSt;         // archRow's st at the nearer row
float gToothBack;       // and how far its ends melt away: the lower row lies against the pink tongue, dark teeth would be spots
vec3  gQ, gQJ, gQE; float gEarS;

float sdEarShape(vec3 e){
  // e: y along the ear, z out of the ear's face.  A thick little disc.
  float d = sdEllipsoid(e - vec3(0.0,0.020,0.0), vec3(0.200,0.190,0.100));
  float bowl = sdEllipsoid(e - vec3(0.0,0.024,0.150), vec3(0.135,0.128,0.085));
  gBowl = bowl;
  return d;      // the bowl is paint only: a carved dish exposes a hard-edged flat patch under this coat
}

// The arm at body-frame point bs, mirrored so it lies at x > 0.  Fur-covered skin goes out, the bare parts in gPad, gClaw.
float gPad, gClaw, gWArm, gWPaw, gArmLen, gArmK, gArmAlong;
vec3  gArmRestP, gArmT, gPq;
float armEval(vec3 bs, ArmPose a, float torsoD){
  // the arm's bound: far from it (farther than its blend with the body reaches) the limb is stood in for by the bound
  vec3 pc = a.W + transpose(a.P)*vec3(0.0, 0.32*PAWS, -0.04*PAWS);
  float ab = min(min(sdTaper(bs, a.S, a.E, 0.40, 0.33), sdTaper(bs, a.E, a.W, 0.33, 0.31)), length(bs - pc) - 0.58);
  gPad = 1e3; gClaw = 1e3; gWPaw = 0.0; gArmAlong = 0.0; gArmLen = 0.10; gArmK = 0.03; gArmRestP = vec3(0.0); gArmT = vec3(0.0,-1.0,0.0); gPq = vec3(9.0);
  float d = ab;
  // the margin: the shoulder melts into the body over 0.30; away from it the limb lies on the body with a short blend and its coat
  float am = length(bs - a.S) < 0.90 ? 0.36 : 0.22;
  for(int it_=0; it_<1 && (ab < am || ab < torsoD + am); it_++){
  vec3 pu = bs - a.S, pf = bs - a.E;
  float hu = sat(dot(pu, a.U)/UARM), hf = sat(dot(pf, a.F)/FARM);
  float du = length(pu - a.U*hu*UARM) - mix(0.320, 0.250, hu);
  float df = length(pf - a.F*hf*FARM) - (mix(0.250, 0.190, hf) + 0.028*sin(PI*min(hf*1.4, 1.0)));
  vec3 pq = a.P*(bs - a.W)/PAWS;
  gPq = pq;
  float palm = sdEllipsoid(pq - vec3(0.0,0.205,0.0), vec3(0.285,0.300,0.150));
  // heel of the paw, carrying the wrist into the palm
  palm = smin(palm, sdEllipsoid(pq - vec3(0.0,0.060,0.005), vec3(0.235,0.170,0.150)), 0.06);
  float bound = length(pq - vec3(0.0,0.37,-0.05)) - 0.42;
  float dd = 1e3; gPad = 1e3; gClaw = 1e3;
  // a loop stops when all its lanes are done: away from the paw this costs one digit, not five
  for(int i=0;i<5;i++){
    vec4 g = DIG[i];
    vec4 cs = i == 0 ? a.cT : (i == 1 ? a.cI : a.cO);
    float sw = 0.0;
    if(i == 0){ g.xy += a.th*vec2(0.030, -0.110); g.z *= 1.0 + 0.20*a.th; sw = 1.20*a.th; }   // the thumb swings out to the side, straight
    vec3 r = pq - vec3(g.x, g.y, 0.028);
    r.xy = rot(a.fan*g.x + sw)*r.xy;
    vec3 d1 = vec3(0.0, cs.x, -cs.y), d2 = vec3(0.0, cs.z, -cs.w);
    vec3 e1 = d1*g.z, e2 = e1 + d2*g.z*0.86;
    dd = min(dd, min(sdTaper(r, vec3(0.0), e1, g.w, 0.94*g.w), sdTaper(r, e1, e2, 0.94*g.w, 0.80*g.w)));
    // a blunt claw off the end of the digit, curving on past it
    vec3 d3 = normalize(d2 + vec3(0.0,-0.10,-0.42));
    gClaw = min(gClaw, sdTaper(r, e2 + vec3(0.0,0.0,0.012), e2 + d2*0.070 + d3*0.055 + vec3(0.0,0.0,0.012), 0.033, 0.012));
    // the digit's pad, on its palm side
    vec3 c = e2 - d2*0.020 - vec3(0.0, cs.w, cs.z)*0.042;
    gPad = min(gPad, length((r - c)/vec3(1.0,1.15,0.72)) - 0.052);
    if(bound > 0.26) break;
  }
  // past the bound the digits not visited are stood in for by the bound itself; inside it, they were all visited
  if(bound > 0.26){ dd = min(dd, bound); gClaw = min(gClaw, bound); gPad = min(gPad, bound); }
  // the big pad across the ball of the paw
  vec3 mp = pq - vec3(0.0,0.225,-0.138);
  mp.x = abs(mp.x);
  float pad = smin(sdEllipsoid(mp, vec3(0.135,0.105,0.052)), sdEllipsoid(mp - vec3(0.118,-0.012,0.010), vec3(0.078,0.088,0.046)), 0.05);
  gPad = min(gPad, pad);
  float paw = smin(palm, dd, 0.040)*PAWS;
  gPad *= PAWS; gClaw *= PAWS;
  float arm = smin(du, df, 0.14);
  d = smin(arm, paw, 0.10);
  // which bone owns this skin
  float wf = smoothstep(-0.13, 0.13, du - df);
  float wp = smoothstep(-0.09, 0.09, df - paw);
  vec3 rq = mix(a.MU*pu + gArmRest.S, a.MF*pf + gArmRest.E, wf);
  rq = mix(rq, a.MP*(bs - a.W) + gArmRest.W, wp);
  gArmRestP = rq;
  gArmT = mix(mix(a.U, a.F, wf), a.D, wp);
  gWPaw = wp;
  gArmAlong = 0.32*hu + 0.34*wf + 0.34*wp;       // 0 at the shoulder .. 1 at the paw, smooth along the whole limb
  float palmSide = smoothstep(-0.11, 0.02, pq.z + 0.30*smoothstep(0.50, 0.30, pq.y));   // the palm has a short nap, the back a coat
  float digit = smoothstep(0.34, 0.46, pq.y);
  gArmLen = mix(mix(0.120, 0.098, wf), mix(0.062, 0.030, digit)*mix(0.32, 1.0, palmSide)*smoothstep(0.0, 0.05, gPad), wp);
  // only the shoulder melts into the body: lower down an arm is a separate thing lying against it
  gArmK = mix(0.30, 0.035, smoothstep(0.10, 0.62, hu)*1.0)*(1.0 - 0.0*wf);
  gArmK = mix(gArmK, 0.030, wf);
  }
  return d;
}

// A bound of everything the head draws (skull, cheeks, jowls, muzzle, open jaw, tongue out, ears in any pose), head space.
// Exact distances to spheres and a capsule: a lower bound of the head's distance
float headGuard(vec3 q){
  float d = length(q - vec3(0.0,-0.05,0.05)) - 1.04;
  vec3 pa = q - vec3(0.0,-0.20,0.30), ba = vec3(0.0,-0.10,0.50);
  d = min(d, length(pa - ba*sat(dot(pa, ba)/dot(ba, ba))) - gHGR);
  return min(d, length(vec3(abs(q.x), q.yz) - EARB) - 0.58);
}

void bearEval(vec3 pw){
  vec3 q = gHR*(pw - gHO) + NECK;
  gQ = q;
  // ---- the head guard.  Far from the head (farther than the body's blends reach) its whole block is skipped: the head is
  // stood in for by its bound, and everything the rest reads from it takes its far value
  float hg = headGuard(q);
  float head = hg, hole = 1e3, upper = hg, lower = hg, wEar = 0.0, hMask = 1.0, hLen = 0.078;
  vec3 hRest = q;
  gE = 1e3; gEU = -1.0; gEL = 1.0; gV = -1.0; gHoleD = 1e3; gDD = 1.0; gLipBand = 0.0; gHeadD = hg;
  gEye = 1e3; gNose = 1e3; gTeeth = 1e3; gTongue = 1e3; gEarD = 1e3; gUp = hg; gLo = hg; gBowl = 1.0;
  gPaintR = q; gQJ = q; gQE = vec3(0.0); gEarS = 1.0; gTgL = 0.0; gTgX = 0.0; gToothSt = vec3(0.0); gToothBack = 0.0;
  for(int it_=0; it_<1 && hg < HG_SKIP; it_++){
  // the mouth's bound: muzzle, lips, jaw, teeth and tongue all lie in this capsule (the head guard's).  Farther from it than
  // the blends reach (the skull, the crown, the ears, the backs of the cheeks) they are stood in for by it
  vec3 mpa = q - vec3(0.0,-0.20,0.30), mba = vec3(0.0,-0.10,0.50);
  float mb = length(mpa - mba*sat(dot(mpa, mba)/dot(mba, mba))) - gHGR;
  bool nearM = mb < 0.32;
  // ---- the aperture at this point
  float xm = q.x - gSmX*smoothstep(0.30, 0.56, q.z);
  float ax = abs(xm);
  float u = ax/gW, uu = u*u;
  float back = sq(sat((0.60 - q.z)/0.28));
  float notch = 0.012*exp(-q.x*q.x*600.0)*smoothstep(0.60, 0.80, q.z);
  float v0 = q.y - (LINEY + gYC + gBend*min(sq(ax/(max(gW, 0.14)*LW/MW)), 1.5) + back*gBackA + gSmY*clamp(xm/gW, -1.3, 1.3));
  float v = v0 - notch*(1.0 - gRing);           // the sealed line, with the philtrum's small notch
  float vl = v0 - notch*gSeal;                  // an open lower lip has none
  float eU = length(vec2(ax, v + gCu)) - gRu;
  float eL = length(vec2(ax, vl - gCl)) - gRl;
  float ap = smax(eU, eL, gRK + 1e-4);
  float cc = v > 0.0 ? gCu : gCl;
  // past a corner the corner itself is nearest.  Only for an arc shallower than a semicircle
  if(cc > 0.0 && gWO*(abs(v) + cc) < cc*ax) ap = max(ap, length(vec2(ax - gWO, v)));
  // a round mouth is one ellipse: no corners to branch on, the ring of lips stays whole
  float rb = 0.5*(gAu + gAl);
  ap = mix(ap, (length(vec2(ax/gW, (v - 0.5*(gAu - gAl))/rb)) - 1.0)*sqrt(gW*rb), gRnd*gRing);
  gE = mix(ap, min(ap, length(vec2(max(ax - gW, 0.0), v))), gPeel);     // the stuck part of the line is still the line
  gEU = eU; gEL = eL; gV = v;
  float g = max(1.0 - uu, 0.0);
  float gw = max(1.0 - sq(ax/max(gW, 0.14)), 0.0);     // the skin's own travel: spread wide, a tight round must not pull a pleat
  float uk = ax/max(gW, 0.14);                         // and so are the blends that hang on the corner
  float cf = exp(-sq(max(u - 1.0, 0.0)/1.10));         // the corners' own travel fades out over the cheek
  float T = gCMin + (1.0 - gCMin)*pow(g + 1e-4, 0.60); // thickness along the curve: thin toward a corner, stopping at a soft end

  // ---- the skin around the mouth.  Both lips gather and come forward together
  float lipZ = smoothstep(0.24, 0.62, q.z);
  float lipY = exp(-sq((q.y + 0.27)/0.25));
  vec3 qc = q;
  qc.x = (qc.x - gSmX*lipZ*lipY)*(1.0 + gGx*lipZ*lipY);
  float pushB = lipZ*lipBell(length(vec2(q.x/0.42, (q.y + 0.30)/0.26)));
  // upper skin follows the upper arc, fading out on the way up to the nose
  vec3 qU = qc;
  qU.y -= (gAu*gw + gYC*cf)*(1.0 - smoothstep(0.0, 0.21, eU));
  qU.z -= gPushU*pushB;
  qU.y -= gNoseW*lipBell(length((q - NOSEP)/vec3(0.30,0.22,0.26)));     // the nose's twitch carries the skin around it

  // ---- skull: huge and round
  float skull = sdEllipsoid(q - vec3(0.0,0.06,-0.04), vec3(0.72,0.63,0.64));
  vec3 qa = vec3(abs(q.x), q.yz);
  float side = sign(q.x);
  // chubby cheeks: they swell and ride up
  float cp = gCheekP + 0.35*gSmile;
  float cheek = sdEllipsoid(qa - vec3(0.45 + 0.03*cp + 0.022*gSp - 0.030*gPk, -0.21 + 0.07*cp - 0.050*gJC + 0.020*gSp, 0.20 + 0.03*cp),
                            vec3(0.34,0.29,0.31)*(1.0 + 0.26*cp));
  skull = smin(skull, cheek, 0.16);
  // shaggy jowls: blunt soft clumps, not the fox's points
  float lift = cp*0.03 - 0.095*gJC;            // the jowls hang from the jaw
  float ruff = sdTaper(qa, vec3(0.36,-0.38+lift,0.10), vec3(0.50,-0.60+lift,0.00), 0.17, 0.09);
  skull = smin(skull, ruff, 0.14);
  // brows: inner ends raised, a gentle worried-kind look
  float bu = (q.x < 0.0) ? gBrowL : gBrowR;
  vec3 bq = qa - vec3(0.285, 0.315 + 0.05*bu - 0.03*gKnit, 0.450);
  bq.xy = gBrowM*bq.xy;
  bq.x += 0.04*gKnit;
  skull = smin(skull, sdEllipsoid(bq, vec3(0.145,0.034,0.09)), 0.09);

  // ---- muzzle (upper): a broad round barrel with two fat lip pads
  float snoutCut = mb;
  for(int m_=0; m_<1 && nearM; m_++){
  float snoutRaw = sdTaperSq(qU, vec3(0.0,-0.06,0.34), vec3(0.0,-0.135,0.625), 0.36, 0.262, 1.12);
  vec3 pa = vec3(sqrt(qU.x*qU.x + sq(0.070*gRing)), qU.yz);     // the two pads run together under a pucker: no cleft through the ring
  float pad = sdEllipsoid(pa - vec3(0.118,-0.222 + gCheekP*0.022 + gSmile*0.018 + 0.012*gSp - 0.012*gJC + 0.60*gSmY*clamp(q.x/0.12, -1.0, 1.0),0.745), vec3(0.192,0.118,0.170));
  snoutRaw = smin(snoutRaw, pad, 0.06);
  snoutCut = smax(snoutRaw, -v, 0.016 + 0.11*gPk*smoothstep(1.0 - 0.9*gRing, 1.5 - 0.7*gRing, uk));   // past a narrow mouth the pad's edge melts away
  }
  float headRaw = smin(skull, snoutCut, 0.14);
  // sockets
  vec3 ea = qa - EYEC;
  headRaw = smax(headRaw, -(length(ea) - (EYER + 0.012)), 0.055);
  upper = headRaw;
  gUp = upper;

  // ---- lower jaw.  The hinge is rigid in qJ.  The skin blends to it: the chin rides the hinge, the lip rides the
  // lower arc, and past the corner the skin stays glued to the upper lip, so the two never part there
  vec3 qJ = jawQ(q + vec3(0.0, gDrop, 0.0));
  gQJ = qJ;
  float wj = sq(sat(1.0 - sq(q.x/0.52)));
  vec3 qL = mix(qc, jawQ(qc + vec3(0.0, gDrop, 0.0)), wj);
  lower = mb;
  for(int m_=0; m_<1 && nearM; m_++){
  qL.y += ((gAl*gw - gYC*cf) - wj*gJF)*(1.0 - smoothstep(0.0, 0.17, eL));
  // near the corners the lip stays out at the skin's own depth instead of swinging back under the upper pad
  qL.z = mix(qL.z, qc.z, (1.0 - smoothstep(0.0, 0.17, eL))*smoothstep(0.10, 0.90, uu));
  qL.z -= gPushL*pushB;
  // labiodental: the lower lip rolls back under the top teeth
  qL.z += gTuck*0.065*smoothstep(0.50, 0.85, q.z)*exp(-sq((eL - 0.020)/0.095));
  float lowerRaw = sdTaperSq(qL, JAW_A, JAW_B, JAW_RA, JAW_RB, 1.05);
  // heavy chin, with a soft dent under the lip
  lowerRaw = smin(lowerRaw, sdEllipsoid(qL - vec3(0.0,-0.432 + gPress*0.020,0.560 + gPress*0.012), vec3(0.205,0.120,0.170)), 0.09);
  float front = smoothstep(0.34, 0.60, qL.z);
  // lower lip pads: the lip comes out to meet the upper pads at the corner, so the corner is one point on one skin
  // and not the edge of an overhang with air behind it
  lowerRaw = smin(lowerRaw, sdEllipsoid(vec3(max(abs(qL.x) - 0.11, 0.0), qL.y + 0.350, qL.z - 0.695), vec3(0.150,0.100,0.128)) + 0.070*(1.0 - gFill), 0.10);
  lowerRaw -= (0.022 - 0.012*gPress)*exp(-sq((eL - 0.032)/0.034))*front*(1.0 - gPk*max(smoothstep(1.0 - 0.9*gRing, 1.4 - 0.6*gRing, uk), gPlug));   // thick soft lower lip
  lowerRaw *= gLS;
  float behind = sat((0.48 - qL.z)/0.20);
  lower = smax(lowerRaw, vl - (0.012*gSeal - 0.002), mix(0.060, 0.016, gOpenK) + 0.07*behind);
  }
  gLo = lower;

  // ---- eyeballs
  gEye = length(ea) - EYER;

  gNose = mb;      // the nose lies in the mouth's bound
  for(int m_=0; m_<1 && nearM; m_++){
  // ---- nose: big, wet, a soft rounded triangle
  vec3 nq = q - NOSEP - vec3(0.0, gNoseW, 0.0) + vec3(0.0, 0.007*gJC - 0.030*gAu*(1.0 - gRnd), 0.0);     // the upper lip's skin hangs off the nose: it gives a little with the lips
  float nose = sdEllipsoid(nq, vec3(0.142,0.086,0.100));
  nose = smin(nose, sdEllipsoid(nq - vec3(0.0,-0.052,-0.012), vec3(0.058,0.062,0.070)), 0.04);
  vec3 nna = vec3(abs(nq.x), nq.yz);
  float nostril = length((nna - vec3(0.072,-0.016,0.082))*vec3(0.8,1.5,1.0)) - (0.021 + 0.010*gNostril);
  nose = smax(nose, -nostril, 0.014);
  gNose = nose;
  }

  // ---- eyelids
  float blink = (q.x < 0.0) ? gBlinkL : gBlinkR;
  vec3 el = ea;
  el.xy = rot(-EYETILT)*el.xy;
  float squint = gSmile*0.10 + gCheekP*0.22 + 0.34*sat(uP[P_SQUINT]) + 0.05*gSp;
  float aU = mix(0.51 + 0.34*gEyeW, -0.32, blink);
  float aL = mix(-0.64 + squint - 0.08*gEyeW, -0.36, blink);
  float bru = (q.x < 0.0) ? gBrowL : gBrowR;
  aU += 0.06*bru;
  // the lids reach no further than their shell: away from the eye they are skipped
  for(int j_=0; j_<1 && length(ea) < EYER + 0.24; j_++){
  float lids = sdEyelids(el, EYER, 0.016, aU, aL);
  // the two lid planes meet in sharp almond corners; side lids trim those, so the opening is round and kind
  float shell = abs(length(el) - (EYER + 0.0144)) - 0.016;
  lids = smin(lids, smax(shell, mix(0.79, 1.15, blink)*EYER - abs(el.x), 0.03), 0.026);
  upper = smin(upper, lids, 0.050 + 0.02*blink);
  }
  gUp = upper;

  // ---- ears: small, round, wide apart
  float s = (q.x < 0.0) ? -1.0 : 1.0;
  vec3 e = q - vec3(s*EARB.x, EARB.y, EARB.z);
  // an ear in any pose lies in this ball: away from it, it is stood in for by the ball
  float ear = length(e) - 0.58, earP = 1e3;      // earP: the ear for weights and paint, far when skipped
  for(int m_=0; m_<1 && ear < 0.30; m_++){
  e.x *= s;
  e.xy = rot(0.62)*e.xy;     // tilted out to the side of the dome
  e.yz = rot(0.20)*e.yz;
  e = ((q.x < 0.0) ? gEarPL : gEarPR)*(e - EPIVP) + EPIVP;
  e = ((q.x < 0.0) ? gEarRL : gEarRR)*(e - EPIV) + EPIV;
  ear = sdEarShape(e);
  earP = ear;
  }
  gQE = e; gEarS = s;
  gEarD = earP;

  // the crease stops at the corner: past it the two bodies are one skin, in any width of mouth
  head = smin(upper, lower, 0.014 + 0.14*max(max(smoothstep(mix(0.82, 0.88, gOpenK) - 0.40*gRnd - 0.46*gRing, mix(1.38, 1.20, gOpenK) - 0.20*gRnd - 0.32*gRing, uk), smoothstep(0.51, 0.34, q.z)), gPlug));
  // the rolls of the lips, on the host
  float wv = smoothstep(-0.004, 0.004, v);
  for(int m_=0; m_<1 && nearM; m_++){
  float lipZk = smoothstep(0.30, 0.46, q.z);
  float rollU = gTU*exp(-sq((eU - 0.6*gHU)/(0.9*gHU + 0.006)));
  float rollL = gTL*exp(-sq((eL - 0.55*gHL)/(0.7*gHL + 0.006)));
  float rollR = gRT*exp(-sq((ap - 0.52*gRH)/(0.60*gRH))) + gCone*exp(-sq(ap/0.17));
  head -= mix(mix(rollL, rollU, wv)*T*(1.0 - smoothstep(1.0, 1.3, u)), rollR, gRing)*lipZk;
  // press: the rolled-in lips bulge behind the seal, the lower one fat, and a fold opens under it above the bunched chin
  float pb = gPress*lipBell(ax/0.30)*lipZk;
  head -= pb*(0.015*lipBell((v - 0.040)/0.050) + 0.022*lipBell((v + 0.044)/0.054));
  head += 0.007*pb*lipBell((v + 0.112)/0.030);
  // corner dimple: deeper when the corners are pulled or pressed
  head += (0.004 + 0.009*gSp + 0.004*gPress)*(1.0 - gPk)*lipBell(length(vec2(ax - gW - 0.012, v)/vec2(0.060, 0.070)));
  head *= 1.0/(1.0 + 0.5*gRing*exp(-sq(ap/(gRH + 0.02))));     // the ring is steep: the march must not step over it
  }
  gDD = head;

  // ---- the carve: closed at rest, wider behind the lips
  hole = 1e3; gTeeth = mb; gTongue = mb;
  for(int m_=0; m_<1 && nearM; m_++){
  float ce = ap - gHole*T - (0.34 - 0.26*gPk)*g*clamp(-head - 0.020, 0.0, 0.16)*gOpenK;
  hole = smax(ce, mix(0.36 + 0.24*min(uu, 1.0), 0.72, gPlug) - q.z, 0.05);        // a bowl: the sides close in toward the back
  gHoleD = hole;
  head = smax(head, -hole, 0.001 + (0.024 + 0.020*gRing)*gOpenK);   // lips roll softly in over the opening
  // teeth and tongue live only inside the carve: well outside it they are stood in for by it (a lower bound)
  gTeeth = hole + 0.006; gTongue = hole + 0.006;
  for(int j_=0; j_<1 && (hole < 0.15 || gTgOut > 0.0); j_++){
  // ---- teeth + tongue (bare): blunt, round, few.  Rigid: the upper row in the head, the lower row on the hinge
  vec3 stU, stL;
  float tlU = LINEY + BENDT*min(sq(q.x/LW), 1.5);
  float teethU = archRow(ax, q.z, q.y - (tlU + gTipU + LU), vec3(0.232,0.270,0.535), 0.066, 7.0, 0.015, LU, 0.014, 0.011, stU);
  float tlL = LINEY + BENDT*min(sq(qJ.x/LW), 1.5);
  float teethL = archRow(abs(qJ.x), qJ.z, (tlL - gTipL - LL) - qJ.y, vec3(0.185,0.205,0.540), 0.052, 3.0, 0.012, LL, 0.008, 0.009, stL);
  gTeeth = min(teethU, teethL);
  gToothSt = teethU < teethL ? stU : stL;
  gToothBack = teethU < teethL ? 1.0 : 0.35;

  // tongue: lies low; tip up behind the upper teeth, tip between the teeth, or its back raised
  // tongue_out: it reaches out past the lips, the tip drooping a little over the lower lip
  float tLen = 0.295 + 0.19*gTgOut;
  vec3 tq = qJ - vec3(0.0, -0.365 + 0.022*gTgOut, 0.40 + 0.19*gTgOut);
  float tl = sat((tq.z + tLen)/(2.0*tLen));                             // 0 root .. 1 tip
  tq.y += 0.075*gTgOut*sq(smoothstep(0.55, 1.0, tl));
  tq.z *= 0.295/tLen;
  tq.x -= 0.22*(0.295 - tq.z);          // its root swings toward the far side: along the view, it stays in the middle of a 3/4 mouth
  tq.z -= gTgGapZ*smoothstep(0.30, 1.0, tl);
  tq.y -= gTgTip*smoothstep(0.42, 1.0, tl) + gTgGap*smoothstep(0.30, 0.95, tl) + gTgBack*0.060*lipBell((tl - 0.34)/0.34);
  // the tip: a broad blade held against the ridge in a nearly shut mouth, a narrow point reaching up in an open one
  float tgUp = max(uP[P_TONGUE_TIP], uP[P_TONGUE_DENTAL]);
  tq.x *= (1.0 + 0.25*tl)*(1.0 + (0.5 + 1.3*smoothstep(0.08, 0.24, uP[P_TEETH_GAP]))*smoothstep(0.34, 0.92, tl)*tgUp);
  gTgL = tl;
  gTgX = tq.x/0.170;
  float tongue = sdEllipsoid(tq, vec3(0.170,0.052,0.295))*0.72;
  // k g: the back of the tongue bunches up and fills the far end of the mouth
  tongue = smin(tongue, sdEllipsoid(qJ - vec3(0.065, -0.365 + 0.050*gTgBack, 0.40), vec3(0.215, 0.040 + 0.055*gTgBack, 0.135)) + 0.09*(1.0 - gTgBack), 0.05);
  tongue += 0.006*smoothstep(0.022, 0.0, abs(tq.x));     // centre groove
  gTongue = tongue;

  gTeeth = max(gTeeth, hole + 0.006);          // only inside the carve, clear of its wall
  gTongue = max(gTongue, mix(hole + 0.006, -1.0, gTgOut*smoothstep(0.66, 0.84, q.z)));   // out past the lips it is not clipped by the opening
  }
  }
  gHoleD = hole;

  head = smin(head, ear, 0.11);
  gHeadD = head;
  // the head's own rest coordinate, fur mask and length
  float wJaw  = smoothstep(0.05, -0.11, v)*smoothstep(0.10, 0.34, q.z);   // which skin: by the lip line.  By distance the two skins' coordinates fold into stripes on the cheek
  wEar  = smoothstep(-0.06, 0.06, min(upper,lower) - earP);
  vec3 rq = mix(qU, qL, wJaw);
  gPaintR = mix(q, qJ, wJaw*wj);
  hRest = mix(rq, vec3(s*(e.x + EARB.x), e.y + EARB.y, e.z + EARB.z), wEar);
  float m = 1.0;
  m *= smoothstep(0.03, 0.10, length((q - NOSEP)/vec3(1.0,0.9,1.0)) - 0.135);
  float lipH = mix(mix(gHL, gHU, wv)*T, gRH, gRing);
  gLipBand = lipH;
  m *= smoothstep(0.012, 0.050, gE - lipH);
  m *= smoothstep(0.005, 0.085, length(ea) - EYER - 0.022);
  m *= smoothstep(-0.015, 0.03, hole);
  m *= mix(0.40, 1.0, smoothstep(0.0, 0.07, abs(min(upper, lower) - earP)));   // short nap where the ear meets the head: no sewn seam
  hMask = m;
  float muzzleShort = mix(0.32, 1.0, smoothstep(0.74, 0.34, hRest.z));
  float jowl = 1.0 + 0.45*smoothstep(0.34, 0.70, abs(q.x))*smoothstep(-0.05, -0.32, q.y);
  hLen = 0.078*muzzleShort*jowl;
  }

  // ---- body, body frame (does not follow the head except a touch of breath).  The bear sits: belly, chest, the back
  // behind them, the thighs coming forward.  The shoulder mound rides the hand, and the skin under it goes along
  bool sideL = pw.x < 0.0;
  ArmPose arm = sideL ? gArmL : gArmR;
  float sd = sideL ? -1.0 : 1.0;
  vec3 b = pw;
  float br = 1.0 + 0.020*gBreath;
  b.y += 0.016*gBreath;
  b.xz = rot(BYAW)*b.xz;
  vec3 bs = vec3(sd*b.x, b.yz);
  gBq = b;
  float wsh = smoothstep(0.15, 0.65, bs.x)*lipBell((b.y + 1.05)/0.90);
  vec3 bo = vec3(sd*arm.off.x, arm.off.yz)*wsh;
  gBodyOff = bo;
  vec3 bw = b - bo;
  float neck  = sdTaper(bw, vec3(0.0,-0.30,-0.30), vec3(0.0,-1.10,-0.38), 0.56, 0.80);
  float sh = sdEllipsoid(bs - arm.S, vec3(0.38,0.44,0.42));
  // everything else of the body lies under y = -0.20 (a raised shoulder carries the chest top up to there): high on the
  // head, farther from it than the blends reach, only the neck and the shoulder are needed
  float bodyD = smin(neck, sh, 0.26);
  gCrease = 1.0;
  for(int j_=0; j_<1 && b.y + 0.20 < neck + 0.32; j_++){
  float belly = sdEllipsoid(bw - BELLY_C, BELLY_R*br);
  float rear  = sdEllipsoid(bw - vec3(0.0,-1.62,-0.55), vec3(0.98,1.06,0.76)*br);
  float chest = sdEllipsoid(bw - CHEST_C, CHEST_R*br);
  vec3 ba = vec3(abs(bw.x), bw.yz);
  bodyD = smin(smin(smin(belly, rear, 0.30), neck, 0.25), chest, 0.25);
  // chest shag
  bodyD = smin(bodyD, sdTaper(ba, vec3(0.30,-0.95,0.18), vec3(0.46,-1.24,0.30), 0.18, 0.05), 0.12);
  bodyD = smin(bodyD, sdTaper(ba, vec3(0.08,-1.02,0.26), vec3(0.18,-1.38,0.44), 0.18, 0.05), 0.12);
  bodyD = smin(bodyD, sh, 0.26);
  // the thighs, resting forward
  float thighD = sdTaper(ba, THIGH_A, THIGH_B, THIGH_RA, THIGH_RB);
  gCrease = smoothstep(0.0, 0.14, abs(thighD - bodyD));          // the coat is shorter where thigh meets belly: no deep dark fold of strands
  bodyD = smin(bodyD, thighD, 0.24);
  }
  bodyD = smax(bodyD, -hole, 0.02);
  gBodyD = bodyD;

  gFur = smin(head, bodyD, 0.20);
  float torsoD = gFur;
  float armD = armEval(bs, arm, torsoD);
  gFur = smin(gFur, armD, gArmK);

  gBare = min(min(gEye, gNose), min(gTeeth, gTongue));
  gBare = min(gBare, min(gPad, gClaw));

  // rest coordinate for fur + paint
  float wBody = smoothstep(-0.10, 0.10, head - bodyD);
  gWBody = wBody;
  gRest = mix(hRest, gBodyR*(pw - gBodyWorld(bo) - NECK) + NECK, wBody);
  // the arm's skin, at rest, in the same frame.  The shoulder blends over a wide band, so no line runs across the coat
  gWArm = smoothstep(-gArmK - 0.02, gArmK + 0.02, torsoD - armD);
  gWBody = max(gWBody, gWArm);                     // no head paint on an arm that has come up beside the head
  vec3 ra = vec3(sd*gArmRestP.x, gArmRestP.yz);
  gRest = mix(gRest, gBodyR*(toWorld(ra) - NECK) + NECK, gWArm);

  // where fur grows, how long
  gMask = hMask;
  gFurLen = mix(mix(hLen, 0.150, wBody)*mix(1.0, 0.68, wEar)*mix(0.30, 1.0, smoothstep(0.0, 0.16, armD))*mix(0.55, 1.0, gCrease), gArmLen, gWArm);   // the coat lies down under a limb
}

// One pole for head and body: blending two poles across the neck sweeps the fur coordinates through many
// periods in a narrow band (sand-like speckle).  Strands leave the nose radially, then droop under gravity,
// so at the silhouette they lie along the edge and overlap like shingles rather than ending in terraces.
vec3 furPt(vec3 r){
  vec2 w = r.xy - POLE.xy;
  float d2 = dot(w, w);
  r.y += 0.62*d2/(1.0 + 0.50*d2);
  // far from the nose the along-strand phase runs slower: long locks on the body, no basket weave.  Radial
  // scaling keeps every strand direction, and the phase stays monotonic, so there is no moire.
  vec3 v = r - POLE;
  float l = length(v), e = max(l - 0.75, 0.0);
  return POLE + v*(1.0 - 0.60*e*e/((e + 0.35)*max(l, 1e-3)));
}

void charEval(vec3 p){
  bearEval(bodyIn(p));
  gcBase = gFur; gcBare = gBare; gcLen = gFurLen*gMask; gcRest = gRest;
}
// Bear coat: ridged strand noise on top of soft clumps.  Same 3 noise evals as furHeight, but the strand crests
// are sharp, so the silhouette breaks into pointed wisps instead of smooth lobes.
float bearFur(vec3 rest){
  vec3 p = furPt(rest);
  float clump = vnoise3(furCoord(p, POLE, F_K, F_L));
  vec3 f = furCoord(p, POLE, F_K*3.6, F_L*1.7);
  float st = vnoise3(f);
  float st2 = vnoise3(f*vec3(2.6,2.6,1.2) + vec3(4.1,1.7,8.3));
  float ridge = 1.0 - abs(2.0*st - 1.0);
  return clamp(clump*0.30 + ridge*0.46 + st2*0.44 - 0.08, 0.0, 1.0);
}
float charTuft(){
  return mix(bearFur(gcRest), 0.92, smoothstep(0.070, 0.030, gcLen));   // short fur is a closed nap: no halftone dots
}

//@shadow-begin
// cheap proxy for shadow rays: a few ellipsoids, the two arms, no fur, no mouth
float charShadow(vec3 pw){
  pw = bodyIn(pw);
  vec3 q = gHR*(pw - gHO) + NECK;
  vec3 qa = vec3(abs(q.x), q.yz);
  float d = sdEllipsoid(q - vec3(0.0,0.05,-0.04), vec3(0.69,0.60,0.60));
  d = min(d, sdEllipsoid(qa - vec3(0.42,-0.22,0.18), vec3(0.30,0.25,0.28)));
  d = smin(d, sdTaper(q, vec3(0.0,-0.06,0.34), vec3(0.0,-0.135,0.60), 0.32, 0.23), 0.10);
  d = min(d, sdTaper(q, vec3(0.0,-0.30,0.16), vec3(0.0,-0.36,0.60), 0.28, 0.18));
  vec3 e = qa - EARB;
  d = min(d, length(e) - 0.17);
  // the body must sit INSIDE the real coat: a proxy larger than the body shadows strand tips at random
  vec3 b = pw;
  b.xz = rot(BYAW)*b.xz;
  d = min(d, sdEllipsoid(b - BELLY_C, 0.90*BELLY_R));
  d = min(d, sdEllipsoid(b - CHEST_C, 0.88*CHEST_R));
  d = min(d, sdTaper(b, vec3(0.0,-0.4,-0.30), vec3(0.0,-1.1,-0.38), 0.42, 0.62));
  // the arms, both of them, wherever the hands are: this shadow follows the arm.  A point on an arm sees them thinner:
  // a ray running out along the arm toward the sun would otherwise graze its own proxy and darken the whole limb
  float kA = 0.075*gSelfArm;
  d = min(d, sdTaper(pw, gWSL, gWEL, 0.25, 0.19) + kA);
  d = min(d, sdTaper(pw, gWEL, gWWL, 0.19, 0.15) + kA);
  d = min(d, length(pw - gWPL) - 0.105 + kA);
  d = min(d, sdTaper(pw, gWSR, gWER, 0.25, 0.19) + kA);
  d = min(d, sdTaper(pw, gWER, gWWR, 0.19, 0.15) + kA);
  d = min(d, length(pw - gWPR) - 0.105 + kA);
  return d;
}
//@shadow-end

const vec3 C_BROWN = vec3(0.082,0.038,0.019);
const vec3 C_DEEP  = vec3(0.036,0.016,0.010);
const vec3 C_WARM  = vec3(0.165,0.078,0.034);
const vec3 C_TAN   = vec3(0.36,0.205,0.105);
const vec3 C_CREAM = vec3(0.50,0.34,0.20);
const vec3 C_DARK  = vec3(0.026,0.017,0.015);

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  bearEval(bodyIn(pw));
  gSelfArm = gWArm;
  vec3 q = gQ, r = gRest;
  vec3 n = s.n;
  bool isEye = gEye < 0.002 && gEye <= gFur + 0.002;
  bool isNose = gNose < 0.004 && gNose <= gFur + 0.004;
  bool isTeeth = gTeeth < 0.003 && gTeeth <= min(gFur, gTongue) + 0.003;
  bool isTongue = gTongue < 0.003 && gTongue <= gFur + 0.003;
  // past the inner edge of the lips everything is mouth, the fur behind it too
  float cm = max(smoothstep(0.009, 0.003, gHoleD), smoothstep(-0.001, -0.006, gE));
  bool inMouth = cm > 0.0;

  // ---- fur paint
  vec3 rf = furPt(r);
  vec3 c1 = furCoord(rf, POLE, 120.0, 13.0);
  float st = furStrand(c1);
  float tuft = furTuft(furCoord(rf, POLE, F_K, F_L));
  float ax = abs(r.x);
  float wob = 0.10*(tuft - 0.5) + 0.05*(st - 0.5);
  // tan muzzle, in rest space
  vec3 mz = (gPaintR - vec3(0.0,-0.205,0.60))/vec3(0.395,0.345,0.420);
  mz.z = min(mz.z, 0.76);               // lips pushed forward stay inside the tan
  float muzzleM = smoothstep(1.05, 0.90, length(mz) + wob)*(1.0 - gWBody);
  // chest crescent
  float bib = smoothstep(1.06, 0.84, length(vec2(abs(gBq.x)/0.62, (gBq.y + 1.56)/0.78)) + wob*1.2)*gWBody*smoothstep(-0.30, 0.05, gBq.z)*(1.0 - gWArm);
  float earW = smoothstep(-0.02, 0.02, min(gUp, gLo) - gEarD)*smoothstep(-0.02, 0.02, gBodyD - gEarD);
  vec3 alb = mix(C_BROWN, C_WARM, 0.30*smoothstep(0.35,0.80,st) + 0.28*smoothstep(0.35,0.75,vnoise3(r*2.7)));
  alb = mix(alb, C_DEEP, 0.32*smoothstep(0.75,0.15,tuft));
  alb = mix(alb, C_DEEP, 0.30*smoothstep(0.30,0.70,r.y)*(1.0 - gWBody));   // the crown is deeper
  alb = mix(alb, mix(C_TAN, C_CREAM, smoothstep(-0.30,-0.14,r.y)*0.45), muzzleM);
  alb = mix(alb, mix(C_TAN, C_CREAM, 0.25), bib);
  // ears: brown backs, soft dusky bowl
  float eY = gQE.y;
  vec3 earC = mix(C_BROWN, C_DEEP, 0.35);
  float bowlM = smoothstep(0.085, -0.02, gBowl + 0.03*(tuft - 0.5));
  earC = mix(earC, mix(vec3(0.20,0.105,0.068), vec3(0.28,0.155,0.10), smoothstep(-0.12,0.14,eY)), bowlM);
  alb = mix(alb, earC, earW);
  // soft brow tuft, a little lighter: paint in head space so it rides `brow` and `brow_knit`
  float bru2 = (q.x < 0.0) ? gBrowL : gBrowR;
  vec3 bt = vec3(abs(q.x), q.yz) - vec3(0.275, 0.335 + 0.075*bru2 - 0.05*gKnit, 0.45);
  bt.xy = rot(-0.30 + 0.34*gKnit)*bt.xy;
  float browM = smoothstep(1.0, 0.45, length(bt.xy/vec2(0.135,0.052)) + wob*1.5)*(1.0 - earW)*(1.0 - gWBody);
  alb = mix(alb, C_WARM*1.2, 0.45*browM);
  // the limbs darken toward the paw, as a bear's do
  alb = mix(alb, mix(C_BROWN, C_DEEP, 0.60), gWArm*smoothstep(0.18, 0.92, gArmAlong));
  alb = mix(alb, C_DEEP, smoothstep(0.10, 0.015, gClaw)*gWPaw);      // the digit darkens into the claw
  // strand contrast
  alb *= 0.90 + 0.20*st + gWBody*0.14*(st - 0.5);

  // eyeliner + dark lid margin
  vec3 ea = vec3(abs(q.x), q.yz) - EYEC;
  float lidr = length(ea);
  // a soft darker patch around the eye, as real bears have: seats the eye, no pale ring
  alb *= mix(0.70, 1.0, smoothstep(EYER + 0.005, EYER + 0.135, lidr + 0.06*(tuft - 0.5)));
  float liner = smoothstep(EYER + 0.026, EYER + 0.015, lidr)*step(0.0, ea.z + 0.05);
  alb = mix(alb, vec3(0.055,0.028,0.018), liner);

  // whisker follicles: three short rows of faint dots on the lip pads
  vec2 wg = vec2((ax - 0.070)/0.052 + 0.5*step(0.5, fract((r.y + 0.225)/0.094)), (r.y + 0.225)/0.047);
  vec2 wc = fract(wg) - 0.5;
  float inPad = step(0.0, wg.x)*step(wg.x, 3.0)*step(0.0, wg.y)*step(wg.y, 3.0)*(1.0 - gWBody)*step(0.60, r.z);
  alb = mix(alb, C_BROWN*1.2, 0.30*smoothstep(0.20, 0.09, length(wc))*inPad);

  // lips: dark soft skin.  A band along the curve, thin at the corners; the inner lower lip is wet and warmer
  float lip = smoothstep(gLipBand + 0.008 + 0.014*gRing, gLipBand - 0.008 - 0.014*gRing, gE);
  vec3 lipC = mix(mix(vec3(0.036,0.021,0.019), vec3(0.075,0.040,0.036), gPk), mix(vec3(0.112,0.056,0.050), vec3(0.060,0.032,0.030), smoothstep(0.040, 0.0, gE)), gRing);   // soft and lighter, darkening into the opening
  lipC = mix(lipC, vec3(0.13,0.058,0.052), max(gOpen, gTuck)*(1.0 - smoothstep(-0.004, 0.004, gV))*smoothstep(0.034, 0.004, gEL));
  lipC = mix(lipC, vec3(0.105,0.046,0.044), gOpen*smoothstep(-0.004, 0.004, gV)*smoothstep(0.030, 0.006, gEU));
  alb = mix(alb, lipC, lip);
  // the lip's edge wraps out of sight round a fat lower lip seen from the side: carry its colour to the contour
  float rimL = gOpenK*(1.0 - gRing)*smoothstep(0.11, 0.03, gE)*smoothstep(0.62, 0.20, dot(n, -rd))*smoothstep(0.30, 0.42, q.z);
  lip = max(lip, rimL);
  alb = mix(alb, lipC, rimL);
  float bareSkin = lip;

  // philtrum: a soft dark seam from the nose down to the lip
  float phil = smoothstep(0.013,0.004,abs(q.x))*smoothstep(0.30,0.8,q.z)*step(0.0, gEU)*step(0.0, gV)*step(q.y, -0.09);
  alb = mix(alb, vec3(0.05,0.03,0.025), phil*0.85*(1.0 - gRing*smoothstep(gLipBand + 0.040, gLipBand + 0.005, gE)));

  s.alb = alb;
  vec3 v = gRest - POLE;
  v = mix(v, 3.0*vec3(gBq.x < 0.0 ? -gArmT.x : gArmT.x, gArmT.yz), gWArm);     // on a limb the coat runs down the bone
  s.T = furFlow(v, n, vec3(0.0,-1.0,0.0));
  float nb = (st - 0.5);
  vec3 b = normalize(cross(s.T, n));
  s.n = normalize(n + b*nb*(0.09 + 0.16*gWBody) + s.T*nb*0.03);
  s.fur = 1.0 - bareSkin;
  s.wrap = 0.26;
  s.sheen = 0.9;
  s.occ = (0.88 + 0.12*smoothstep(0.1,0.7,tuft))*(0.94 + 0.10*st);
  s.alb *= mix(vec3(0.88,0.80,0.74), vec3(1.0), smoothstep(0.05,0.6,tuft));
  s.trans = earW*0.16;
  s.transCol = vec3(1.0,0.45,0.20);
  if(bareSkin > 0.5){ s.spec = 0.30; s.shin = 45.0; s.sheen = 0.2; }

  if(inMouth){
    float dz = (1.0 - 0.55*gPlug)*smoothstep(-0.06, 0.0, gDD)*(1.0 - smoothstep(-0.03, 0.0, gEU))*(1.0 - 0.85*gRnd*(1.0 - 0.35*gPlug))*mix(smoothstep(0.95, 0.60, abs(gQ.x)/gW), 1.0, gPlug);    // only the lip's own inside catches light; the lining up behind the lip stays in shadow
    vec3 c = mix(vec3(0.040,0.010,0.012), vec3(0.20,0.055,0.055), dz*dz)*(1.0 - 0.6*gRnd*(1.0 - gPlug));
    c += vec3(0.062,0.014,0.016)*smoothstep(-0.20, -0.02, gV)*(1.0 - gRnd)*(1.0 - gDark);     // the roof of the mouth catches a little light: an open mouth has depth, it is not a black hole
    s.alb = mix(s.alb, c, cm);
    s.fur *= 1.0 - cm; s.sheen *= 1.0 - cm; s.spec = mix(s.spec, 0.08, cm); s.shin = mix(s.shin, 30.0, cm); s.wrap = mix(s.wrap, 0.3, cm);
    s.occ = mix(s.occ, mix(0.45, 1.0, dz), cm); s.trans *= 1.0 - cm;
    s.n = normalize(mix(s.n, n, cm));
    s.emit = c*0.20*cm;                 // a little fill so the cavity stays red, not black
  }
  if(isTongue){
    float tb = smoothstep(0.10, 0.52, gQJ.z);
    s.alb = mix(vec3(0.26,0.075,0.080), vec3(0.66,0.235,0.245), tb)*(0.88 + 0.22*vnoise3(gQJ*40.0));
    s.alb = mix(s.alb, vec3(0.10,0.026,0.032), smoothstep(0.50, 1.05, abs(gTgX)));      // its edges sink into the dark
    s.alb *= mix(0.22, 1.0, smoothstep(0.03, 0.34, dot(n, -rd)));                       // and so does its horizon
    s.alb *= mix(1.0, mix(0.42, 1.15, smoothstep(0.62, 0.92, gTgL)), uP[P_TONGUE_TIP]);    // raised, its dark underside faces us and only the tip is bright
    s.alb *= 1.0 - 0.60*gTgBack*smoothstep(0.66, 0.46, gTgL);                              // bunched up at the back it sits in the shadow of the mouth
    s.alb *= mix(1.0, smoothstep(-0.19, -0.05, gV), uP[P_TONGUE_TIP]);                     // and a raised blade falls into the dark toward the floor
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.25; s.shin = 30.0; s.wrap = 0.35; s.occ = mix(0.6, 1.0, tb); s.trans = 0.0; s.n = n;
    s.emit = s.alb*(0.14 + 0.26*max(uP[P_TONGUE_TIP], uP[P_TONGUE_DENTAL]));     // raised, it faces the viewer and away from the sky
  }
  if(isTeeth){
    // the row melts into the warm dark toward the corner, sooner while the lips gather (the rear teeth stack up
    // along the view there); the lip shades the gum side, a soft crease parts the teeth
    float back = gToothBack*smoothstep(0.26 - 0.12*gPk, 0.74 - 0.40*gPk, gToothSt.x);
    s.alb = mix(vec3(0.020,0.009,0.009), vec3(0.76,0.70,0.58), pow(1.0 - back, 2.5));
    s.alb *= mix(0.70, 1.0, smoothstep(0.0, 0.022, -gEU))*mix(1.0, 0.70, smoothstep(0.70, 1.0, abs(gToothSt.y)));
    // above the crown it is gum: where the lip rides high the teeth stay short and round
    float gumM = smoothstep(-0.003, 0.007, gToothSt.z - 0.012*(1.0 - gToothSt.y*gToothSt.y));
    s.alb = mix(s.alb, vec3(0.026,0.012,0.012), gumM);
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.45; s.shin = 60.0; s.wrap = 0.25; s.occ = 1.0; s.trans = 0.0; s.n = n;
    s.emit = s.alb*0.16;
    s.spec *= 1.0 - gumM;
  }
  // while the mouth is closed nothing inside may read: teeth, tongue and cavity are all lip-dark, and they
  // stay dark in a thin slit; teeth come up to white as they clear the lip, never as bright slivers
  float dk = max(isTeeth ? 1.0 - smoothstep(0.08, 0.45, gShow) : gDark*(isTongue ? 1.0 - 0.85*max(uP[P_TONGUE_TIP], uP[P_TONGUE_DENTAL]) : 1.0), smoothstep(0.3, 0.9, gTunnel)*step(0.5, float(isTeeth || isTongue)));
  if((inMouth || isTeeth || isTongue) && dk > 0.0){
    s.alb = mix(s.alb, vec3(0.036,0.021,0.019)*(isTeeth ? 0.3 : 1.0), dk);     // no shadow reaches in here: teeth facing the sun would still read
    s.emit *= 1.0 - dk;
    s.spec *= 1.0 - dk;
  }
  if(gPad < 0.004 && gPad <= min(gFur, gClaw) + 0.004){
    // pads: thick dark leather, a soft shine, a fine pebbling
    float fine = vnoise3(gPq*90.0);
    s.alb = vec3(0.040,0.029,0.030)*(0.80 + 0.40*fine);
    s.n = normalize(n + 0.10*(fine - 0.5)*s.T);
    s.fur = 0.0; s.sheen = 0.10; s.spec = 0.35; s.shin = 28.0; s.wrap = 0.20; s.occ = 1.0; s.trans = 0.0;
  }
  if(gClaw < 0.004 && gClaw <= gFur + 0.004){
    s.alb = mix(vec3(0.030,0.024,0.022), vec3(0.115,0.092,0.076), smoothstep(0.012, 0.0, gClaw + 0.012));
    s.n = n;
    s.fur = 0.0; s.sheen = 0.0; s.spec = 0.55; s.shin = 40.0; s.wrap = 0.15; s.occ = 1.0; s.trans = 0.0;
  }
  if(isNose){
    float fine = vnoise3(q*120.0);
    s.alb = vec3(0.016,0.013,0.013)*(0.85 + 0.3*fine);
    s.n = n;
    s.fur = 0.0; s.sheen = 0.25; s.spec = 0.9; s.shin = 80.0; s.wrap = 0.1; s.occ = 1.0; s.trans = 0.0; s.mirror = 0.6;
  }
  if(isEye){
    // gaze space
    vec3 toCam = rootGaze(gHR*(bodyIn(uCamPos) - gHO)) + NECK - vec3(sign(q.x)*EYEC.x, EYEC.y, EYEC.z);
    vec3 eyeF = normalize(normalize(toCam) + vec3(gGaze.x*0.30, gGaze.y*0.24, 0.0));
    vec3 ex = normalize(cross(vec3(0,1,0), eyeF)), ey = cross(eyeF, ex);
    mat3 toE = transpose(mat3(ex, ey, eyeF));
    vec3 ee = vec3(q.x - sign(q.x)*EYEC.x, q.y - EYEC.y, q.z - EYEC.z)/EYER;
    vec3 e = toE*ee;
    vec3 v = toE*(gHR*(gBM*rd));
    float irisM;
    vec3 c = eyeColour(e, v, vec3(0.27,0.125,0.045), vec3(0.105,0.046,0.022), vec3(0.15,0.075,0.04), 0.965, 0.62, irisM);
    s.alb = c;
    // shadow cast by the upper lid
    s.occ = mix(0.68, 1.0, smoothstep(0.75, 0.15, ee.y));
    s.fur = 0.0; s.sheen = 0.0; s.spec = 1.5; s.shin = 900.0; s.wrap = 0.0; s.trans = 0.0;
    s.mirror = 0.55*smoothstep(0.80, 0.35, ee.y);
    s.n = n;
    // painted catch-lights, locked to the light rig
    vec3 r2 = reflect(rd, n);
    float k1 = smoothstep(0.962, 0.975, dot(r2, normalize(uSunDir + vec3(0.05,0.2,0.3))));
    float k2 = smoothstep(0.985, 0.993, dot(r2, normalize(vec3(0.5,0.1,0.85))));
    s.emit = vec3(1.0,0.97,0.92)*(3.6*k1 + 1.3*k2)*smoothstep(0.92, 0.70, ee.y);
  }
}

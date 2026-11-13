// ---------------------------------------------------------------------------
//  animals/cube.glsl — a chunky glazed ceramic block with a face.  Head space: origin at the centre of the
//  block, +x screen right, +y up, +z out of the face.  The block sits turned SHAPE_YAW0, so the camera sees the
//  face, its right flank and the top.  Mechanics live in engine/lips.glsl and engine/face.glsl (docs/SHAPE_GUIDE.md).
// ---------------------------------------------------------------------------
#define CUBE_H 0.70                                 // half size
#define CUBE_R 0.15                                 // edge radius

#define SHAPE_PIVOT vec3(0.0,-0.70,0.0)
#define SHAPE_WORLD vec3(0.0,-0.70,0.0)
#define SHAPE_YAW0 (-0.34)
#define SHAPE_SQUASH 0.020
#define SHAPE_JIGGLE 0.012
#define EYE_R 0.215
#define EYE_C vec3(0.305,0.135,0.543)              // flush like buttons: 0.73 EYE_R under the surface
#define EYE_AU 0.72
#define EYE_AL (-0.74)
#define EYE_AC (-0.34)
#define EYE_TILT 0.10
#define EYE_IRIS 0.92
#define EYE_PUPIL 0.60
#define EYE_COL_A vec3(0.74,0.40,0.12)
#define EYE_COL_B vec3(0.15,0.070,0.030)
#define BROW_C vec3(0.300,0.395,0.0)
#define BROW_L 0.185
#define BROW_W 0.046
#define BROW_H 0.030
#define BROW_RISE 0.150
#define BROW_COL vec3(0.040,0.022,0.024)
#define LINER_COL vec3(0.006,0.028,0.038)
#define LIP_TILT (-0.32)                            // the camera is above the mouth ...
#define LIP_AIM 0.47                                // ... and off to its right: 0.34 of turn, 0.15 of offset, a shade under.  See cubeEval
#define LIP_LEVEL 0.09                              // the aimed frame sits level on screen, and the block's horizontals do not: roll it onto them
#define LIP_TOOTH_Z (-0.190*LIPU)                   // the aimed frame runs out of the flat face toward one corner
#define LIP_DEPTH (0.56*LIPU)
#define LIP_O vec3(0.0,-0.272,0.70)
#define LIP_W 0.318                                 // a broad graphic slot: half the face wide
#define LIP_HU (0.052*LIPU)                         // the two lips are one thick rolled rim, like the lip of a pot
#define LIP_HL (0.056*LIPU)
#define LIP_TU (0.030*LIPU)
#define LIP_TL (0.032*LIPU)
#define LIP_SQUARE 0.78
#define LIP_BOXR 0.62
#define LIP_RIM 1.0
#define LIP_BOW 0.0
#define LIP_LIFT 0.21
#define LIP_OPEN_Y 0.62                             // the block is short: the chin stays on the flat of the face
#define LIP_BEND 0.55                               // a straighter line: drawn with a ruler, nearly
#define LIP_SPREAD 0.17                             // the block has no cheeks to pull: the blush stays clear
#define LIP_SKIN 0.25                               // fired clay: nothing soft around the mouth
#define LIP_BLEND (0.010*LIPU)
#define LIP_PUSH (0.070*LIPU)
#define LIP_SCALLOP 0.0                             // one band of white slip
#define LIP_GROOVE 0.30
#define LIP_TEETH_H (0.066*LIPU)
#define LIP_TOOTH_GAIN 0.74                         // a white band on a dark block is the highest contrast in the film: keep it down
#define LIP_FLOW 0.40
#define LIP_COL_U vec3(0.030,0.200,0.205)
#define LIP_COL_L vec3(0.030,0.215,0.215)
#define LIP_COL_LINE vec3(0.010,0.045,0.060)
#define LIP_COL_CAVE vec3(0.46,0.095,0.060)         // coral underglaze, the same hand as the cheeks
#define LIP_COL_DEEP vec3(0.100,0.022,0.024)
#define LIP_COL_GUM  vec3(0.52,0.13,0.085)
#define LIP_COL_TONGUE vec3(0.92,0.34,0.20)
#define LIP_COL_TOOTH vec3(0.93,0.89,0.79)
#define LIP_SPEC 0.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.0
#define LIP_WRAP 0.06
#define LIP_EMIT vec3(0.0)
#pragma include face
#pragma include lips
// real 3-D arms with mitten hands on the contract's hand columns (engine/shape_arms.glsl, docs/RIG.md)
#define SA_SHO vec3(0.660,-0.150,0.150)
#define SA_REST vec3(0.120,-0.360,0.220)
#define SA_LEN 0.850
#define SA_R 0.090
#define SA_HAND 0.180
#define SA_ALB vec3(0.008,0.15,0.17)
#define SA_HALB vec3(0.012,0.19,0.19)
#define SA_SPEC 0.6
#define SA_MIRROR 0.25
#pragma include shape_arms

float gSkin, gBare;
vec3  gQ;
mat3  gLipAim;

float cubeHost(vec3 q){
  vec3 d = abs(q) - (CUBE_H - CUBE_R);
  return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0) - CUBE_R;
}

void charInit(){
  faceInit(); lipsInit();
  gLipAim = rotZ(LIP_LEVEL)*rotY(-LIP_AIM);
  gHO = SHAPE_WORLD;           // it sits on the floor: head_x, y, z would slide it off its contact.  It lives by rocking
  // body_bounce: up is a hop off the floor, down a squash onto it
  float bb = clamp(uP[P_BODY_BOUNCE], -1.0, 1.0);
  gHO.y += 0.10*max(bb, 0.0);
  gSqY *= 1.0 + 0.07*min(bb, 0.0); gSqX = 1.0/sqrt(gSqY);
  // head_roll and head_pitch rock the block on a bottom edge: that edge stays put and the far one lifts.
  // The pivot slides across soft, so the rock through rest has no kink.
  float sx = uP[P_HEAD_ROLL]/(abs(uP[P_HEAD_ROLL]) + 0.008), sz = uP[P_HEAD_PITCH]/(abs(uP[P_HEAD_PITCH]) + 0.008);
  vec3 b = vec3(sx, 0.0, sz)*(CUBE_H - 0.5*CUBE_R);
  gHO += rotY(SHAPE_YAW0 + uP[P_HEAD_YAW])*b - transpose(gHR)*b;
  // the eyes aim at the lens; from a face turned away that is a sideways look.  Most of the way there reads warmer
  gGaze.x -= 0.52;
  saInit();       // last: the arms hang off the shape as posed
}

void cubeEval(vec3 pw){
  vec3 q = shapeSpace(pw);
  gQ = q;
  float host = cubeHost(q);
  // The rig tilts its frame about x only, and this face also looks away to the left.  Handing it a point turned about
  // the mouth centre swings the frame's z toward the camera in yaw too.  Depth still comes from the host, so the lips
  // stay on the flat face; what changes is that a pucker's tunnel and the mouth interior are seen down their axis
  float skin = lipsEval(LIP_O + gLipAim*(q - LIP_O), host);
  skin = faceEval(q, skin, host);
  gSkin = skin;
  gBare = min(gLipBare, gFEye);
}
void charEval(vec3 p){
  cubeEval(p);
  gcBase = smin(min(gSkin, gBare - 0.0007), saDist(p), SA_SKIN); gcBare = gBare; gcRest = gQ; gcLen = -0.0006;
}
float charTuft(){ return 0.0; }

//@shadow-begin
float charShadowHost(vec3 pw){
  vec3 q = gHR*(pw - gHO) + SHAPE_PIVOT;
  vec3 d = abs(q) - (0.665 - 0.13);
  return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0) - 0.13;
}
float charShadow(vec3 pw){ return min(charShadowHost(pw), saShadow(pw)); }
//@shadow-end

// A tall window in the glaze, drawn on a plane of reflection directions, so it slides when the block rocks.  W0 = where
// it hangs, g = tangent coordinates around that, WIN_S = half width and pane height in g.  One lies across the flank, big
// and clean.  One stands in the upper corner of the face, above the brow.  Near an eye it reads as a tear: it stays out
float winGlint(vec3 r, vec3 W0, vec2 S, float tilt, float top, float low, float ka, vec2 bw, float period){
  vec3 u = normalize(cross(vec3(0.0,1.0,0.0), W0)), v = cross(W0, u);
  float k = max(dot(r, W0), 1e-3);
  vec2 c = rot(tilt)*vec2(dot(r, u), dot(r, v))/k/S;         // 1 = half width, 1 = one pane in y
  c.x -= period*clamp(floor(c.x/max(period, 1e-3) + 0.5), -1.0, 1.0);
  float d = c.y < top ? max(abs(c.x) - 1.0, -c.y - low) : length(vec2(c.x, (c.y - top)*ka)) - 1.0;
  float bars = min(abs(c.x)*S.x/bw.x, abs(fract(c.y + 0.5) - 0.5)*S.y/bw.y);
  return smoothstep(0.03, -0.03, d)*smoothstep(0.55, 1.0, bars);
}

// the bay: three panes by two, bent a little by the surface and fading away from the window side.  q units, about
float bayGlint(vec3 r){
  vec3 W0 = vec3(-0.9314,0.1788,0.3172);
  vec3 u = normalize(cross(vec3(0.0,1.0,0.0), W0)), v = cross(W0, u);
  float k = max(dot(r, W0), 1e-3);
  vec2 c = rot(-0.19)*vec2(dot(r, u), dot(r, v))/k*vec2(1.0,1.19) - vec2(-0.048,0.036);
  c.y += 0.65*c.x*c.x;
  float d = max(abs(c.x) - 0.118, abs(c.y) - 0.066);
  float bx = abs(abs(c.x) - 0.040), by = abs(c.y - 0.006);
  float m = smoothstep(0.0015, -0.0015, d)*smoothstep(0.0024, 0.0046, min(bx, by));
  return m*mix(1.0, 0.34, smoothstep(-0.12, 0.12, c.x));
}

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  cubeEval(pw);
  if(saDist(pw) < gSkin){ saSurface(pw, rd, s); return; }
  vec3 q = gQ, n = s.n;

  // ---- where on the block: which face, how near an edge
  vec3 dq = abs(q) - (CUBE_H - CUBE_R), dp = max(dq, 0.0);
  float e2 = max(min(dp.x, dp.y), max(min(dp.x, dp.z), min(dp.y, dp.z)));        // >0 on a rounded edge
  float ins = max(min(dq.x, dq.y), max(min(dq.x, dq.z), min(dq.y, dq.z)));        // <0 how far in from the nearest edge
  vec3 nq = dot(dp, dp) > 1e-9 ? normalize(dp)*sign(q) : vec3(0.0, 0.0, 1.0);
  vec3 nh = normalize(nq*gHR);
  float nv = clamp(dot(nh, -rd), 0.0, 1.0);
  // every flat face is painted a little convex: the geometry stays flat for the mouth, the light does not
  vec3 tq = q - nq*dot(q, nq);
  vec3 tw = tq*gHR;
  s.n = normalize(n + 0.13*tw);

  // ---- the glaze: its thickness decides its colour
  vec3 qf = q; qf.xy -= lipsFlow();                                                      // the glaze's marks ride with the mouth
  float cloud = fbm3(qf*vec3(1.7,1.0,1.7) + 3.7);
  float run = vnoise3(qf*vec3(7.0,1.1,7.0) + 1.9);                                       // it ran, a little, in the kiln
  float lo = vnoise3(qf*5.3 + 2.0)*0.6 + vnoise3(qf*11.0)*0.4;                           // the slow wander of the edge
  float grit = vnoise3(qf*23.0)*0.55 + vnoise3(qf*57.0)*0.45;
  float th = 0.50 + 1.10*(cloud - 0.5) + 0.20*(run - 0.5) + 0.14*smoothstep(0.25, -0.65, q.y);
  vec3 mid  = vec3(0.0050,0.125,0.150);
  vec3 deep = vec3(0.0015,0.038,0.066);
  vec3 c = mix(mid*1.22, deep, smoothstep(0.28, 0.86, th));
  c = mix(c, vec3(0.008,0.215,0.200), 0.34*smoothstep(0.58, 0.30, th));                  // thin: a little greener
  // it pulls away from an edge, thin and pale at the crown, and gathers just inside it
  float ec = e2/(0.707*CUBE_R);
  float brk = smoothstep(0.74, 1.02, ec + 0.30*(lo - 0.5) + 0.06*(grit - 0.5));
  float pool = exp(-pow(max(-ins, 0.0)/0.115, 2.0))*(1.0 - smoothstep(0.0, 0.62, ec));
  c = mix(c, deep*0.55, 0.78*pool);
  c = mix(c, vec3(0.23,0.60,0.53), brk);
  // and it gathers in every crease of the face
  float cr = exp(-max(gLRing, 0.0)/0.024)*0.85;
  cr = max(cr, 0.55*exp(-max(gFLid, 0.0)/0.016)*step(0.0, gFEye));
  cr = max(cr, exp(-max(gFBrow, 0.0)/0.016)*0.8);
  c = mix(c, deep*0.42, 0.70*cr*step(0.0, q.z));

  // speckle: pale flecks and a few iron specks, the sparkle that holds the eye on a smooth surface
  vec3 g = qf*31.0, gi = floor(g), gf = fract(g) - 0.5, h = hash33(gi);
  float sp = smoothstep(0.15, 0.07, length(gf - (h - 0.5)*0.6) + 0.16*h.y)*step(0.72, h.x);
  vec3 g2 = qf*79.0, gi2 = floor(g2), h2 = hash33(gi2 + 7.0);
  float sp2 = smoothstep(0.20, 0.10, length(fract(g2) - 0.5 - (h2 - 0.5)*0.5))*step(0.58, h2.x);
  c = mix(c, h.z > 0.45 ? vec3(0.24,0.42,0.37) : vec3(0.030,0.018,0.014), 0.62*sp);
  c = mix(c, h2.z > 0.5 ? vec3(0.17,0.34,0.31) : vec3(0.012,0.022,0.024), 0.50*sp2);

  // crackle: the fine craze of a glaze that shrank a little faster than its clay.  Per face: it only has to be a net
  vec2 uv = abs(nq.z) > 0.5 ? qf.xy : (abs(nq.x) > 0.5 ? q.zy : q.xz);
  vec2 cg = uv*25.0 + 0.16*vec2(sin(uv.y*31.0), cos(uv.x*27.0)), cc = floor(cg), cf = fract(cg);
  float F1 = 9.0, F2 = 9.0;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 o = vec2(float(i), float(j)), rr = o + hash22(cc + o) - cf;
    float dd = dot(rr, rr);
    if(dd < F1){ F2 = F1; F1 = dd; } else if(dd < F2) F2 = dd;
  }
  float craze = smoothstep(0.070, 0.018, sqrt(F2) - sqrt(F1))*smoothstep(0.42, 0.74, vnoise3(q*3.1 + 8.0));
  c = mix(c, c*1.5 + vec3(0.012,0.030,0.028), 0.34*craze);

  // the foot: raw clay under the glaze line
  float fy = -CUBE_H + 0.062 + 0.014*sin(q.x*7.0 + 1.3*sin(q.z*5.0)) + 0.012*sin(q.z*13.0 + q.x*9.0) + 0.020*(run - 0.5);
  float foot = smoothstep(0.004, -0.004, q.y - fy);
  float roll = exp(-pow((q.y - fy - 0.012)/0.016, 2.0))*(1.0 - foot);
  c = mix(c, deep*0.5, 0.75*roll);

  // cheeks: a dab of coral underglaze, painted, not airbrushed.  Teal and coral grey each other out in a soft mix
  vec2 ck = vec2(abs(qf.x) - 0.470, qf.y + 0.076);
  float rc = length(ck*vec2(0.80,1.22)) - 0.058 - 0.016*gLCheek - 0.008*gLSmile;
  float blush = smoothstep(0.006, -0.006, rc)*step(0.5, q.z);
  c = mix(c, vec3(0.66,0.135,0.082)*(0.94 + 0.12*grit), 0.86*blush);

  s.alb = mix(c, vec3(0.42,0.27,0.17)*(0.75 + 0.5*grit), foot);
  s.spec = 0.0; s.mirror = 0.0; s.sheen = 0.0; s.wrap = 0.06; s.fur = 0.0;
  s.occ = mix(0.70, 1.0, smoothstep(-0.70, 0.15, q.y));

  // ---- the shine.  A hand-made surface: the reflection wobbles, its edges stay sharp
  vec3 wob = vec3(vnoise3(q*3.1), vnoise3(q*3.1 + 7.1), vnoise3(q*3.1 + 13.7)) - 0.5;
  vec3 wob2 = vec3(vnoise3(q*9.7 + 3.0), vnoise3(q*9.7 + 11.0), vnoise3(q*9.7 + 19.0)) - 0.5;
  vec3 nr = normalize((nq + 0.30*tq/CUBE_H + 0.036*wob + 0.0085*wob2)*gHR + 0.30*(n - nh));
  vec3 r = reflect(rd, nr);
  float Fr = 0.042 + 0.86*pow(1.0 - nv, 5.0);
  vec3 env = roomEnv(pw, r, 0.0)*mix(1.0, 0.62, smoothstep(0.5, 0.8, nq.x));          // the flank must not outshine the face
  float flat_ = exp(-dot(n - nh, n - nh)*60.0)*(1.0 - cr);          // the window lies on the open glaze, not on a lid
  // a bay of panes high on the forehead, running up and over the rounded edge; one tall window across the flank
  float glint = bayGlint(r)*flat_;
  float glintF = winGlint(r, vec3(0.6212,-0.1556,-0.7681), vec2(0.156,0.060), -0.14, 1.4, 2.6, 0.71, vec2(0.010,0.0034), 0.0);
  s.emit = (env*Fr + vec3(0.86,0.98,0.95)*0.54*glint + vec3(0.80,0.98,0.94)*0.30*glintF)*(1.0 - foot);
  // the panes that land on the top of the block
  float lid = smoothstep(0.75, 0.95, nh.y);
  float pane = beamMask(toRoom(pw) + vec3(-0.15,-1.30,0.0));
  s.alb *= 1.0 + 1.4*pane*lid;
  s.emit += env*Fr*0.35*pane*lid;

  vec3 em0 = s.emit;
  lipsPaint(n, s);
  // the rim is the same glaze as the block, and on its crest it breaks pale exactly as it does on the block's edges
  float lipN = gPLip*(1.0 - gPLine);
  if(lipN > 0.0){
    float lr = gLE/max(gLH, 1e-3);                                        // 0 inner edge .. 1 outer
    float crest = exp(-lipsSq((lr - 0.40 + 0.34*(lo - 0.5))/0.15))*smoothstep(0.30, 0.52, lo + 0.25*(run - 0.5))*smoothstep(-0.10, 0.45, gLDepthS);
    vec3 g = mix(s.alb, deep*0.7, 0.7*smoothstep(0.75, 1.1, lr));
    g = mix(g, vec3(0.23,0.60,0.53)*(0.85 + 0.3*grit), 0.80*crest);
    s.alb = mix(s.alb, g, lipN);
    vec3 rl = reflect(rd, n);
    // no room reflected on the grazing walls either side of it, nor on the rim's underside, which this high camera sees
    float top = smoothstep(-0.10, 0.45, gLDepthS);
    float onCrest = smoothstep(0.02, 0.28, lr)*smoothstep(0.92, 0.70, lr)*top;
    s.emit = mix(s.emit, roomEnv(pw, rl, 0.0)*(0.05 + 0.90*onCrest*pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 4.0)), lipN);
    // the underside belongs to the inside of the mouth
    float inside = (1.0 - top)*smoothstep(0.90, 0.62, lr)*lipN;
    s.alb = mix(s.alb, LIP_COL_DEEP, inside); s.emit = mix(s.emit, LIP_COL_DEEP*0.40, inside);
  }
  // the carve, wherever the rig's wall paint does not reach: the mouth's dark, never glaze.  Deeper than any lip can sit
  float sunk = smoothstep(0.010, 0.026, -cubeHost(q) - 0.050*LIPU*gLTuck)*(1.0 - gPTeeth)*(1.0 - gPTongue);
  vec3 cv = mix(LIP_COL_CAVE, LIP_COL_DEEP, smoothstep(-0.015*LIPU, -0.20*LIPU, gLQ.z));
  cv = mix(cv, LIP_COL_DEEP*0.7, max(gLDark, gLTgDim*0.8));
  s.alb = mix(s.alb, cv, sunk); s.emit = mix(s.emit, cv*0.34*(1.0 - gLDark), sunk);
  s.spec *= 1.0 - sunk; s.mirror *= 1.0 - sunk; s.sheen *= 1.0 - sunk; s.occ = mix(s.occ, 0.55, sunk);
  facePaint(q, rd, n, gSkin, s);
}

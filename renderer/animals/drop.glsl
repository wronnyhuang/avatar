// ---------------------------------------------------------------------------
//  animals/drop.glsl — a plump teardrop of clear golden-amber liquid, held in surface tension on a huge leaf.
//  docs/SHAPE_GUIDE.md is the recipe; animals/orb.glsl the worked example.  Head space: origin at the centre of
//  the bulb, +x screen right, +y up, +z toward the camera, and at rest head space IS world space.
//
//  The liquid is faked in charSurface, no rays: a two-bounce ball lens worked out from the host normal alone
//  gives the upside-down world in s.emit, Beer-Lambert along the same chord gives the amber.
//
//  The tip: one frame's params hold no history, so a true lag cannot be read here.  ear_l and ear_r are the tip's
//  channels (x and z, in units of TIP_REACH): real params bring a slow independent sway through them, and
//  tools/drop_tip_lag.py rewrites them to the spring-lagged head motion.  On top of that the tip holds partly
//  upright against whatever head_roll and head_pitch do now.
// ---------------------------------------------------------------------------
const float R = 0.66;                               // the bulb, a sphere where the face sits
#define BASE_Y (-0.575)                              // the leaf

#define SHAPE_PIVOT vec3(0.0,BASE_Y,0.0)
#define SHAPE_WORLD vec3(0.0,BASE_Y,0.0)
#define SHAPE_SQUASH 0.020                          // breath: a slow swell
#define SHAPE_JIGGLE 0.008
// a storybook face: high forehead (the whole tip), big flush button eyes low and wide, no nose
#define EYE_R 0.172
#define EYE_C vec3(0.248,0.030,0.482)
#define EYE_AU 0.76
#define EYE_AL (-0.74)
#define EYE_AC (-0.34)
#define EYE_TILT 0.18
#define EYE_IRIS 0.88
#define EYE_PUPIL 0.54
#define EYE_COL_A vec3(0.20,0.42,0.20)
#define EYE_COL_B vec3(0.018,0.060,0.040)
#define BROW_W 0.017
#define BROW_L 0.118
#define BROW_COL vec3(0.30,0.115,0.035)
#define BROW_C vec3(0.250,0.280,0.0)
#define LINER_COL vec3(0.13,0.045,0.025)
#define LIP_TILT (-0.06)                            // aims at the camera, which looks a little down at the drop
#define LIP_O vec3(0.0,-0.2570,0.6079)              // on the surface
#define LIP_W 0.262                                 // a child's mouth still, wide enough to read
#define LIP_DEPTH (0.54*LIPU)                       // the tongue must not meet the cavity's end wall
#define LIP_TOOTH_Z (-0.095*LIPU)                   // deeper than the default: the frame runs out of the sphere below
#define LIP_HU (0.056*LIPU)                         // swollen with surface tension, not drawn on
#define LIP_HL (0.072*LIPU)
#define LIP_TU (0.036*LIPU)
#define LIP_TL (0.044*LIPU)
#define LIP_BOW 0.45
#define LIP_SQUARE 0.42
#define LIP_OPEN_Y 0.86                             // the leaf is close under the chin
#define LIP_BLEND (0.026*LIPU)                      // they run into the body like a meniscus
#define LIP_SKIN 0.60
#define LIP_TEETH_H (0.060*LIPU)
#define LIP_TOOTH_GAIN 1.30                         // the body glows: the wax has to stand up to it
#define LIP_COL_U vec3(0.34,0.092,0.012)
#define LIP_COL_L vec3(0.40,0.115,0.014)
#define LIP_COL_LINE vec3(0.085,0.020,0.006)
#define LIP_COL_CAVE vec3(0.30,0.075,0.012)         // treacle: dark amber, never red
#define LIP_COL_DEEP vec3(0.075,0.018,0.006)
#define LIP_COL_GUM  vec3(0.44,0.13,0.03)
#define LIP_COL_TONGUE vec3(0.92,0.38,0.14)
#define LIP_COL_TOOTH vec3(0.98,0.88,0.62)           // honeycomb wax
#define LIP_SPEC 0.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.0
#define LIP_EMIT vec3(0.0)
#pragma include face
#pragma include lips
// real 3-D arms with mitten hands on the contract's hand columns (engine/shape_arms.glsl, docs/RIG.md)
#define SA_SHO vec3(0.570,-0.280,0.050)
#define SA_REST vec3(0.140,-0.340,0.220)
#define SA_LEN 0.780
#define SA_R 0.082
#define SA_HAND 0.160
#define SA_ALB vec3(0.80,0.42,0.06)
#define SA_HALB vec3(0.88,0.55,0.14)
#define SA_SPEC 0.6
#define SA_MIRROR 0.20
#pragma include shape_arms

#define TIP_Y 1.10                                  // centre of the tip's round end
#define TIP_R 0.058
#define TIP_CURL 0.15                               // the tip's rest curl toward screen right
#define TIP_REACH 0.20                              // ear_l/ear_r = 1 moves the tip this far

const vec3  SIGMA  = vec3(0.36,1.22,3.30);          // absorption per unit of path
const vec3  CAUS_D = vec3(0.470,-0.838,0.276);      // head of the caustic: low, opposite the sun

float gSkin, gBare;
vec3  gQ;
vec2  gTipA;                                        // axis of the tip: its offset at unit height above the shoulder
float gRipA, gRipP;                                 // the voice's ripple: amplitude and phase

void charInit(){
  faceInit(); lipsInit();
  float T = uTime;
  float H = 1.7;                                    // shape height: what a head turn moves the tip by
  gTipA = vec2(TIP_CURL + 0.022*sin(T*0.71) + 0.012*sin(T*1.27 + 1.0), 0.016*sin(T*0.59 + 2.0));
  gTipA += TIP_REACH*vec2(uP[P_EAR_L], uP[P_EAR_R]);
  gTipA += vec2(0.05*(uP[P_EAR_FLICK_R] - uP[P_EAR_FLICK_L]), 0.0);
  gTipA -= 0.45*H*vec2(uP[P_HEAD_ROLL] + 0.25*uP[P_HEAD_X], uP[P_HEAD_PITCH]);
  gRipA = 0.017*clamp(uP[P_VOICE], 0.0, 1.0);
  gRipP = T*14.0;
  gLBend += 0.011*(1.0 - 0.55*smoothstep(0.0, 0.6, clamp(uP[P_LIP_OPEN], 0.0, 1.0)));   // a shade more of a smile
  saInit();       // last: the arms hang off the shape as posed
}

float sdRoundCone2(vec2 q, float r1, float r2, float h){
  float b = (r1 - r2)/h, a = sqrt(1.0 - b*b);
  float k = dot(q, vec2(-b, a));
  if(k < 0.0) return length(q) - r1;
  if(k > a*h) return length(q - vec2(0.0, h)) - r2;
  return dot(q, vec2(a, b)) - r1;
}

// the plain host, head space.  Exactly a sphere from the eyes down, so the lips ride a true distance
float dropHost(vec3 q){
  float up = max(q.y - 0.10, 0.0);
  float rip = gRipA*sin(7.5*q.y - gRipP)*smoothstep(0.16, 1.05, q.y);
  vec2 ax = gTipA*up*up + vec2(rip, 0.0);
  vec3 b = vec3(q.x - ax.x, q.y, q.z - ax.y);
  float bulb = length(q) - R;
  float tip = sdRoundCone2(vec2(length(b.xz), b.y - 0.18), 0.355, TIP_R, TIP_Y - 0.18);
  return smin(bulb, tip, 0.34);
}

void dropEval(vec3 pw){
  vec3 q = shapeSpace(pw);
  gQ = q;
  float host = smax(dropHost(q), BASE_Y + 0.004 - pw.y, 0.075);      // sits flat on the leaf whatever the head does
  float skin = lipsEval(q, host);
  // the rig's brow leans its inner end up, which on this face reads as worry: shear the brow band the other way
  vec3 qb = q;
  qb.y -= 0.13*(abs(q.x) - BROW_C.x)*smoothstep(0.20, 0.235, q.y);
  skin = faceEval(qb, skin, host);
  gSkin = skin;
  gBare = min(gLipBare, gFEye);
}
void charEval(vec3 p){
  dropEval(p);
  gcBase = smin(min(gSkin, gBare - 0.0007), saDist(p), SA_SKIN); gcBare = gBare; gcRest = gQ; gcLen = -0.0006;
}
float charTuft(){ return 0.0; }

//@shadow-begin
float charShadowHost(vec3 pw){
  vec3 q = gHR*(pw - gHO) + SHAPE_PIVOT;
  float tip = sdRoundCone2(vec2(length(q.xz), q.y - 0.18), 0.33, 0.03, 0.90);
  return min(length(q) - R*0.96, tip);
}
float charShadow(vec3 pw){ return min(charShadowHost(pw), saShadow(pw)); }
//@shadow-end

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  dropEval(pw);
  if(saDist(pw) < gSkin){ saSurface(pw, rd, s); return; }
  vec3 q = gQ, n = s.n;
  // everything optical follows the HOST's normal, not the shaded one: features must not grow a rainbow fringe
  vec2 he = vec2(1.0,-1.0)*0.006;
  vec3 gh = he.xyy*dropHost(q + he.xyy) + he.yyx*dropHost(q + he.yyx) + he.yxy*dropHost(q + he.yxy) + he.xxx*dropHost(q + he.xxx);
  vec3 nh = normalize(gh*gHR);
  float nv = clamp(dot(nh, -rd), 0.0, 1.0);
  float fr = 1.0 - nv;

  // a slow current in the skin, and the voice shivering it: the world in the drop is never quite still
  vec3 wob = vec3(vnoise3(q*2.1 + vec3(0.0, uTime*0.13, 0.0)), vnoise3(q*2.1 + vec3(5.2, 1.3 - uTime*0.11, 0.0)), 0.5) - 0.5;
  vec3 nl = normalize(nh + wob*(0.050 + 0.040*gRipA/0.017));

  // ---- the ball lens: in through the front, out through the back.  With a sphere the exit normal is the entry
  // normal mirrored about the inner ray, so it needs no centre and no radius
  const float ETA = 1.36;
  vec3 r1 = refract(rd, nl, 1.0/ETA);
  vec3 nb = reflect(nl, r1);
  vec3 r2 = refract(r1, -nb, ETA);
  if(dot(r2, r2) < 0.5) r2 = rd;
  float hy = smoothstep(0.05, 1.10, q.y);
  float Rl = mix(R, 0.07, hy);                                       // the local radius of the form
  float L = 2.0*Rl*max(-dot(r1, nl), 0.0);
  vec3 T = exp(-SIGMA*L);

  gHzSoft = 0.16; gVeinK = 0.50;                                     // soft enough never to draw lines across the face
  r2 = normalize(r2 + vec3(0.19,-0.24,0.0));                         // art direction: the sun's image and the sky sit low and right
  vec3 seen = setSky(r2);
  gHzSoft = 0.015; gVeinK = 1.0;
  float lum = min(dot(seen, vec3(0.30,0.55,0.15)), 0.60);            // no sun disc on the face: the core below is its image
  // honey: how much light gets through sets the colour.  The sky is the bright half, and the lens turns it upside down
  // and the light gathers on the side away from the sun: the reverse of a solid, which is what reads as clear
  float away = smoothstep(-0.35, 0.92, dot(nh, normalize(vec3(0.56,-0.70,0.44))));
  float x = (lum*(0.70 + 0.80*away) + 0.92*smoothstep(-0.10, 0.55, r2.y)*away)*exp(-0.74*L);
  vec3 body = mix(vec3(0.21,0.048,0.004), vec3(0.74,0.205,0.010), smoothstep(0.0, 0.30, x));
  body = mix(body, vec3(0.98,0.30,0.018), smoothstep(0.24, 0.62, x));
  body = mix(body, vec3(1.22,0.48,0.045), smoothstep(0.62, 1.30, x));
  // the world's own colour survives: the leaf green and low in the tip, the flower upside down low on the left
  body *= mix(vec3(1.0), seen/(lum + 0.06), 0.07 + 0.12*smoothstep(0.05, 0.60, q.y));
  float pink = clamp((seen.r - seen.g)*2.4, 0.0, 1.0);
  body = mix(body, seen*vec3(1.15,0.60,0.46) + vec3(0.14,0.02,0.0), 0.62*pink);

  // ---- the sun gathered: a soft hot core low in the bulb, opposite the sun, seen along the ray inside
  vec3 cw = gHO + (vec3(0.0) - SHAPE_PIVOT)*gHR;                     // the bulb's centre, world
  vec3 f = cw + CAUS_D*R*0.80;
  vec3 w = f - pw;
  float s0 = clamp(dot(w, r1), 0.0, L);
  float dd = dot(w - r1*s0, w - r1*s0);
  float cor = exp(-dd/0.018), glow = exp(-dd/0.13);
  // and the crescent it throws on the inside of the skin
  float cd = dot(nh, CAUS_D);
  float ripple = vnoise3(q*6.0 + vec3(0.0, uTime*0.31, 0.7))*0.6 + vnoise3(q*13.0 + vec3(uTime*0.23, 0.0, 2.0))*0.4;
  float cres = smoothstep(0.52 + 0.14*ripple, 0.86, cd)*smoothstep(0.995, 0.90, cd + 0.10*(1.0 - nv));
  // fine moving filaments where the light lands, as on the floor of a pool
  float na = vnoise3(q*vec3(5.0,7.0,5.0) + vec3(0.0, uTime*0.27, uTime*0.19));
  float nc = vnoise3(q*vec3(9.0,12.0,9.0) + vec3(uTime*0.21, 3.0, -uTime*0.17));
  float fil = pow(1.0 - abs(2.0*na - 1.0), 7.0)*0.7 + pow(1.0 - abs(2.0*nc - 1.0), 9.0)*0.6;
  float fm = smoothstep(0.46, 0.92, away)*smoothstep(-0.14, -0.44, q.y);
  body += vec3(0.90,0.28,0.016)*glow*0.85 + vec3(2.0,0.98,0.16)*cor + vec3(1.7,0.82,0.13)*cres*0.8 + vec3(0.95,0.46,0.07)*fil*fm;

  // darker amber where the path is long is T; a refracted dark band runs just inside the silhouette
  float band = smoothstep(0.58, 0.82, fr)*smoothstep(1.02, 0.90, fr);
  body *= 1.0 - 0.62*band;

  // ---- the skin mirrors the world
  vec3 r = reflect(rd, nh);
  vec3 env = setSky(r);
  float Fr = 0.024 + 0.976*pow(fr, 5.0);
  // one sharp window of sky on the sun side, inside a small bloom, clear of the face
  vec3 W = normalize(vec3(-0.70,0.71,0.08));
  vec3 Wx = normalize(cross(W, vec3(0.0,1.0,0.0))), Wy = cross(Wx, W);
  float wz = max(dot(r, W), 1e-3);
  vec2 wv = vec2(dot(r, Wx), dot(r, Wy))/wz;
  wv.x += 0.32*wv.y*wv.y;
  vec2 wd = abs(wv) - vec2(0.12, 0.21);
  float wsd = length(max(wd, 0.0)) + min(max(wd.x, wd.y), 0.0) - 0.05;
  float win = smoothstep(0.012, -0.012, wsd)*step(0.0, dot(r, W));
  vec3 gloss = vec3(1.0,0.98,0.92)*(win*(0.34 + 0.62*smoothstep(-0.3, 0.3, wv.y)) + 0.10*pow(max(dot(r, W), 0.0), 9.0));
  // a smaller second light, low on the far side: the bright leaf
  gloss += vec3(1.0,0.95,0.70)*0.34*smoothstep(0.9925, 0.9965, dot(r, normalize(vec3(0.71,-0.67,-0.21))));
  // a long thin bright crescent down the far side: the mist behind it, bent round the edge
  float crs = smoothstep(0.70, 0.76, fr)*smoothstep(0.86, 0.80, fr)*smoothstep(0.10, 0.55, nh.x)*smoothstep(-0.75, -0.35, nh.y)*smoothstep(0.75, 0.30, nh.y);
  gloss += vec3(1.0,0.86,0.56)*0.42*crs;

  s.alb = vec3(0.115,0.052,0.010)*(0.55 + 0.45*smoothstep(-0.2, 0.7, dot(nh, uSunDir)));   // just enough to carry the soft moves
  // cheeks, a soft rose that rises with `cheek`
  vec2 qk = q.xy - lipsFlow();
  vec2 ck = vec2(abs(qk.x) - 0.405, qk.y + 0.170);
  float blush = exp(-dot(ck*vec2(1.0,1.2), ck*vec2(1.0,1.2))/0.017);
  float bl = (0.22 + 0.30*gLCheek + 0.10*gLSmile)*blush;
  body = mix(body, body*vec3(1.04,0.62,0.56) + vec3(0.10,0.012,0.022), bl);
  s.alb = mix(s.alb, vec3(0.70,0.22,0.20), 0.55*bl);

  s.occ = 1.0; s.spec = 0.0; s.mirror = 0.0; s.sheen = 0.0; s.wrap = 0.18; s.fur = 0.0;
  s.emit = mix(body, env*1.05, Fr) + gloss;

  // the mouth is opaque: no gold may survive on the carved walls, whose grazing normals turn it into flecks
  float cav = smoothstep(0.024*LIPU, 0.007*LIPU, gLHoleD);
  s.emit *= 1.0 - cav;
  s.alb = mix(s.alb, LIP_COL_DEEP, cav);

  // the skin settles dark against each eye all the way round: the inner edge meets a dark line, never bare gold
  float sock = smoothstep(0.024, 0.004, gFEye);
  s.emit *= mix(1.0, 0.30, sock);
  s.alb = mix(s.alb, LINER_COL, 0.55*sock);

  vec3 em0 = s.emit;
  lipsPaint(n, s);
  // the lips are the honey itself, swollen.  Light comes through their crest, and where the surface turns off the body's
  // own curve the path through it runs long and goes dark, as along the drop's edge.  One small wet light rides each
  float lipN = gPLip*(1.0 - gPLine);
  float lrun = gLE/max(gLH, 1e-3);                                    // never on the closed line: it would read as a pearl
  float bend = 1.0 - clamp(dot(n, nh), 0.0, 1.0);
  vec3 honey = mix(vec3(1.10,0.52,0.070), vec3(0.42,0.085,0.008), smoothstep(0.02, 0.26, bend));
  s.emit = mix(s.emit, honey*(0.80 + 0.45*smoothstep(0.05, 0.60, q.y + 0.70)) + 0.35*em0, lipN);
  float lg = max(dot(reflect(rd, n), normalize(vec3(-0.22,0.70,0.68))), 0.0);
  s.emit += vec3(1.0,0.92,0.80)*gPLip*smoothstep(0.20, 0.42, lrun)*smoothstep(1.0, 0.72, lrun)*(0.70*pow(lg, 34.0) + 0.06*pow(lg, 6.0));
  facePaint(q, rd, n, gSkin, s);
  // the eye right at the socket's edge has broken normals: no catch-light may fire there, even with the head turned
  float eyeEdge = step(gFEye, 0.002)*smoothstep(0.018, 0.004, gSkin);
  s.emit *= 1.0 - eyeEdge; s.spec *= 1.0 - eyeEdge;
}

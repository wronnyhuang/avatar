// ---------------------------------------------------------------------------
//  animals/star.glsl — a puffy five-pointed stuffed star, golden velvet, with real lips.  Head space: origin at
//  the middle of the pillow, +x screen right, +y up, +z toward the camera.  docs/SHAPE_GUIDE.md is the recipe.
//
//  The host is two things blended: a domed pillow (an ellipsoid) that owns the face, and five round cones, flatter
//  than they are wide, that sink inside that dome.  Over the face the dome is the nearest surface, so the host
//  there is a true distance with no crease; the arms only come out past the edge of the mouth and eyes.
//
//  Tips move by a warp of the rest space (gR), not by moving the cones: paint and nap stick to the cloth.
//  The warp is exactly 0 where the face sits.
// ---------------------------------------------------------------------------
const float DOME_R = 0.665, DOME_Z = 0.545;          // the pillow
const float ARM_R0 = 0.600, ARM_ZS = 0.84;           // radius of each arm at the centre, its depth squash
const float ARM_CUT = 0.52;                          // where along its axis each arm begins
const float NAP = 0.024;                             // pile height
const float PUFF = 0.0125;                           // breath: the whole pillow swells, about its middle
const float SEAM_IN = 0.058;                         // how far inside the outline the seam runs
const float SP_TAU = 6.2831853/5.0;

// arm directions, clockwise from the top; a hand-sewn star is never quite regular
const vec2  ARM_D[5] = vec2[5](vec2(0.0,1.0), vec2(0.95106,0.30902), vec2(0.58779,-0.80902), vec2(-0.58779,-0.80902), vec2(-0.95106,0.30902));
const float ARM_L[5] = float[5](0.985, 0.975, 0.955, 0.965, 0.990);   // centre of the tip
const float ARM_T[5] = float[5](0.205, 0.215, 0.225, 0.220, 0.210);   // tip radius
// where each tip rests, off the regular star, and how each one answers the voice
const vec3  TIP_REST[5] = vec3[5](vec3(0.045,-0.012,0.050), vec3(0.000,-0.040,0.085), vec3(0.018,0.012,0.060), vec3(-0.012,0.006,0.090), vec3(0.008,0.024,0.060));
const float TIP_HZ[5]  = float[5](2.05, 2.45, 1.85, 2.25, 2.60);
const float TIP_LAG[5] = float[5](0.00, 0.085, 0.170, 0.125, 0.045);  // seconds behind the top tip

#define SHAPE_PIVOT vec3(0.0,0.0,0.0)
#define SHAPE_WORLD vec3(0.0,0.0,0.0)
#define SHAPE_BOB 0.020
#define SHAPE_SQUASH 0.0                            // breath and voice do their own thing here, see charInit
#define SHAPE_JIGGLE 0.0
// the storybook face from the orb, set on the pillow
#define EYE_R 0.205
#define EYE_C vec3(0.2307,0.0364,0.3518)
#define EYE_AU 0.72
#define EYE_AL (-0.76)
#define EYE_AC (-0.34)
#define EYE_TILT 0.13
#define EYE_OUT 1.02
#define EYE_IRIS 0.95
#define EYE_PUPIL 0.66
#define EYE_COL_A vec3(0.66,0.25,0.040)
#define EYE_COL_B vec3(0.070,0.028,0.014)
#define LINER_COL vec3(0.22,0.075,0.030)
#define BROW_W 0.030
#define BROW_L 0.155
#define BROW_H 0.024
#define BROW_COL vec3(0.34,0.145,0.060)
#define BROW_C vec3(0.292,0.318,0.0)
#define LIP_TILT 0.08                               // aims at the camera; the dome's normal here leans 0.34
#define LIP_O vec3(0.0,-0.2780,0.4950)              // on the dome
#define LIP_W 0.292                                 // a toy's mouth: wide, flat, sewn on
#define LIP_HU (0.056*LIPU)                         // a strip of felt, not a cushion
#define LIP_HL (0.064*LIPU)
#define LIP_TU (0.017*LIPU)
#define LIP_TL (0.019*LIPU)
#define LIP_BOW 0.10
#define LIP_SQUARE 0.42
#define LIP_BOXR 0.72
#define LIP_RIM 0.65
#define LIP_FLOW 0.65
#define LIP_OPEN_Y 1.06
#define LIP_BLEND (0.010*LIPU)
#define LIP_SKIN 0.70
#define LIP_TOOTH_Z (-0.070*LIPU)                   // deeper than the default: the tilted frame runs out of the dome below
#define LIP_SCALLOP 0.0                             // one strip of cream felt
#define LIP_GROOVE 0.0
#define LIP_TOOTH_GAIN 1.20
#define LIP_TEETH_H (0.066*LIPU)
#define LIP_COL_U vec3(0.44,0.040,0.105)
#define LIP_COL_L vec3(0.52,0.060,0.125)
#define LIP_COL_LINE vec3(0.090,0.012,0.040)
#define LIP_COL_CAVE vec3(0.24,0.040,0.105)         // the dark felt behind
#define LIP_COL_DEEP vec3(0.060,0.012,0.042)
#define LIP_COL_GUM  vec3(0.30,0.05,0.10)
#define LIP_COL_TONGUE vec3(0.94,0.34,0.28)
#define LIP_COL_TOOTH vec3(0.96,0.86,0.66)
#define LIP_SPEC 0.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.55
#define LIP_WRAP 0.45
#define LIP_EMIT vec3(0.10,0.05,0.06)
#pragma include face
#pragma include lips
// real 3-D arms with mitten hands on the contract's hand columns (engine/shape_arms.glsl, docs/RIG.md)
#define SA_SHO vec3(0.520,-0.170,0.100)
#define SA_REST vec3(0.200,-0.380,0.250)
#define SA_LEN 0.800
#define SA_R 0.084
#define SA_HAND 0.170
#pragma include shape_arms

float sqr2(float x){ return x*x; }
float gSkin, gBare, gFurM, gBlend;
vec3  gQ, gR;
vec3  gTipD[5];

void charInit(){
  faceInit(); lipsInit();
  float T = uTime, br = uP[P_BREATH], v = clamp(uP[P_VOICE], 0.0, 1.0);
  // hangs in the air, a little off true, rocking very slowly on top of what head_* do
  gHR = gHR*rotZ(0.042 + 0.016*sin(T*0.59 + 0.7) + 0.007*sin(T*1.37));
  gSqX = gSqY = 1.0 + PUFF*br;
  gBlend = 0.180 + 0.016*br;                       // a full breath fills the notches a touch
  // stuffing has no fast moves: a slow sway all the time, and each tip shaken by the voice at its own rate, its
  // own time behind the top one.  The params hold one instant, so the lag is a phase, not a delay line.
  for(int i=0;i<5;i++){
    float fi = float(i);
    vec2 ax = ARM_D[i], tg = vec2(ax.y, -ax.x);
    float ph = 6.2832*TIP_HZ[i]*(T - TIP_LAG[i]);
    float w = sin(ph) + 0.35*sin(1.63*ph + fi*2.1);
    float idle = 0.0055*sin(T*(0.61 + 0.13*fi) + fi*1.9);
    float a = 0.023*v*w + idle;
    gTipD[i] = TIP_REST[i] + vec3(tg*a + ax*0.28*a, 0.014*v*sin(ph*0.77 + fi));
  }
  // the two side tips are this character's ears
  gTipD[4] += vec3(0.0, 0.016*uP[P_EAR_L] + 0.020*uP[P_EAR_FLICK_L], 0.0);
  gTipD[1] += vec3(0.0, 0.016*uP[P_EAR_R] + 0.020*uP[P_EAR_FLICK_R], 0.0);
  saInit();       // last: the arms hang off the shape as posed
}

// an arm: a round cone, r0 at the centre to rt at L, in (radius across, length along), flattened in z
float starArm(vec3 r, vec2 ax, float L, float rt, float r0){
  vec2 c = vec2(r.x*ax.y - r.y*ax.x, r.z/ARM_ZS);
  vec2 q = vec2(length(c), dot(r.xy, ax));
  float b = (r0 - rt)/L, a = sqrt(1.0 - b*b);
  float k = dot(q, vec2(-b, a));
  float d = dot(q, vec2(a, b)) - r0;
  if(k < 0.0) d = length(q) - r0;
  if(k > a*L) d = length(q - vec2(0.0, L)) - rt;
  // the squash, undone to first order: the field keeps a gradient near 1
  d *= mix(1.0, ARM_ZS, c.y*c.y/(dot(c, c) + 1e-4));
  // the arm starts out under the dome.  Five cones meeting at the centre would pile their blends up and swell the face
  return smax(d, ARM_CUT - q.y, 0.16);
}
float starHost(vec3 r){
  float a = 1e3;
  for(int i=0;i<5;i++) a = smin(a, starArm(r, ARM_D[i], ARM_L[i], ARM_T[i], ARM_R0), 0.15);
  return smin(sdEllipsoid(r, vec3(DOME_R, DOME_R, DOME_Z)), a, gBlend);
}

void starEval(vec3 pw){
  vec3 q = shapeSpace(pw);
  gQ = q;
  vec3 r = q;
  for(int i=0;i<5;i++){
    float w = smoothstep(0.46, 1.16, dot(q.xy, ARM_D[i]));
    r -= gTipD[i]*w*w;
  }
  gR = r;
  // the tuck is a real groove in the front panel: its near lip catches the key, its far lip falls into shade.  The
  // host cut at z = 0 is the outline, so the groove stays parallel to the edge, tips included; it never reaches the face
  float seam = starHost(vec3(r.xy, 0.0)) + SEAM_IN;
  float host = starHost(r) + 0.014*exp(-seam*seam/(0.025*0.025))*smoothstep(-0.02, 0.02, r.z);
  float skin = lipsEval(q, host);
  skin = faceEval(q, skin, host);
  gSkin = skin;
  gBare = min(gLipBare, gFEye);
  // no pile on the lips, the eyes or the brows, nor anywhere inside the mouth
  // measured along the skin, not through the air: the height above whatever is here comes off, or the pile would
  // stand on a bubble over the eye
  float z = min(min(gLRing, gLHoleD), min(min(gFEye, gFLid), gFBrow));
  gFurM = smoothstep(0.003, 0.030, z - max(min(skin, gBare), 0.0));
}
void charEval(vec3 p){
  starEval(p);
  gcBase = smin(min(gSkin, gBare - 0.0007), saDist(p), SA_SKIN); gcBare = gBare; gcRest = gR; gcLen = NAP*gFurM - 0.0006;
}
// short dense pile, brushed outward along the arms: the noise is stretched along the radius
float charTuft(){
  vec3 c = gcRest;
  vec2 R = c.xy/(length(c.xy) + 0.05);
  c -= vec3(R, 0.0)*dot(c.xy, R)*0.60;
  float a = vnoise3(c*125.0);
  float b = vnoise3(c*215.0 + 4.1);
  float h = clamp(0.36 + 0.46*a + 0.34*b, 0.0, 1.0);
  return mix(0.86, h, smoothstep(0.006, 0.018, gcLen));
}

//@shadow-begin
float charShadowHost(vec3 pw){
  vec3 q = gHR*(pw - gHO) + SHAPE_PIVOT;
  float d = length(q) - 1.34;
  // inside the real body: the cones and the dome, shrunk.  A loop, so rays far from the star skip it
  for(int k=0;k<1 && d < 0.25;k++){
    float a = 1e3;
    for(int i=0;i<5;i++) a = min(a, starArm(q, ARM_D[i], ARM_L[i], ARM_T[i]*0.90, ARM_R0*0.95));
    d = min(a, length(q*vec3(1.0,1.0,1.22)) - 0.640);
  }
  return d;
}
float charShadow(vec3 pw){ return min(charShadowHost(pw), saShadow(pw)); }
//@shadow-end

// where the seam runs: distance along it, folded onto one arm so every tip gets its stitches.  e runs along the arm's
// straight edge, which also crosses the notch cleanly; round the tip it is arc length about the tip's centre
float seamCoord(vec2 p){
  float a = atan(p.x, p.y);
  float k = floor(a/SP_TAU + 0.5);
  vec2 l = rot(k*SP_TAU)*p;
  l.x = abs(l.x);
  const vec2 e = vec2(-0.455,0.890), o = vec2(0.890,0.455), c = vec2(0.0,0.975);
  vec2 v = l - c;
  float ve = dot(v, e);
  return ve < 0.0 ? dot(l, e) : dot(c, e) + 0.150*atan(ve, dot(v, o));
}

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  starEval(pw);
  if(saDist(pw) < gSkin){ saSurface(pw, rd, s); return; }
  vec3 q = gQ, r = gR, n = s.n;
  r.xy -= lipsFlow();                               // the cloth rides with the mouth: it is stretched, the lips do not slide over it
  // the cloth's own normal, without the features
  vec3 nq = vec3(0.0);
  for(int i=0;i<4;i++){
    vec3 k = 0.5773*(2.0*vec3(float(((i+3)>>1)&1), float((i>>1)&1), float(i&1)) - 1.0);
    nq += k*starHost(r + k*0.012);
  }
  nq = normalize(nq);
  vec3 nh = nq*gHR;                                 // head -> world
  float nv = clamp(dot(nh, -rd), 0.0, 1.0), fr = 1.0 - nv;

  // ---- soft lumps of stuffing, and a soft crease running in from each notch where the cloth is gathered
  float lm = vnoise3(r*3.4 + 1.3);
  vec3 lg = vec3(vnoise3(r*3.4 + 1.3 + vec3(0.05,0.0,0.0)), vnoise3(r*3.4 + 1.3 + vec3(0.0,0.05,0.0)), vnoise3(r*3.4 + 1.3 + vec3(0.0,0.0,0.05))) - lm;
  float an = atan(r.x, r.y) - SP_TAU*0.5;
  float kn = floor(an/SP_TAU + 0.5);
  vec2 ln = rot(kn*SP_TAU + SP_TAU*0.5)*r.xy;       // +y out through the notch, x across it
  float wob = 0.018*sin(ln.y*11.0 + kn*2.0);
  float crease = exp(-sqr2((ln.x + wob)/0.024))*smoothstep(0.47, 0.70, ln.y)*smoothstep(0.84, 0.74, ln.y);
  crease *= 1.0 - smoothstep(0.30, 0.06, abs(r.x))*step(r.y, 0.0);   // none under the chin: it reads as a cleft
  vec3 ncr = vec3(rot(-(kn*SP_TAU + SP_TAU*0.5))*vec2(-sign(ln.x)*smoothstep(0.0, 0.03, abs(ln.x + wob))*crease, 0.0), 0.0);

  // ---- velvet: pile that has been stroked this way and that, so patches take the light differently
  vec3 c = r;
  vec2 Rd = c.xy/(length(c.xy) + 0.05);
  c -= vec3(Rd, 0.0)*dot(c.xy, Rd)*0.60;
  float grain = vnoise3(c*125.0)*0.6 + vnoise3(c*215.0 + 4.1)*0.4;
  float strokes = vnoise3(r*vec3(2.6,2.6,3.4) + 3.0)*0.62 + vnoise3(r*vec3(6.7,6.7,8.0) + 9.0)*0.38;
  float crush = smoothstep(0.34, 0.66, strokes);
  vec3 gold = mix(vec3(0.97,0.410,0.034), vec3(1.0,0.480,0.050), crush);
  gold = mix(gold, vec3(1.0,0.56,0.095), 0.45*smoothstep(0.55, 0.92, nv));   // a paler bloom face on, on top of the form
  gold *= 0.875 + 0.25*grain;

  // the tips and the edge go a deeper, redder gold: the form turns away and the pile closes up
  float edge = smoothstep(0.30, 0.96, fr);
  gold = mix(gold, gold*vec3(0.80,0.54,0.34), 0.72*edge);
  float rr = length(r.xy);
  gold = mix(gold, gold*vec3(0.98,0.84,0.70), 0.45*smoothstep(0.72, 1.20, rr));

  // ---- the seam: a tuck that runs round the edge, a running stitch in it, the cloth gathered either side
  float o2 = starHost(vec3(r.xy, 0.0));
  // the blend in the notches stretches this field: true the distance, or the tuck and the thread go fat there
  float g2 = length(vec2(starHost(vec3(r.x + 0.01, r.y, 0.0)), starHost(vec3(r.x, r.y + 0.01, 0.0))) - o2)/0.01;
  float sd = (o2 + SEAM_IN)/max(g2, 0.4);
  float ss = seamCoord(r.xy);
  float front = smoothstep(-0.02, 0.02, r.z);
  float P = 0.070, fs = (fract(ss/P) - 0.5)*P;
  float gath = cos(fs*6.2832/P);                    // +1 at the middle of a stitch, -1 at the holes
  float bandW = 0.0120 + 0.0034*gath;
  float tuck = exp(-sd*sd/(bandW*bandW))*front;
  float wide = exp(-sd*sd/(0.040*0.040))*front;
  gold = mix(gold, gold*vec3(0.68,0.50,0.36), 0.85*wide);
  gold = mix(gold, vec3(0.170,0.044,0.008), 0.92*tuck);
  // a stitch: a short round thread, a touch aslant, laid along the tuck; its ends sink into dark holes
  vec2 st = rot(0.12)*vec2(fs, sd);
  float dt = length(vec2(max(abs(st.x) - 0.0165, 0.0), st.y));
  const float TH = 0.0052;
  float thread = smoothstep(TH + 0.0012, TH - 0.0012, dt)*front;
  float round_ = sqrt(max(1.0 - dt*dt/(TH*TH), 0.0));
  gold = mix(gold, vec3(0.98,0.52,0.13)*(0.45 + 0.60*round_), 0.95*thread);

  // cheeks, a warm coral low and wide of the eyes
  vec2 ck = vec2(abs(r.x) - 0.452, r.y + 0.158);
  float blush = exp(-dot(ck*vec2(1.0,1.25), ck*vec2(1.0,1.25))/0.022);
  gold = mix(gold, vec3(0.86,0.140,0.075), (0.42 + 0.34*gLCheek + 0.12*gLSmile)*blush);

  gold = mix(gold, gold*vec3(0.76,0.58,0.44), 0.50*crease);

  s.alb = gold;
  // ---- how it takes the light
  s.T = furFlow(vec3(r.xy, 0.05), n, vec3(0.0,-0.10,0.0));
  s.fur = gFurM;
  s.wrap = 0.30; s.spec = 0.0; s.shin = 16.0; s.mirror = 0.0; s.sheen = 0.28;
  s.n = normalize(n + gFurM*((lg*vec3(1.0,1.0,0.3))*2.2 + ncr*0.55)*gHR);
  float keyside = dot(nh, normalize(vec3(-0.60,0.62,0.50)));
  s.occ = mix(0.30, 1.0, smoothstep(-0.60, 0.62, keyside));
  s.occ *= (0.88 + 0.12*grain)*(1.0 - 0.55*tuck)*(1.0 - 0.20*wide);

  // the pile lifts where the surface turns away: a warm sheen that follows the form, broken by the strokes,
  // the moon's pale edge on its side, and the cold of the sky on the other
  vec3 nb = normalize(nh + 1.25*(vec3(vnoise3(r*5.1 + 7.0), vnoise3(r*5.1 + 13.0), vnoise3(r*5.1 + 19.0)) - 0.5)*gHR);
  float frb = 1.0 - clamp(dot(nb, -rd), 0.0, 1.0);
  float nap = (0.65 + 0.70*crush)*(0.70 + 0.60*grain);
  float pf = pow(fr, 2.6) + 0.55*pow(frb, 3.0);
  vec3 sheen = vec3(1.00,0.50,0.11)*0.34*pf*nap*smoothstep(-0.70, 0.60, keyside);
  // velvet's lustre: a broad soft lobe on the key side, broken into patches by how the pile was stroked
  float lu = max(dot(nb, normalize(uSunDir - rd)), 0.0);
  sheen += vec3(1.0,0.60,0.15)*0.17*lu*lu*lu*lu*lu*lu*(0.6 + 0.8*crush);
  // the pile's own sparkle: single fibres catching the key
  float spark = smoothstep(0.66, 0.92, grain)*smoothstep(0.1, 0.8, keyside);
  sheen += vec3(1.0,0.60,0.20)*0.085*spark;
  vec3 moonDir = normalize(vec3(0.74,0.10,-0.42));
  float mside = smoothstep(-0.25, 0.75, dot(nh, moonDir));
  vec3 moonRim = vec3(1.00,0.60,0.20)*1.35*pow(fr, 5.0)*mside*(0.6 + 0.4*nap);
  float cold = smoothstep(0.10, -0.75, dot(nh, moonDir))*smoothstep(-0.2, 0.7, nh.y + 0.25);
  vec3 skyRim = vec3(0.22,0.20,0.62)*0.55*pow(fr, 4.0)*cold;
  // it glows a little from inside: gold against the blue
  vec3 inner = s.alb*vec3(0.100,0.050,0.016)*(0.5 + 0.5*nv);
  s.emit = (sheen + moonRim + skyRim)*(1.0 - 0.8*tuck) + inner;

  lipsPaint(n, s);
  // the lips are felt, sewn on: a matted nap, and a running stitch round the outer edge in the seam's own thread
  float lipN = gPLip*(1.0 - gPLine);
  if(lipN > 0.0){
    float felt = vnoise3(q*vec3(170.0,170.0,90.0))*0.55 + vnoise3(q*61.0 + 2.0)*0.45;
    s.alb *= mix(1.0, 0.72 + 0.60*felt, lipN);
    s.n = normalize(s.n + lipN*0.20*(vec3(vnoise3(q*140.0), vnoise3(q*140.0 + 5.0), 0.5) - 0.5));
    float lr = gLE/max(gLH, 1e-3);
    float wl = gLW + 0.8*gLH;                                       // length along the curve: flat at rest, a ring in a pucker
    float ang = wl*asin(clamp(gLQ.x/wl, -1.0, 1.0))/(0.026*LIPU);   // so the stitches ride the lip
    float dash = smoothstep(0.34, 0.26, abs(fract(ang) - 0.5));
    float st0 = smoothstep(0.055, 0.0, abs(lr - 0.74))*dash*smoothstep(1.02, 0.86, gLU);
    s.alb = mix(s.alb, vec3(0.98,0.60,0.24), 0.85*st0*lipN);
    s.emit += vec3(0.20,0.11,0.04)*st0*lipN;
  }
  // the eye sits under a dome that leans away, so a pupil aimed at the camera lands inboard of the opening and the
  // face goes cross-eyed.  Each eye's rest aim swings out to the middle of its opening
  gGaze.x += (q.x < 0.0 ? -1.0 : 1.0)*EYE_OUT;
  facePaint(q, rd, n, gSkin, s);

  // ---- the brows are satin stitch: slanted threads laid side by side
  vec2 qb = vec2(abs(q.x), q.y - BROW_RISE*(q.x < 0.0 ? gBrowL : gBrowR));
  float bm = smoothstep(0.0075, 0.0015, gFBrow);
  float ph = (qb.x*0.88 + 0.46*qb.y)*6.2832/0.0235;
  float thr = 0.5 + 0.5*cos(ph);
  s.alb = mix(s.alb, s.alb*(0.62 + 0.72*thr), bm);
  s.emit += BROW_COL*0.05*thr*bm;
}

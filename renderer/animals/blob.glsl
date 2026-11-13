// ---------------------------------------------------------------------------
//  animals/blob.glsl — a gummy jelly lump with a face, resting on a mossy stump.  A shape character on the lip
//  rig (docs/SHAPE_GUIDE.md).  Head space: +x screen right, +y up, +z toward the camera, the foot at y = BASE_Y.
//  Translucency is faked: the view and sun chords through the host drive Beer-Lambert absorption into s.emit.
// ---------------------------------------------------------------------------
#define BASE_Y (-0.80)
#define SHAPE_PIVOT vec3(0.0,BASE_Y,0.0)
#define SHAPE_WORLD vec3(0.0,BASE_Y,0.0)
#define SHAPE_SQUASH 0.030                          // breath: a big slow swell
#define SHAPE_JIGGLE 0.020                          // voice
// a storybook face: high forehead, big flush eyes low and wide, no nose
#define EYE_R 0.185
#define EYE_C vec3(0.258,0.072,0.486)
#define EYE_AU 0.74
#define EYE_AL (-0.78)
#define EYE_AC (-0.34)
#define EYE_TILT 0.10
#define EYE_IRIS 0.86
#define EYE_PUPIL 0.46
#define EYE_COL_A vec3(0.88,0.50,0.11)
#define EYE_COL_B vec3(0.26,0.11,0.030)
#define BROW_W 0.036
#define BROW_L 0.165
#define BROW_H 0.024
#define BROW_RISE 0.150
#define BROW_COL vec3(0.070,0.115,0.040)
#define BROW_C vec3(0.272,0.335,0.0)
#define LIP_TILT (-0.03)                            // aims at the camera; the host leans -0.04 here, so the mouth sits flat
#define LIP_O vec3(0.0,-0.250,0.7300)
#define LIP_W 0.322                                 // a big soft gummy mouth
#define LIP_HU (0.066*LIPU)
#define LIP_HL (0.082*LIPU)
#define LIP_TU (0.042*LIPU)                         // two fat rolls of the same jelly
#define LIP_TL (0.052*LIPU)
#define LIP_BOW 0.15
#define LIP_SQUARE 0.44
#define LIP_OPEN_Y 0.92
#define LIP_LIFT 0.24
#define LIP_BLEND (0.034*LIPU)                      // they melt into the body
#define LIP_SKIN 1.20                               // everything around the mouth wobbles along
#define LIP_TOOTH_Z (-0.075*LIPU)
#define LIP_SCALLOP 0.7
#define LIP_GROOVE 0.05
#define LIP_TOOTH_GAIN 1.02
#define LIP_COL_U vec3(0.10,0.46,0.13)
#define LIP_COL_L vec3(0.14,0.54,0.14)
#define LIP_COL_LINE vec3(0.010,0.100,0.045)
#define LIP_COL_CAVE vec3(0.040,0.21,0.095)          // deep bottle green: the jelly going dark
#define LIP_COL_DEEP vec3(0.010,0.060,0.038)
#define LIP_COL_GUM  vec3(0.10,0.34,0.14)
#define LIP_COL_TONGUE vec3(0.98,0.32,0.36)          // the one spot of raspberry
#define LIP_COL_TOOTH vec3(0.93,0.98,0.76)
#define LIP_SPEC 0.30
#define LIP_SHIN 40.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.10
#define LIP_WRAP 0.45
#define LIP_EMIT vec3(0.0)
#pragma include face
#pragma include lips
// real 3-D arms with mitten hands on the contract's hand columns (engine/shape_arms.glsl, docs/RIG.md)
#define SA_SHO vec3(0.700,-0.220,0.100)
#define SA_REST vec3(0.160,-0.300,0.250)
#define SA_LEN 0.850
#define SA_R 0.090
#define SA_HAND 0.180
#define SA_ALB vec3(0.16,0.62,0.18)
#define SA_HALB vec3(0.26,0.74,0.28)
#define SA_SPEC 0.5
#define SA_MIRROR 0.10
#pragma include shape_arms

float gSkin, gBare;
vec3  gQ;
float gSwayX, gSwayZ, gBob, gRip, gRipP, gSwell, gT;

void charInit(){
  faceInit(); lipsInit();
  float T = uTime, v = clamp(uP[P_VOICE], 0.0, 1.0);
  gT = T;
  // slow soft jiggle, always on, plus low wobble with the voice.  All of it from (frame, params): no state
  gSwayX = 0.020*sin(T*1.31) + 0.012*sin(T*2.17 + 1.0) + v*(0.030*sin(T*17.3) + 0.016*sin(T*11.9 + 2.0));
  gSwayZ = 0.014*sin(T*1.07 + 2.0) + v*0.018*sin(T*14.7 + 1.0);
  gBob   = v*0.012*sin(T*21.1 + 0.7);
  gRip   = 0.022*v;                                 // amplitude from the params; phase from the frame alone
  gRipP  = T*12.5;
  gSwell = 0.040*uP[P_BREATH];
  saInit();       // last: the arms hang off the shape as posed
}

// the foot stays on the stump: head_* lean the mass, and only the part above the mouth is rigid
vec3 blobSpace(vec3 pw){
  vec3 qh = shapeSpace(pw);
  vec3 qb = (pw - SHAPE_WORLD)/vec3(gSqX, gSqY, gSqX) + SHAPE_PIVOT;
  vec3 q = mix(qb, qh, smoothstep(BASE_Y + 0.02, -0.52, qb.y));
  float h = clamp((q.y - BASE_Y)/1.55, 0.0, 1.0), hh = h*h;
  q.x -= hh*gSwayX; q.z -= hh*gSwayZ; q.y -= h*gBob;
  return q;
}

// the lump.  Real distance near the face: blended ellipsoids, corrected by the fitted gradient
float blobHost(vec3 q){
  float d = sdEllipsoid(q - vec3(0.0,0.17,0.0), vec3(0.635,0.565,0.59));
  d = smin(d, sdEllipsoid(q - vec3(0.0,-0.32,0.0), vec3(0.80,0.54,0.72)), 0.42);
  d = smin(d, sdEllipsoid(q - vec3(0.0,-0.72,0.0), vec3(0.98,0.13,0.90)), 0.30);       // the skirt, where it rests
  d = smin(d, sdEllipsoid(q - vec3(0.62,-0.68,0.20), vec3(0.36,0.12,0.34)), 0.20);     // droops, not symmetric
  d = smin(d, sdEllipsoid(q - vec3(-0.64,-0.72,0.02), vec3(0.32,0.09,0.30)), 0.16);
  d = smin(d, length(q - vec3(-0.17,0.68,-0.02)) - 0.095, 0.15);                       // a soft tip
  d = smin(d, length(q - vec3(0.70,-0.16,0.10)) - 0.095, 0.16);                        // a stubby nub
  d = smax(d, BASE_Y - q.y, 0.05);
  // the blends leave the gradient at 0.6-0.9 across the face; lips and brows take their depth from it, so correct it
  float gy = q.y - 0.05;
  d /= 1.0 - 0.115*exp(-gy*gy/0.09) - 0.37*smoothstep(-0.32, -0.46, q.y)*smoothstep(-0.66, -0.52, q.y);
  // life.  The face patch keeps still skin under the lips and eyes; the rest of the mass moves
  float fm = smoothstep(0.08, 0.46, q.z)*smoothstep(1.22, 0.82, length(vec2(q.x/0.66, (q.y + 0.03)/0.58)));
  float belly = exp(-(q.y + 0.42)*(q.y + 0.42)/0.09);
  d -= gSwell*(0.35 + 0.65*belly)*(1.0 - 0.75*fm);
  d += (1.0 - fm)*(gRip*sin(5.6*q.y + 1.7*q.x - gRipP) + 0.006*sin(5.6*q.y - gT*2.3) + 0.005*sin(3.3*q.x + 2.1*q.z + gT*1.1)*sin(2.9*q.y - gT*0.9));
  return d;
}
void blobEval(vec3 pw){
  vec3 q = blobSpace(pw);
  gQ = q;
  float host = blobHost(q);
  float skin = lipsEval(q, host);
  skin = faceEval(q, skin, host);
  gSkin = skin;
  gBare = min(gLipBare, gFEye);
}
void charEval(vec3 p){
  blobEval(p);
  gcBase = smin(min(gSkin, gBare - 0.0007), saDist(p), SA_SKIN); gcBare = gBare; gcRest = gQ; gcLen = -0.0006;
}
float charTuft(){ return 0.0; }

//@shadow-begin
float charShadowHost(vec3 pw){
  vec3 q = gHR*(pw - gHO) + SHAPE_PIVOT;
  float d = sdEllipsoid(q - vec3(0.0,0.20,0.0), vec3(0.58,0.52,0.54));
  d = min(d, sdEllipsoid(q - vec3(0.0,-0.36,0.0), vec3(0.84,0.50,0.76)));
  return d;
}
float charShadow(vec3 pw){ return min(charShadowHost(pw), saShadow(pw)); }
//@shadow-end

// how far the ray stays inside the body
float blobChord(vec3 o, vec3 d){
  float t = 0.03;
  for(int i=0;i<4;i++) t += max(-blobHost(o + d*t), 0.018);
  return t;
}

#define NB 26
const vec4 BUB[NB] = vec4[NB](
  vec4(-0.60,-0.50,0.26,.009), vec4(-0.46,-0.36,0.36,.008), vec4(-0.28,-0.50,0.46,.010), vec4( 0.04,-0.56,0.44,.008),
  vec4( 0.24,-0.62,0.44,.010), vec4( 0.46,-0.42,0.40,.009), vec4( 0.60,-0.24,0.30,.008), vec4(-0.26, 0.28,0.34,.007),
  vec4( 0.28, 0.46,0.20,.008), vec4(-0.56,-0.08,0.26,.008),
  vec4(-0.52,-0.42,0.30,.030), vec4(-0.40,-0.55,0.42,.018), vec4(-0.16,-0.62,0.40,.026), vec4( 0.10,-0.67,0.48,.015),
  vec4( 0.36,-0.52,0.44,.034), vec4( 0.54,-0.60,0.28,.020), vec4( 0.62,-0.34,0.30,.016), vec4( 0.50,-0.10,0.22,.024),
  vec4(-0.50, 0.04,0.22,.022), vec4(-0.34, 0.44,0.22,.028), vec4(-0.10, 0.55,0.12,.016), vec4( 0.20, 0.52,0.18,.020),
  vec4( 0.42, 0.32,0.20,.014), vec4( 0.02, 0.42,0.30,.012), vec4(-0.64,-0.20,0.20,.014), vec4( 0.70,-0.56,0.10,.022));

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  blobEval(pw);
  if(saDist(pw) < gSkin){ saSurface(pw, rd, s); return; }
  vec3 q = gQ, n = s.n;
  // everything optical follows the HOST's normal, so features grow no fringe
  vec2 ee = vec2(0.006,-0.006);
  vec3 nq = normalize(ee.xyy*blobHost(q + ee.xyy) + ee.yyx*blobHost(q + ee.yyx) + ee.yxy*blobHost(q + ee.yxy) + ee.xxx*blobHost(q + ee.xxx));
  vec3 nh = normalize(nq*gHR);
  vec3 rq = gHR*rd, Lq = gHR*uSunDir;
  float nv = clamp(dot(nh, -rd), 0.0, 1.0), fr = 1.0 - nv;
  vec3 ri = refract(rq, nq, 0.75);
  // chords start on the host: lids, lips and sockets are off it, and would read as thin
  vec3 qs = q - nq*blobHost(q);
  float chord = blobChord(qs, ri), toSun = blobChord(qs, Lq);

  // the body colour shifts across the form: lime on the sun side, mint on the far side and low
  float across = dot(nq, normalize(vec3(0.55,0.62,-0.2))) + 0.45*q.y;
  vec3 body = mix(vec3(0.12,0.80,0.44), vec3(0.50,0.92,0.04), smoothstep(-0.75, 0.55, across));
  // Beer-Lambert: green passes, red and blue die with depth
  const vec3 SIG = vec3(4.6,0.62,3.2);
  vec3 Tv = exp(-SIG*(chord + 0.10)), Ts = exp(-SIG*toSun);
  float ck = setCookie(pw);
  float ckc = smoothstep(0.25, 0.85, ck), dap = smoothstep(0.60, 0.92, ck);
  // a soft denser core, and clouds that drift with parallax
  vec3 C0 = vec3(0.0,-0.40,0.0);
  float coreD = length(cross(ri, C0 - qs));
  float core = smoothstep(0.66, 0.10, coreD);
  float cloud = vnoise3((qs + ri*0.20)*3.1 + 1.7)*0.6 + vnoise3((qs + ri*0.55)*2.2 + 5.0)*0.4;

  s.alb = body*0.17;
  s.spec = 0.0; s.mirror = 0.0; s.sheen = 0.0; s.wrap = 0.35; s.fur = 0.0; s.trans = 0.0;
  float keyside = dot(nh, normalize(vec3(-0.30,0.70,0.60)));
  s.occ = mix(0.45, 1.0, smoothstep(-0.8, 0.5, keyside));

  // light that has come through the body from behind, and light that scattered in from the sun
  float side = clamp(dot(nh, normalize(uRimDir + vec3(0.0,0.0,0.7)))*0.5 + 0.5, 0.0, 1.0);
  vec3 through = vec3(0.92,0.97,0.42)*Tv*mix(0.28, 1.20, side*side);
  vec3 scat = body*(0.13 + 0.15*cloud) + vec3(0.58,1.0,0.08)*Ts*(0.07 + 0.34*ckc);
  // light that came down through the mass gathers low at the front
  scat += vec3(0.40,0.95,0.30)*0.20*exp(-(q.y + 0.60)*(q.y + 0.60)/0.022)*smoothstep(0.0, 0.5, nq.z);
  vec3 inner = (through + scat)*mix(vec3(1.0), vec3(0.42,0.66,0.62), 0.85*core);

  // the shell: sky above, grass below, a wet highlight on the sun side
  vec3 r = reflect(rd, nh);
  vec3 env = mix(SP_BOUNCE*0.5, setSky(r), smoothstep(-0.10, 0.25, r.y));
  float Fre = 0.020 + 0.42*pow(fr, 4.0);
  // wet: a pill of window light on the crown that bends with the surface, and a soft bloom round it
  vec3 wc = normalize(vec3(-0.44,0.88,-0.20)), wa = normalize(cross(vec3(0.0,0.0,1.0), wc)), wb = cross(wc, wa);
  vec2 wv = vec2(dot(r, wa), dot(r, wb));
  float pill = length(max(abs(wv) - vec2(0.22,0.040), 0.0));
  float hl = max(dot(r, wc), 0.0);
  vec3 gloss = vec3(1.0,1.0,0.94)*(0.12*pow(hl, 8.0) + 1.5*smoothstep(0.075, 0.035, pill)*smoothstep(0.2, 0.6, dot(r, wc)));

  // suspended bubbles, seen along the ray inside the jelly
  vec3 bub = vec3(0.0);
  for(int i=0;i<NB;i++){
    vec3 oc = BUB[i].xyz + vec3(0.0, 0.006*sin(gT*0.7 + float(i)), 0.0) - qs;
    float b = dot(oc, ri);
    vec3 w = (ri*b - oc)/BUB[i].w;
    float u = length(w);
    float m = smoothstep(1.06, 0.92, u)*step(0.0, b);
    if(m > 0.0){
      float rimB = smoothstep(0.50, 0.96, u);
      float dot1 = smoothstep(0.34, 0.14, length(w.xy - vec2(-0.32,0.36)));
      bub += m*(vec3(0.75,1.0,0.60)*(0.08 + 0.34*rimB) + vec3(1.0)*0.8*dot1)*exp(-1.1*b);
    }
  }

  // moving dapples: the sun through the leaves lands on the jelly and spreads
  float faceQuiet = 1.0 - 0.55*smoothstep(1.05, 0.70, length(vec2(q.x/0.62, (q.y + 0.03)/0.52)));   // the mouth stays readable
  dap *= faceQuiet;
  vec3 glow = vec3(0.86,1.0,0.34)*(0.62*pow(fr, 2.4)*side*side);                                    // the lit edge
  s.emit = inner*mix(0.90, 1.05, dap) + vec3(0.19,0.27,0.035)*dap + glow + env*Fre + gloss + bub;
  // the glow is unshaded, so give the soft features (cheeks, dimples, lids) their relief from the true normal
  float rel = clamp(dot(n, nh), 0.0, 1.0);
  s.emit *= mix(0.52, 1.0, rel*rel)*(0.93 + 0.11*n.y);
  // the foot sinks into the moss
  float foot = smoothstep(BASE_Y - 0.01, BASE_Y + 0.15, q.y);
  s.emit *= mix(0.42, 1.0, foot); s.alb *= mix(0.55, 1.0, foot);

  // no body glow in the lip line: at rest that crease is 11 seconds of film, and it must stay dark
  s.emit *= 1.0 - smoothstep(0.0080*LIPU, 0.0015*LIPU, gLE);
  // the carve, wherever the rig's wall paint does not reach: the mouth's dark, never jelly
  float sunk = smoothstep(0.010, 0.026, -blobHost(q) - 0.050*LIPU*gLTuck)*smoothstep(1.40, 1.15, gLU)*smoothstep(0.13, 0.09, gLQ.y)
             *smoothstep(0.14*LIPU, 0.06*LIPU, gLE);                       // never the skin fold under a tucked lip
  s.alb = mix(s.alb, LIP_COL_DEEP, sunk); s.emit *= 1.0 - sunk;
  vec3 em0 = s.emit;
  lipsPaint(n, s);
  // the lips are the same jelly, thicker: the body's glow comes through them, and goes richer and darker where they
  // turn away, because the path through the jelly runs long there
  float lipN = gPLip*(1.0 - gPLine);
  float bend = 1.0 - clamp(dot(n, nh), 0.0, 1.0);
  s.emit = mix(s.emit, em0*mix(vec3(0.92,1.02,0.70), vec3(0.20,0.46,0.20), smoothstep(0.03, 0.36, bend)), lipN);
  facePaint(q, rd, n, gSkin, s);
  // the sun rakes the lip ends: no flecks at the corners
  float cornerQuiet = smoothstep(1.02, 0.72, gLU);
  s.spec *= cornerQuiet; s.mirror *= cornerQuiet;
  // the sun is behind the lips too: a soft card in front gives them form and one wet highlight
  float lipM = smoothstep(0.0075*LIPU, 0.0015*LIPU, gLRing);
  vec3 Lf = normalize(vec3(-0.28,0.62,0.74)), hf = normalize(Lf - rd);
  s.emit += lipM*(s.alb*0.24*max(dot(n, Lf), 0.0) + vec3(1.0,1.0,0.92)*0.36*pow(max(dot(n, hf), 0.0), 26.0)*smoothstep(1.0, 0.72, gLU));
  // the eyes sit in the body's glow, and the sun is behind them: warm the iris and give them their own lights
  bool isEye = gFEye < 0.002 && gFEye <= gSkin + 0.002;
  if(isEye){
    float lc = min(smoothstep(0.0, 0.022, gFLid), smoothstep(0.001, 0.016, gSkin));
    vec3 r2 = reflect(rd, n);
    float k1 = smoothstep(0.962, 0.976, dot(r2, normalize(vec3(-0.34,0.40,0.85))));
    float k2 = smoothstep(0.990, 0.996, dot(r2, normalize(vec3(0.30,-0.28,0.92))));
    s.emit = s.alb*(0.06 + 0.34*lc) + vec3(1.0,0.98,0.92)*(2.6*k1 + 0.8*k2)*lc;
    s.spec = 0.0;
  }
}

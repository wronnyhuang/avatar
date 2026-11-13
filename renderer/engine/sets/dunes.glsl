// ---------------------------------------------------------------------------
//  engine/sets/dunes.glsl — rolling desert dunes at golden hour.  Long raking sun from screen left, sharp
//  crests with wind ripples, deep blue-violet slip faces, a pale gold sky with long clouds, far pyramids and a
//  mesa in the haze, dry grass at the frame edge.  Implements the SET API documented in engine/sets/autumn.glsl.
//
//  The sun the camera sees (SP_SUN_VIS, off frame left, behind the subject) and the key light (uSunDir, in front
//  of the subject) differ on purpose, as in the lagoon: the key keeps the subject's face lit, the visible sun
//  warms the sky and the haze.
//
//  Sand for the subject.  The engine composites the character over the background, so the set cannot hide its
//  foot.  sandNear() is therefore compiled into every pass (SET_SAND tells a character it exists): the
//  character removes everything of itself below this surface, and the sand line lands exactly where the baked
//  sand is.  The dunes fade out around SP_SUBJECT so that sandNear() is the true ground there.
//
//  Life.  gWind drifts the cloud shadows of PASS_LIVE across the dunes.  Blowing sand is painted in PASS_VOL:
//  setInit computes one screen-space density per fragment (gPlume), so the 24 fixed-depth samples of the
//  god-ray loop cost nothing extra.  Their additive sun-coloured light is exactly what sunlit sand in the air is.
//
//  Cost.  As in the lagoon: everything heavy is inside #ifdef PASS_BG, and each expensive part sits behind a
//  one-shot loop with a bound, the only thing llvmpipe skips.
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-0.86)                          // the sand around the subject
#endif
#ifndef SP_SUBJECT
#define SP_SUBJECT vec4(0.0,0.0,0.90,-0.30)          // footprint: centre x, z, half width, yaw
#endif
#ifndef SP_BANK_A
#define SP_BANK_A 0.13                               // sand banked up against the footprint
#endif
#ifndef SP_SUN_VIS
#define SP_SUN_VIS vec3(-0.80,0.10,-0.60)            // the sun the camera nearly sees
#endif
#ifndef SP_WIND
#define SP_WIND vec2(0.92,0.38)                      // downwind: crests run into the distance, slip faces look right
#endif
#ifndef SP_SKY_HZ
#define SP_SKY_HZ vec3(1.42,0.84,0.36)               // gold at the horizon
#endif
#ifndef SP_SKY_GOLD
#define SP_SKY_GOLD vec3(1.05,0.64,0.27)             // pale gold
#endif
#ifndef SP_SKY_HIGH
#define SP_SKY_HIGH vec3(0.72,0.48,0.32)             // pale, faintly blue, overhead
#endif
#ifndef SP_SUN_GLOW
#define SP_SUN_GLOW vec3(1.0,0.66,0.32)
#endif
#ifndef SP_AMB_SKY
#define SP_AMB_SKY vec3(0.125,0.150,0.385)              // shadows are blue-violet: the sky is all that lights them
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.66,0.40,0.20)               // light thrown up off the sunlit sand
#endif
#ifndef SP_SAND_A
#define SP_SAND_A vec3(0.60,0.250,0.105)               // deep orange sand
#endif
#ifndef SP_SAND_B
#define SP_SAND_B vec3(0.76,0.365,0.160)               // gold sand
#endif
#ifndef SP_SAND_C
#define SP_SAND_C vec3(0.90,0.56,0.30)               // pale, wind-sorted crest sand
#endif
#ifndef SP_FOG_AWAY
#define SP_FOG_AWAY vec3(0.62,0.44,0.60)             // haze away from the sun: dusty rose-lilac
#endif
#ifndef SP_FOG_SUN
#define SP_FOG_SUN vec3(1.45,0.92,0.46)              // haze toward the sun: gold
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#ifndef SP_PLUME_A
#define SP_PLUME_A vec4(-1.27,0.285,0.74,0.0)         // blowing sand: anchor in screen uv, length, seed
#endif
#ifndef SP_PLUME_B
#define SP_PLUME_B vec4(1.02,-0.54,0.76,3.0)
#endif
#ifndef SP_PLUME_GAIN
#define SP_PLUME_GAIN 1.45
#endif

#define SET_SAND 1
const float G0 = SP_GROUND_Y;
const vec3  SUNV = normalize(SP_SUN_VIS);
const vec2  WD = normalize(SP_WIND);

#define M_CANOPY 99.0     // unused: nothing in this set ignores direct shadowing
#define M_SAND   1.0
#define M_ROCK   2.0
#define M_STRAW  3.0
#define M_TWIG   4.0

vec2  gWind;
float gSurfDone = 0.0;    // PASS_BG: set once a thin plant is painted: the engine's 5-unit AO would blacken it
float gPlume = 0.0;       // PASS_VOL: blowing sand along this view ray
float gStrawT = 0.0;      // PASS_BG: 0 root .. 1 tip of the grass blade hit
vec3  gAmbK = vec3(1.0);  // PASS_BG: orange sand has almost no blue to give back, so a sky-lit shadow would be brown, and
                          // a blue ambient strong enough to fix that washes out the sunlit sand.  So sand answers the
                          // ambient by its own sun visibility: the violet goes only where the sun does not

#ifdef PASS_VOL
// Additive light clips to white on the pale sky, so a plume only reads over something dark: each one is routed down
// across a shadow.  fall = its slope, uv units per uv unit; P = crest anchor (uv), length, seed
float plumeAt(vec2 uv, vec4 P, float fall, float widK, float T){
  vec2 d = uv - P.xy;
  float a = d.x/P.z;                                 // 0 at the crest .. 1 where it has thinned to nothing
  float env = smoothstep(0.0, 0.10, a)*smoothstep(1.0, 0.25, a);
  float rise = d.y + fall*a*P.z + 0.022*sin(a*6.0 + P.w + T*0.21);
  float wid = mix(0.010, 0.095, a)*widK;
  float v = rise/wid;
  float n = vnoise2(vec2(a*5.0 - T*1.3 + P.w, v*0.7 + P.w))*0.60 + vnoise2(vec2(a*15.0 - T*2.6, v*1.6 + 4.0 + P.w))*0.40;
  float gust = 0.62 + 0.38*sin(T*0.73 + P.w*1.9)*sin(T*0.41 + 1.0 + P.w);
  return env*exp(-v*v*0.8)*smoothstep(0.28, 0.70, n)*gust;
}
float groundDrift(vec2 uv, float T){
  // grains skimming the near sand: thin fast streaks lying on the ground, in gusts
  float y = uv.y;
  float band = smoothstep(-1.02, -0.92, y)*smoothstep(-0.52, -0.66, y);
  float k = 34.0/(1.0 + 1.6*(y + 1.0));
  float n = vnoise2(vec2(uv.x*1.3 - T*1.5, y*k))*vnoise2(vec2(uv.x*3.1 - T*2.3 + 3.0, y*k*1.7 + 2.0));
  float gust = smoothstep(0.35, 0.75, vnoise2(vec2(uv.x*0.5 - T*0.33, 0.4*T + 2.0)));
  return band*smoothstep(0.16, 0.42, n)*gust;
}
#endif

void setInit(){
  float T = uTime;
  gWind = vec2(T*0.012, T*0.004);                    // slow, one way: cloud shadows cross the dunes
#ifdef PASS_VOL
  vec2 uv = (2.0*gl_FragCoord.xy - uRes)/uRes.y;
  gPlume = (plumeAt(uv, SP_PLUME_A, 0.86, 1.0, T) + 1.1*plumeAt(uv, SP_PLUME_B, -0.03, 0.34, T) + 0.42*groundDrift(uv, T))*SP_PLUME_GAIN;   // near ones have fewer samples in front of the surface
#endif
}

// ------------------------------------------------------------------ the sand at the subject, every pass
float sandNear(vec2 p){
  float h = 0.040*sin(p.x*0.7 + 0.4)*sin(p.y*0.9 - 0.6) + 0.024*sin(p.x*1.7 - p.y*0.6 + 1.1);
  vec2 q = rot(SP_SUBJECT.w)*(p - SP_SUBJECT.xy);
  vec2 a = abs(q) - SP_SUBJECT.z;
  float d = length(max(a, vec2(0.0)));
  float bank = 1.0 + 0.55*sin(q.x*2.3 + q.y*1.7 + 1.0) + 0.30*sin(q.x*5.1 - q.y*3.3 + 2.0);   // drifts, not a skirt
  return G0 + h + SP_BANK_A*bank*exp(-d*d*1.6 - d*1.5);
}

// ------------------------------------------------------------------ sky, light (per frame on the character)
vec3 skyBase(vec3 rd){
  float y = max(rd.y, 0.0);
  vec3 c = mix(SP_SKY_HZ, SP_SKY_GOLD, smoothstep(0.0, 0.075, y));
  c = mix(c, SP_SKY_HIGH, smoothstep(0.05, 0.34, y));
  float sd = max(dot(rd, SUNV), 0.0);
  float sd2 = sd*sd, sd4 = sd2*sd2, sd8 = sd4*sd4, sd16 = sd8*sd8;
  c += SP_SUN_GLOW*(0.34*sd4 + 0.55*sd16 + 1.1*sd16*sd16);
  return c;
}
#ifdef PASS_BG
vec3 setSky(vec3 rd){
  vec3 c = skyBase(rd);
  float sd = max(dot(rd, SUNV), 0.0);
  // long clouds, lying along the horizon, under-lit where the sun is near
  float y = max(rd.y, 0.0);
  vec2 uv = rd.xz/(y + 0.10);
  float n = fbm2(uv*vec2(0.34,1.55) + vec2(1.7,0.4));
  float n2 = fbm2(uv*vec2(1.1,5.2) + vec2(4.0,2.0));
  float cl = smoothstep(0.44, 0.62, n*0.66 + n2*0.34);
  cl *= smoothstep(0.016, 0.060, y)*smoothstep(0.72, 0.22, y);
  float thin = smoothstep(0.42, 0.62, n2);
  vec3 away = vec3(0.56,0.40,0.46);
  vec3 lit = mix(away, vec3(1.80,1.02,0.46), pow(sd, 3.0));
  c = mix(c, lit, cl*(0.52 + 0.40*thin));
  return c;
}
#else
vec3 setSky(vec3 rd){ return skyBase(rd); }
#endif

vec3 setAmbient(vec3 p, vec3 n){
  vec3 sky = SP_AMB_SKY*(0.66 + 0.34*n.y);
  vec3 bounce = SP_BOUNCE*clamp(0.46 - 0.60*n.y, 0.0, 1.0);
  float w = clamp(dot(n, normalize(SUNV + vec3(0.0,0.2,0.0)))*0.5 + 0.5, 0.0, 1.0);
  return (sky + bounce + vec3(1.0,0.62,0.34)*0.17*w*w)*gAmbK;
}
// a thin high cloud crossing the sun: long and slow
float setCookie(vec3 p){
  float n = vnoise2(p.xz*0.17 + vec2(uTime*0.031, -uTime*0.010));
  return smoothstep(0.26, 0.60, n);
}
float setVol(vec3 p){
#ifdef PASS_VOL
  return gPlume;
#else
  return 0.04;
#endif
}
float setFog(vec3 p){
#ifdef PASS_VOL
  return 0.0090;                                      // the plumes' scale: density is all in setVol
#else
  float h = max(p.y - G0, 0.0);
  return (0.0030*exp(-h*0.22) + 0.00035)*SP_FOG_GAIN;
#endif
}
vec3 setFogAmb(vec3 rd){
  float k = 0.5 + 0.5*dot(rd, SUNV);
  return mix(SP_FOG_AWAY, SP_FOG_SUN, k*k);
}

// ------------------------------------------------------------------ the world, bake only
#ifdef PASS_BG
float bgShadow(vec3 ro, vec3 rd);      // scene_main
// one family of transverse dunes.  Long windward rise, sharp crest at S, short straight slip face.  Each row has
// its own height and the height changes along the crest, so rows end, merge and hook like real barchans.
float duneRow(vec2 p, float lam, float warp, float seed, float S){
  float a = dot(p, WD), b = dot(p, vec2(-WD.y, WD.x));
  float u = (a + warp*(sin(b*2.1/lam + seed) + 0.55*sin(b*4.7/lam + seed*2.3 + 1.0) + 1.5*(vnoise2(p*0.6/lam + seed) - 0.5)))/lam;
  float id = floor(u), f = u - id;
  float r = f/S, k = (f - S)/(1.0 - S);
  float h = f < S ? r*r*(1.7 - 0.7*r) : 1.0 - k;
  float ar = 0.55 + 0.45*hash11(id*1.37 + seed);
  float al = smoothstep(0.16, 0.70, vnoise2(vec2(b*0.9/lam, id*3.1 + seed*5.0)));
  return h*ar*al;
}
float sandH(vec2 p){
  float dist = length(p - SP_SUBJECT.xy);
  float m = smoothstep(2.2, 8.5, dist)*smoothstep(2.6, -2.0, p.y);    // nothing rises at the subject or near the camera
  float far_ = smoothstep(14.0, 120.0, dist);
  float big = duneRow(p + vec2(7.0,3.0), 31.0, 9.0, 1.3, 0.85)*mix(2.7, 4.2, far_);
  float mid = duneRow(p + vec2(5.0,9.0), 11.5, 3.2, 4.1, 0.84)*mix(1.35, 1.5, far_);
  float sml = duneRow(p + vec2(4.0,2.0), 4.3, 1.1, 7.7, 0.72)*0.07;
  float swell = 0.9*sin(p.x*0.11 + 0.6)*sin(p.y*0.08 + 1.0);
  float rise = 7.0*smoothstep(55.0, 195.0, dist);                     // the last long ridge makes the horizon
  return sandNear(p) + m*(big + mid + sml + swell + rise);
}

// ---- far off, in the haze
float sdFarPyr(vec3 p, vec3 c, float hw, float h){
  vec3 q = p - c;
  q.xz = rot(0.6)*q.xz;
  float m = max(abs(q.x), abs(q.z));
  float k = hw/h, sl = 1.0/sqrt(1.0 + k*k);
  return max((m + q.y*k - hw)*sl, -q.y);
}
float sdMesa(vec3 p, vec3 c, vec3 r){
  vec3 q = p - c;
  q.x += 0.08*r.x*sin(q.z*0.05 + 1.0);
  float wob = 1.0 + 0.10*sin(q.x*0.09 + 2.0) + 0.07*sin(q.x*0.23);
  float talus = max(q.y + r.y, 0.0)/(2.0*r.y);               // 0 foot .. 1 top
  float top = r.y*(0.86 + 0.14*sin(q.x*0.31 + 1.0)*sin(q.x*0.13)) ;
  float side = max(abs(q.x)/wob + talus*talus*r.x*0.42 - r.x, abs(q.z) - r.z);
  return max(side*0.8, max(q.y - top, -q.y - r.y));
}
vec2 farShapes(vec3 p){
  float d = sdFarPyr(p, vec3(-30.0,1.0,-158.0), 9.5, 10.5);
  d = min(d, sdFarPyr(p, vec3(-42.0,1.0,-170.0), 6.5, 7.0));
  d = min(d, sdFarPyr(p, vec3(-20.0,1.0,-178.0), 4.2, 4.6));
  d = min(d, sdMesa(p, vec3(28.0,5.5,-186.0), vec3(8.5,7.5,9.0)));
  d = min(d, sdMesa(p, vec3(39.0,4.5,-198.0), vec3(4.5,6.0,7.0)));
  return vec2(d, M_ROCK);
}

// ---- dry grass at the frame edge.  One straight blade per loop in a bent space: smooth arcs for the price of one taper
float grassTuft(vec3 p, vec3 c, float seed, int N, float tall, float best){
  float d = best;
  for(int i=0;i<N;i++){
    vec3 h = hash31(float(i)*1.73 + seed), h2 = hash31(float(i)*3.11 + seed + 2.0);
    float az = h.x*6.2832, sp = h.y*h.y*0.13;
    vec3 q = p - (c + vec3(cos(az)*sp, -0.04, sin(az)*sp*0.8));
    float H = tall*mix(0.42, 1.0, h.z);
    vec2 lean = (WD*mix(0.25, 1.0, h2.x) + vec2(cos(az), sin(az))*0.46)*mix(0.7, 1.25, h2.y);    // downwind, fanned
    q.xz -= lean*q.y*q.y/H;
    float R = mix(0.0025, 0.0042, h2.z);
    float dm = sdTaper(q, vec3(0.0), vec3(0.0,H,0.0), R, 0.0005);
    if(dm < d){ d = dm; gStrawT = clamp(q.y/H, 0.0, 1.0); }
  }
  return d*0.72;
}
// a dead bush: stiff grey twigs forking out of the sand
float deadBush(vec3 p, vec3 c, float seed, float best){
  float d = best;
  for(int i=0;i<9;i++){
    vec3 h = hash31(float(i)*2.37 + seed);
    float az = h.x*6.2832, el = mix(0.45, 1.25, h.y), L = mix(0.24, 0.50, h.z);
    vec3 dir = vec3(cos(az)*cos(el), sin(el), sin(az)*cos(el));
    vec3 a = c + vec3(cos(az), 0.0, sin(az))*0.03, b = a + dir*L;
    d = min(d, sdTaper(p, a, b, 0.0085, 0.0036));
    for(int k=0;k<2;k++){
      vec3 h2 = hash31(float(i*2 + k)*5.91 + seed + 1.0);
      vec3 o = mix(a, b, 0.45 + 0.3*float(k));
      vec3 dir2 = normalize(dir + 1.05*(h2 - 0.5) + vec3(0.0,0.1,0.0));
      d = min(d, sdTaper(p, o, o + dir2*L*mix(0.36, 0.66, h2.x), 0.0040, 0.0010));
    }
  }
  return d;
}

const vec2 TUFT_A = vec2(-1.46,2.05), TUFT_B = vec2(-1.84,1.38), TUFT_C = vec2(1.42,2.20), TUFT_D = vec2(2.06,0.62);
vec2 nearMap(vec3 p, float G, vec2 res){
  float gy = sandNear(p.xz);
  vec3 cA = vec3(TUFT_A.x, gy, TUFT_A.y), cB = vec3(TUFT_B.x, gy, TUFT_B.y), cC = vec3(TUFT_C.x, gy, TUFT_C.y), cD = vec3(TUFT_D.x, gy, TUFT_D.y);
  float bA = min(length(p - vec3(cA.x, gy + 0.28, cA.z)) - 0.86, length(p - vec3(cB.x, gy + 0.22, cB.z)) - 0.80);
  res = opU(res, vec2(max(bA, G), M_STRAW));
  for(int i=0;i<1 && bA < G;i++){
    float d = grassTuft(p, cA, 3.1, 34, 0.88, 1e3);
    d = min(d, grassTuft(p, cB, 9.4, 24, 0.60, d));
    res = opU(res, vec2(d*0.9, M_STRAW));
  }
  float bB = min(length(p - vec3(cC.x, gy + 0.26, cC.z)) - 0.72, length(p - vec3(cD.x, gy + 0.20, cD.z)) - 0.62);
  res = opU(res, vec2(max(bB, G), M_STRAW));
  for(int i=0;i<1 && bB < G;i++){
    float d = deadBush(p, cC, 4.7, 1e3);
    res = opU(res, vec2(d*0.9, M_TWIG));
    float e = grassTuft(p, cD, 12.3, 18, 0.42, 1e3);
    res = opU(res, vec2(e*0.9, M_STRAW));
  }
  return res;
}

vec2 setMap(vec3 p, bool shadow){
  if(gSurfDone > 0.5) return vec2(shadow ? 1.0 : 1e3, M_SAND);   // a plant: no AO or self-shadow zebra
  float h = sandH(p.xz);
  vec2 res = vec2((p.y - h)*0.72, M_SAND);
  res = opU(res, farShapes(p));
  // the sun is so low that a plant's shadow would smear across the sand: they cast none, setSurface paints a contact
  if(shadow) return res;
  return nearMap(p, 0.20, res);
}

// ---- wind ripples: three scales, all across the wind, sharp on the lee.  Faded by the view distance
float rippleRow(vec2 p, float lam, float seed){
  vec2 w = p/lam;
  float a = dot(w, WD), b = dot(w, vec2(-WD.y, WD.x));
  float u = a + 0.30*sin(b*0.19 + seed) + 0.10*sin(b*0.71 + a*0.14 + seed*2.0) + 1.3*(vnoise2(w*0.07 + seed) - 0.5) + 0.15*(vnoise2(w*0.5) - 0.5);
  float f = fract(u);
  float tri = f < 0.62 ? f/0.62 : (1.0 - f)/0.38;
  return tri*tri*(3.0 - 2.0*tri);
}
float gRipT = 0.0;
float rippleH(vec2 p, vec3 k){
  float patchy = 0.45 + 0.55*smoothstep(0.25, 0.65, vnoise2(p*0.23 + 7.0));
  float h = 0.0034*k.x*rippleRow(p, 0.074, 1.1)*patchy;
  h += 0.0056*k.y*rippleRow(p + vec2(3.0,1.0), 0.205, 2.7)*(1.25 - patchy);
  h += 0.0075*k.z*rippleRow(p + vec2(9.0,5.0), 0.52, 5.3);
  h += 0.0160*k.z*smoothstep(9.0, 22.0, gRipT)*rippleRow(p + vec2(2.0,7.0), 1.45, 3.9);
  return h;
}

void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){
  // a ray out of steps has no gradient, so the engine's normal is NaN, and one NaN pixel turns the whole FFT lens
  // into NaN
  if(!(dot(n, n) > 0.5)) n = vec3(0.0,1.0,0.0);
  s.n = n;
  if(m == M_SAND){
    vec2 xz = p.xz;
    float slip = smoothstep(0.92, 0.80, n.y);                       // steep lee faces lie smooth
    vec3 k = vec3(smoothstep(11.0, 5.0, t), smoothstep(36.0, 11.0, t), smoothstep(70.0, 26.0, t));
    k *= mix(1.0, 0.22, slip);
    float e = 0.0045 + 0.0010*t;
    gRipT = t;
    float h0 = rippleH(xz, k);
    vec2 gr = vec2(rippleH(xz + vec2(e,0.0), k) - h0, rippleH(xz + vec2(0.0,e), k) - h0)/e;
    s.n = normalize(n - vec3(gr.x, 0.0, gr.y));
    float fb = fbm2(xz*0.075 + 2.0);
    float fine = vnoise2(xz*0.9);
    vec3 a = mix(SP_SAND_A, SP_SAND_B, smoothstep(0.30, 0.68, fb*0.75 + fine*0.25));
    // ripple crests are paler: the wind sorts the grains
    float ri = smoothstep(0.0, 1.0, h0/max(0.0034*k.x + 0.0056*k.y + 0.0075*k.z, 1e-3));
    a = mix(a, SP_SAND_C, 0.24*ri);
    a *= 0.95 + 0.08*vnoise2(xz*31.0);
    // dark mineral grains, in the near sand
    float gr2 = smoothstep(0.76, 0.86, vnoise2(xz*170.0))*smoothstep(9.0, 3.0, t);
    a = mix(a, vec3(0.30,0.20,0.15), 0.34*gr2);
    a = mix(a, a*vec3(1.0,0.84,0.74), slip*0.5);                    // lee sand is a deeper red-orange
    s.alb = a;
    float lit = bgShadow(p + n*0.02, uSunDir)*smoothstep(0.0, 0.13, dot(n, uSunDir));
    gAmbK = mix(vec3(1.0,1.20,2.70), vec3(1.0,0.96,0.72), lit);
    // the sand holds the subject's foot in a little shade of its own
    vec2 q = rot(SP_SUBJECT.w)*(xz - SP_SUBJECT.xy);
    vec2 da = abs(q) - SP_SUBJECT.z;
    float dd = length(max(da, vec2(0.0)));
    s.occ = mix(0.46, 1.0, smoothstep(0.03, 0.46, dd));
    // the foot, on the sun side too: a shade in the sun term, and a little at each plant
    s.alb *= mix(0.74, 1.0, smoothstep(0.04, 0.34, dd));
    float tf = min(min(length(xz - TUFT_A), length(xz - TUFT_B)), min(length(xz - TUFT_C), length(xz - TUFT_D)));
    s.alb *= mix(0.70, 1.0, smoothstep(0.02, 0.26, tf));
    s.wrap = 0.04; s.sheen = 0.12; s.spec = 0.02; s.shin = 12.0;
  } else if(m == M_ROCK){
    float band = 0.5 + 0.5*sin(p.y*0.9 + 2.0*vnoise2(p.xz*0.06));
    s.alb = mix(vec3(0.46,0.26,0.16), vec3(0.70,0.44,0.26), band*0.6 + 0.4*vnoise2(p.xz*0.25));
    s.wrap = 0.1; s.spec = 0.0; s.sheen = 0.0;
  } else {
    nearMap(p, 0.20, vec2(1e3, M_SAND));                            // again, for gStrawT
    float v = vnoise3(p*vec3(9.0,3.0,9.0));
    if(m == M_STRAW){
      s.alb = mix(vec3(0.36,0.21,0.10), vec3(0.74,0.53,0.27), smoothstep(0.05, 0.9, gStrawT)*0.7 + 0.3*v);
      s.wrap = 0.42; s.trans = 0.65; s.transCol = vec3(1.0,0.66,0.28); s.sheen = 0.5; s.spec = 0.08; s.shin = 22.0;
    } else {
      s.alb = mix(vec3(0.22,0.17,0.15), vec3(0.50,0.41,0.34), v);
      s.wrap = 0.25; s.sheen = 0.35; s.spec = 0.05; s.shin = 16.0;
    }
    s.occ = mix(0.62, 1.0, smoothstep(0.0, 0.32, p.y - G0));
    gSurfDone = 1.0;
  }
}
#else
vec2 setMap(vec3 p, bool shadow){ return vec2(1.0, 0.0); }
void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){}
#endif

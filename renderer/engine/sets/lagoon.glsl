// ---------------------------------------------------------------------------
//  engine/sets/lagoon.glsl — a sunrise lagoon.  Calm mirror water, sea-stacks and low islands layered in
//  mist, a soft sun low on the horizon to the right of the subject, reeds, cattails and a water lily at
//  the frame edges.  Implements the SET API documented in engine/sets/autumn.glsl.  WL is the water level.
//
//  The sun the camera sees (SUNV, SP_SUN_VIS) and the key light (uSunDir) are different on purpose: the
//  key stays in front of the subject so its face stays lit, the visible sun feeds the sky, the mist, the
//  glitter on the water and the warm fill in setAmbient.
//
//  Layers.  Water and the far rocks are shaded to RADIANCE in setSurface, not lit by the engine: they are
//  lit by the visible sun, which the engine does not know.  scene_main multiplies every term by s.alb, so
//  the carrier is:  s.alb = what the SUN layer should hold,  setAmbient() = gRad = wanted radiance / s.alb.
//  That keeps reflections out of the drifting light and puts only the sun's glitter path in it.
//
//  Cost.  llvmpipe does not skip `if`, only loops, so every expensive part sits in a one-shot loop behind
//  a bound (see `G` in setMap) and everything heavy is inside #ifdef PASS_BG: this file is also compiled
//  into the character, live, god-ray and sprite passes, where setSky, setAmbient, setCookie, setVol and
//  setFog are plain arithmetic.
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-0.86)                          // the water level
#endif
#ifndef SP_SUN_VIS
#define SP_SUN_VIS vec3(0.24,0.07,-0.97)             // the sun the camera sees, not the key light
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.30,0.40,0.47)               // cool teal-lilac light off the water
#endif
#ifndef SP_AMB_SKY
#define SP_AMB_SKY vec3(0.50,0.36,0.42)              // rosy sky
#endif
#ifndef SP_SKY_HZ
#define SP_SKY_HZ vec3(1.62,0.60,0.36)               // peach at the horizon
#endif
#ifndef SP_SKY_ROSE
#define SP_SKY_ROSE vec3(1.08,0.27,0.35)             // rose band
#endif
#ifndef SP_SKY_LILAC
#define SP_SKY_LILAC vec3(0.42,0.26,0.68)            // pale lilac
#endif
#ifndef SP_SKY_BLUE
#define SP_SKY_BLUE vec3(0.18,0.32,0.88)             // soft blue overhead
#endif
#ifndef SP_SUN_GLOW
#define SP_SUN_GLOW vec3(1.0,0.60,0.30)
#endif
#ifndef SP_FOG_ROSE
#define SP_FOG_ROSE vec3(1.10,0.46,0.42)             // mist away from the sun
#endif
#ifndef SP_FOG_SUN
#define SP_FOG_SUN vec3(2.25,0.98,0.50)              // mist where the sun is
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#ifndef SP_WATER_C
#define SP_WATER_C vec3(0.040,0.085,0.105)           // body colour of the water
#endif
#ifndef SP_REFL_SQUASH
#define SP_REFL_SQUASH 0.34                          // the subject hovers: see waterRad
#endif
#ifndef SP_REFL_GAIN
#define SP_REFL_GAIN 0.33
#endif

const float WL = SP_GROUND_Y;
const vec3  SUNV = normalize(SP_SUN_VIS);

#define M_CANOPY 99.0     // unused: nothing in this set ignores direct shadowing
#define M_WATER  1.0
#define M_ROCK   2.0
#define M_VEG    3.0
#define M_BLADE  4.0
#define M_STEM   5.0
#define M_HEAD   6.0
#define M_REED   7.0
#define M_PAD    8.0
#define M_PETAL  9.0
#define M_STAMEN 10.0

vec2  gWind;
float gRadShade = 0.0;     // PASS_BG: 1 while this pixel is shaded to radiance, see the header
vec3  gRad = vec3(1.0);
float gSurfDone = 0.0;     // PASS_BG: set once a solid is painted.  Then the engine's AO (5 units wide) and reed-on-reed
                           // sun shadows are off: the first blackens all near the reeds, the second draws zebra marks
float gGlit = 0.0;
float gMirrorRay = 0.0;
float gRockCrk = 0.0;         // PASS_BG: > 0 on water that carries sun glitter: needs the sun term's visibility

void setInit(){
  float T = uTime;
  gWind = 0.05*vec2(sin(T*0.31), cos(T*0.27)) + 0.015*vec2(sin(T*0.83 + 1.0), sin(T*0.71));
}

// ------------------------------------------------------------------ sky, light (per frame on the character)
vec3 skyBase(vec3 rd){
  float y = max(rd.y, 0.0);
  vec3 c = mix(SP_SKY_HZ, SP_SKY_ROSE, smoothstep(0.0, 0.060, y));
  c = mix(c, SP_SKY_LILAC, smoothstep(0.040, 0.150, y));
  c = mix(c, SP_SKY_BLUE, smoothstep(0.135, 0.275, y));
  float sd = max(dot(rd, SUNV), 0.0);
  float sd2 = sd*sd, sd4 = sd2*sd2, sd16 = sd4*sd4; sd16 *= sd16;
  c += SP_SUN_GLOW*(0.10*sd16 + 0.34*pow(sd, 90.0) + 0.95*pow(sd, 700.0));
  return c;
}
#ifdef PASS_BG
float sqr_(float x){ return x*x; }
vec3 setSky(vec3 rd){
  vec3 c = skyBase(rd);
  float sd = max(dot(rd, SUNV), 0.0);
  // thin soft streaks, under-lit by the sun, a little deeper than the sky behind them
  float y = max(rd.y, 0.0);
  vec2 uv = rd.xz/(y + 0.11);
  float n = fbm2(uv*vec2(0.40,1.7) + vec2(1.7,0.4));
  float n2 = fbm2(uv*vec2(1.2,5.5) + vec2(4.0,2.0));
  float cl = smoothstep(0.50, 0.74, n*0.74 + n2*0.26);
  cl *= smoothstep(0.012, 0.070, y)*smoothstep(0.65, 0.18, y);
  vec3 lit = mix(vec3(1.05,0.44,0.52), vec3(2.30,1.05,0.55), pow(sd, 10.0));
  c = mix(c, lit, cl*0.55);
  // the sun itself: bright, but behind mist, no hard disc.  Not in the mirror: the water's sun is the glitter
  float direct = 1.0 - gMirrorRay;
  c += direct*vec3(11.0,7.0,3.8)*pow(sd, 5200.0);
  c += direct*vec3(1.7,0.90,0.42)*exp(-sqr_(rd.y - SUNV.y)/0.0009)*exp(-sqr_(rd.x - SUNV.x)/0.02);   // flattened by the haze
  return c;
}
#else
// the pearl mirrors this: a bright horizon band, so a curved surface draws a clean bright line
vec3 setSky(vec3 rd){
  float b = rd.y/0.10;
  return skyBase(rd) + vec3(0.55,0.34,0.26)*exp(-b*b);
}
#endif

vec3 setAmbient(vec3 p, vec3 n){
#ifdef PASS_BG
  if(gRadShade > 0.5) return gRad;
#endif
  vec3 sky = SP_AMB_SKY*(0.66 + 0.34*n.y);
  vec3 bounce = SP_BOUNCE*clamp(0.58 - 0.72*n.y, 0.0, 1.0);
  vec3 toSun = normalize(SUNV + vec3(0.0,0.16,0.0));
  float w = clamp(dot(n, toSun)*0.5 + 0.5, 0.0, 1.0);
  return sky + bounce + vec3(1.0,0.64,0.40)*0.44*w*w*w;
}
// open sky: a long slow breath, no dapples
float setCookie(vec3 p){ return 0.96 + 0.04*sin(uTime*0.11 + p.x*0.21); }
float setVol(vec3 p){ return 0.012; }
float setFog(vec3 p){
  float h = max(p.y - WL, 0.0);
  float d = 0.0015*exp(-h*0.11);
#ifdef PASS_BG
  // soft bands lying on the water, thicker toward the horizon
  float far_ = smoothstep(6.0, 55.0, length(p.xz));
  float n1 = vnoise2(p.xz*vec2(0.035,0.14) + vec2(3.1,1.7));
  float n2 = vnoise2(p.xz*vec2(0.11,0.42) + vec2(8.2,4.4));
  float band = smoothstep(0.30, 0.74, n1*0.68 + n2*0.32);
  d += (0.0030 + 0.030*band)*(0.35 + 0.65*far_)*exp(-h/(0.30 + 0.85*n1));
#else
  d += 0.0100*exp(-h*1.3);
#endif
  return d*SP_FOG_GAIN;
}
vec3 setFogAmb(vec3 rd){
  float sd = clamp(dot(rd, SUNV), 0.0, 1.0);
  float k = sd*sd; k *= k; k *= k; k *= k; k *= k;      // ^32
  return mix(SP_FOG_ROSE, SP_FOG_SUN, 0.20 + 0.80*k);
}

// =========================================================================== bake only
#ifdef PASS_BG
// -------- sea-stacks and low islands.  The screen's half width at distance d is 0.346*d (d = 5.6 - z).
#define NST 12
// x, z, half width, height
const vec4 ST_A[NST] = vec4[NST](
  vec4( -4.55,-19.0, 1.12,3.70),    // the tall one left of the subject
  vec4( -6.05,-22.5, 0.80,2.60),    // its companion
  vec4( -3.65,-16.5, 0.90,0.86),    // low shoulder
  vec4(-10.50,-35.0, 2.40,4.90),    // behind, paler
  vec4( -6.80,-41.0, 1.90,2.60),
  vec4(  5.50,-27.0, 1.05,2.95),    // right of the subject, in the sun's haze
  vec4(  4.85,-30.0, 1.15,1.30),
  vec4( 10.00,-22.0, 1.55,3.60),    // right edge
  vec4( 16.00,-55.0, 9.00,2.10),    // the long island under the sun
  vec4( -2.70,-64.0,15.00,1.95),    // the far island behind the subject
  vec4( -3.40,-10.5, 0.62,0.38),    // a low wet rock, close
  vec4(  5.05,-12.0, 0.70,0.46)
);
// depth/width ratio, rotation, seed, vegetation (0 none, else size of the cap)
const vec4 ST_B[NST] = vec4[NST](
  vec4(0.72, 0.50, 1.1,0.85), vec4(0.85, 1.20, 2.3,0.00), vec4(0.80, 0.30, 3.7,0.00),
  vec4(0.70, 0.20, 4.9,1.00), vec4(0.60,-0.15, 6.1,1.00), vec4(0.80, 0.70, 7.3,0.80),
  vec4(0.70, 0.10, 8.7,0.00), vec4(0.75,-0.40, 9.9,0.90), vec4(0.22, 0.04,11.3,1.40),
  vec4(0.16,-0.03,12.7,1.40), vec4(0.70, 0.40,13.9,0.00), vec4(0.60,-0.50,15.1,0.00)
);

vec2 stacks(vec3 p, float G, vec2 res){
  for(int i=0;i<NST;i++){
    vec4 a = ST_A[i], b = ST_B[i];
    float Rx = a.z, Rz = a.z*b.x, H = a.w, Rm = min(Rx, Rz), sd = b.z;
    vec3 q = p - vec3(a.x, WL, a.y);
    vec2 xz = rot(b.y)*q.xz;
    float bd = max((length(xz/vec2(Rx, Rz)) - 1.45)*Rm, q.y - H*1.30);
    if(bd >= G) res.x = min(res.x, bd);
    for(int k=0;k<1 && bd<G;k++){
      float F = 88.0/(5.6 - a.y);               // features of about the same size on screen
      vec2 e = xz/vec2(Rx, Rz);
      float v = clamp(q.y/H, 0.0, 1.0);
      float prof = 1.0 - 0.60*pow(v, 1.6);
      prof *= 1.0 + 0.11*sin(v*8.0 + sd*3.0) + 0.05*sin(v*17.0 + sd);    // shoulders and ledges
      float az = atan(e.y, e.x);
      prof *= 1.0 + 0.14*sin(az*3.0 + sd) + 0.08*sin(az*5.0 + sd*2.0);
      float d = (length(e) - prof)*Rm;
      float top = H*(0.86 + 0.20*vnoise2(xz*F*0.32 + sd));
      d = smax(d, q.y - top, 0.35*min(Rm, H*0.4));
      // columns, cracks, strata: three scales, all coarse enough to survive the depth blur
      float col = vnoise2(xz*F*0.55 + sd*3.0);
      float crk = vnoise3(vec3(xz*F*1.5, q.y*F*0.35 + sd));
      float rid = vnoise3(q*F*0.75 + sd);
      float fis = 1.0 - abs(2.0*vnoise2(xz*F*0.85 + sd*5.0 + vec2(0.0, 0.25*sin(q.y*F*0.5))) - 1.0);
      fis = fis*fis*fis;
      float led = fract(q.y*F*0.62 + 1.4*vnoise2(xz*F*0.26 + sd));
      d += (0.70*(col - 0.5) + 0.34*(crk - 0.5) + 0.42*(rid - 0.5))/F;
      d += 0.62*fis/F + 0.20*smoothstep(0.0, 0.35, led)*smoothstep(1.0, 0.65, led)/F;
      float coat = d - 0.60/F;
      d *= 0.50;
      if(d < res.x){ res = vec2(d, M_ROCK); gRockCrk = fis; }
      if(b.w > 0.0){
        vec3 c = vec3(xz.x, q.y - H*0.92, xz.y);
        float isl = step(b.x, 0.3);
        vec3 r = mix(vec3(Rx*0.70*b.w, 0.12*H, Rz*0.72*b.w), vec3(Rx*0.62, 0.42*H, Rz*0.80), isl);
        float dv = sdEllipsoid(c + vec3(0.0, isl*0.30*H, 0.0), r);
        dv += (1.25*(vnoise3(q*F*1.05 + sd*2.0) - 0.5) + 0.62*(vnoise3(q*F*2.6 + sd) - 0.5))/F*(1.0 + 0.3*isl);
        dv = (isl > 0.5 ? dv : smax(dv, coat, 0.12))*0.6;       // hugs the rock, does not overhang
        if(dv < res.x) res = vec2(dv, M_VEG);
      }
    }
  }
  return res;
}

// -------- near dressing: reeds, cattails, a thicket behind them, pads, the lily
vec3  gNearQ; float gNearId;          // what the painter needs: local coordinate, hash of the piece
vec3  gTmpQ;  float gTmpId;

// an arc: rooted at o, curving toward az.  q.x bends, q.z is out of its plane
float arcS(vec3 q, float Ra, out float u){
  float th = atan(q.y, Ra - q.x);
  u = length(vec2(q.x - Ra, q.y)) - Ra;
  return th*Ra;
}
// a leaf blade: a thin ribbon along the arc, turning slowly so a blade seen edge on still shows a face
float sdBlade(vec3 p, vec3 o, float az, float L, float Ra, float W, float tw, out vec2 lc){
  vec3 q = p - o;
  q.xz = rot(az)*q.xz;
  float u; float sr = arcS(q, Ra, u);
  float s = clamp(sr, 0.0, L), t = s/L;
  vec2 c = rot(tw + 1.5*t)*vec2(u, q.z);
  float w = W*pow(1.0 - t, 0.62)*(0.72 + 0.28*smoothstep(0.0, 0.30, t));
  float cx = c.x - 0.30*abs(c.y);                       // a shallow V: the fold along the midrib
  float dsec = length(vec2(max(abs(c.y) - w, 0.0), cx)) - mix(0.0040, 0.0014, t);
  lc = vec2(c.y/max(w, 1e-4), t);
  return max(dsec, sr - L)*0.62;
}
// a cattail: a round stem with the velvet head on it and the thin spike above
float sdTail(vec3 p, vec3 o, float az, float L, float Ra, float RH, out float tOut){
  vec3 q = p - o;
  q.xz = rot(az)*q.xz;
  float u; float sr = arcS(q, Ra, u);
  float s = clamp(sr, 0.0, L), t = s/L;
  const float H0 = 0.64, H1 = 0.87;
  float hd = smoothstep(H0 - 0.006, H0 + 0.010, t)*smoothstep(H1 + 0.006, H1 - 0.010, t);
  float r = mix(mix(0.0100, 0.0052, t), RH, hd);
  r *= smoothstep(1.0, 0.90, t)*0.8 + 0.2;
  tOut = t;
  return max(length(vec2(u, q.z)) - r, sr - L)*0.8;
}
// c = root of the clump, sp = spread in x and z, H = height, dirx = the way the blades bend,
// NB blades and NT cattails, Rk = curvature radius in lengths: smaller droops more
vec2 reedClump(vec3 p, vec3 c, vec2 sp, float H, float dirx, float sd, int NB, int NT, float Rk){
  float d = 1e9; float mt = M_BLADE;
  for(int i=0;i<NB;i++){
    float fi = float(i);
    vec3 h = hash31(fi*1.37 + sd), h2 = hash31(fi*2.91 + sd + 4.0);
    vec3 o = c + vec3((h.x - 0.5)*sp.x, 0.0, (h.y - 0.5)*sp.y);
    bool inward = (i & 7) == 7 && Rk < 1.0;                 // a few lean back into the frame, more upright
    float az = (inward ? 3.14159 - dirx : dirx) + (h2.x - 0.5)*1.0;
    float L = H*mix(0.62, 1.12, h.z);
    float Ra = L*mix(Rk, Rk + 1.2, h2.y)*(inward ? 1.9 : 1.0);
    vec2 lc;
    float db = sdBlade(p, o, az, L, Ra, mix(0.014, 0.026, h2.z), 1.05 + h.x*1.2, lc);
    if(db < d){ d = db; mt = M_BLADE; gTmpQ = vec3(lc, 0.0); gTmpId = h.y; }
  }
  for(int i=0;i<NT;i++){
    float fi = float(i);
    vec3 h = hash31(fi*5.13 + sd + 9.0), h2 = hash31(fi*3.77 + sd + 2.0);
    vec3 o = c + vec3((h.x - 0.5)*sp.x, 0.0, (h.y - 0.5)*sp.y);
    float L = H*mix(0.78, 1.00, h.z);
    float tt;
    float dt = sdTail(p, o, dirx + (h2.x - 0.5)*0.8, L, L*mix(3.0, 6.0, h2.y), mix(0.026, 0.037, h2.z), tt);
    if(dt < d){ d = dt; mt = (tt > 0.635 && tt < 0.875) ? M_HEAD : M_STEM; gTmpQ = vec3(0.0, tt, 0.0); gTmpId = h.y; }
  }
  return vec2(d, mt);
}
// the thicket: many thin stems behind the hero blades, one jittered grid, a constant 4 stems per lookup
#define NCL 4
const vec3 CLP[NCL] = vec3[NCL](      // x, z, radius
  vec3(-2.60,-2.2,0.95), vec3(-3.35,-3.9,1.25), vec3( 2.92,-2.6,0.62), vec3( 3.75,-4.6,0.95)
);
float clumpMask(vec2 xz){
  float m = 0.0;
  for(int i=0;i<NCL;i++){
    vec2 v = (xz - CLP[i].xy)/CLP[i].z;
    m = max(m, 1.0 - dot(v, v));
  }
  return clamp(m*1.6, 0.0, 1.0);
}
vec2 thicket(vec3 pw, float cell, float H, float rad, float seed){
  float y = pw.y - WL;
  float Hn = y/H;
  vec2 xz = pw.xz - vec2(-sign(pw.x)*0.10 + 0.07*(vnoise2(pw.xz*0.7) - 0.5), 0.04)*Hn*Hn*H;
  vec2 g = xz/cell - 0.5; vec2 c0 = floor(g);
  float d = 1e9; float mt = M_REED;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, seed));
    vec2 base = (ci + 0.5 + (h.xy - 0.5)*0.74)*cell;
    float pres = clumpMask(base);
    if(pres < 0.02 || h.z > pres*0.66) continue;
    float h2 = fract(h.x*7.13 + h.y*3.1);
    float Hi = H*(0.52 + 0.60*h2);
    float v = clamp(y/Hi, 0.0, 1.0);
    vec2 q = xz - base - (h.yx - 0.5)*cell*0.55*v*v;
    float dd = (length(q) - rad*(0.10 + 0.90*sqrt(max(1.0 - v, 0.0))));
    dd = max(dd, y - Hi);
    if(dd < d){ d = dd; mt = M_REED; gTmpId = h.y; gTmpQ = vec3(0.0, h2, 0.0); }
    if(fract(h.z*31.7) < 0.032){
      float head = sdCapsule(vec3(q.x, y, q.y), vec3(0.0,Hi*0.74,0.0), vec3(0.0,Hi*0.90,0.0), rad*3.0);
      if(head < d){ d = head; mt = M_HEAD; }
    }
  }
  // only the four nearest cells were looked at: a stem one cell further is at least 0.36*cell away, so the
  // step must not pass that, or rays tunnel through stems and heads in stripes
  return vec2(min(d*0.72, 0.30*cell), mt);
}
float thicketBound(vec3 p){
  float d = 1e9;
  for(int i=0;i<NCL;i++) d = min(d, length(p.xz - CLP[i].xy) - CLP[i].z - 0.20);
  return max(d, p.y - WL - 2.30);
}

// pads: x, z, radius, turn
#define NPD 8
const vec4 PD[NPD] = vec4[NPD](
  vec4(-1.19, 0.10,0.23,0.4), vec4(-1.52, 0.42,0.20,2.1), vec4(-1.07,-0.42,0.13,4.0), vec4(-1.72,-0.28,0.16,1.0),
  vec4( 1.17, 0.78,0.19,5.1), vec4( 1.43, 0.34,0.15,3.0), vec4( 1.14,-0.14,0.11,0.9), vec4(-1.34, 0.86,0.15,5.7)
);
vec2 gPadC; float gPadR;
float sdPads(vec3 p){
  float d = 1e9;
  for(int i=0;i<NPD;i++){
    vec2 v = p.xz - PD[i].xy;
    float R = PD[i].z;
    float a = atan(v.y, v.x) + PD[i].w;
    float notch = smoothstep(0.0, 0.16, abs(mod(a, 6.28318) - 3.14159));
    float rad = R*(0.90 + 0.10*notch)*(0.975 + 0.025*sin(a*6.0 + PD[i].w*4.0));
    float l = length(v);
    float lift = 0.012 + 0.009*smoothstep(0.82, 1.0, l/R);    // the rim turns up
    float dd = max(l - rad, abs(p.y - WL - lift) - 0.006);
    if(dd < d){ d = dd; gPadC = PD[i].xy; gPadR = R; }
  }
  return d*0.8;
}
// the water lily: three rings of pointed cupped petals on a golden heart
const vec3 LILY = vec3(-1.20, -0.838, 0.09);     // on the first pad
const float LILY_S = 0.165;
float sdPetals(vec3 u, float n, float off, float el, float L, float W, out vec2 lc){
  float seg = 6.28318/n;
  float a = atan(u.z, u.x) + off;
  float am = mod(a, seg) - seg*0.5;
  float r = length(u.xz);
  vec3 q = vec3(cos(am)*r, u.y, sin(am)*r);
  q.xy = rot(-el)*q.xy;                       // the petal lies along +x, rising by el
  q.y -= 0.42*q.z*q.z/W - 0.10*sin(q.x/L*3.14159);    // cupped, and curled along its length
  float t = clamp(q.x/L, 0.0, 1.0);
  float w = W*pow(sin(3.14159*pow(t, 0.62)), 0.80);
  lc = vec2(q.z/max(w, 1e-3), t);
  float d = max(abs(q.z) - w, max(-q.x, q.x - L));
  return max(d*0.7, abs(q.y) - 0.012)*0.8;
}
vec2 sdLily(vec3 p){
  vec3 u = (p - LILY)/LILY_S;
  vec2 lc, best = vec2(0.0);
  float d = sdPetals(u, 9.0, 0.3, 0.30, 1.05, 0.30, lc); best = lc; float ring = 0.0;
  float d2 = sdPetals(u - vec3(0.0,0.03,0.0), 8.0, 0.9, 0.62, 0.92, 0.27, lc);
  if(d2 < d){ d = d2; best = lc; ring = 1.0; }
  d2 = sdPetals(u - vec3(0.0,0.06,0.0), 7.0, 1.7, 1.02, 0.70, 0.22, lc);
  if(d2 < d){ d = d2; best = lc; ring = 2.0; }
  float heart = sdEllipsoid(u - vec3(0.0,0.20,0.0), vec3(0.20,0.15,0.20));
  heart += 0.035*sin(atan(u.z, u.x)*19.0)*smoothstep(0.0, 0.10, u.y - 0.14);
  gTmpQ = vec3(best, ring); gTmpId = 0.0;
  if(heart < d) return vec2(heart*LILY_S*0.8, M_STAMEN);
  return vec2(d*LILY_S, M_PETAL);
}

vec2 opUn(vec2 a, vec2 b){ if(b.x < a.x){ gNearQ = gTmpQ; gNearId = gTmpId; return b; } return a; }

const vec3 RC_L = vec3(-1.62, WL - 0.02, 0.60);
const vec3 RC_R = vec3( 1.90, WL - 0.02, 0.55);

vec2 nearMap(vec3 p, bool shadow, float G){
  vec2 res = vec2(1e5, M_REED);
  // hero reeds at each edge
  float bL = sdCapsule(p, RC_L + vec3(-0.15,0.0,0.0), RC_L + vec3(-0.45,2.2,0.0), 1.12);
  if(bL >= G) res.x = min(res.x, bL);
  for(int i=0;i<1 && bL<G;i++){ res = opUn(res, reedClump(p, RC_L, vec2(0.86,1.70), 2.35, 3.14159, 1.3, 26, 4, 0.72)); }
  float bR = sdCapsule(p, RC_R + vec3(0.1,0.0,0.0), RC_R + vec3(-0.25,2.0,0.0), 1.05);
  if(bR >= G) res.x = min(res.x, bR);
  for(int i=0;i<1 && bR<G;i++){ res = opUn(res, reedClump(p, RC_R, vec2(0.70,1.25), 1.90, 3.14159, 5.9, 18, 3, 1.55)); }
  // the thicket behind them
  float bT = thicketBound(p);
  if(bT >= G) res.x = min(res.x, bT);
  for(int i=0;i<1 && bT<G;i++){
    res = opUn(res, thicket(p, 0.118, 1.95, 0.0125, 3.0));
    res = opUn(res, thicket(p + vec3(0.05,0.0,0.07), 0.155, 1.60, 0.0140, 7.0));
  }
  // pads and the lily stay out of the sun-shadow and mirror rays: too flat to matter there
  if(!shadow){
    float bP = max(abs(abs(p.x) - 1.42) - 0.62, max(abs(p.z - 0.26) - 0.92, p.y - WL - 0.34));
    if(bP >= G) res.x = min(res.x, bP);
    for(int i=0;i<1 && bP<G;i++){
      res = opU(res, vec2(sdPads(p), M_PAD));
      res = opUn(res, sdLily(p));
    }
  }
  return res;
}

// distance along a camera ray to the water: every primary ray starts at the camera, so similar triangles
// give the exact step and the grazing plane needs no marching.  Shadow and mirror rays only ever go up.
float planeStep(vec3 p, float y){
  float den = uCamPos.y - WL - y;
  return (den > 1e-3) ? max(y, 0.0)*length(p - uCamPos)/den + min(y, 0.0) : 1e3;
}

vec2 setMap(vec3 p, bool shadow){
  // already shaded: nothing below may wake up.  Radiance pixels return "in shadow" so the 56-step shadow loop
  // stops at once, except water that carries glitter, whose sun term needs its visibility.
  bool off = gRadShade > 0.5 || gSurfDone > 0.5;
  float G = off ? -1e6 : 0.22;
  float y = p.y - WL;
  vec2 res = vec2(shadow ? 1e3 : planeStep(p, y), M_WATER);
  res = stacks(p, G, res);
  res = opU(res, nearMap(p, shadow, G));
  if(off) res = vec2((shadow && gRadShade > 0.5 && gGlit < 1e-4) ? -1.0 : 1e3, M_WATER);
  return res;
}

// -------- paint
vec3 rockRad(vec3 p, vec3 n, float m){
  float y = p.y - WL;
  float f = fbm3(p*vec3(0.9,0.42,0.9));
  float str = 0.5 + 0.5*sin(y*6.0 + 3.0*vnoise2(p.xz*0.55));
  vec3 a = mix(vec3(0.150,0.100,0.135), vec3(0.340,0.235,0.250), smoothstep(0.30, 0.72, f));
  a *= 0.80 + 0.30*str;
  a = mix(a, vec3(0.085,0.070,0.100), smoothstep(0.75, 0.0, y));      // wet dark foot
  a *= 1.0 - 0.72*gRockCrk;                                             // the fissures stay dark
  float scrub = smoothstep(0.30, 0.72, n.y)*smoothstep(0.40, 0.56, vnoise3(p*1.9));   // tufts on the ledges
  vec3 green = mix(vec3(0.060,0.082,0.082), vec3(0.150,0.170,0.125), vnoise3(p*2.3));
  a = mix(a, green, m == M_VEG ? 1.0 : scrub*0.85);
  vec3 toSun = normalize(SUNV + vec3(0.0,0.22,0.0));
  float w = clamp(dot(n, toSun)*0.5 + 0.5, 0.0, 1.0);
  float lam = clamp(dot(n, vec3(-0.38,0.62,0.68))*0.5 + 0.5, 0.0, 1.0);   // a soft sky side and a shade side
  vec3 amb = vec3(0.62,0.47,0.64)*mix(0.42, 1.25, lam*lam) + SP_BOUNCE*0.5*clamp(-n.y, 0.0, 1.0);
  w *= w;
  return a*(amb + vec3(3.4,1.65,0.86)*w*w);
}

void solidSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){
  float y = p.y - WL;
  // the visible sun comes through anything thin
  float sd = max(dot(rd, SUNV), 0.0);
  float sd2 = sd*sd, sd4 = sd2*sd2;
  float against = sd4*sd4*sd2;
  vec3 glow = mix(vec3(1.0), vec3(0.62,0.54,0.56), against);
  if(m == M_BLADE){
    vec3 q = gNearQ;
    float k = gNearId;
    float v = q.y;
    vec3 a = mix(vec3(0.050,0.098,0.088), vec3(0.150,0.200,0.120), smoothstep(0.02, 0.52, v));
    a = mix(a, mix(vec3(0.40,0.25,0.21), vec3(0.32,0.28,0.22), k), smoothstep(0.58, 1.04, v)*(0.35 + 0.65*k));
    a *= 0.46 + 0.86*fract(k*7.3);
    a *= mix(0.40, 1.0, smoothstep(0.0, 1.35, y));             // the clump's own shade: no reed shadows here
    float rib = smoothstep(0.16, 0.0, abs(q.x));
    float fine = 0.92 + 0.08*sin(q.x*19.0);
    a = mix(a*fine, a*1.32 + 0.012, rib*0.55);
    a *= smoothstep(-0.05, 0.22, y)*0.55 + 0.45;
    s.alb = a*glow; s.wrap = 0.5; s.trans = 0.7; s.transCol = a; s.spec = 0.35; s.shin = 36.0; s.sheen = 0.5 + 0.3*against;
  } else if(m == M_STEM || m == M_REED){
    float v = clamp(y/2.2, 0.0, 1.0);
    float k = m == M_REED ? gNearId : hash12(vec2(gNearId, 3.0));
    vec3 a = mix(vec3(0.060,0.095,0.082), vec3(0.165,0.180,0.120), smoothstep(0.05, 0.60, v));
    a = mix(a, vec3(0.36,0.25,0.19), smoothstep(0.50, 0.95, v)*(0.35 + 0.65*k));
    a *= (0.55 + 0.60*k)*mix(0.34, 1.0, smoothstep(0.0, 1.1, y))*0.82;
    s.alb = a*glow; s.wrap = 0.45; s.trans = 0.7; s.transCol = a; s.spec = 0.2; s.shin = 28.0; s.sheen = 0.3;
  } else if(m == M_HEAD){
    float f = vnoise3(p*vec3(70.0,26.0,70.0));
    float g = vnoise3(p*vec3(190.0,80.0,190.0));
    s.alb = mix(vec3(0.115,0.052,0.036), vec3(0.285,0.145,0.080), f*0.65 + g*0.35);
    s.n = normalize(n + 0.30*(vec3(f, g, vnoise3(p*150.0)) - 0.5));
    s.wrap = 0.45; s.sheen = 0.9; s.spec = 0.05; s.shin = 12.0;
  } else if(m == M_PAD){
    vec2 v = p.xz - gPadC;
    float r = length(v)/gPadR;
    float a = atan(v.y, v.x);
    float vein = 0.5 + 0.5*cos(a*17.0 + 2.0*r);
    float k = fbm2_3(p.xz*9.0);
    vec3 c = mix(vec3(0.050,0.125,0.100), vec3(0.150,0.235,0.125), smoothstep(0.30, 0.72, k));
    c *= 0.84 + 0.26*vein*smoothstep(0.05, 0.3, r);
    c = mix(c, vec3(0.170,0.095,0.100), smoothstep(0.92, 0.99, r));         // the red rolled rim
    s.alb = c + vec3(0.12,0.08,0.10)*smoothstep(0.75, 0.2, r)*0.5;      // waxy: it holds a little sky
    s.spec = 0.9; s.shin = 46.0; s.sheen = 0.7; s.wrap = 0.35;
  } else if(m == M_PETAL){
    vec3 q = gNearQ;
    float tip = smoothstep(0.30, 1.02, q.y);
    vec3 c = mix(vec3(0.84,0.70,0.68), vec3(0.80,0.26,0.42), tip*(0.62 + 0.20*q.z));
    c = mix(c, vec3(0.70,0.54,0.26), smoothstep(0.26, 0.0, q.y)*0.6);    // gold at the heart
    c *= 0.95 + 0.05*sin(q.x*24.0);                                       // fine veins
    c *= 1.0 - 0.32*smoothstep(0.55, 1.0, abs(q.x))*(1.0 - tip);
    s.alb = c*glow; s.wrap = 0.6; s.trans = 0.8; s.transCol = vec3(1.0,0.52,0.55); s.sheen = 0.5; s.spec = 0.18; s.shin = 26.0;
    s.occ = 0.68 + 0.32*smoothstep(0.0, 0.42, q.y);
  } else if(m == M_STAMEN){
    s.alb = mix(vec3(0.74,0.40,0.07), vec3(0.86,0.62,0.13), vnoise3(p*180.0));
    s.wrap = 0.5; s.sheen = 0.4; s.occ = 0.8;
  }
}

// a mirror ray off the water: goes up, so it never needs the plane
vec3 mirrorTrace(vec3 ro, vec3 rd){
  float t = 0.015; float m = -1.0;
  for(int i=0;i<96;i++){
    vec3 p = ro + rd*t;
    vec2 h = setMap(p, true);
    if(h.x < 0.0011*t){ m = h.y; break; }      // out of steps in the thicket's fringe counts as a miss
    t += h.x;
    if(t > 230.0 || p.y > WL + 7.0) break;
  }
  vec3 col;
  float tEnd = 150.0;
  if(m < 0.0){
    gMirrorRay = 1.0;
    col = setSky(rd);
  } else {
    vec3 p = ro + rd*t;
    tEnd = min(t, 150.0);
    vec3 n = vec3(0.0);
    float e = 0.002*max(t, 1.0);
    for(int i=0;i<4;i++){
      vec3 k = 0.5773*(2.0*vec3(float(((i+3)>>1)&1), float((i>>1)&1), float(i&1)) - 1.0);
      n += k*setMap(p + k*e, true).x;
    }
    n = normalize(n + vec3(0.0,1e-6,0.0));     // out of steps in the thicket every tap is the step limit: no gradient
    if(m == M_ROCK || m == M_VEG){
      col = rockRad(p, n, m);
    } else {
      nearMap(p, true, 0.22);
      Surf s = surfDefault(n);
      solidSurface(p, rd, t, m, n, s);
      float dif = clamp((dot(n, uSunDir) + s.wrap)/(1.0 + s.wrap), 0.0, 1.0);
      col = s.alb*(setAmbient(p, n) + uSunCol*dif*0.7);
    }
  }
  // the same mist on the way back
  vec3 fa = setFogAmb(rd), ins = vec3(0.0);
  float tr = 1.0, dt = tEnd/7.0;
  for(int i=0;i<7;i++){
    float dens = setFog(ro + rd*(float(i) + 0.5)*dt);
    ins += tr*dens*dt*fa;
    tr *= exp(-dens*dt);
  }
  if(m < 0.0) tr = mix(tr, 1.0, 0.35);
  return col*tr + ins;
}

// how much sun this water pixel sends the camera: the slope a facet needs to mirror the visible sun,
// through the slope statistics of a nearly calm surface.  World x and log depth keep the sparkles the
// right shape from the frame's bottom to the mist.
float glitter(vec3 p, vec3 rd, float t){
  vec3 m = SUNV - rd;
  vec2 tilt = m.xz/max(m.y, 1e-3);
  vec2 q = tilt/vec2(0.25, 0.42);
  float G = exp(-dot(q, q));
  float lt = log(t);
  vec2 gs = vec2(p.x*44.0, lt*88.0);
  float n = vnoise2(gs)*0.60 + vnoise2(gs*vec2(2.1,1.7) + 7.0)*0.40 + 0.20*(vnoise2(vec2(p.x*2.6, lt*9.0)) - 0.5);
  float sp = smoothstep(0.63, 0.84, n);
  return G*(0.15 + 2.6*sp);
}

vec3 waterRad(vec3 p, vec3 rd, float t, out float gl){
  vec2 w = p.xz;
  float near = smoothstep(30.0, 4.2, t);
  // long calm lanes and a slope field drawn out across the view; ripples grow toward the camera
  float lane = smoothstep(0.30, 0.72, vnoise2(w*vec2(0.06,0.52) + vec2(9.0,2.0)));
  const float E = 0.025;
  vec2 f1 = w*vec2(1.0,3.1), f2 = w*vec2(2.9,9.3);
  float n0 = fbm2_3(f1) + 0.34*vnoise2(f2);
  float nx = fbm2_3((w + vec2(E,0.0))*vec2(1.0,3.1)) + 0.34*vnoise2((w + vec2(E,0.0))*vec2(2.9,9.3));
  float nz = fbm2_3((w + vec2(0.0,E))*vec2(1.0,3.1)) + 0.34*vnoise2((w + vec2(0.0,E))*vec2(2.9,9.3));
  vec2 g = vec2(nx - n0, nz - n0)/E;
  g *= mix(0.0016, 0.0125, near)*(0.35 + 0.65*lane + 0.35*near);
  vec3 N = normalize(vec3(-g.x, 1.0, -g.y));
  vec3 R = reflect(rd, N);
  R.y = abs(R.y) + 0.001;
  R = normalize(R);
  vec3 refl = mirrorTrace(p + vec3(0.0,0.012,0.0), R);
  float subjA = 0.0; vec3 subjC = vec3(0.0), subjH = vec3(0.0);
#ifdef SP_SUBJECT
  // The subject in the mirror.  Taken literally its reflection starts at the very bottom edge of the frame,
  // because it hovers 0.2 above water seen at only 7 degrees: it is drawn closer to the water (a squash
  // about the water plane) so the top of the reflection reads, the way ripples stretch a real one.
  {
    vec4 S = SP_SUBJECT;
    vec3 Rs = normalize(vec3(R.x, R.y/SP_REFL_SQUASH, R.z));
    vec3 oc = (p - S.xyz)/S.w;
    float b = dot(oc, Rs);
    vec3 cp = oc - Rs*b;
    float dd = length(cp);
    float a = smoothstep(1.0, 0.70, dd)*step(b, 0.0);
    vec3 pearl = mix(vec3(1.05,0.82,0.70), vec3(0.82,0.78,0.90), smoothstep(-0.95, -0.30, cp.y));
    float halo = exp(-max(dd - 0.9, 0.0)*2.6)*step(b, 0.0);
    subjA = a*SP_REFL_GAIN; subjC = pearl;
    subjH = vec3(0.30,0.21,0.23)*halo*SP_REFL_GAIN;                 // it glows: a pool of light on the water
  }
#endif
  float nv = clamp(dot(-rd, N), 0.0, 1.0);
  float F = 0.02 + 0.98*pow(1.0 - nv, 5.0);
  F = clamp(0.70 + F, 0.0, 0.985);                 // a calm mirror, more than physical
  vec3 col = mix(SP_WATER_C, refl, F);
  // weight at the bottom of the frame
  float dark = smoothstep(14.0, 4.5, t);
  col = mix(col, col*vec3(0.13,0.205,0.27) + SP_WATER_C*0.60, dark*0.96);
  col *= 0.97 + 0.06*lane;
  col = mix(col, subjC, subjA) + subjH;
  gl = glitter(p, rd, t);
  return col;
}

void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){
  // a ray out of steps in the thicket has no gradient, so the engine's normal is NaN, and one NaN pixel turns
  // the whole FFT lens into NaN
  if(!(dot(n, n) > 0.5)) n = vec3(0.0,0.0,1.0);
  s.n = n;
  if(m == M_WATER){
    float gl;
    vec3 col = waterRad(p, rd, t, gl);
    gGlit = gl;
    s.n = vec3(0.0,1.0,0.0); s.wrap = 1.0;
    float dif = (dot(s.n, uSunDir) + 1.0)*0.5;
    s.alb = vec3(1.0,0.64,0.36)*gl/(uSunCol*dif) + 1e-4;
    gRad = col/s.alb;
    s.spec = 0.0; s.sheen = 0.0; s.trans = 0.0;
    gRadShade = 1.0;
  } else if(m == M_ROCK || m == M_VEG){
    gRad = rockRad(p, n, m)/1e-4;
    s.alb = vec3(1e-4);
    s.spec = 0.0; s.sheen = 0.0; s.trans = 0.0; s.wrap = 0.0;
    gGlit = 0.0;
    gRadShade = 1.0;
  } else {
    nearMap(p, false, 0.22);        // again, for the painter's local coordinates
    solidSurface(p, rd, t, m, n, s);
    s.occ *= 0.58 + 0.42*smoothstep(0.0, 0.9, p.y - WL);    // painted: the water's edge sits in the reeds' shade
    gSurfDone = 1.0;
  }
}
#else
vec2 setMap(vec3 p, bool shadow){ return vec2(1.0, 0.0); }
void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){}
#endif

// ---------------------------------------------------------------------------
//  engine/sets/river.glsl — a golden-hour river.  Calm mirror water, hero reeds and papyrus at the
//  frame edges, lily pads, an acacia bank, soft hills, a pink-gold sky and a low hazy sun.
//  Implements the SET API documented in engine/sets/forest.glsl.  World units; WL is the water level.
//
//  The sun the camera sees (SUNP, painted into the sky) and the key light (uSunDir) are different on
//  purpose: the key stays in front of the character, the painted sun gives the backdrop its glow and
//  feeds the rim.  Everything heavy is inside #ifdef PASS_BG, because this file is also compiled into
//  the sprite vertex shader.
//
//  Water, two halves:
//   bake   reflections are traced into the ambient layer (the sun layer stays black on water).
//   live   PASS_LIVE has no set hook, so the #define at the end of this file reroutes its texture()
//          calls through riverLive(): water pixels slide along moving waves, which makes the baked
//          reflections shimmer, and pick up glints and ring ripples around the character.
//   The character's reflection is in the bake too: mirror rays also hit charShadow(), grown a little.
//   Ring ripples and a contact shadow around the body are added live in riverLive().
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-1.25)                          // the water level
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.70,0.45,0.30)               // light off the water
#endif
#ifndef SP_SUN_PAINT
#define SP_SUN_PAINT vec3(0.232,0.074,-1.0)          // where the visible sun sits
#endif
#ifndef SP_SKY_LOW
#define SP_SKY_LOW vec3(1.70,0.46,0.15)              // gold at the horizon
#endif
#ifndef SP_SKY_MID
#define SP_SKY_MID vec3(1.20,0.29,0.235)              // pink band
#endif
#ifndef SP_SKY_HIGH
#define SP_SKY_HIGH vec3(0.36,0.25,0.46)             // dusty lilac overhead
#endif
#ifndef SP_SUN_GLOW
#define SP_SUN_GLOW vec3(1.0,0.46,0.14)
#endif
#ifndef SP_AMB_SKY
#define SP_AMB_SKY vec3(0.37,0.335,0.50)
#endif
#ifndef SP_WATER_C
#define SP_WATER_C vec3(0.075,0.070,0.060)           // body colour of the river
#endif
#ifndef SP_FOG_LO
#define SP_FOG_LO vec3(1.05,0.36,0.27)
#endif
#ifndef SP_FOG_HI
#define SP_FOG_HI vec3(1.55,0.55,0.20)
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#ifndef SP_RING_C
#define SP_RING_C vec2(-0.05,-0.80)                  // the character's footprint on the water: centre
#endif
#ifndef SP_RING_S
#define SP_RING_S vec2(1.62,0.98)                    // ... and half size.  Ring ripples start at its edge
#endif
#ifndef SP_REFL_GROW
#define SP_REFL_GROW 0.05                            // shadow proxy -> real skin, for the reflection
#endif
#ifndef SP_REFL_ALB
#define SP_REFL_ALB vec3(0.17,0.135,0.205)           // the character's colour in the reflection
#endif
#ifndef SP_REFL_LEG
#define SP_REFL_LEG vec3(0.300,0.155,0.155)          // a moving leg in the water, drawn live: what the bake holds for the body
#endif

const float WL = SP_GROUND_Y;
const vec3  SUNP = normalize(SP_SUN_PAINT);

#define M_CANOPY 99.0     // unused here: nothing in this set ignores direct shadowing
#define M_WATER 1.0
#define M_BANK  2.0
#define M_REED  3.0
#define M_CAT   4.0
#define M_PAPY  5.0
#define M_UMBEL 6.0
#define M_TREE  7.0
#define M_LILY  8.0
#define M_LOTUS 9.0
#define M_BUSH  10.0
#define M_CHAR  12.0

vec2  gWind;
float gWaterShade = 0.0;     // PASS_BG: 1 while lighting a water pixel, see setMap

void setInit(){
  float T = uTime;
  gWind = 0.06*vec2(sin(T*0.50), cos(T*0.41)) + 0.02*vec2(sin(T*1.3 + 1.0), sin(T*1.1));
}
float sqr(float x){ return x*x; }

// ------------------------------------------------------------------ the water surface in motion
// .xy = surface slope, .z = ring crest brightness.  Cheap: four noise lookups and a sine.
vec3 riverWaves(vec2 xz, float T){
  vec2 a = vec2(xz.x*0.80 + T*0.07, xz.y*2.3 + T*0.30);
  vec2 b = vec2(xz.x*1.90 - T*0.11, xz.y*5.0 + T*0.46);
  const float E = 0.11;
  float n0 = vnoise2(a) + 0.5*vnoise2(b);
  float nx = vnoise2(a + vec2(E,0.0)) + 0.5*vnoise2(b + vec2(E,0.0));
  float ny = vnoise2(a + vec2(0.0,E)) + 0.5*vnoise2(b + vec2(0.0,E));
  vec2 g = vec2(nx - n0, ny - n0)/E;
  vec2 dv = (xz - SP_RING_C)/SP_RING_S;
  float d = length(dv);
  float wob = 0.05*sin(atan(dv.y, dv.x)*3.0 + T*0.4);
  float ph = (d + wob)*13.0 - T*1.9;
  float env = exp(-max(d - 1.0, 0.0)*1.9)*smoothstep(0.86, 1.03, d);
  float ring = sin(ph)*env;
  g += (dv/max(d, 1e-3))*cos(ph)*env*1.4;
  return vec3(g, ring);
}

// ------------------------------------------------------------------ sky, light
vec3 skyBase(vec3 rd){
  float y = max(rd.y + 0.012, 0.0);
  vec3 c = mix(SP_SKY_LOW, SP_SKY_MID, smoothstep(0.015, 0.15, y));
  c = mix(c, SP_SKY_HIGH, smoothstep(0.11, 0.48, y));
  float sd = max(dot(rd, SUNP), 0.0);
  c += SP_SUN_GLOW*(0.34*pow(sd, 7.0) + 0.85*pow(sd, 48.0) + 1.9*pow(sd, 420.0));
  return c;
}
#ifdef PASS_BG
vec3 setSky(vec3 rd){
  vec3 c = skyBase(rd);
  float sd = max(dot(rd, SUNP), 0.0);
  // long sunset cloud bars, under-lit gold near the sun, pink away from it
  if(rd.y > 0.0){
    vec2 uv = rd.xz/(rd.y + 0.10);
    float n = fbm2(uv*vec2(0.55,2.4) + vec2(1.7,0.4));
    float n2 = fbm2(uv*vec2(1.5,6.5) + vec2(4.0,2.0));
    float cl = smoothstep(0.50, 0.74, n*0.75 + n2*0.25 + 0.05*sin(uv.x*0.9));
    cl *= smoothstep(0.015, 0.09, rd.y)*smoothstep(0.75, 0.25, rd.y);
    vec3 lit = mix(vec3(1.50,0.40,0.34), vec3(2.2,0.80,0.26), pow(sd, 6.0));
    vec3 dark = vec3(0.42,0.20,0.34);
    c = mix(c, mix(dark, lit, smoothstep(0.55, 0.8, n2 + 0.25)), cl*0.78);
  }
  // the sun: a soft hazy disc you can look at
  float sq = smoothstep(0.99972, 0.99990, dot(rd, SUNP));
  c = mix(c, vec3(7.0,3.7,1.25), sq);
  return c;
}
#else
vec3 setSky(vec3 rd){ return skyBase(rd); }
#endif

vec3 setAmbient(vec3 p, vec3 n){
#ifdef PASS_BG
  if(gWaterShade > 0.5) return vec3(1.0);     // the water's albedo already is its radiance
#endif
  vec3 sky = SP_AMB_SKY*(0.62 + 0.38*n.y);
  vec3 bounce = SP_BOUNCE*0.55*clamp(0.55 - 0.65*n.y, 0.0, 1.0);
  vec3 toSun = normalize(SUNP + vec3(0.0,0.25,0.0));
  vec3 glow = vec3(1.0,0.66,0.34)*0.42*pow(clamp(dot(n, toSun)*0.5 + 0.5, 0.0, 1.0), 3.0);
  return sky + bounce + glow;
}
// open sky: the light does not dapple, a long slow cloud shadow only
float setCookie(vec3 p){ return 0.93 + 0.07*sin(uTime*0.13 + p.x*0.21); }
float setVol(vec3 p){ return 0.05; }
float setFog(vec3 p){
  float h = max(p.y - WL, 0.0);
  return (0.0034*exp(-h*0.10) + 0.0090*exp(-h*0.42))*SP_FOG_GAIN;   // thins with height: the sky keeps its pink
}
vec3 setFogAmb(vec3 rd){
  float k = pow(clamp(dot(rd, SUNP)*0.5 + 0.5, 0.0, 1.0), 2.5);
  return mix(SP_FOG_LO, SP_FOG_HI, k)*0.95;
}

// =========================================================================== bake only
#ifdef PASS_BG
float charShadow(vec3 p);       // the animal's proxy, at rest: the reflection in the water

// -------- shore and hills
float shoreZ(float x){
  float z = -20.0 - 12.0*smoothstep(-14.0, 30.0, x);
  z += 9.0*exp(-sqr((x + 13.5)/7.5));          // a spit of land at frame left
  z += 1.3*sin(x*0.23 + 1.0) + 0.7*sin(x*0.61 + 2.0);
  return z;
}
float landH(vec2 xz){
  float d = shoreZ(xz.x) - xz.y;               // + inland
  float h = -0.45 + 1.05*smoothstep(-2.0, 7.0, d) + 0.9*smoothstep(7.0, 40.0, d);
  h += 0.6*(vnoise2(xz*0.13) - 0.5)*smoothstep(0.0, 8.0, d);
  h += (0.8 + 4.2*fbm2_3(vec2(xz.x*0.017 + 4.0, 1.3)))*smoothstep(42.0, 85.0, d);
  h += (1.5 + 9.5*fbm2_3(vec2(xz.x*0.0105 + 9.0, 7.7)))*smoothstep(95.0, 150.0, d);
  return h;
}

// -------- reeds: a jittered grid of thin tapered blades, bent by one slow field so a clump leans together
const int NCL = 9;
const vec3 CLP[NCL] = vec3[NCL](          // x, z, radius: the frame's half width is 0.36*(6.7 - z)
  vec3(-2.42, 0.55,0.46), vec3(-2.95,-0.55,0.62), vec3(-3.45,-1.95,0.78), vec3(-4.20,-3.70,1.05), vec3(-5.60,-6.10,1.70),
  vec3( 2.72, 0.20,0.40), vec3( 3.30,-1.30,0.62), vec3( 4.15,-3.40,0.95), vec3( 5.70,-6.20,1.60)
);
float clumpMask(vec2 xz){
  float m = 0.0;
  for(int i=0;i<NCL;i++){
    vec2 v = (xz - CLP[i].xy)/CLP[i].z;
    m = max(m, 1.0 - dot(v, v));
  }
  return clamp(m*1.5, 0.0, 1.0);
}
float shoreMask(vec2 xz){
  float d = shoreZ(xz.x) - xz.y;
  float band = smoothstep(-3.0, -1.6, d)*smoothstep(4.0, 1.5, d);
  return band*smoothstep(0.34, 0.50, vnoise2(vec2(xz.x*0.12, 3.1)));
}
vec2 reedShear(vec2 xz){       // bends toward the screen middle at the sides, opposite to the frame edge
  float s = sign(xz.x + 0.001);
  return vec2(-0.20*s + 0.10*(vnoise2(xz*0.6) - 0.5), 0.05);
}
// returns distance, material
vec2 reedGrid(vec3 pw, float cell, float H, float rad, float seed, float shore, float bend){
  float y = pw.y - WL;
  vec2 sh = reedShear(pw.xz);
  float Hn = y/H;
  vec2 xz = pw.xz - sh*Hn*Hn*H*0.32*bend;
  vec2 g = xz/cell - 0.5; vec2 c0 = floor(g);
  float d = 1e9; float mt = M_REED;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, seed));
    vec2 base = (ci + 0.5 + (h.xy - 0.5)*0.72)*cell;
    float pres = (shore > 0.5) ? shoreMask(base) : clumpMask(base);
    if(pres < 0.02 || h.z > pres*0.62) continue;
    float h2 = fract(h.x*7.13 + h.y*3.1);
    float Hi = H*(0.55 + 0.55*h2);
    float v = clamp(y/Hi, 0.0, 1.0);
    vec2 q = xz - base - (h.yx - 0.5)*cell*0.55*v*v;
    float flat_ = 0.55 + 0.45*(1.0 - v);        // blades flatten toward the tip
    float r = rad*(0.10 + 0.90*sqrt(max(1.0 - v, 0.0)));
    float dd = (length(q*vec2(1.0, 1.0/flat_)) - r)*flat_;
    dd = max(dd, y - Hi);
    if(dd < d){ d = dd; mt = M_REED; }
    if(fract(h.z*31.7) < 0.14){                  // bulrush heads on some stems
      float y0 = Hi*0.76, y1 = Hi*0.92;
      float cat = sdCapsule(vec3(q.x, y, q.y), vec3(0.0,y0,0.0), vec3(0.0,y1,0.0), rad*1.9);
      if(cat < d){ d = cat; mt = M_CAT; }
    }
  }
  return vec2(d*0.72, mt);
}

// -------- papyrus: tall bare stems under a burst of rays
const int NPY = 13;
const vec4 PY[NPY] = vec4[NPY](           // x, z, height above water, umbel radius
  vec4(-2.28, 0.72,1.98,0.31), vec4(-2.62,-0.30,2.46,0.38), vec4(-2.34,-0.95,1.84,0.30),
  vec4(-3.04,-1.25,2.82,0.42), vec4(-3.28,-2.45,2.26,0.36), vec4(-3.62,-0.45,2.64,0.40),
  vec4(-4.10,-3.30,3.10,0.46), vec4(-3.40,-4.20,2.40,0.38),
  vec4( 2.42, 0.34,2.04,0.30), vec4( 2.88,-0.82,2.42,0.36), vec4( 3.42,-1.70,2.80,0.42),
  vec4( 3.12,-2.70,2.18,0.34), vec4( 4.15,-3.40,3.05,0.46)
);
float sdUmbel(vec3 u, float R, float seed){
  float r = length(u);
  vec3 dir = u/max(r, 1e-4);
  float el = asin(clamp(dir.y, -1.0, 1.0));
  const float RINGS = 6.0, E0 = -0.42, ER = 1.96;
  float ri = clamp(floor((el - E0)/ER*RINGS), 0.0, RINGS - 1.0);
  float elc = (ri + 0.5)/RINGS*ER + E0;
  float cnt = floor(mix(30.0, 5.0, (ri + 0.5)/RINGS));
  float off = ri*1.71 + seed;
  float az = atan(u.z, u.x) + off;
  float ai = floor(az/6.28318*cnt);
  float azc = (ai + 0.5)/cnt*6.28318 - off;
  float hh = hash12(vec2(ai + 1.0, ri + seed*7.0));
  vec3 c = vec3(cos(elc)*cos(azc), sin(elc), cos(elc)*sin(azc));
  float len = R*(0.74 + 0.26*hh);
  float along = clamp(dot(u, c), 0.0, len);
  // rays droop toward their tips
  vec3 ctr = c*along; ctr.y -= 0.22*sqr(along/R)*R*(1.0 - 0.5*hh);
  float d = length(u - ctr) - mix(0.0078, 0.0040, along/R);
  // a feathery tuft at each tip
  d = smin(d, length(u - ctr) + 0.024 - 0.042*smoothstep(0.76, 1.0, along/len), 0.01);
  return d;
}

// -------- flat-topped acacias on the bank
const int NTR = 9;
const vec4 TRE[NTR] = vec4[NTR](           // x, z, height, canopy radius
  vec4(-11.8,-16.5,3.1,2.5), vec4(-17.5,-20.0,3.9,3.1), vec4(-7.6,-30.5,3.4,2.5),
  vec4(  2.5,-41.0,3.5,2.8), vec4( 19.0,-44.0,4.2,3.3), vec4( 27.0,-52.0,3.6,3.0),
  vec4( -2.5,-55.0,3.2,2.4), vec4(-27.0,-36.0,4.6,3.6), vec4( 33.0,-62.0,4.2,3.4)
);
vec2 sdAcacia(vec3 p, float H, float R, float seed){
  vec3 q = p;
  q.xz = rot(seed)*q.xz;
  vec3 c = q - vec3(0.0,H,0.0);
  float d = sdEllipsoid(c, vec3(R,R*0.17,R));
  d = smin(d, sdEllipsoid(c - vec3( R*0.42, R*0.07, R*0.20), vec3(R*0.52,R*0.13,R*0.5)), R*0.14);
  d = smin(d, sdEllipsoid(c - vec3(-R*0.40,-R*0.02,-R*0.26), vec3(R*0.55,R*0.14,R*0.5)), R*0.14);
  d += 0.30*R*0.3*(vnoise3(p*1.3 + seed) - 0.5) + 0.20*R*0.14*(vnoise3(p*3.7) - 0.5);
  float t = sdTaper(q, vec3(0.0,-0.2,0.0), vec3(0.08*R,H*0.52,0.0), 0.070*H, 0.036*H);
  t = smin(t, sdTaper(q, vec3(0.08*R,H*0.5,0.0), vec3( R*0.60,H*0.95, R*0.20), 0.032*H, 0.012*H), 0.2);
  t = smin(t, sdTaper(q, vec3(0.08*R,H*0.5,0.0), vec3(-R*0.52,H*0.95, R*0.30), 0.030*H, 0.012*H), 0.2);
  t = smin(t, sdTaper(q, vec3(0.08*R,H*0.5,0.0), vec3(-R*0.15,H*0.97,-R*0.55), 0.028*H, 0.012*H), 0.2);
  return (d < t) ? vec2(d*0.8, M_TREE) : vec2(t*0.9, M_BANK + 0.5);
}

// -------- lily pads and lotus
float padMask(vec2 c){
  float keep = smoothstep(1.25, 1.75, length((c - SP_RING_C)/SP_RING_S));   // none under the character
  float patch_ = smoothstep(0.40, 0.62, vnoise2(c*0.46 + vec2(2.6,7.3)));
  float behind = smoothstep(-17.0, -6.0, c.y);
  return keep*patch_*behind;
}
vec2 gPadC; float gPadR;
vec2 sdPads(vec3 pw){
  const float CS = 0.86;
  vec2 g = pw.xz/CS - 0.5; vec2 c0 = floor(g);
  float d = 1e9;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, 41.0));
    vec2 base = (ci + 0.5 + (h.xy - 0.5)*0.55)*CS;
    if(h.z > padMask(base)*0.95) continue;
    float R = 0.13 + 0.17*fract(h.x*9.1);
    vec2 v = pw.xz - base;
    float a = atan(v.y, v.x) + h.y*6.28;
    float notch = smoothstep(0.0, 0.14, abs(mod(a, 6.28318) - 3.14159));
    float rad = R*(0.86 + 0.14*notch)*(0.97 + 0.03*sin(a*7.0 + h.x*4.0));
    float dd = max(length(v) - rad, abs(pw.y - WL - 0.012) - 0.008);
    if(dd < d){ d = dd; gPadC = base; gPadR = R; }
  }
  return vec2(d*0.8, M_LILY);
}
const int NLO = 5;
const vec3 LOT[NLO] = vec3[NLO](          // x, z, scale
  vec3(-2.05,2.15,0.082), vec3(-2.62,0.92,0.098), vec3(2.34,1.72,0.090), vec3(2.05,-0.50,0.070), vec3(-1.95,-1.35,0.062)
);
float sdLotus(vec3 pw){
  float d = 1e9;
  for(int i=0;i<NLO;i++){
    vec3 u = (pw - vec3(LOT[i].x, WL + 0.030, LOT[i].y))/LOT[i].z;
    float a = atan(u.z, u.x) + float(i);
    float seg = 6.28318/8.0;
    float am = mod(a, seg) - seg*0.5;
    float l = length(u.xz);
    vec3 pe = vec3(cos(am)*l, u.y, sin(am)*l);
    float p1 = sdEllipsoid(pe - vec3(0.72,0.42,0.0), vec3(0.74,0.24,0.34));     // outer petals, cupped
    float a2 = mod(a + seg*0.5, seg) - seg*0.5;
    vec3 pf = vec3(cos(a2)*l, u.y, sin(a2)*l);
    float p2 = sdEllipsoid(pf - vec3(0.40,0.82,0.0), vec3(0.34,0.60,0.27));     // inner petals, upright
    float core = length(u - vec3(0.0,0.55,0.0)) - 0.20;
    d = min(d, min(min(p1, p2), core)*LOT[i].z);
  }
  return d;
}

// -------- the whole map
vec2 setMap(vec3 p, bool shadow){
  if(gWaterShade > 0.5) return shadow ? vec2(-1.0, M_WATER) : vec2(1000.0, M_WATER);  // no sun layer, no AO on water
  float y = p.y - WL;
  // the plane is over-relaxed on primary + mirror rays: they never come steeper than 20 degrees, and the
  // plain distance needs hundreds of steps at a grazing angle
  vec2 res = vec2(shadow ? y : y*2.4, M_WATER);
  float h = landH(p.xz);
  float dg = (y - h)*0.55;
  if(dg < res.x) res = vec2(dg, M_BANK);
  // reeds near the camera, reeds along the shore
  float near = max(abs(p.x) - 6.8, p.z - 2.0);
  near = max(near, -6.9 - p.z);
  near = max(near, y - 4.0);
  if(near < res.x){
    vec2 r1 = reedGrid(p, 0.125, 2.05, 0.0092, 3.0, 0.0, 1.0);
    res = opU(res, vec2(max(r1.x, near - 0.01), r1.y));
    vec2 r2 = reedGrid(p + vec3(0.07,0.0,0.05), 0.16, 2.75, 0.0105, 5.0, 0.0, 1.0);
    res = opU(res, vec2(max(r2.x, near - 0.01), r2.y));
    // low arching sedge: the blades that cross the corners of the frame, sharp, in the plane of focus
    vec2 r5 = reedGrid(p + vec3(0.03,0.0,0.11), 0.070, 1.16, 0.0100, 13.0, 0.0, 3.4);
    res = opU(res, vec2(max(r5.x, near - 0.01)*0.6, r5.y));
    float st = 1e9, um = 1e9;
    for(int i=0;i<NPY;i++){
      vec3 q = p - vec3(PY[i].x, WL, PY[i].y);
      float v = clamp(q.y/PY[i].z, 0.0, 1.0);
      vec2 bend = vec2(0.20*sign(q.x + PY[i].x*0.9 + 0.01)*-1.0, 0.06)*v*v*PY[i].w*1.8;
      q.xz -= bend;
      st = min(st, max(length(q.xz) - mix(0.0245, 0.0105, v), q.y - PY[i].z));
      um = min(um, sdUmbel(q - vec3(0.0,PY[i].z,0.0), PY[i].w, float(i)*1.9));
    }
    res = opU(res, vec2(st*0.85, M_PAPY));
    res = opU(res, vec2(um*0.7, M_UMBEL));
  }
  float shoreBound = max(abs(p.z - shoreZ(p.x)) - 6.0, y - 2.4);
  if(shoreBound < res.x){
    vec2 r3 = reedGrid(p, 0.24, 1.5, 0.017, 7.0, 1.0, 1.0);
    res = opU(res, vec2(max(r3.x, shoreBound - 0.01), r3.y));
    vec2 r4 = reedGrid(p + vec3(0.2,0.0,0.15), 0.19, 1.05, 0.014, 9.0, 1.0, 1.0);
    res = opU(res, vec2(max(r4.x, shoreBound - 0.01), r4.y));
  }
  // bushes along the bank, behind the reeds
  float dland = shoreZ(p.x) - p.z;
  if(dland > 1.0 && y < 9.0){
    float bm = smoothstep(0.42, 0.62, vnoise2(p.xz*0.17 + 5.0));
    float b = y - h - bm*(1.7 + 1.0*vnoise2(p.xz*0.9)) - 0.5*(vnoise3(p*1.1) - 0.5)*bm;
    if(b*0.45 < res.x) res = vec2(b*0.45, M_BUSH);
  }
  // acacias
  for(int i=0;i<NTR;i++){
    vec3 q = p - vec3(TRE[i].x, WL + landH(TRE[i].xy), TRE[i].y);
    vec2 t = sdAcacia(q, TRE[i].z, TRE[i].w, float(i)*1.3);
    res = opU(res, t);
  }
  if(!shadow){
    res = opU(res, sdPads(p));
    res = opU(res, vec2(sdLotus(p), M_LOTUS));
  }
  return res;
}

// -------- paint
void solidSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){
  float y = p.y - WL;
  float sd = max(dot(rd, SUNP), 0.0);        // the painted sun comes through anything thin
  float glow = 1.0 + 1.1*pow(sd, 5.0);
  if(m == M_BANK){
    float n1 = fbm2(p.xz*0.35);
    float n2 = vnoise2(p.xz*2.1);
    vec3 a = mix(vec3(0.17,0.085,0.035), vec3(0.36,0.19,0.06), smoothstep(0.3,0.7,n1));
    a = mix(a, vec3(0.13,0.10,0.04), 0.6*smoothstep(0.52,0.72,fbm2(p.xz*0.11 + 7.0)));
    a *= 0.8 + 0.3*n2;
    a = mix(a, vec3(0.12,0.09,0.07), smoothstep(0.55,0.05,y));    // wet mud at the edge
    s.alb = a; s.wrap = 0.25;
  } else if(m == M_BANK + 0.5){
    s.alb = vec3(0.075,0.052,0.040); s.wrap = 0.1;
  } else if(m == M_BUSH){
    float k = vnoise3(p*1.7);
    s.alb = mix(vec3(0.05,0.05,0.025), vec3(0.19,0.13,0.04), k)*(0.7 + 0.5*vnoise3(p*4.3));
    s.wrap = 0.35;
  } else if(m == M_TREE){
    float k = vnoise3(p*0.9);
    s.alb = mix(vec3(0.04,0.05,0.022), vec3(0.13,0.105,0.04), k)*(0.65 + 0.6*vnoise3(p*3.1));
    s.wrap = 0.45; s.trans = 0.6; s.transCol = vec3(0.7,0.6,0.2);
  } else if(m == M_REED){
    float v = clamp(y/2.6, 0.0, 1.0);
    float k = hash12(floor(p.xz*13.0));
    vec3 a = mix(vec3(0.055,0.055,0.022), vec3(0.15,0.125,0.035), smoothstep(0.0,0.55,v));
    a = mix(a, vec3(0.42,0.23,0.065), smoothstep(0.42,1.0,v)*(0.5 + 0.5*k));
    a *= 0.75 + 0.5*k;
    a *= smoothstep(-0.05,0.25,y)*0.45 + 0.55;
    s.alb = a*glow; s.wrap = 0.45; s.trans = 0.8; s.transCol = a; s.spec = 0.2; s.shin = 24.0;
  } else if(m == M_CAT){
    s.alb = mix(vec3(0.20,0.095,0.04), vec3(0.34,0.17,0.07), vnoise3(p*41.0)); s.wrap = 0.3;
  } else if(m == M_PAPY){
    float v = clamp(y/3.0, 0.0, 1.0);
    s.alb = mix(vec3(0.060,0.070,0.026), vec3(0.21,0.17,0.05), v)*glow; s.wrap = 0.4; s.spec = 0.3; s.shin = 30.0;
    s.alb *= smoothstep(-0.05,0.20,y)*0.45 + 0.55;
  } else if(m == M_UMBEL){
    float k = hash12(floor(p.xz*37.0 + p.y*29.0));
    s.alb = mix(vec3(0.20,0.17,0.05), vec3(0.58,0.32,0.09), k*0.8 + 0.2*vnoise3(p*5.0))*glow;
    s.wrap = 0.6; s.trans = 0.9; s.transCol = vec3(1.0,0.72,0.28);
  } else if(m == M_LILY){
    float k = vnoise2(p.xz*7.0);
    float vein = 0.5 + 0.5*sin(atan(p.z - gPadC.y, p.x - gPadC.x)*19.0);
    s.alb = mix(vec3(0.030,0.060,0.030), vec3(0.11,0.125,0.04), k)*(0.82 + 0.30*vein);
    s.alb = mix(s.alb, vec3(0.34,0.11,0.09), smoothstep(0.82, 1.0, length(p.xz - gPadC)/gPadR));    // the red rolled rim
    s.spec = 0.8; s.shin = 40.0; s.sheen = 0.5; s.wrap = 0.3;
  } else if(m == M_LOTUS){
    float k = clamp((y - 0.03)/0.12, 0.0, 1.0);
    s.alb = mix(vec3(0.96,0.70,0.68), vec3(0.98,0.30,0.42), k);
    s.wrap = 0.6; s.trans = 0.7; s.transCol = vec3(1.0,0.5,0.5); s.sheen = 0.4;
  }
}

// a mirror ray off the water: simple lighting, the blur comes from the lens and the bake's samples
vec3 mirrorTrace(vec3 ro, vec3 rd){
  float t = 0.03; float m = -1.0;
  for(int i=0;i<110;i++){
    vec3 p = ro + rd*t;
    vec2 h = setMap(p, true);
    float hc = charShadow(p) - SP_REFL_GROW;
//body-root{
#ifdef BG_NO_CHAR
    hc = 1e9;
#endif
//body-root}
    if(hc < h.x) h = vec2(hc, M_CHAR);
    m = h.y;
    if(abs(h.x) < 0.0012*t) break;
    t += max(h.x, 0.004);             // a ray that starts on the bank's skin must not walk backward
    if(t > 230.0 || p.y > WL + 40.0){ m = -1.0; break; }
  }
  vec3 col;
  if(m < 0.0){
    col = setSky(rd);
  } else {
    vec3 p = ro + rd*t;
    vec2 e = vec2(1,-1)*0.002*max(t, 1.0);
    #define MAPC(q) min(setMap(q,true).x, charShadow(q) - SP_REFL_GROW)
    vec3 n = normalize(e.xyy*MAPC(p+e.xyy) + e.yyx*MAPC(p+e.yyx) + e.yxy*MAPC(p+e.yxy) + e.xxx*MAPC(p+e.xxx));
    Surf s = surfDefault(n);
    if(m == M_CHAR){ s.alb = SP_REFL_ALB; s.wrap = 0.3; }
    else solidSurface(p, rd, t, m, n, s);
    float dif = clamp((dot(n, uSunDir) + s.wrap)/(1.0 + s.wrap), 0.0, 1.0);
    col = s.alb*(setAmbient(p, n) + uSunCol*dif*0.75);
    vec3 fa = setFogAmb(rd);
    float tr = exp(-0.0060*SP_FOG_GAIN*min(t, 180.0));
    col = col*tr + fa*0.30*(1.0 - tr)*3.0;
  }
  return col;
}

void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){
  if(m == M_WATER){
    // a still slope field: long low swells, foreshortened into streaks by the view
    vec2 w = p.xz;
    float E = 0.03;
    float n0 = fbm2_3(w*vec2(0.9,2.6));
    float nx = fbm2_3((w + vec2(E,0.0))*vec2(0.9,2.6));
    float nz = fbm2_3((w + vec2(0.0,E))*vec2(0.9,2.6));
    vec2 g = vec2(nx - n0, nz - n0)/E*0.016;
    // sample jitter: the fine chop the bake averages into soft vertical streaks
    vec2 jit = hash22(gl_FragCoord.xy*0.731 + uJitter*57.0) - 0.5;
    vec3 N = normalize(vec3(-g.x + jit.x*0.0035, 1.0, -g.y + jit.y*0.016));
    vec3 R = reflect(rd, N);
    R.y = abs(R.y) + 0.002;
    R = normalize(R);
    vec3 refl = mirrorTrace(p + vec3(0.0,0.015,0.0), R);
    float nv = clamp(dot(-rd, N), 0.0, 1.0);
    float F = 0.02 + 0.98*pow(1.0 - nv, 5.0);
    F = clamp(0.56 + F, 0.0, 0.97);               // a storybook mirror, more than physical
    float depthTint = smoothstep(0.0, 14.0, t);
    vec3 body = SP_WATER_C*(1.0 + 0.6*depthTint);
    s.alb = mix(body, refl, F);
    s.spec = 0.0; s.sheen = 0.0; s.wrap = 0.0; s.trans = 0.0;
    gWaterShade = 1.0;
  } else {
    solidSurface(p, rd, t, m, n, s);
  }
}
#else
vec2 setMap(vec3 p, bool shadow){ return vec2(1.0, 0.0); }
void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){}
#endif

// =========================================================================== live water
#ifdef PASS_LIVE
vec2  gLiveUv;
vec3  gLiveAdd;
float gLiveMul;
float gLiveCalls = 0.0;
#ifdef SP_CHAR_WATER
// the animal may draw what moves in the water: its legs' reflection, and rings where they go in.  See the end of animals/hippo.glsl
void  charInit();
float charWaterRefl(vec3 X, vec3 m);
vec2  charWaterDist(vec3 X);
float gLiveRefl = 0.0, gLiveX = 0.0;
#endif
// Called for the ambient layer first (it knows the depth), then for the sun layer: both shift the same way.
vec4 riverLive(sampler2D tex, sampler2D sunL, vec2 uv){
  if(gLiveCalls < 0.5){
    gLiveUv = uv; gLiveAdd = vec3(0.0); gLiveMul = 1.0;
    float dep = texture(tex, uv).a;
    vec2 ndc = 2.0*uv - 1.0;
    vec3 rd = normalize(ndc.x*uRes.x/uRes.y*uCamU + ndc.y*uCamV + uFl*uCamW);
    if(rd.y < -0.0005){
      float tw = (WL - uCamPos.y)/rd.y;
      float r = dep/tw;
      float isW = smoothstep(0.86, 0.97, r)*smoothstep(1.14, 1.03, r);
      vec2 X = (uCamPos + rd*tw).xz;
      vec3 w = riverWaves(X, uTime);
      float near = 1.0/(1.0 + tw*0.05);
      gLiveUv += isW*near*vec2(w.x*0.0022, w.y*0.0100);
#ifdef SP_CHAR_WATER
      charInit();
      gLiveX = X.x;
      vec2 wd = charWaterDist(vec3(X.x, WL, X.y));
      float wring = smoothstep(0.35, 1.0, sin(wd.x*42.0 - uTime*3.1 + 0.8*sin(X.x*2.3)))*exp(-max(wd.x, 0.0)*5.2)*wd.y*smoothstep(-0.01, 0.04, wd.x);
      gLiveUv.y += isW*near*0.0065*wring;
      // the mirrored ray of the same wobbled point the bake is read at, so the leg shimmers with everything else in the water
      vec2 nw = 2.0*gLiveUv - 1.0;
      vec3 rw = normalize(nw.x*uRes.x/uRes.y*uCamU + nw.y*uCamV + uFl*uCamW);
      if(rw.y < -0.0005){
        vec3 Xw = uCamPos + rw*(WL - uCamPos.y)/rw.y;
        vec3 sn = texture(sunL, gLiveUv).rgb;            // nothing floating takes the reflection: the water's sun layer is black
        gLiveRefl = isW*charWaterRefl(Xw, vec3(rw.x, -rw.y, rw.z))*smoothstep(0.012, 0.0015, sn.r + sn.g + sn.b);
      }
#endif
      // crests catch the sky, troughs darken the mirror a touch
      float crest = w.y*0.5 + w.z*0.8;
      float sun = pow(clamp(dot(normalize(vec3(rd.x, -rd.y, rd.z)), SUNP), 0.0, 1.0), 90.0);
      float gl = smoothstep(0.86, 0.98, vnoise2(vec2(X.x*6.0, X.y*17.0) + vec2(uTime*0.9, uTime*1.7)));
      gLiveAdd = isW*(vec3(1.0,0.72,0.50)*(0.030*crest*near + 0.16*w.z*near) + vec3(1.0,0.78,0.42)*gl*(0.05 + 1.2*sun));
      // the water darkens against the body
      float dv = length((X - SP_RING_C)/SP_RING_S);
      gLiveMul = 1.0 - isW*0.42*exp(-max(dv - 0.96, 0.0)*9.0);
#ifdef SP_CHAR_WATER
      gLiveMul *= 1.0 - isW*0.40*wd.y*exp(-max(wd.x, 0.0)*11.0);
      gLiveAdd += isW*wd.y*vec3(1.0,0.74,0.52)*(0.20*wring*near + 0.13*exp(-sqr(wd.x/0.014)));
#endif
    }
  }
  vec4 c = texture(tex, gLiveUv);
  c.rgb *= gLiveMul;
  if(gLiveCalls < 0.5) c.rgb = max(c.rgb + gLiveAdd, vec3(0.0));
#ifdef SP_CHAR_WATER
  c.rgb = (gLiveCalls < 0.5) ? mix(c.rgb, SP_REFL_LEG*(0.80 + 0.32*smoothstep(-1.4, 1.4, gLiveX)), 0.93*gLiveRefl) : c.rgb*(1.0 - 0.93*gLiveRefl);
#endif
  gLiveCalls += 1.0;
  return c;
}
#define texture(s, uv) riverLive(s, uBgS, uv)
#endif

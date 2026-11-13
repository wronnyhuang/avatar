// ---------------------------------------------------------------------------
//  engine/sets/jungle.glsl — art-directed jungle for a fixed head-and-shoulders portrait.
//  Same SET API as forest.glsl.  Every look lives in `set_params`; layout is authored in screen
//  coordinates (SCR) for camera pos (0,0.02,6.0) -> target (0,-0.06,0), fov 22.
//
//  Layers, near to far:
//    near props   z -0.4..-1.6   mossy log, ferns, toadstools, bromeliad, sprigs: nearDress()
//    hero leaves  z -0.5..-2.0   frame edges, table LEAF_TABLE
//    middle       z -2.4..-10    big-leaf plants (same table), tree ferns, lianas, flowers
//    far          z -12..-40     domain-repeated leaf walls, buttressed trunks
//    sky                         painted silhouettes, the bright mist behind the head (bake only)
//
//  Cost notes (llvmpipe): an `if` never skips work, a loop does, and a loop always runs at least one
//  iteration.  So every group below is one loop whose count drops to zero when the point is far from
//  that group's depth slab (slab()), with the slab distance returned as a bound.  Shadow rays that the
//  canopy blocks stop on their first step (gDap in setMap).  setSky only paints its far leaves in the
//  bake: PASS_CHAR calls it for the eyes on every sample.
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-3.6)
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.10,0.17,0.08)              // green light bounced up from the understory
#endif
#ifndef SP_LEAF_DEEP
#define SP_LEAF_DEEP vec3(0.012,0.070,0.055)        // blue-green shadow leaf
#endif
#ifndef SP_LEAF_MID
#define SP_LEAF_MID vec3(0.035,0.190,0.075)
#endif
#ifndef SP_LEAF_LIME
#define SP_LEAF_LIME vec3(0.30,0.56,0.06)           // young growth, sunlit blade
#endif
#ifndef SP_GLOW_LEAF
#define SP_GLOW_LEAF vec3(2.3,2.9,0.75)             // light through a blade
#endif
#ifndef SP_BARK_A
#define SP_BARK_A vec3(0.060,0.048,0.038)
#endif
#ifndef SP_BARK_B
#define SP_BARK_B vec3(0.34,0.27,0.19)
#endif
#ifndef SP_MOSS
#define SP_MOSS vec3(0.10,0.27,0.05)
#endif
#ifndef SP_FOG_LO
#define SP_FOG_LO vec3(0.05,0.17,0.15)              // mist at the frame edge
#endif
#ifndef SP_FOG_HI
#define SP_FOG_HI vec3(0.56,0.74,0.32)              // mist behind the head
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#ifndef SP_GLOW_C
#define SP_GLOW_C vec2(-0.10,0.24)                  // screen uv of the bright opening
#endif
#ifndef SP_GLOW_R
#define SP_GLOW_R 1.30
#endif
#ifndef SP_AMB_LOW
#define SP_AMB_LOW vec3(0.10,0.23,0.24)             // ambient on down-facing set surfaces
#endif
#ifndef SP_AMB_HIGH
#define SP_AMB_HIGH vec3(0.30,0.46,0.40)
#endif
#ifndef SP_AMB_CHAR
#define SP_AMB_CHAR vec3(0.54,0.58,0.48)            // ambient on the character
#endif
#ifndef SP_SUN_OPEN
#define SP_SUN_OPEN 0.50                            // canopy threshold: higher = fewer sun patches
#endif
#ifndef SP_CANOPY_Y
#define SP_CANOPY_Y 12.0                            // height of the plane the gaps live on
#endif
#ifndef SP_DAPPLE_FREQ
#define SP_DAPPLE_FREQ 1.0
#endif

const float G0 = SP_GROUND_Y;
const float CANY = SP_CANOPY_Y;
#define TANH 0.19438
#define CAMZ 6.0
// world point for screen uv (u,v) at depth plane z
#define SCR(u,v,z) vec3((u)*(CAMZ-(z))*TANH, (v)*(CAMZ-(z))*TANH + 0.02 - 0.01333*(CAMZ-(z)), (z))

#define M_GROUND 1.0
#define M_BARK   2.0
#define M_LEAF   3.0
#define M_FERN   4.0
#define M_VINE   5.0
#define M_PETAL  6.0
#define M_LOG    20.0
#define M_MOSS   21.0
#define M_NFERN  22.0
#define M_TWIG   23.0
#define M_BLEAF  24.0
#define M_CAP    25.0
#define M_STIPE  26.0
#define M_BROM   27.0
#define M_CANOPY 99.0   // unused: nothing in this set skips direct shadowing

// ------------------------------------------------------------------ leaf table
// One row: base (u,v,z), tip (u,v,z), the way the top face looks (fx,fy,1), half width, droop, fold,
// shape: skew (0 centred .. 1 widest near the base), oblong (-1 pointed lens, 0 elliptic, 1 strap), heart (cordate lobes),
// cut (0 none, .05 banana tears, .24 philodendron lobes), seed.
#define LF_P(bu,bv,bz,tu,tv,tz,fx,fy,W,dr,fo,sk,ob,he,cu,se) vec4(SCR(bu,bv,bz), length(SCR(tu,tv,tz)-SCR(bu,bv,bz)))
#define LF_A(bu,bv,bz,tu,tv,tz,fx,fy,W,dr,fo,sk,ob,he,cu,se) vec4(normalize(SCR(tu,tv,tz)-SCR(bu,bv,bz)), W)
#define LF_N(bu,bv,bz,tu,tv,tz,fx,fy,W,dr,fo,sk,ob,he,cu,se) vec4(normalize(vec3(fx,fy,1.0) - normalize(SCR(tu,tv,tz)-SCR(bu,bv,bz))*dot(vec3(fx,fy,1.0), normalize(SCR(tu,tv,tz)-SCR(bu,bv,bz)))), dr)
#define LF_K(bu,bv,bz,tu,tv,tz,fx,fy,W,dr,fo,sk,ob,he,cu,se) vec4(sk,ob,he,cu)
#define LF_S(bu,bv,bz,tu,tv,tz,fx,fy,W,dr,fo,sk,ob,he,cu,se) vec2(fo,se)

#define NHERO 7
#define NLEAF 25
#define LEAF_TABLE(X) \
  /* hero, frame left */ \
  X(-2.36, 1.30,-0.9, -1.18, 0.52,-0.6,   0.25,-0.60, 0.38, 0.16,0.20, 0.15,1.0,0.00,0.05, 0.13), \
  X(-2.36, 0.30,-1.2, -1.44,-0.20,-0.7,  -0.10, 0.20, 0.50, 0.10,0.16, 0.90,0.0,0.30,0.00, 0.37), \
  X(-2.00,-1.30,-1.95,-1.42,-0.50,-1.55,  0.20, 0.10, 0.52, 0.10,0.20, 0.70,0.3,0.20,0.24, 0.71), \
  /* hero, frame right */ \
  X( 2.36, 1.26,-0.8,  1.26, 0.60,-0.5,  -0.25,-0.50, 0.40, 0.18,0.20, 0.15,1.0,0.00,0.05, 0.23), \
  X( 2.12, 0.72,-1.35, 1.72,-0.12,-1.0,  -0.30, 0.00, 0.30, 0.10,0.22, 0.40,0.5,0.00,0.00, 0.53), \
  X( 2.32,-0.50,-1.1,  1.48,-0.12,-0.7,   0.05, 0.20, 0.50, 0.10,0.16, 0.90,0.0,0.30,0.00, 0.67), \
  X( 1.98,-1.30,-2.0,  1.46,-0.54,-1.6,  -0.20, 0.10, 0.52, 0.10,0.20, 0.70,0.3,0.20,0.24, 0.83), \
  /* middle: top sprays */ \
  X(-0.74, 1.24,-3.0, -1.06, 0.74,-2.8,   0.30,-0.40, 0.24, 0.14,0.18, 0.60,0.2,0.10,0.00, 0.17), \
  X(-0.60, 1.24,-3.2, -0.56, 0.90,-3.0,   0.00,-0.40, 0.20, 0.14,0.18, 0.60,0.2,0.10,0.00, 0.29), \
  X( 0.62, 1.24,-3.3,  0.92, 0.72,-3.1,  -0.30,-0.40, 0.24, 0.14,0.18, 0.60,0.2,0.10,0.00, 0.41), \
  /* middle: left plant */ \
  X(-1.26,-1.28,-3.6, -1.70,-0.50,-3.3,  -0.30, 0.25, 0.34, 0.14,0.18, 0.90,0.0,0.28,0.00, 0.47), \
  X(-1.20,-1.28,-3.9, -1.18,-0.46,-3.6,   0.10, 0.25, 0.36, 0.14,0.18, 0.90,0.0,0.28,0.00, 0.59), \
  X(-1.12,-1.28,-3.7, -0.86,-0.70,-3.5,   0.30, 0.25, 0.28, 0.14,0.18, 0.90,0.0,0.28,0.00, 0.61), \
  X(-1.30,-1.20,-4.3, -1.82,-0.92,-4.1,  -0.40, 0.30, 0.30, 0.14,0.18, 0.60,0.3,0.15,0.24, 0.73), \
  /* middle: right plant */ \
  X( 1.34,-1.28,-4.6,  1.70,-0.44,-4.3,   0.30, 0.25, 0.34, 0.14,0.18, 0.15,1.0,0.00,0.05, 0.79), \
  X( 1.28,-1.28,-4.9,  1.26,-0.36,-4.6,  -0.10, 0.25, 0.34, 0.14,0.18, 0.15,1.0,0.00,0.05, 0.89), \
  X( 1.20,-1.28,-4.7,  0.92,-0.68,-4.5,  -0.30, 0.25, 0.30, 0.14,0.18, 0.90,0.0,0.28,0.00, 0.97), \
  X( 1.38,-1.20,-5.3,  1.84,-0.88,-5.1,   0.40, 0.30, 0.32, 0.14,0.18, 0.60,0.3,0.15,0.24, 0.11), \
  /* middle: deep flanks */ \
  X(-1.62, 0.92,-6.0, -1.02, 0.30,-5.7,   0.30,-0.10, 0.30, 0.12,0.20, 0.15,1.0,0.00,0.05, 0.19), \
  X( 1.60, 0.86,-8.0,  1.08, 0.34,-7.7,  -0.30,-0.10, 0.32, 0.12,0.20, 0.15,1.0,0.00,0.05, 0.27), \
  X(-1.74, 0.20,-7.0, -1.24,-0.12,-6.8,   0.20, 0.10, 0.36, 0.12,0.18, 0.90,0.0,0.28,0.00, 0.33), \
  X( 1.74, 0.20,-8.6,  1.26,-0.08,-8.4,  -0.20, 0.10, 0.38, 0.12,0.18, 0.90,0.0,0.28,0.00, 0.39), \
  X(-1.02,-1.20,-9.0, -0.98,-0.60,-8.8,   0.10, 0.20, 0.40, 0.12,0.18, 0.60,0.3,0.15,0.24, 0.43), \
  X( 1.02,-1.20,-9.4,  0.98,-0.62,-9.2,  -0.10, 0.20, 0.42, 0.12,0.18, 0.60,0.3,0.15,0.24, 0.49), \
  X(-0.42, 1.26,-9.2, -0.70, 0.84,-9.0,   0.20,-0.30, 0.40, 0.12,0.18, 0.60,0.2,0.10,0.00, 0.57)

const vec4 LP[NLEAF] = vec4[NLEAF](LEAF_TABLE(LF_P));
const vec4 LA[NLEAF] = vec4[NLEAF](LEAF_TABLE(LF_A));
const vec4 LN[NLEAF] = vec4[NLEAF](LEAF_TABLE(LF_N));
const vec4 LK[NLEAF] = vec4[NLEAF](LEAF_TABLE(LF_K));
const vec2 LS[NLEAF] = vec2[NLEAF](LEAF_TABLE(LF_S));

// ------------------------------------------------------------------ other hand-placed things
#define NTRUNK 7
const vec4 TRK[NTRUNK] = vec4[NTRUNK](      // x, z, radius, seed
  vec4(-3.70,-12.6, 0.74, 0.4),
  vec4( 5.60,-13.4, 0.84, 1.3),
  vec4(-9.5,-25.0, 1.5, 2.2),
  vec4(10.5,-27.0, 1.6, 3.1),
  vec4(-6.6,-50.0, 1.1, 4.0),
  vec4( 7.8,-54.0, 1.2, 5.2),
  vec4(-15.0,-33.0, 1.8, 6.1)
);
#define NFERN 2
const vec4 FERN[NFERN] = vec4[NFERN](       // crown xyz, frond length
  vec4(SCR( 1.04, 0.36, -8.0), 1.55),
  vec4(SCR(-0.86, 0.80,-12.0), 1.75)
);
#define NVINE 7
const vec4 VINE[NVINE] = vec4[NVINE](       // x, z, sway, lowest y
  vec4(-2.55,-4.0, 0.22,-2.8),
  vec4(-1.80,-6.5, 0.16, 0.2),
  vec4( 1.95,-3.4, 0.20,-3.0),
  vec4( 3.00,-6.0, 0.24,-1.6),
  vec4(-3.80,-8.5, 0.20,-3.2),
  vec4( 1.05,-9.0, 0.14, 1.2),
  vec4(-0.95,-5.5, 0.12, 0.9)
);
#define NHIB 3
const vec4 HIB[NHIB] = vec4[NHIB](          // centre, radius
  vec4(SCR(-1.12,-0.50,-3.0), 0.30),
  vec4(SCR( 1.31,-0.40,-3.7), 0.28),
  vec4(SCR( 1.38, 0.30,-3.2), 0.20)
);
const vec3 HIBN[NHIB] = vec3[NHIB](         // facing
  normalize(vec3( 0.25,0.25,1.0)),
  normalize(vec3(-0.30,0.15,1.0)),
  normalize(vec3( 0.35,0.05,1.0))
);
#define NHEL 2
const vec4 HEL[NHEL] = vec4[NHEL](          // top, length
  vec4(SCR(-1.04, 0.70,-3.4), 1.00),
  vec4(SCR( 0.98, 1.08,-4.8), 0.95)
);

vec2  gWind;
vec4  gL;            // nearest leaf: along 0..1, across -1..1, which face, id
vec3  gLA, gLS;      // its axes
vec4  gF;            // nearest frond / petal paint coordinates
float gGlow;         // light through the blade, added in setAmbient; 0 outside setSurface
float gDap;          // canopy opening along the sun ray at the surface being shaded
float gShN;          // shadow-map calls so far in this fragment
float gBladeE, gLE;  // 0 at the midrib .. 1 at the leaf margin

void setInit(){
  float T = uTime;
  gWind = 0.10*vec2(sin(T*0.80), cos(T*0.65)) + 0.035*vec2(sin(T*2.1+1.0), sin(T*1.7));
  gGlow = 0.0; gDap = 1.0; gShN = 0.0; gBladeE = 0.0; gLE = 0.0;
  gL = vec4(0.0); gLA = vec3(0,1,0); gLS = vec3(1,0,0); gF = vec4(0.0);
}

// Static canopy: 0..1 how open the leaves far above are, along the sun ray through p.
vec2 sunPlane(vec3 p){ return p.xz + uSunDir.xz*((CANY - p.y)/uSunDir.y); }
float canopyOpen(vec3 p){
  vec2 q = sunPlane(p);
  float a = fbm2_3(q*0.42 + 3.1);
  float b = vnoise2(q*1.7 + 7.0);
  return smoothstep(SP_SUN_OPEN - 0.06, SP_SUN_OPEN + 0.05, a*0.78 + b*0.22 + 0.03);
}

// A group living between depths zb..zf: how many of its n items to loop over here.  Outside the slab
// the loop is skipped and the distance to the slab stands in, always > 0 at the switch so it never
// becomes a surface.
int slab(vec3 p, float zb, float zf, int n, inout vec2 res){
  float dz = max(p.z - zf, zb - p.z);
  if(dz > 0.4) res.x = min(res.x, dz);
  return dz > 0.4 ? 0 : n;
}

// A leaf blade in its own frame: u along from the base, v across, w normal.
float sdBlade(float u, float v, float w, float L, float W, float dr, float fo, vec4 K, float se, float th, float pet, float ribR,
              out float ul, out float vn){
  ul = u/L;
  float cl = clamp(ul, -0.25, 1.2);
  float av = abs(v)/W;
  vn = v/W;
  float wr = w + dr*L*cl*cl - fo*abs(v) + 0.10*W*av*av*sin(cl*9.0 + se*40.0 + sign(v)*1.3);
  float x = cl + K.z*min(av*av, 1.6);
  float x2 = mix(x, x*(2.0 - x), K.x);
  float P = clamp(4.0*x2*(1.0 - x2), 0.0, 1.0);
  float sq = sqrt(P);
  float hw = W*(K.y > 0.0 ? mix(sq, sqrt(sq), K.y) : mix(sq, P, -K.y));
  float xt = ul + K.z*min(av*av, 1.6);
  float d2 = max(abs(v) - hw, max(-xt, xt - 1.0)*L);
  gBladeE = abs(v)/max(hw, 0.02*W);
  // tears (banana) and lobes (philodendron) follow the side veins in from the edge
  float NC = mix(7.0, 16.0, max(K.y, 0.0));
  float c = cl*NC - 1.1*av;
  float fc = abs(fract(c) - 0.5)*L/NC;
  float on = K.w > 0.15 ? 1.0 : step(0.55, fract(sin(floor(c) + sign(v)*17.0 + se*91.0)*43758.5));
  float sw = K.w*on*smoothstep(K.w > 0.15 ? 0.16 : 0.45, K.w > 0.15 ? 0.72 : 1.0, av)*L/NC*2.0;
  d2 = max(d2, sw - fc);
  float e = d2*0.7, f = abs(wr) - th;
  float d = length(max(vec2(e, f), 0.0)) + min(max(e, f), 0.0);
  float du = u - clamp(u, -pet, L);
  float rr = mix(ribR, ribR*0.2, clamp(ul, 0.0, 1.0));
  return min(d, length(vec3(du, v, wr + rr*0.4)) - rr)*0.8;
}

void tableLeaves(vec3 p, inout vec2 res){
  // hero leaves live in z -0.2..-2.3, the middle in z -2.6..-10.2
  int nh = slab(p, -2.3, -0.2, NHERO, res);
  int nm = slab(p, -10.2, -2.6, NLEAF - NHERO, res);
  int i0 = nh > 0 ? 0 : NHERO;
  int i1 = nm > 0 ? NLEAF : (nh > 0 ? NHERO : 0);
  for(int i=i0; i<i1; i++){
    vec4 P = LP[i], A = LA[i], N = LN[i];
    vec3 S = cross(N.xyz, A.xyz);
    vec3 q = p - P.xyz;
    float ul, vn;
    float d = sdBlade(dot(q,A.xyz), dot(q,S), dot(q,N.xyz), P.w, A.w, N.w, LS[i].x, LK[i], LS[i].y,
                      i < NHERO ? 0.006 : 0.012, 1.2, i < NHERO ? 0.016 : 0.022, ul, vn);
    if(d < res.x){ res = vec2(d, M_LEAF); gL = vec4(ul, vn, sign(dot(q,N.xyz)), float(i)); gLA = A.xyz; gLS = S; gLE = gBladeE; }
  }
}

// ------------------------------------------------------------------ tree fern
void treeFerns(vec3 p, inout vec2 res){
  int n = slab(p, -14.2, -5.8, NFERN, res);
  const float K = 11.0;
  for(int i=0; i<n; i++){
    vec4 F = FERN[i];
    float R = F.w;
    vec3 c = p - F.xyz;
    // trunk, leaning a little
    vec3 tb = vec3(F.x + 0.25, G0, F.z - 0.3);
    float dt = sdTaper(p, F.xyz + vec3(0.0,-0.05,-0.15), tb, 0.10, 0.17);
    if(dt < res.x){ res = vec2(dt, M_BARK); gF = vec4(atan(p.z - F.z, p.x - F.x)*0.6, p.y, 9.0 + float(i), 0.0); }
    // crown tipped toward the camera so the fronds radiate on screen
    vec3 q = c;
    q.yz = rot(-0.86)*q.yz;
    float x = atan(q.z, q.x)*K/6.28318;
    float s0 = floor(x);
    for(int j=0; j<2; j++){
      float s = s0 + float(j);
      float sm = mod(s, K);
      float h = hash11(sm*0.37 + float(i)*3.1);
      float a = (sm + 0.32*(h - 0.5))*6.28318/K;
      vec2 r = rot(-a)*q.xz;
      float Ri = R*(0.80 + 0.28*h);
      float along = max(r.x, 0.0)/Ri;
      float yr = Ri*(0.30*along*(1.0 + 0.2*h) - 0.62*along*along);
      float yh = (q.y - yr)*0.62;
      float t = clamp(along, 0.0, 1.0);
      float B = 0.31*sqrt(1.0 - t)*min(t*5.0 + 0.1, 1.0)*(0.9 + 0.2*h)*R/1.55;
      float side = r.y;
      float sn = abs(side)/max(B, 0.02);
      const float DP = 0.088;
      float m = r.x/DP - sn*1.1;
      float fm = abs(fract(m) - 0.5);
      float comb = (fm - 0.40*(1.0 - 0.75*sn*sn))*DP;
      float e = max(max(abs(side) - B, comb), max(r.x - Ri, 0.12 - r.x));
      e *= 0.6;
      float f = abs(yh) - 0.012;
      float d = length(max(vec2(e, f), 0.0)) + min(max(e, f), 0.0);
      float rach = length(vec2(side, yh)) - mix(0.018, 0.005, t);
      d = min(d, max(rach, r.x - Ri));
      if(d < res.x){ res = vec2(d*0.8, M_FERN); gF = vec4(t, sn, h, float(i)); }
    }
  }
}

// ------------------------------------------------------------------ lianas with small heart leaves
vec2 vineOff(float y, float s, float amp){
  return amp*vec2(sin(y*0.75 + s*5.0) + 0.45*sin(y*1.9 + s*11.0), 0.7*cos(y*0.55 + s*3.0));
}
void lianas(vec3 p, inout vec2 res){
  int n = slab(p, -9.8, -2.8, NVINE, res);
  for(int i=0; i<n; i++){
    vec4 V = VINE[i];
    float s = float(i)*0.618 + 0.1;
    vec2 q = p.xz - V.xy - vineOff(p.y, s, V.z);
    float dv = length(q) - (0.016 + 0.006*sin(p.y*3.0 + s));
    dv = max(dv, V.w - p.y);
    if(dv < res.x) res = vec2(dv*0.9, M_VINE);
    const float CELL = 0.30;
    float n0 = floor(p.y/CELL);
    for(int j=0; j<2; j++){
      float nn = n0 + float(j);
      float yc = (nn + 0.5)*CELL;
      float side = mod(nn, 2.0)*2.0 - 1.0;
      float hh = hash11(nn*0.73 + s*9.0);
      vec3 c = vec3(V.xy + vineOff(yc, s, V.z), yc).xzy;
      vec3 k = p - c;
      vec3 A = normalize(vec3(side*(0.70 + 0.5*hh), -0.75, 0.10));
      vec3 N = normalize(vec3(-A.x*0.3 + 0.22*(hh - 0.5), 0.30, 1.0));
      vec3 S = normalize(cross(N, A));
      N = cross(A, S);
      float ul, vn;
      float L = 0.25 + 0.13*hh;
      float d = sdBlade(dot(k,A), dot(k,S), dot(k,N), L, 0.47*L, 0.20, 0.14, vec4(0.42,0.45,0.30,0.0), hh, 0.006, 0.06, 0.008, ul, vn);
      if(yc < V.w) continue;                          // the vine ends on a whole leaf
      if(d < res.x){ res = vec2(d, M_LEAF); gL = vec4(ul, vn, sign(dot(k,N)), 40.0 + float(i)); gLA = A; gLS = S; gLE = gBladeE; }
    }
  }
}

// ------------------------------------------------------------------ flowers
void flowers(vec3 p, inout vec2 res){
  int n = slab(p, -5.6, -2.6, NHIB, res);
  for(int i=0; i<n; i++){
    vec4 H = HIB[i];
    vec3 q = p - H.xyz, N = HIBN[i];
    vec3 S = normalize(cross(vec3(0,1,0), N)), T = cross(N, S);
    float x = dot(q,S), y = dot(q,T), z = dot(q,N);
    float R = H.w, r = length(vec2(x,y))/R;
    float a = atan(y, x) + float(i)*1.1;
    float fa = fract(a*5.0/6.28318);                     // 0..1 across one petal, pinwheel order
    float lobe = 4.0*fa*(1.0 - fa);
    float rim = 0.80 + 0.20*sqrt(lobe) + 0.03*sin(a*23.0)*r;
    float d2 = (r - rim)*R;
    float zs = R*(0.36*r*r - 0.60*exp(-r*r*8.0) + 0.05*sin(a*17.0)*r*r + 0.14*r*(fa - 0.5));
    float e = d2*0.7, f = abs(z - zs) - 0.008;
    float d = (length(max(vec2(e,f),0.0)) + min(max(e,f),0.0))*0.7;
    if(d < res.x){ res = vec2(d, M_PETAL); gF = vec4(r, lobe, 0.0, float(i)); }
    // stamen column
    vec3 tip = H.xyz + N*R*0.85 + T*R*0.15;
    float ds = sdTaper(p, H.xyz - N*R*0.5, tip, 0.032*R/0.3, 0.018*R/0.3);
    if(ds < res.x){ res = vec2(ds, M_PETAL); gF = vec4(length(p - tip)/R, 0.0, 1.0, float(i)); }
  }
  // heliconia: hanging zigzag of boat-shaped bracts
  int m = n > 0 ? NHEL : 0;
  for(int i=0; i<m; i++){
    vec4 H = HEL[i];
    vec3 q = p - H.xyz;
    q.x += 0.05*sin(q.y*3.0 + float(i));
    float Y = -q.y/H.w;
    float da = length(vec2(q.x, q.z)) - 0.016;
    da = max(da, max(q.y - 1.0, (Y - 1.0)*H.w));
    if(da < res.x) res = vec2(da, M_VINE);
    const float DY = 0.135;
    float n0 = floor(-q.y/DY);
    for(int j=0; j<2; j++){
      float nn = clamp(n0 + float(j), 0.0, floor(H.w/DY));
      float side = mod(nn, 2.0)*2.0 - 1.0;
      vec3 k = q - vec3(0.0, -(nn + 0.5)*DY, 0.0);
      k.xy = rot(side*0.42)*k.xy;
      k.x = k.x*side;
      float taper = 1.0 - 0.72*nn*DY/H.w;
      float len = 0.40*taper;
      k.x -= len*0.85;
      float d = sdEllipsoid(k, vec3(len, 0.100*taper + 0.022, 0.090*taper + 0.024));
      float cut = k.y - 0.25*(0.100*taper);
      d = max(d, cut);
      if(d < res.x){ res = vec2(d*0.9, M_PETAL); gF = vec4(clamp(k.x/len*0.5 + 0.5, 0.0, 1.0), taper, 2.0, float(i)); }
    }
  }
}

// ------------------------------------------------------------------ near-focus dressing
// Crisp props at the frame edges, all behind the subject: a mossy fallen log along the bottom with toadstools
// and a bromeliad on it, ferns rising behind it, a leafy sprig in each top corner.  Each part sits in a
// one-shot loop guarded by a lower bound (clamped to G so a bound is never a surface).
const vec3 LOG_A = vec3(-3.80,-2.40,-0.46), LOG_B = vec3(3.80,-2.50,-0.34);
const float LOG_R = 0.44;
#define NMU 6
const vec4 MU_P[NMU] = vec4[NMU](       // base xyz, stem height
  vec4(-2.26,-1.97,-0.34,0.30), vec4(-2.06,-1.98,-0.30,0.19), vec4(-2.44,-1.97,-0.36,0.13),
  vec4( 2.22,-2.05,-0.26,0.17), vec4( 2.38,-2.05,-0.24,0.11), vec4( 2.10,-2.05,-0.26,0.09));
const float MU_R[NMU] = float[NMU](0.15, 0.10, 0.07, 0.09, 0.065, 0.05);
const vec3 MUC_L = vec3(-2.26,-1.84,-0.34), MUC_R = vec3(2.22,-1.96,-0.26);
const vec4 BROM = vec4(2.90,-1.98,-0.40, 0.56);     // centre, leaf length
const vec3 FERN_L = vec3(-2.42,-2.20,-1.52), FERN_R = vec3(2.86,-2.24,-1.60);
#define NFR 7
#define NTW 3
const vec3 TWR[NTW+1] = vec3[NTW+1](SCR( 1.62, 1.26,-0.94), SCR( 1.20, 1.09,-0.90), SCR( 0.98, 0.98,-0.86), SCR( 0.74, 0.78,-0.80));
const vec3 TWL[NTW+1] = vec3[NTW+1](SCR(-1.70, 1.26,-1.00), SCR(-1.28, 1.08,-0.96), SCR(-1.04, 1.00,-0.92), SCR(-0.82, 0.82,-0.86));
#define NLF 11

void logAndMoss(vec3 p, inout vec2 res){
  vec3 ba = LOG_B - LOG_A, pa = p - LOG_A;
  float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  vec3 c = pa - ba*h;
  float a = atan(c.z, c.y), r = length(c), u = h*6.8;
  float rad = LOG_R*(1.0 + 0.05*sin(u*1.3 + 1.0));
  rad += 0.014*sin(a*17.0 + 1.8*sin(u*1.4)) + 0.045*(vnoise2(vec2(a*5.0, u*2.2)) - 0.5) + 0.012*(vnoise2(vec2(a*24.0, u*3.0)) - 0.5);
  float d = (r - rad)*0.75;
  // wet moss wraps most of the way round, in thick soft cushions
  float up = c.y/max(r, 1e-3);
  float mz = smoothstep(0.02, 0.42, up + 0.95*(fbm3(p*vec3(2.1,3.6,2.1) + 0.5*sin(p.yzx*4.0)) - 0.5));
  float dm = d - mz*(0.018 + 0.095*vnoise3(p*vec3(3.1,5.0,3.1)) + 0.030*vnoise3(p*11.0) + 0.010*vnoise3(p*38.0));
  float dd = min(d, dm)*0.9;
  if(dd < res.x) res = vec2(dd, dm < d - 0.002 ? M_MOSS : M_LOG);
}
void toadstools(vec3 p, inout vec2 res){
  for(int i=0; i<NMU; i++){
    float H = MU_P[i].w, Rc = MU_R[i], fi = float(i);
    vec3 q = p - MU_P[i].xyz;
    q.xz += vec2(sin(fi*2.4), cos(fi*1.7))*0.28*q.y;
    float st = sdTaper(q, vec3(0.0,-0.04,0.0), vec3(0.0,H,0.0), Rc*0.30, Rc*0.18)*0.85;
    vec3 cq = q - vec3(0.0, H, 0.0);
    cq.xy = rot(0.20*sin(fi*3.1))*cq.xy;
    float cap = sdEllipsoid(cq, vec3(Rc, Rc*0.60, Rc));
    cap = smax(cap, -cq.y - Rc*0.10 + 0.10*length(cq.xz), Rc*0.25)*0.85;
    if(st < res.x){ res = vec2(st, M_STIPE); gF = vec4(q/Rc, fi); }
    if(cap < res.x){ res = vec2(cap, M_CAP); gF = vec4(cq/Rc, fi); }
  }
}
// one arching frond: an arc of radius Ra, length L, half width W, its blade twisted to face the camera
float sdFrond(vec3 p, vec3 o, float az, float L, float Ra, float W){
  vec3 q = p - o;
  q.xz = rot(az)*q.xz;
  vec2 v = vec2(q.x - Ra, q.y);
  float th = atan(q.y, Ra - q.x);
  float s = clamp(th, 0.0, L/Ra)*Ra;
  float u = length(v) - Ra;
  vec2 uz = rot(1.12)*vec2(u, q.z);
  u = uz.x; float z = uz.y;
  float tt = s/L;
  u -= 0.55*z*z;
  float rach = length(vec2(u, z)) - mix(0.0065, 0.0015, tt);
  float ds = th*Ra - s;
  rach = length(vec2(max(rach, 0.0), ds));
  const float SP = 0.047;
  float x = abs(z);
  float n = (s - x*0.55)/SP;
  float id = floor(n), f = fract(n) - 0.5;
  float tc = clamp((id + 0.5)*SP/L, 0.0, 1.0);
  float w = W*smoothstep(0.0, 0.16, tc)*pow(1.0 - tc, 0.75);
  float rel = clamp(x/max(w, 1e-3), 0.0, 1.0);
  float pw = 0.40*SP*sqrt(max(1.0 - rel*rel, 0.0))*(1.0 + 0.16*sin(x*150.0));
  float d2 = max(abs(f)*SP - pw, x - w);
  float d = max(d2*0.75, abs(u) - 0.0030);
  d = max(d, max(-s, s - L));
  return min(d*0.6, rach);
}
void groundFern(vec3 p, vec3 c, float sd, float dirx, inout vec2 res){
  for(int i=0; i<NFR; i++){
    vec3 h = hash31(float(i)*3.17 + sd);
    float az = ((i & 3) == 3 ? 3.14159 - dirx : dirx) + (h.x - 0.5)*1.1;     // most arch into the frame
    vec3 o = c + vec3(h.y - 0.5, 0.0, h.z - 0.5)*0.40 + vec3(0.0, 0.0, -0.05*float(i));
    float d = sdFrond(p, o, az, mix(1.05, 1.50, h.x), mix(0.62, 1.15, h.y), mix(0.15, 0.22, h.z));
    if(d < res.x){ res = vec2(d, M_NFERN); gF = vec4(p - o, h.y); }
  }
}
// a hanging twig and its small glossy leaves
void leafySprig(vec3 p, vec3 a, vec3 b, vec3 c, vec3 e, float sd, inout vec2 res){
  float tw = min(min(sdTaper(p, a, b, 0.024, 0.016), sdTaper(p, b, c, 0.016, 0.010)), sdTaper(p, c, e, 0.010, 0.0045));
  if(tw < res.x) res = vec2(tw, M_TWIG);
  for(int i=0; i<NLF; i++){
    float fi = float(i);
    vec3 h = hash31(fi*1.93 + sd), h2 = hash31(fi*4.37 + sd + 3.0);
    float t = 0.22 + 0.80*(fi + 0.5*h.x)/float(NLF);
    vec3 o = t < 0.5 ? mix(b, c, t*2.0) : mix(c, e, min(t*2.0 - 1.0, 1.0));
    float sgn = (i & 1) == 0 ? 1.0 : -1.0;
    vec3 dir = normalize(vec3(sgn*(0.25 + 0.75*h2.x), -0.65 - 0.9*h2.y, 0.35*(h.y - 0.3)));
    vec3 nr = normalize(vec3(0.5*(h2.z - 0.5), 0.45*(h.z - 0.3) + 0.25, 1.0));
    vec3 xa = normalize(cross(dir, nr)), za = cross(xa, dir);
    vec3 v = p - o;
    vec3 lq = vec3(dot(v, xa), dot(v, dir), dot(v, za));
    float S = mix(0.23, 0.34, h.z), W = S*mix(0.27, 0.34, h.x);
    float stalk = sdTaper(lq, vec3(0.0), vec3(0.0,0.05,0.0), 0.0045, 0.0035);
    if(stalk < res.x) res = vec2(stalk, M_TWIG);
    lq.y -= 0.045;
    lq.z += 0.55*lq.x*lq.x/W - 0.5*sin(lq.y/S*3.14159)*0.04;
    float tl = clamp(lq.y/S, 0.0, 1.0);
    float wd = W*pow(sin(3.14159*pow(tl, 0.72)), 0.85)*(1.0 + 0.05*sin(tl*44.0));
    float d = max(abs(lq.x) - wd, max(-lq.y, lq.y - S));
    d = max(d*0.72, abs(lq.z) - 0.0032)*0.8;
    if(d < res.x){ res = vec2(d, M_BLEAF); gF = vec4(lq.x/max(wd, 1e-3), tl, 0.0, h2.x); }
  }
}
// strap leaves in a rosette with a hot inner cup, tipped toward the camera
void bromeliad(vec3 p, inout vec2 res){
  vec3 q = p - BROM.xyz;
  q.yz = rot(-0.50)*q.yz;
  const float K = 9.0;
  float x = atan(q.z, q.x)*K/6.28318;
  for(int w=0; w<2; w++){
    float fw = float(w), off = fw*0.5;
    float s0 = floor(x + off);
    float R = BROM.w*mix(1.0, 0.50, fw);
    for(int j=0; j<2; j++){
      float sm = mod(s0 + float(j), K);
      float h = hash11(sm*0.37 + fw + 0.2);
      float a = (sm - off + 0.3*(h - 0.5))*6.28318/K;
      vec2 r = rot(-a)*q.xz;
      float L = R*(0.85 + 0.30*h);
      float ul, vn;
      float d = sdBlade(r.x, r.y, (q.y - r.x*mix(0.62, 1.15, fw))*0.75, L, 0.14*L, mix(0.66, 0.60, fw), 0.22, vec4(0.2,0.35,0.0,0.0), h, 0.006, 0.0, 0.006, ul, vn);
      if(d < res.x){ res = vec2(d, M_BROM); gF = vec4(ul, vn, fw, h); }
    }
  }
}
void nearDress(vec3 p, inout vec2 res){
  const float G = 0.16;
  float bL = sdCapsule(p, LOG_A, LOG_B, LOG_R + 0.13);
  res.x = min(res.x, max(bL, G));
  for(int i=0; i<1 && bL < G; i++) logAndMoss(p, res);

  float bM = min(length(p - MUC_L) - 0.56, length(p - MUC_R) - 0.44);
  res.x = min(res.x, max(bM, G));
  for(int i=0; i<1 && bM < G; i++) toadstools(p, res);

  float bB = length(p - BROM.xyz) - BROM.w*1.25;
  res.x = min(res.x, max(bB, G));
  for(int i=0; i<1 && bB < G; i++) bromeliad(p, res);

  float bFl = length(p - FERN_L - vec3(0.0,0.85,0.0)) - 1.60, bFr = length(p - FERN_R - vec3(0.0,0.85,0.0)) - 1.60;
  res.x = min(res.x, max(min(bFl, bFr), G));
  for(int i=0; i<1 && bFl < G; i++) groundFern(p, FERN_L, 1.3, 0.0, res);
  for(int i=0; i<1 && bFr < G; i++) groundFern(p, FERN_R, 4.1, 3.14159, res);

  float bTr = sdCapsule(p, TWR[1], TWR[3], 0.62), bTl = sdCapsule(p, TWL[1], TWL[3], 0.62);
  res.x = min(res.x, max(min(bTr, bTl), G));
  for(int i=0; i<1 && bTr < G; i++) leafySprig(p, TWR[0], TWR[1], TWR[2], TWR[3], 2.2, res);
  for(int i=0; i<1 && bTl < G; i++) leafySprig(p, TWL[0], TWL[1], TWL[2], TWL[3], 7.7, res);
}

// ------------------------------------------------------------------ far: leaf walls + trunks
void leafWall(vec3 p, float zc, float zr, float cs, float sd, float Lk, float hole, inout vec2 res){
  int n = slab(p, zc - zr - 0.7*cs, zc + zr + 0.7*cs, 4, res);
  vec2 g = p.xy/cs - 0.5;
  vec2 c0 = floor(g);
  for(int k=0; k<n; k++){
    vec2 ci = c0 + vec2(float(k & 1), float(k >> 1));
    vec3 h = hash33(vec3(ci, sd));
    vec2 c2 = (ci + 0.5 + (h.xy - 0.5)*0.8)*cs;
    // keep a window of bright mist behind the head
    float gap = length((c2 - SP_GLOW_C*(CAMZ - zc)*TANH)/vec2(1.3, 1.0))/((CAMZ - zc)*TANH);
    float pres = smoothstep(hole - 0.25, hole + 0.35, gap + 0.5*(h.y - 0.5));
    if(h.z > pres) continue;
    vec3 h2 = hash33(vec3(ci, sd + 5.0));
    float ang = -1.57 + (h2.x - 0.5)*2.9;
    vec2 dir = vec2(cos(ang), sin(ang));
    vec3 q = p - vec3(c2, zc + (h2.y - 0.5)*zr*2.0);
    float L = cs*Lk*(0.75 + 0.5*h2.z);
    float u = dot(q.xy, dir) + 0.5*L;
    float v = -q.x*dir.y + q.y*dir.x;
    float w = q.z + (h.x - 0.5)*1.5*q.x + (h.y - 0.35)*1.5*q.y;
    float ul, vn;
    float d = sdBlade(u, v, w*0.72, L, L*(0.16 + 0.10*h.z), 0.14, 0.15, vec4(0.5,-0.8,0.04,0.0), h.z, 0.02, 0.0, 0.016, ul, vn);
    if(d < res.x){ res = vec2(d, M_LEAF); gL = vec4(ul, vn, 1.0, 100.0 + h.z*50.0 + 50.0*step(0.0, zc + 22.0));
                   gLA = vec3(dir, 0.0); gLS = vec3(-dir.y, dir.x, 0.0); gLE = gBladeE; }
  }
}

void trunks(vec3 p, inout vec2 res){
  int n = slab(p, -60.0, -8.0, NTRUNK, res);
  for(int i=0; i<n; i++){
    vec4 k = TRK[i];
    vec3 q = p - vec3(k.x, G0, k.y);
    float yy = max(q.y, 0.0);
    q.x += 0.25*k.z*sin(q.y*0.22 + k.w);
    q.z += 0.20*k.z*sin(q.y*0.17 + k.w*1.7);
    float a = atan(q.z, q.x), l = length(q.xz);
    float rad = k.z*(1.0 + 0.40*exp(-yy/(1.4*k.z)));
    float fin = 0.5 + 0.5*cos(a*5.0 + k.w*3.0);
    rad *= 1.0 + 1.7*exp(-yy/(1.5*k.z))*fin*fin*fin;
    rad += k.z*0.05*sin(a*13.0 + 2.0*sin(yy*0.7 + k.w));
    float d = (l - rad)*0.55;
    if(d < res.x){ res = vec2(d, M_BARK); gF = vec4(a, yy, float(i), 0.0); }
  }
}

float groundH(vec2 xz){
  float back = max(-xz.y - 25.0, 0.0);              // falls away behind the last leaf wall: no horizon line
  return 0.5*sin(xz.x*0.31 + 0.4)*sin(xz.y*0.27 + 1.1) + 0.6*(vnoise2(xz*0.5) - 0.5) - 0.02*back*back;
}

vec2 setMap(vec3 p, bool shadow){
  if(shadow){
    gShN += 1.0;
    if(gShN < 1.5) return vec2(0.003*gDap, 0.0);   // canopy decides first; a blocked ray stops at step 1
  }
  vec2 res = vec2((p.y - G0 - groundH(p.xz))*0.6, M_GROUND);
  tableLeaves(p, res);
  treeFerns(p, res);
  lianas(p, res);
  flowers(p, res);
  nearDress(p, res);
  trunks(p, res);
  if(!shadow){
    leafWall(p, -13.6, 0.8, 0.62, 3.0, 1.75, 0.98, res);
    leafWall(p, -17.0, 1.2, 0.75, 4.0, 1.75, 0.90, res);
    leafWall(p, -22.5, 1.6, 0.95, 5.0, 1.75, 0.78, res);
    leafWall(p, -31.0, 2.2, 1.30, 7.0, 1.75, 0.62, res);
  }
  return res;
}

// ------------------------------------------------------------------ sky: the farthest jungle, painted
vec3 setSky(vec3 rd){
  vec2 s = rd.xy/max(abs(rd.z), 0.08)/TANH;
  vec2 c = s - SP_GLOW_C;
  float g = exp(-dot(c, c)/(SP_GLOW_R*SP_GLOW_R));
  vec3 col = mix(vec3(0.08,0.22,0.17), vec3(0.62,0.80,0.36), g);
  col = mix(col, vec3(0.75,0.95,0.75), smoothstep(0.1, 0.6, rd.y));
#ifdef PASS_BG
  // three flat layers of far leaves, paler toward the glow
  for(int l=0; l<3; l++){
    float fl = float(l);
    float cs = 0.34 - fl*0.07;
    vec2 g2 = s/cs + fl*3.7;
    vec2 c0 = floor(g2 - 0.5);
    float ins = 1e9;
    for(int k=0; k<4; k++){
      vec2 ci = c0 + vec2(float(k & 1), float(k >> 1));
      vec3 h = hash33(vec3(ci, 31.0 + fl));
      vec2 q = g2 - (ci + 0.5 + (h.xy - 0.5)*0.8);
      q = rot(-1.57 + (h.z - 0.5)*2.8)*q;
      float L = 0.72 + 0.3*h.x;
      float x = clamp(q.x/L*0.5 + 0.5, 0.0, 1.0);
      float dl = abs(q.y) - 0.34*sqrt(4.0*x*(1.0 - x));
      dl = max(dl, abs(q.x) - L);
      if(h.y > 0.28) ins = min(ins, dl);
    }
    float a = smoothstep(0.03, -0.03, ins);
    vec3 lc = mix(vec3(0.10,0.26,0.20), vec3(0.50,0.68,0.46), fl*0.5)*mix(1.0, 2.0, g);
    col = mix(col, lc, a*(0.55 - 0.08*fl)*(1.0 - 0.6*g));
  }
#endif
  return col;
}

// ------------------------------------------------------------------ paint
void paintLeaf(vec3 p, vec3 rd, vec3 n, inout Surf s){
  float id = gL.w;
  bool wall = id >= 99.5;
  float u = clamp(gL.x, 0.0, 1.0), v = gL.y, av = abs(v);
  float h = fract(id*0.618 + 0.13);
  float kindV = id < float(NLEAF) ? LK[int(id)].y : 0.2;
  float NV = mix(8.0, 34.0, kindV);
  float c = u*NV - mix(1.5, 2.2, kindV)*av - 0.6*av*av;
  float fc = abs(fract(c) - 0.5);
  float vein = smoothstep(0.075, 0.015, fc)*(1.0 - 0.5*av);
  float rib = smoothstep(0.045, 0.012, av);
  float mott = fbm2_3(vec2(u*5.0, v*3.0) + h*40.0);
  vec3 a = mix(SP_LEAF_DEEP, SP_LEAF_MID, 0.25 + 0.75*h);
  a = mix(a, SP_LEAF_LIME, (0.12 + 0.30*fract(h*7.0))*smoothstep(0.2, 1.0, u + 0.4*(mott - 0.5)));
  a = mix(a, a*vec3(0.55,0.92,1.25), smoothstep(0.45, 0.85, fract(h*5.3)));   // some lean blue-green
  a *= 0.78 + 0.45*mott;
  a = mix(a, a*1.3 + vec3(0.01,0.03,0.005), vein*0.5);
  a = mix(a, SP_LEAF_LIME*0.62 + vec3(0.10,0.10,0.04), rib*0.8);
  if(gL.x < 0.0) a = SP_LEAF_MID*1.2 + vec3(0.04,0.03,0.0);     // petiole
  a *= 0.94 + 0.10*smoothstep(0.0, 0.5, fc);                    // quilting between the veins
  if(id < float(NHERO)){
    float fine = sin(c*6.2832*7.0);                             // fine parallel venation
    float streak = vnoise2(vec2(c*3.0, av*5.0 + h*30.0));       // long pale streaks along the veins
    vec2 bq = vec2(u*7.0, v*2.4) + 0.5*sin(vec2(v*5.0, u*9.0) + h*20.0);    // warped: no lattice in the blotches
    float blotch = fbm2_3(bq + h*70.0);
    a *= 0.92 + 0.05*fine + 0.22*streak;
    a = mix(a, a*vec3(0.74,0.72,0.56), smoothstep(0.48, 0.74, blotch)*0.6);
    a = mix(a, a*vec3(1.5,1.22,0.55), 0.55*smoothstep(0.86, 1.0, gLE));   // margins yellow a little
  }
  // the underside of a leaf is paler and matte
  float under = step(0.0, -gL.z);
  a = mix(a, a*vec3(1.1,1.2,1.0) + vec3(0.01,0.02,0.015), under*0.6);
  s.alb = a;
  // ribbing: the blade rises and falls between the side veins
  if(!wall) s.n = normalize(n + 0.06*sin(c*6.2832)*gLA + 0.16*(mott - 0.5)*gLS + 0.05*sin(c*44.0)*gLA);
  s.occ = 0.72 + 0.28*smoothstep(0.0, 0.25, av) ;
  s.spec = under > 0.5 ? 0.25 : 0.9; s.shin = 46.0; s.wrap = 0.25; s.sheen = 0.5;
  // light through the blade: sun on the far side, veins stay dark
  float ndl = dot(s.n, uSunDir);
  float thru = smoothstep(0.02, 0.55, -ndl);
  float vis = gDap + 0.05;
  gGlow = thru*vis*(1.0 - 0.55*vein)*(1.0 - 0.7*rib);
  // and the bright mist behind lights the blades seen against it
  vec2 sc = rd.xy/max(-rd.z, 0.1)/TANH - SP_GLOW_C;
  gGlow += 0.14*exp(-dot(sc, sc)/1.6)*(1.0 - 0.5*vein)*(1.0 - 0.7*rib);
}

void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){
  setMap(p, false);       // recover the nearest leaf's coordinates at the hit point
  gDap = canopyOpen(p);
  if(m == M_LEAF){
    paintLeaf(p, rd, n, s);
  } else if(m == M_FERN){
    float t0 = gF.x, sn = gF.y;
    vec3 a = mix(SP_LEAF_MID*1.1, SP_LEAF_LIME, 0.25 + 0.6*t0*t0 + 0.18*gF.z);
    a *= 0.75 + 0.5*smoothstep(0.0, 1.0, sn);
    a = mix(a, vec3(0.10,0.12,0.03), smoothstep(0.10, 0.03, sn)*0.7);
    s.alb = a;
    s.wrap = 0.35; s.spec = 0.25; s.shin = 24.0; s.sheen = 0.3;
    gGlow = 0.04 + 0.6*smoothstep(0.02, 0.5, -dot(n, uSunDir))*max(gDap, 0.1);
  } else if(m == M_BARK){
    // long fibres follow the trunk; the moss coordinates are warped so no noise lattice shows
    float ang = gF.x, yy = gF.y, sd = gF.z;
    float f1 = vnoise2(vec2(ang*4.0 + sd*7.0, yy*0.16));
    float f2 = vnoise2(vec2(ang*13.0 + sd*3.0, yy*0.45 + 4.0));
    float f3 = vnoise2(vec2(ang*37.0, yy*1.3 + sd));
    float fib = f1*0.5 + f2*0.32 + f3*0.18;
    vec3 a = mix(SP_BARK_A, SP_BARK_B, smoothstep(0.30, 0.72, fib));
    vec3 mp = vec3(ang*1.6, yy*0.30, sd) + 0.7*sin(vec3(yy*0.5 + sd, ang*2.3, yy*0.31));
    float mn = vnoise3(mp*1.3)*0.6 + vnoise3(mp*3.1 + 5.0)*0.28 + vnoise3(mp*7.3 + 9.0)*0.12;
    float mb = smoothstep(0.42, 0.56, mn + 0.12*n.y + 0.14*smoothstep(4.0, 0.0, p.y - G0));
    vec3 moss = SP_MOSS*(0.55 + 0.95*vnoise2(p.xy*5.0)*(0.6 + 0.8*f3));
    a = mix(a, moss, mb);
    vec3 tang = normalize(cross(n, vec3(0.0,1.0,0.0)) + vec3(1e-4));
    s.n = normalize(n + (1.0 - 0.7*mb)*0.55*(f2 - 0.5 + 0.5*(f3 - 0.5))*tang);
    s.alb = a; s.occ = 0.7 + 0.3*smoothstep(0.25, 0.6, fib); s.wrap = 0.25; s.spec = 0.03; s.shin = 8.0;
  } else if(m == M_VINE){
    s.alb = mix(vec3(0.05,0.045,0.03), vec3(0.09,0.14,0.04), vnoise2(p.xy*9.0));
    s.wrap = 0.3;
  } else if(m == M_PETAL){
    float kind = gF.z;
    if(kind < 0.5){
      float r = gF.x, lobe = gF.y;
      float i = gF.w;
      vec3 c0 = i < 0.5 ? vec3(1.0,0.05,0.30) : (i < 1.5 ? vec3(1.0,0.16,0.03) : vec3(1.0,0.08,0.42));
      vec3 c1 = vec3(0.30,0.0,0.05);
      vec3 a = mix(c1, c0, smoothstep(0.12, 0.42, r));
      a = mix(a, a + vec3(0.0,0.16,0.14), smoothstep(0.65, 1.0, r)*0.5);
      a *= 0.94 + 0.08*sin(gF.y*40.0 + r*6.0);           // radial veins
      a *= 0.70 + 0.30*smoothstep(0.0, 0.12, gF.y);      // the crease where petals overlap
      s.alb = a; s.wrap = 0.5; s.spec = 0.3; s.shin = 22.0; s.sheen = 0.5;
      gGlow = 0.55;
    } else if(kind < 1.5){
      s.alb = mix(vec3(1.0,0.75,0.05), vec3(1.0,0.28,0.06), smoothstep(0.0, 0.5, gF.x));
      s.wrap = 0.5; gGlow = 0.3;
    } else {
      s.alb = mix(vec3(0.92,0.035,0.022), vec3(1.0,0.66,0.05), smoothstep(0.58, 0.92, gF.x));
      s.wrap = 0.4; s.spec = 0.5; s.shin = 40.0; s.sheen = 0.6;
      gGlow = 0.35;
    }
    s.occ = 1.0;
  } else if(m == M_LOG){
    vec3 ba = LOG_B - LOG_A, pa = p - LOG_A;
    float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
    vec3 c = pa - ba*h;
    float a = atan(c.z, c.y), u = h*6.8;
    float gr = vnoise2(vec2(a*16.0, u*3.0)), gr2 = vnoise2(vec2(a*42.0, u*5.0));
    vec3 bk = mix(SP_BARK_A*0.8, SP_BARK_B*0.62, smoothstep(0.25, 0.8, gr*0.7 + gr2*0.3));
    s.alb = bk*(0.8 + 0.4*vnoise3(p*9.0));
    s.occ = 0.45 + 0.55*smoothstep(0.2, 0.6, gr);
    s.wrap = 0.25; s.spec = 0.10; s.shin = 18.0; s.sheen = 0.2;
  } else if(m == M_MOSS){
    float f1 = fbm3(p*26.0), f2 = fbm3(p*74.0), f3 = fbm3(p*2.7);
    vec3 mo = mix(vec3(0.030,0.115,0.040), vec3(0.17,0.40,0.055), smoothstep(0.32, 0.68, f1*0.55 + f2*0.45));
    s.alb = mix(mo, mo*vec3(1.35,1.10,0.60), smoothstep(0.44, 0.64, f3));
    s.n = normalize(n + 0.40*(vec3(vnoise3(p*90.0), vnoise3(p*90.0 + 5.0), vnoise3(p*90.0 + 9.0)) - 0.5));
    s.occ = 0.55 + 0.45*smoothstep(0.25, 0.65, f2);
    s.wrap = 0.55; s.sheen = 0.8; s.spec = 0.02; s.shin = 8.0;
  } else if(m == M_NFERN){
    float tp = smoothstep(0.1, 1.4, length(gF.xyz));
    vec3 a1 = mix(SP_LEAF_MID*1.1, SP_LEAF_MID*1.7, gF.w);
    vec3 a2 = mix(SP_LEAF_LIME*0.75, SP_LEAF_LIME, gF.w);
    s.alb = mix(a1, a2, tp)*(0.78 + 0.40*vnoise3(p*30.0));
    s.wrap = 0.55; s.spec = 0.14; s.shin = 24.0; s.sheen = 0.3;
    gGlow = 0.10 + 0.55*smoothstep(0.02, 0.5, -dot(n, uSunDir))*(gDap + 0.05);
  } else if(m == M_TWIG){
    s.alb = mix(vec3(0.05,0.04,0.025), vec3(0.17,0.13,0.07), vnoise3(p*vec3(60.0,20.0,60.0)));
    s.wrap = 0.2; s.spec = 0.05; s.shin = 14.0;
  } else if(m == M_BLEAF){
    float x = gF.x, y = gF.y, id = gF.w;
    vec3 c1 = id < 0.45 ? SP_LEAF_DEEP*1.6 : (id < 0.80 ? SP_LEAF_MID : SP_LEAF_LIME*0.70);
    vec3 c2 = id < 0.45 ? SP_LEAF_MID : (id < 0.80 ? SP_LEAF_MID*1.5 : SP_LEAF_LIME);
    vec3 ca = mix(c1, c2, smoothstep(0.30, 0.68, fbm3(p*11.0)));
    float vein = smoothstep(0.045, 0.012, abs(x));
    float side = abs(fract(y*9.0 - abs(x)*2.2) - 0.5);
    vein = max(vein, 0.65*smoothstep(0.07, 0.02, side)*smoothstep(1.0, 0.25, abs(x)));
    s.alb = mix(ca, ca*vec3(1.25,1.25,0.8) + 0.04, vein*0.6);
    s.alb *= mix(0.74, 1.0, smoothstep(1.0, 0.72, abs(x)));
    s.wrap = 0.4; s.spec = 0.7; s.shin = 50.0; s.sheen = 0.35;
    gGlow = 0.65*smoothstep(0.02, 0.5, -dot(n, uSunDir))*(gDap + 0.08)*(1.0 - 0.6*vein);
  } else if(m == M_CAP){
    vec3 q = gF.xyz;
    float cr = length(q.xz);
    vec3 ca = mix(vec3(0.74,0.46,0.10), vec3(0.95,0.80,0.40), smoothstep(0.15, 1.0, cr));
    float under = smoothstep(0.02, -0.14, q.y);
    float gill = 0.72 + 0.28*sin(atan(q.z, q.x)*46.0);
    s.alb = mix(ca, vec3(0.86,0.78,0.56)*gill, under);
    s.occ = mix(1.0, 0.6, under);
    s.spec = 0.35; s.shin = 40.0; s.wrap = 0.35; s.sheen = 0.35;
    gGlow = 0.16;
  } else if(m == M_STIPE){
    s.alb = vec3(0.84,0.78,0.60)*(0.90 + 0.10*sin(gF.y*34.0))*mix(0.70, 1.0, smoothstep(0.0, 0.5, gF.y));
    s.wrap = 0.4; s.sheen = 0.3; s.spec = 0.05;
  } else if(m == M_BROM){
    float u = clamp(gF.x, 0.0, 1.0), av = abs(gF.y), h = gF.w;
    vec3 a;
    if(gF.z < 0.5){
      a = mix(vec3(0.34,0.03,0.09), mix(SP_LEAF_MID*1.4, SP_LEAF_LIME*0.8, h), smoothstep(0.12, 0.55, u));
      a = mix(a, vec3(0.75,0.08,0.22), smoothstep(0.78, 1.0, u)*0.75);    // pink-dipped tips
      a *= 0.86 + 0.20*sin(av*16.0);                                     // lengthwise stripes
      gGlow = 0.10;
    } else {
      a = mix(vec3(1.0,0.05,0.30), vec3(1.0,0.30,0.08), u);
      gGlow = 0.40;
    }
    s.alb = a; s.wrap = 0.4; s.spec = 0.5; s.shin = 36.0; s.sheen = 0.4;
  } else {
    float a = fbm2(p.xz*0.9);
    s.alb = mix(vec3(0.02,0.05,0.03), vec3(0.07,0.17,0.05), smoothstep(0.3, 0.7, a));
    s.wrap = 0.2;
  }
}

vec3 setAmbient(vec3 p, vec3 n){
#ifdef PASS_BG
  // clamped: a blade edge can give a zero gradient, and one NaN here blanks the frame through the FFT lens
  vec3 sky = mix(SP_AMB_LOW, SP_AMB_HIGH, clamp(0.5 + 0.5*n.y, 0.0, 1.0));
  vec3 bounce = SP_BOUNCE*clamp(0.6 - 0.5*n.y, 0.0, 1.0);
  return sky + bounce + SP_GLOW_LEAF*gGlow;
#else
  vec3 sky = SP_AMB_CHAR*(0.72 + 0.28*n.y);
  vec3 bounce = SP_BOUNCE*0.8*clamp(0.6 - 0.5*n.y, 0.0, 1.0);
  return sky + bounce;
#endif
}

// ------------------------------------------------------------------ per-frame light
float setCookie(vec3 p){
  vec2 q = sunPlane(p);
  vec2 w = gWind;
  float a = vnoise2(q*1.7*SP_DAPPLE_FREQ + w*2.5);
  float b = vnoise2(q*3.9*SP_DAPPLE_FREQ - w*3.5 + 5.0);
  return smoothstep(0.34, 0.66, a*0.66 + b*0.34);
}
// Art-directed shafts: ellipses on the canopy plane, long along the view so PASS_VOL's 3.75 unit samples
// never miss them.  A beam only shows in front of the scene depth, so the near ones are what cross the foliage.
#define NSHAFT 7
const vec4 SHAFT[NSHAFT] = vec4[NSHAFT](    // centre on the sun plane, half width, half length
  vec4( -9.55,  2.8, 0.26, 3.5),            // near: in front of the far leaves, side strips
  vec4( -8.95,  1.6, 0.17, 3.0),
  vec4( -3.42,  2.6, 0.26, 3.5),
  vec4( -5.25,  1.4, 0.17, 3.0),
  vec4( -8.9,-12.0, 0.42, 7.0),             // far: above the head
  vec4( -4.9,-14.0, 0.50, 6.0),
  vec4( -1.0,-12.6, 0.55, 6.0)
);
float setVol(vec3 p){
  vec2 q = sunPlane(p) + gWind*0.5;
  float acc = 0.0;
  for(int i=0; i<NSHAFT; i++){
    vec4 S = SHAFT[i];
    vec2 d = (q - S.xy)/S.zw;
    acc += (i < 4 ? 0.42 : 0.85)*exp(-dot(d, d));
  }
  // the shafts stay in the world: PASS_COMP adds 15% of the volume over the character, so they fade out
  // around the portrait (head about 0.17 in these units) and stay strong at the sides and upper corners
  vec3 v = p - uCamPos;
  vec2 s = v.xy/max(-v.z, 0.1) - vec2(0.0, -0.02);
  float keep = smoothstep(0.15, 0.26, length(s*vec2(1.0, 0.85)));
  return clamp(acc, 0.0, 1.0)*keep*smoothstep(CANY + 1.0, CANY - 3.0, p.y);
}
float setFog(vec3 p){
  float hy = max(p.y - G0, 0.0);
  float d = 0.0155*(0.45 + 0.55*exp(-hy*0.11));
  return d*smoothstep(1.0, -5.0, p.z)*SP_FOG_GAIN;
}
vec3 setFogAmb(vec3 rd){
  vec2 c = rd.xy/max(abs(rd.z), 0.1)/TANH - SP_GLOW_C;
  float g = exp(-dot(c, c)/(SP_GLOW_R*SP_GLOW_R));
  return mix(SP_FOG_LO, SP_FOG_HI, g);
}

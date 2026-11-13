// ---------------------------------------------------------------------------
//  engine/sets/meadow.glsl — sunlit wildflower meadow, late morning.  Rabbit's-eye view.
//  Layers, near to far:
//    HERO     hand-placed plants at the frame edges, real ribbons and petals, near focus
//    MID      SDF blades + flowers on stalks on jittered grids, up to the first swell (z about -45)
//    FAR      height field, painted colour patches, flower speckles, soft cloud shadows
//    HILLS    three ridges with tree clumps; aerial perspective comes from setFog
//    SKY      painted gradient + hand-placed cumulus (bake only; setSky stays cheap per frame)
//  Same API as engine/sets/forest.glsl.  setMap/setSurface are bake only.
//
//  llvmpipe has no branch skipping: only a loop stops early, so WHEN(c){...} (a loop that runs zero
//  or one times) is how expensive parts are really skipped.  Never `continue` inside a WHEN.
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-3.2)
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.46,0.50,0.28)              // warm green light off the grass
#endif
#ifndef SP_AMB_SKY
#define SP_AMB_SKY vec3(0.84,0.97,1.20)             // open-sky ambient, facing up
#endif

#define WHEN(c) for(int _w = ((c) ? 0 : 1); _w < 1; _w++)

const float G0 = SP_GROUND_Y;
#define MD_TAU 6.28318531

#define M_CANOPY  99.0      // no canopy here
#define M_GROUND  1.0
#define M_BLADE   2.0
#define M_MIDFL   3.0
#define M_MIDSTEM 4.0
#define M_TREE    5.0
#define M_RIBBON  6.0
#define M_STEM    7.0
#define M_DAISY   8.0
#define M_DISC    9.0
#define M_POPPY   10.0
#define M_POPC    11.0
#define M_CORN    12.0
#define M_BUTTER  13.0
#define M_CLOVER  14.0
#define M_SPIKE   15.0
#define M_CALYX   16.0

vec2  gWind;
bool  mdShade;                  // true after setSurface: the AO + shadow calls of the same hit
float mdU, mdV, mdH;              // hero stash: along, across, per-part hash
vec3  mdQ;                      // hero stash: flower-local point
float mdYn, mdWn, mdBH;           // mid blade stash
vec3  mdFV;                     // mid flower stash: x = r/R, y = axial/R, z = azimuth
float mdFS, mdFH;                // mid flower stash: species, hash

void setInit(){
  float T = uTime;
  gWind = 0.05*vec2(sin(T*0.5), cos(T*0.41));
  mdShade = false;
}

// ------------------------------------------------------------------ terrain
float mdGauss(float z, float c, float w){ float u = (z - c)/w; return exp(-u*u); }
float z1Line(float x){ return -100.0 + 11.0*sin(x*0.031 + 2.0) + 5.0*sin(x*0.073 + 0.5); }
float z2Line(float x){ return -152.0 + 13.0*sin(x*0.024 + 4.1) + 5.0*sin(x*0.061 + 1.2); }
float terrainH(vec2 q){
  float x = q.x, z = q.y;
  float h = 0.20*sin(x*0.23 + 0.4)*sin(z*0.19 + 1.0) + 0.14*sin(x*0.51 + z*0.33 + 2.0);
  h -= 0.50*mdGauss(z, -13.0, 12.0);                                         // the ground falls away a little
  h += (1.30 + 0.35*sin(x*0.06 + 1.0))*mdGauss(z, -44.0 + 6.0*sin(x*0.045 + 0.6), 15.0);   // first swell
  h += (5.0 + 1.5*sin(x*0.025 + 0.5))*mdGauss(z, z1Line(x), 25.0);           // meadow ridge
  h += (8.6 + 1.8*sin(x*0.020 + 3.1) + 0.9*sin(x*0.055 + 1.0))*mdGauss(z, z2Line(x), 33.0);   // tree ridge
  h += smoothstep(-172.0, -226.0, z)*(19.5 + 4.5*sin(x*0.016 + 1.0) + 2.0*sin(x*0.047 + 4.0));   // far hills
  return h;
}
float grassZone(vec2 q){ return smoothstep(-70.0, -52.0, q.y); }
// blade height: taller next to the character, lower toward the swell
float grassH(vec2 q){ return mix(2.2, 2.85, smoothstep(-15.0, -3.0, q.y)); }

// ------------------------------------------------------------------ mid meadow
vec2 windBend(vec2 q){
  return 0.30*vec2(sin(q.x*0.19 + q.y*0.07 + 1.0) + 0.55*sin(q.y*0.31 + 0.4) + 0.5,
                   0.7*cos(q.y*0.17 + q.x*0.11) + 0.4*sin(q.x*0.37 + 2.0));
}
vec2 sdBlades(vec3 p, float gh, float zone, float cd, bool shadow){
  float y = p.y - gh + 0.06;
  float Hm = grassH(p.xz)*zone;
  float ny = clamp(y/2.6, 0.0, 1.0);
  vec2 q = p.xz - windBend(p.xz)*ny*ny;
  const float CS = 0.60;
  vec2 c0 = floor(q/CS);
  float d = 1.2;
  float T = 0.010 + 0.0016*cd;
  int NK = shadow ? 2 : 4;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, 5.0));
    vec2 base = (ci + 0.5 + (h.xy - 0.5)*0.85)*CS;
    if(base.y > -1.45) continue;                     // nothing from this grid in front of the character
    for(int k=0;k<NK;k++){
      float fk = float(k);
      float r1 = fract(h.x*7.13 + fk*0.377), r2 = fract(h.y*5.71 + fk*0.613), r3 = fract(h.z*3.37 + fk*0.541);
      float a = h.z*MD_TAU + fk*2.4 + r1;
      vec2 dir = vec2(cos(a), sin(a));
      float Hb = Hm*(0.58 + 0.42*r1);
      float L = min(0.60, Hb*(0.08 + 0.20*r2));
      float W = 0.048 + 0.038*r3;
      vec2 e = q - (base + (vec2(r2, r3) - 0.5)*0.42*CS + vec2(dir.y, -dir.x)*0.10*(fk - 1.5));
      float s = dot(e, dir), wc = dot(e, vec2(-dir.y, dir.x));
      float yn = clamp(y/max(Hb, 0.02), 0.0, 1.0);
      float sp = L*yn*yn*yn;
      float hw = W*sqrt(max(1.0 - yn, 0.0))*(0.75 + 0.25*smoothstep(0.0, 0.3, yn));
      vec2 v = vec2(abs(wc) - hw, abs(s - sp + 0.35*abs(wc)) - T);
      float db = min(max(v.x, v.y), 0.0) + length(max(v, 0.0));
      db = length(vec2(max(db, 0.0), max(y - Hb, 0.0))) + min(db, 0.0);
      if(db < d){ d = db; mdYn = yn; mdWn = wc/max(hw, 1e-3); mdBH = fract(r1*3.1 + r3*7.7 + h.y); }
    }
  }
  return vec2(d*0.80, M_BLADE);
}
// species picked by smooth drifts, dithered per plant: clusters, not confetti
float flowerSpecies(vec2 b, float dither){
  float f = 0.5 + 0.5*sin(b.x*0.070 + 1.5*sin(b.y*0.052 + 0.7))*cos(b.y*0.061 + 1.2*sin(b.x*0.043 + 2.0));
  return floor(6.0*fract(1.45*f + 0.075*dither + 0.13));
}
float flowerDensity(vec2 b){
  float f = sin(b.x*0.41 + 1.4*sin(b.y*0.27 + 1.0))*sin(b.y*0.33 + 1.2*sin(b.x*0.23));
  return 0.05 + 0.78*smoothstep(0.05, 0.62, f);
}
void sdMidFlowers(vec3 p, float gh, float zone, float cd, bool shadow, inout vec2 res){
  const float FC = 1.0;
  vec2 c0 = floor(p.xz/FC);
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, 9.0));
    vec3 h2 = hash33(vec3(ci, 21.0));
    vec2 b = (ci + 0.5 + (h.xy - 0.5)*0.9)*FC;
    if(b.y > -2.2 || h.z > flowerDensity(b)) continue;
    float sp = flowerSpecies(b, h2.x - 0.5);
    float R = 0.135, Hf = 1.00;                      // ox-eye daisy
    if(sp == 1.0){ R = 0.085; Hf = 0.97; }           // buttercup
    else if(sp == 2.0){ R = 0.120; Hf = 1.06; }      // poppy
    else if(sp == 3.0){ R = 0.100; Hf = 1.08; }      // cornflower
    else if(sp == 4.0){ R = 0.095; Hf = 0.84; }      // clover
    else if(sp == 5.0){ R = 0.030; Hf = 1.20; }      // seed grass
    R *= (0.85 + 0.35*h2.y)*(1.0 + cd*0.011);
    Hf *= (0.90 + 0.18*h2.z)*zone*grassH(b);
    vec2 lean = vec2(h2.z - 0.5, h.z - 0.5)*0.42 + windBend(b)*0.7;
    vec3 c = vec3(b.x + lean.x, gh + Hf, b.y + lean.y);
    vec3 ax = normalize(vec3((h2.y - 0.5)*0.9 + 0.12, 0.72, 0.60 + (h.x - 0.5)*0.7));
    if(sp == 5.0) ax = normalize(vec3(lean.x, 2.0, lean.y));
    float ds = sdCapsule(p, vec3(b.x, gh - 0.05, b.y), c, 0.013 + 0.0006*cd);
    if(ds*0.9 < res.x) res = vec2(ds*0.9, M_MIDSTEM);
    vec3 v = p - c;
    float ya = dot(v, ax);
    float r = length(v - ax*ya);
    float Ry = (sp == 0.0) ? 0.24 : (sp == 5.0) ? 7.0 : (sp == 4.0) ? 1.0 : 0.62;
    float dh = 1e3;
    WHEN(!shadow && dot(v, v) < 1.2){
      vec3 ux = normalize(vec3(ax.y, -ax.x, 0.0)), uz = cross(ux, ax);
      float az = atan(dot(v, uz), dot(v, ux));
      float n = (sp == 0.0) ? 13.0 : (sp == 3.0) ? 8.0 : (sp == 2.0) ? 2.0 : 5.0;
      float Re = R*(1.0 + ((sp == 4.0 || sp == 5.0) ? 0.0 : 0.13)*cos(n*az));
      dh = sdEllipsoid(vec3(r, ya, 0.0), vec3(Re, R*Ry, Re));
      if(dh < res.x){ mdFV = vec3(r/R, ya/R, az); mdFS = sp; mdFH = h.x; }
    }
    WHEN(shadow || dot(v, v) >= 1.2){
      dh = sdEllipsoid(vec3(r, ya, 0.0), vec3(R, R*Ry, R));
    }
    if(dh < res.x) res = vec2(dh, M_MIDFL);
  }
}

// ------------------------------------------------------------------ distant trees
float mdCrown;     // nearest real crown: leaf noise goes on this, never on an empty-cell bound
// a chain of round crowns along a line; bx = spacing, keep = presence, rr = crown radius range
float crownRow(vec3 p, float zc, float S, float seed, float keep, vec2 rr){
  float ci = floor(p.x/S);
  float d = max(max(1.1*S - 1.6*rr.y, 1.6), max(abs(p.z - zc) - 14.0 - 1.3*rr.y, p.y - terrainH(p.xz) - 2.3*rr.y));
  for(int i=-1;i<=1;i++){
    float c = ci + float(i);
    vec3 h = hash33(vec3(c, seed, 3.0));
    if(h.z > keep) continue;
    float x = (c + 0.5 + (h.x - 0.5)*0.7)*S;
    float z = zc + (h.y - 0.5)*5.0;
    float R = mix(rr.x, rr.y, fract(h.x*5.3 + h.y));
    float y0 = terrainH(vec2(x, z));
    vec3 q = p - vec3(x, y0 + R*0.95, z);
    float e = sdEllipsoid(q, vec3(R*1.12, R*1.0, R*1.05));
    e = smin(e, sdEllipsoid(q - vec3(R*0.62, -R*0.25, 0.0), vec3(R*0.62, R*0.55, R*0.7)), R*0.3);
    e = smin(e, sdEllipsoid(q - vec3(-R*0.66, -R*0.32, R*0.1), vec3(R*0.55, R*0.5, R*0.6)), R*0.3);
    d = min(d, e);
    mdCrown = min(mdCrown, e);
  }
  return d;
}
float sdTrees(vec3 p, float aboveT, bool shadow){
  float z2 = z2Line(p.x), z1 = z1Line(p.x), z3 = -219.0 + 4.0*sin(p.x*0.03);
  // a true lower bound to every crown; rows further than it can't matter, so they are really skipped
  float bz = min(min(abs(p.z - z2 + 9.0) - 13.0, abs(p.z - z3) - 11.0), abs(p.z - z1 + 5.0) - 7.0);
  float bh = aboveT - 9.5;
  float bd = max(bz, bh);
  float d = bd > 0.0 ? bd + 1.0 : 40.0;          // +1: this bound is a gate, it must never pass the hit test itself
  mdCrown = 1e3;
  WHEN(abs(p.z - z2 + 9.0) < d + 16.0){
    d = min(d, crownRow(p, z2 - 5.0, 5.0, 1.0, 0.88, vec2(2.0, 3.1)));                 // treeline on the ridge
    d = min(d, crownRow(p, z2 - 14.0, 6.0, 2.0, 0.80, vec2(2.4, 3.6)));
  }
  WHEN(abs(p.z - z3) < d + 12.0){
    d = min(d, crownRow(p, z3, 4.2, 3.0, 0.97, vec2(2.2, 3.5)));                       // woods on the far hills
  }
  WHEN(abs(p.z - z1 + 5.0) < d + 10.0){
    d = min(d, crownRow(p, z1 - 5.0, 15.0, 4.0, 0.42, vec2(1.3, 2.3)));                // hedge clumps on the near ridge
  }
  WHEN(!shadow && mdCrown < 1.6){
    float dn = mdCrown + 1.1*(fbm3(p*0.70) - 0.5) + 0.42*(vnoise3(p*2.6) - 0.5);
    d = (mdCrown <= d) ? dn : min(d, dn);
  }
  return d*0.85;
}

// ------------------------------------------------------------------ hero plants
// MD_SCR: a point given in the z=0 picture plane (frame x about -2.5..2.5, y -1.56..1.25 since round three opened the
// lens from the same camera position), pushed to depth z along the camera ray.  The keep-out of the character is a rule
// in picture space, so this is the natural unit.  Blades and stems root below the frame, at MD_ROOT.
#define MD_SCR(sx,sy,z) vec3((sx)*(5.15-(z))/5.15, 0.21+((sy)-0.21)*(5.15-(z))/5.15, (z))
#define MD_ROOT (-1.85)

#define MD_NRIB 18
// ribbon blades and leaves: root, control, tip; xyz + (half width | twist | kind: 0 blade, 1 leaf)
const vec4 MD_RA[MD_NRIB] = vec4[MD_NRIB](
  vec4(MD_SCR(-1.42,MD_ROOT, 0.50),0.075), vec4(MD_SCR(-1.76,MD_ROOT, 0.10),0.060), vec4(MD_SCR(-1.98,MD_ROOT, 0.70),0.095),
  vec4(MD_SCR(-1.43,MD_ROOT,-0.40),0.055), vec4(MD_SCR(-2.10,MD_ROOT,-0.50),0.070), vec4(MD_SCR(-1.62,MD_ROOT, 0.30),0.060),
  vec4(MD_SCR(-1.52,MD_ROOT,-1.00),0.050), vec4(MD_SCR(-1.84,MD_ROOT,-0.90),0.060),
  vec4(MD_SCR( 1.46,MD_ROOT, 0.60),0.080), vec4(MD_SCR( 1.78,MD_ROOT, 0.20),0.065), vec4(MD_SCR( 2.00,MD_ROOT, 0.70),0.095),
  vec4(MD_SCR( 1.44,MD_ROOT,-0.30),0.055), vec4(MD_SCR( 1.58,MD_ROOT, 0.20),0.060), vec4(MD_SCR( 1.92,MD_ROOT,-0.70),0.060),
  vec4(MD_SCR( 2.15,MD_ROOT, 0.30),0.070), vec4(MD_SCR( 1.66,MD_ROOT,-1.10),0.050),
  // two daisy leaves
  vec4(MD_SCR(-1.66,-1.00, 0.05),0.090), vec4(MD_SCR( 1.60,-1.05, 0.25),0.085));
const vec4 MD_RB[MD_NRIB] = vec4[MD_NRIB](
  vec4(MD_SCR(-1.50, 0.05, 0.50), 1.2), vec4(MD_SCR(-1.73, 0.25, 0.10),-0.9), vec4(MD_SCR(-2.00,-0.20, 0.70), 1.4),
  vec4(MD_SCR(-1.44,-0.40,-0.40),-0.8), vec4(MD_SCR(-2.02, 0.45,-0.50), 0.7), vec4(MD_SCR(-1.63,-0.60, 0.30), 1.5),
  vec4(MD_SCR(-1.47,-0.50,-1.00), 0.6), vec4(MD_SCR(-1.86, 0.10,-0.90),-1.1),
  vec4(MD_SCR( 1.50,-0.15, 0.60),-1.2), vec4(MD_SCR( 1.75, 0.35, 0.20), 0.9), vec4(MD_SCR( 2.02,-0.30, 0.70),-1.4),
  vec4(MD_SCR( 1.45,-0.60,-0.30), 0.8), vec4(MD_SCR( 1.60,-0.80, 0.20),-1.4), vec4(MD_SCR( 1.86,-0.10,-0.70),-0.7),
  vec4(MD_SCR( 2.10, 0.15, 0.30), 1.0), vec4(MD_SCR( 1.63,-0.40,-1.10), 0.6),
  vec4(MD_SCR(-1.56,-0.72, 0.05), 0.7), vec4(MD_SCR( 1.48,-0.80, 0.25),-0.6));
const vec4 MD_RC[MD_NRIB] = vec4[MD_NRIB](
  vec4(MD_SCR(-1.95, 0.58, 0.30),0.0), vec4(MD_SCR(-1.40, 1.00,-0.20),0.0), vec4(MD_SCR(-1.62, 0.22, 0.60),0.0),
  vec4(MD_SCR(-1.55, 0.10,-0.60),0.0), vec4(MD_SCR(-1.72, 1.28,-0.80),0.0), vec4(MD_SCR(-1.88,-0.22, 0.35),0.0),
  vec4(MD_SCR(-1.38, 0.42,-1.20),0.0), vec4(MD_SCR(-2.02, 0.92,-1.10),0.0),
  vec4(MD_SCR( 1.92, 0.36, 0.50),0.0), vec4(MD_SCR( 1.42, 0.84,-0.10),0.0), vec4(MD_SCR( 1.66, 0.02, 0.60),0.0),
  vec4(MD_SCR( 1.62,-0.12,-0.50),0.0), vec4(MD_SCR( 1.37,-0.40, 0.30),0.0), vec4(MD_SCR( 1.72, 0.58,-0.90),0.0),
  vec4(MD_SCR( 1.84, 0.96, 0.10),0.0), vec4(MD_SCR( 1.92, 0.70,-1.20),0.0),
  vec4(MD_SCR(-1.34,-0.56, 0.00),1.0), vec4(MD_SCR( 1.82,-0.48, 0.30),1.0));

#define MD_NPL 20
#define MD_D0 0
#define MD_D1 5      // daisies   MD_D0..MD_D1-1
#define MD_P1 8      // poppies   MD_D1..MD_P1-1
#define MD_C1 11     // cornflowers
#define MD_B1 14     // buttercups
#define MD_L1 16     // clovers
#define MD_T1 20     // seed heads
// head xyz + radius; axis + seed; root (below the frame)
const vec4 MD_PH[MD_NPL] = vec4[MD_NPL](
  vec4(MD_SCR(-1.62, 0.64,-0.20),0.245), vec4(MD_SCR(-1.70, 0.16,-1.20),0.20), vec4(MD_SCR( 1.68,-0.44, 0.20),0.205),
  vec4(MD_SCR( 1.76,-0.04,-1.10),0.21), vec4(MD_SCR(-1.84,-0.32, 0.40),0.22),
  vec4(MD_SCR( 1.76, 0.38,-0.30),0.225), vec4(MD_SCR(-1.82,-0.30, 0.35),0.205), vec4(MD_SCR( 1.90, 0.74,-1.00),0.175),
  vec4(MD_SCR(-1.50, 0.02,-0.70),0.18), vec4(MD_SCR( 1.49,-0.02,-0.40),0.19), vec4(MD_SCR(-1.96, 0.40,-0.40),0.17),
  vec4(MD_SCR(-1.52,-0.62, 0.10),0.13), vec4(MD_SCR( 1.76,-0.64, 0.30),0.14), vec4(MD_SCR( 1.48,-0.24,-1.00),0.12),
  vec4(MD_SCR(-1.76,-0.66, 0.15),0.15), vec4(MD_SCR( 1.53,-0.66, 0.30),0.15),
  vec4(MD_SCR(-1.38, 1.28,-0.30),0.00), vec4(MD_SCR( 1.64, 0.98,-0.60),0.00), vec4(MD_SCR(-1.90, 1.12,-0.80),0.00),
  vec4(MD_SCR( 1.92, 0.40, 0.00),0.00));
const vec4 MD_PA[MD_NPL] = vec4[MD_NPL](
  vec4(0.30, 0.45, 0.84, 1.3), vec4(-0.25,0.60, 0.75, 4.1), vec4(-0.35,0.50, 0.80, 7.9),
  vec4(-0.20,0.55, 0.80, 2.6), vec4(0.25, 0.65, 0.72, 5.5),
  vec4(-0.30,0.62, 0.72, 3.3), vec4(0.34, 0.66, 0.66, 6.2), vec4(-0.30,0.72, 0.62, 1.1),
  vec4(0.30, 0.60, 0.74, 2.2), vec4(-0.34,0.66, 0.66, 8.8), vec4(0.40, 0.55, 0.73, 3.7),
  vec4(0.22, 0.72, 0.66, 4.4), vec4(-0.28,0.68, 0.68, 9.1), vec4(-0.15,0.75, 0.64, 2.9),
  vec4(0.0,  1.0,  0.0,  1.7), vec4(0.0,  1.0,  0.0,  5.1),
  vec4(0.12, 1.0,  0.05, 1.0), vec4(0.45, 0.85, 0.10, 2.0), vec4(-0.25,0.95, 0.05, 3.0),
  vec4(0.55, 0.80, 0.15, 4.0));
const vec3 MD_PR[MD_NPL] = vec3[MD_NPL](
  MD_SCR(-1.56,MD_ROOT,-0.20), MD_SCR(-1.62,MD_ROOT,-1.20), MD_SCR( 1.72,MD_ROOT, 0.20),
  MD_SCR( 1.70,MD_ROOT,-1.10), MD_SCR(-1.76,MD_ROOT, 0.40),
  MD_SCR( 1.80,MD_ROOT,-0.30), MD_SCR(-1.74,MD_ROOT, 0.35), MD_SCR( 1.84,MD_ROOT,-1.00),
  MD_SCR(-1.54,MD_ROOT,-0.70), MD_SCR( 1.52,MD_ROOT,-0.40), MD_SCR(-1.86,MD_ROOT,-0.40),
  MD_SCR(-1.50,MD_ROOT, 0.10), MD_SCR( 1.72,MD_ROOT, 0.30), MD_SCR( 1.52,MD_ROOT,-1.00),
  MD_SCR(-1.70,MD_ROOT, 0.15), MD_SCR( 1.56,MD_ROOT, 0.30),
  MD_SCR(-1.50,MD_ROOT,-0.30), MD_SCR( 1.72,MD_ROOT,-0.60), MD_SCR(-2.00,MD_ROOT,-0.80),
  MD_SCR( 2.02,MD_ROOT, 0.00));

// nearest point on the quadratic bezier A,B,C by Newton from the chord projection
float mdBezNear(vec3 p, vec3 A, vec3 B, vec3 C, out vec3 v, out vec3 T){
  vec3 a = B - A, b = A - 2.0*B + C, ca = C - A;
  float t = clamp(dot(p - A, ca)/dot(ca, ca), 0.0, 1.0);
  for(int i=0;i<3;i++){
    vec3 r = A + t*(2.0*a + t*b) - p;
    vec3 d1 = 2.0*(a + t*b);
    float f = dot(r, d1), fp = dot(d1, d1) + 2.0*dot(r, b);
    t = clamp(t - f/max(fp, 1e-3), 0.0, 1.0);
  }
  v = p - (A + t*(2.0*a + t*b));
  T = normalize(a + t*b);
  return t;
}
float mdRibbon(vec3 p, vec4 A, vec4 B, vec4 C, out float u, out float wn){
  vec3 v, T;
  float t = mdBezNear(p, A.xyz, B.xyz, C.xyz, v, T);
  vec3 W = normalize(cross(C.xyz - A.xyz, B.xyz - A.xyz + vec3(0.0, 0.0, 1e-3)));
  vec3 N = cross(T, W);
  float ph = B.w*t, c = cos(ph), s = sin(ph);
  float vw = dot(v, W), vn = dot(v, N);
  float x = vw*c + vn*s, y = -vw*s + vn*c;
  float along = dot(v, T);
  float prof = (C.w > 0.5) ? pow(max(sin(3.14159*pow(t, 0.75)), 0.0), 0.8)
                           : pow(max(1.0 - t, 0.0), 0.62)*(0.72 + 0.28*smoothstep(0.0, 0.25, t));
  float hw = A.w*prof + 0.0015;
  // shallow V with a raised rib: the midrib shows up in the shading, not only in the paint
  float xr = x/(0.14*A.w);
  float yy = y - 0.30*abs(x) + 0.010*exp(-xr*xr);
  vec2 q = vec2(abs(x) - hw, abs(yy) - 0.0055);
  float d = min(max(q.x, q.y), 0.0) + length(max(q, 0.0));
  u = t; wn = x/hw;
  return length(vec2(max(d, 0.0), along)) + min(d, 0.0);
}
// local frame of a flower head: y along the axis
vec3 mdToHead(vec3 p, vec3 H, vec3 ax){
  vec3 q = p - H;
  vec3 ux = normalize(cross(ax, vec3(0.31, 0.17, 0.93)));
  vec3 uz = cross(ux, ax);
  return vec3(dot(q, ux), dot(q, ax), dot(q, uz));
}
float mdStrapPetal(float sx, float sy, float yy, float r0, float L, float w, float th){
  float cx = 0.5*(L + r0), hl = 0.5*(L - r0);
  vec2 e = abs(vec2(sx - cx, sy)) - vec2(max(hl - w, 0.0), 0.0);
  float d2 = length(max(e, 0.0)) + min(max(e.x, e.y), 0.0) - w;
  vec2 w2 = vec2(d2, abs(yy) - th);
  return min(max(w2.x, w2.y), 0.0) + length(max(w2, 0.0));
}

void mdHeroMap(vec3 p, bool shadow, inout vec2 res){
  // ribbons
  for(int i=0;i<MD_NRIB;i++){
    vec3 c = 0.5*(MD_RA[i].xyz + MD_RC[i].xyz);
    float bd = length(p - c) - 0.62*length(MD_RC[i].xyz - MD_RA[i].xyz) - 0.15;
    WHEN(bd < res.x){
      float u, wn;
      float d = mdRibbon(p, MD_RA[i], MD_RB[i], MD_RC[i], u, wn)*0.85;
      if(d < res.x){ res = vec2(d, M_RIBBON); mdU = u; mdV = wn; mdH = float(i)*0.137 + MD_RC[i].w; }
    }
  }
  // stems
  for(int i=0;i<MD_NPL;i++){
    vec3 H = MD_PH[i].xyz, A = MD_PR[i];
    vec3 ax = normalize(MD_PA[i].xyz);
    float bd = length(p - 0.5*(H + A)) - 0.58*length(H - A) - 0.08;
    WHEN(bd < res.x){
      vec3 v, T;
      vec3 B = H - ax*0.45*length(H - A);
      float t = mdBezNear(p, A, B, H, v, T);
      float d = (length(v) - mix(0.021, 0.013, t))*0.9;
      if(d < res.x){ res = vec2(d, M_STEM); mdU = t; mdH = MD_PA[i].w; }
    }
  }
  // ox-eye daisies
  for(int i=MD_D0;i<MD_D1;i++){
    float R = MD_PH[i].w;
    WHEN(shadow && length(p - MD_PH[i].xyz) - R*1.2 < res.x){
      vec3 q = mdToHead(p, MD_PH[i].xyz, normalize(MD_PA[i].xyz));
      res.x = min(res.x, sdEllipsoid(q + vec3(0.0, 0.30*R, 0.0), vec3(0.70*R, 0.05*R, 0.70*R)));
    }
    WHEN(!shadow && length(p - MD_PH[i].xyz) - R*1.2 < res.x){
      vec3 q = mdToHead(p, MD_PH[i].xyz, normalize(MD_PA[i].xyz));
      float seed = MD_PA[i].w;
      float r = length(q.xz), a = atan(q.z, q.x);
      const float N = 21.0;
      float sec = MD_TAU/N;
      float id = floor(a/sec + 0.5);
      float fa = a - id*sec;
      float hv = hash11(mod(id, N)*1.37 + seed);
      float sx = r*cos(fa), sy = r*sin(fa);
      float L = R*(0.92 + 0.11*hv), r0 = 0.18*R;
      float xn = clamp((sx - r0)/(L - r0), 0.0, 1.0);
      float w = R*(0.095 + 0.035*smoothstep(0.1, 0.8, xn));
      float yy = q.y + R*(0.20*xn*xn*(0.5 + hv) - 0.10*sin(xn*3.14) + 0.05*(hv - 0.5));
      float notch = 0.016*R*smoothstep(0.84, 1.0, xn)*(1.0 + cos(sy/w*7.5));
      float d = mdStrapPetal(sx + notch, sy, yy, r0, L, w, 0.018*R)*0.8;
      if(d < res.x){ res = vec2(d, M_DAISY); mdU = xn; mdV = sy/w; mdH = hv; }
      vec3 qd = q - vec3(0.0, 0.03*R, 0.0);
      float dd = sdEllipsoid(qd, vec3(0.30*R, 0.17*R, 0.30*R))*0.9;
      if(dd < res.x){ res = vec2(dd, M_DISC); mdQ = q/R; }
      float dc = sdEllipsoid(q + vec3(0.0, 0.07*R, 0.0), vec3(0.24*R, 0.09*R, 0.24*R));
      if(dc < res.x){ res = vec2(dc, M_CALYX); mdU = 0.0; }
    }
  }
  // poppies: four cupped petals, two outer, two inner
  for(int i=MD_D1;i<MD_P1;i++){
    float R = MD_PH[i].w;
    WHEN(shadow && length(p - MD_PH[i].xyz) - R*1.45 < res.x){
      vec3 q = mdToHead(p, MD_PH[i].xyz, normalize(MD_PA[i].xyz));
      res.x = min(res.x, sdEllipsoid(q + vec3(0.0, 0.16*R, 0.0), vec3(0.72*R, 0.07*R, 0.72*R)));
    }
    WHEN(!shadow && length(p - MD_PH[i].xyz) - R*1.45 < res.x){
      vec3 q = mdToHead(p, MD_PH[i].xyz, normalize(MD_PA[i].xyz));
      float seed = MD_PA[i].w;
      for(int k=0;k<4;k++){
        float fk = float(k), inner = mod(fk, 2.0);
        vec3 qr = q; qr.xz = rot(fk*1.5708 + 0.3 + 0.2*sin(seed + fk))*qr.xz;
        float ang = atan(qr.z, qr.x);
        float Rc = R*mix(1.16, 0.98, inner);
        vec3 cc = vec3(0.0, R*mix(1.10, 0.96, inner), 0.0);
        float ruff = 0.040*sin(ang*4.0 + seed + fk*2.0) + 0.018*sin(ang*11.0 + seed*2.0);
        float sph = length(qr - cc) - Rc*(1.0 + ruff);
        float shell = abs(sph) - 0.012*R;
        float lobe = sdEllipsoid(qr - vec3(R*0.56, R*mix(0.40, 0.52, inner), 0.0),
                                 vec3(R*mix(0.78, 0.68, inner), R*0.95, R*mix(1.02, 0.86, inner)));
        lobe += R*(0.035*sin(ang*7.0 + seed*3.0 + fk) + 0.012*sin(ang*19.0 + seed));
        float d = smax(shell, lobe, 0.03*R)*0.7;
        if(d < res.x){ res = vec2(d, M_POPPY); mdU = length(q.xz)/R; mdV = ang; mdH = fk + 10.0*hash11(seed); mdQ = qr/R; }
      }
      // seed capsule + flat stigma + ring of dark stamens
      float dc = sdEllipsoid(q - vec3(0.0, 0.19*R, 0.0), vec3(0.12*R, 0.17*R, 0.12*R));
      if(dc < res.x){ res = vec2(dc, M_POPC); mdU = 0.0; mdQ = q/R; }
      float ds = sdEllipsoid(q - vec3(0.0, 0.34*R, 0.0), vec3(0.15*R, 0.028*R, 0.15*R));
      if(ds < res.x){ res = vec2(ds, M_POPC); mdU = 1.0; mdQ = q/R; }
      float a = atan(q.z, q.x); float sec = MD_TAU/17.0;
      float fa = a - floor(a/sec + 0.5)*sec;
      float r = length(q.xz);
      float st = sdEllipsoid(vec3(r*cos(fa) - 0.27*R, q.y - 0.22*R, r*sin(fa)), vec3(0.042*R, 0.02*R, 0.022*R))*0.8;
      if(st < res.x){ res = vec2(st, M_POPC); mdU = 2.0; mdQ = q/R; }
    }
  }
  // cornflowers: toothed trumpets around a dark tuft, on a scaly bulb
  for(int i=MD_P1;i<MD_C1;i++){
    float R = MD_PH[i].w;
    WHEN(shadow && length(p - MD_PH[i].xyz) - R*1.3 < res.x){
      vec3 q = mdToHead(p, MD_PH[i].xyz, normalize(MD_PA[i].xyz));
      res.x = min(res.x, sdEllipsoid(q + vec3(0.0, 0.24*R, 0.0), vec3(0.45*R, 0.30*R, 0.45*R)));
    }
    WHEN(!shadow && length(p - MD_PH[i].xyz) - R*1.3 < res.x){
      vec3 q = mdToHead(p, MD_PH[i].xyz, normalize(MD_PA[i].xyz));
      float r = length(q.xz), a = atan(q.z, q.x) + MD_PA[i].w;
      float sec = MD_TAU/9.0;
      float id = floor(a/sec + 0.5);
      float fa = a - id*sec + 0.12*sin(id*2.3);
      float sx = r*cos(fa), sy = r*sin(fa);
      float xn = clamp(sx/R, 0.0, 1.0);
      float w = R*(0.05 + 0.27*xn*xn);
      float tooth = abs(fract(sy/w*2.0 + 0.5) - 0.5)*2.0;
      float Lt = R*(0.76 + 0.26*tooth);
      float yy = q.y - 0.30*sx + 0.32*xn*xn*R*0.6;
      vec2 e = vec2(max(max(abs(sy) - w, sx - Lt), 0.10*R - sx), abs(yy) - 0.022*R);
      float d = (min(max(e.x, e.y), 0.0) + length(max(e, 0.0)))*0.7;
      if(d < res.x){ res = vec2(d, M_CORN); mdU = xn; mdV = sy/w; mdH = 0.0; }
      float sec2 = MD_TAU/14.0;
      float fb = a*1.0 - floor(a/sec2 + 0.5)*sec2;
      float sx2 = r*cos(fb), sy2 = r*sin(fb);
      float dt = sdCapsule(vec3(sx2, q.y, sy2), vec3(0.03*R, 0.03*R, 0.0), vec3(0.33*R, 0.36*R, 0.0), 0.030*R);
      if(dt < res.x){ res = vec2(dt, M_CORN); mdU = length(q.xz)/R; mdV = 0.0; mdH = 1.0; }
      float db = sdEllipsoid(q + vec3(0.0, 0.24*R, 0.0), vec3(0.25*R, 0.36*R, 0.25*R));
      if(db < res.x){ res = vec2(db, M_CALYX); mdU = 1.0; mdQ = q/R; }
    }
  }
  // buttercups: five glossy cupped petals
  for(int i=MD_C1;i<MD_B1;i++){
    float R = MD_PH[i].w;
    WHEN(shadow && length(p - MD_PH[i].xyz) - R*1.3 < res.x){
      vec3 q = mdToHead(p, MD_PH[i].xyz, normalize(MD_PA[i].xyz));
      res.x = min(res.x, sdEllipsoid(q + vec3(0.0, 0.14*R, 0.0), vec3(0.60*R, 0.08*R, 0.60*R)));
    }
    WHEN(!shadow && length(p - MD_PH[i].xyz) - R*1.3 < res.x){
      vec3 q = mdToHead(p, MD_PH[i].xyz, normalize(MD_PA[i].xyz));
      float r = length(q.xz), a = atan(q.z, q.x) + MD_PA[i].w;
      float sec = MD_TAU/5.0;
      float fa = a - floor(a/sec + 0.5)*sec;
      float sx = r*cos(fa), sy = r*sin(fa);
      float sph = length(q - vec3(0.0, 0.92*R, 0.0)) - 1.02*R;
      float lobe = sdEllipsoid(vec3(sx - 0.56*R, q.y - 0.30*R, sy), vec3(0.56*R, 0.70*R, 0.56*R));
      float d = smax(abs(sph) - 0.016*R, lobe, 0.03*R)*0.75;
      if(d < res.x){ res = vec2(d, M_BUTTER); mdU = sx/R; mdV = sy/R; mdH = 0.0; }
      float dc = sdEllipsoid(q - vec3(0.0, 0.03*R, 0.0), vec3(0.20*R, 0.12*R, 0.20*R));
      if(dc < res.x){ res = vec2(dc, M_BUTTER); mdU = r/R; mdV = 0.0; mdH = 1.0; }
    }
  }
  // clover balls: rows of little florets on a sphere, drooping older ones at the bottom
  for(int i=MD_B1;i<MD_L1;i++){
    float R = MD_PH[i].w;
    WHEN(length(p - MD_PH[i].xyz) - R*1.3 < res.x){
      vec3 q = p - MD_PH[i].xyz;
      float rr = length(q);
      float lat = acos(clamp(q.y/max(rr, 1e-4), -1.0, 1.0));
      float lon = atan(q.z, q.x);
      float row = floor(lat*5.5);
      float fl = lat*5.5 - row;
      float nn = max(floor(sin((row + 0.5)/5.5)*16.0), 3.0);
      float cell = (lon/MD_TAU + 0.5*mod(row, 2.0)/nn)*nn;
      float fc = fract(cell) - 0.5;
      float rd = length(vec2(fc, (fl - 0.5)*0.75));
      float bump = 0.17*R*smoothstep(0.62, 0.05, rd);
      float d = (rr - R*0.88 - bump)*0.6;
      if(d < res.x){ res = vec2(d, M_CLOVER); mdU = lat/3.14159; mdV = rd; mdH = hash12(vec2(row, floor(cell))); }
    }
  }
  // grass seed heads: spike or drooping panicle at the end of the stem's arc
  for(int i=MD_L1;i<MD_T1;i++){
    vec3 H = MD_PH[i].xyz, A = MD_PR[i];
    vec3 ax = normalize(MD_PA[i].xyz);
    WHEN(length(p - H) - 0.62 < res.x){
      vec3 v, T;
      vec3 B = H - ax*0.45*length(H - A);
      float t = mdBezNear(p, A, B, H, v, T);
      float tt = smoothstep(0.78, 0.84, t);
      float ang = atan(dot(v, vec3(0.8, 0.0, 0.6)), dot(v, vec3(-0.6, 0.0, 0.8)));
      float prof = (1.0 - 0.75*smoothstep(0.88, 1.0, t));
      float bump = 0.009*tt*(0.5 + 0.5*sin(t*420.0 + ang*5.0))*(0.5 + 0.5*sin(ang*13.0 - t*160.0));
      float d = (length(v) - 0.036*tt*prof - bump - 0.006)*0.7;
      if(d < res.x){ res = vec2(d, tt > 0.5 ? M_SPIKE : M_STEM); mdU = t; mdV = ang; mdH = MD_PA[i].w; }
    }
  }
}

// ------------------------------------------------------------------ map
vec2 setMap(vec3 pw, bool shadow){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z);
  float cd = length(pw - uCamPos);
  vec2 res = vec2(1e3, M_GROUND);
  bool aoCall = mdShade && !shadow;
  float hT = terrainH(p.xz);
  float zone = grassZone(p.xz);
  float he = p.y - hT - zone*(grassH(p.xz) + 0.62);    // height above everything on the grid
  float E = 0.0006*cd;
  if(!aoCall){
    // Above the grid envelope a ray only needs to come down: a generous extra step keeps grazing rays
    // from running out near the horizon.  It vanishes within a few epsilons of a surface, so normals are exact.
    float add = shadow ? 0.0 : 0.022*cd*smoothstep(E, 3.0*E, he);
    float dG = (p.y - hT)*0.94;
    res.x = (he > 0.0) ? min(he*0.94 + 0.14, dG) + add : dG;     // the envelope itself must never pass the hit test
    WHEN(he < 0.0){
      res = opU(res, sdBlades(p, hT, zone, cd, shadow));
      if(!shadow) sdMidFlowers(p, hT, zone, cd, shadow, res);
      if(!shadow) res.x += 0.0030*cd*smoothstep(3.0*E, 6.0*E, res.x);    // grazing rays through sparse tips must get through
    }
    float aboveT = p.y - hT;
    float tb = max(p.z + 62.0, aboveT - 14.0);
    WHEN(tb < res.x){
      float dt = sdTrees(p, aboveT, shadow);
      if(dt < res.x) res = vec2(dt, M_TREE);
    }
  }
  float hb = max(abs(pw.z + 0.35) - 1.75, max(1.05 - abs(pw.x), pw.y - 1.9));
  WHEN(hb < res.x){ mdHeroMap(pw, shadow, res); }
  return res;
}

// ------------------------------------------------------------------ sky
#ifdef PASS_BG
#define MD_NCLD 9
// picture-space cumulus: centre x, base, half width, height (units: 6*azimuth rad, 6*sin elevation)
const vec4 MD_CLD[MD_NCLD] = vec4[MD_NCLD](
  vec4(-1.40, 0.70, 0.80, 0.36), vec4(-0.40, 0.98, 0.55, 0.22), vec4( 0.92, 0.60, 0.95, 0.38),
  vec4( 1.80, 1.02, 0.55, 0.20), vec4(-1.95, 1.12, 0.50, 0.15), vec4( 0.15, 0.56, 0.50, 0.13),
  vec4( 1.35, 0.46, 0.40, 0.08), vec4(-0.95, 0.50, 0.40, 0.08), vec4( 0.55, 1.20, 0.60, 0.10));
float cloudShape(vec2 c, float f, float g){
  float s = 0.0;
  for(int i=0;i<MD_NCLD;i++){
    vec4 k = MD_CLD[i];
    float xx = (c.x - k.x)/k.z, yy = (c.y - k.y)/k.w;
    float top = (1.0 - xx*xx)*(1.0 + 0.25*sin(xx*3.0 + k.x));
    float e = top - yy + 1.5*(f - 0.5) + 0.30*(g - 0.5);
    float base = smoothstep(-0.25, 0.15, yy + 0.5*(g - 0.5));
    s = max(s, smoothstep(0.0, 0.20, e)*base);
  }
  return s;
}
#endif
vec3 setSky(vec3 rd){
  float y = max(rd.y, 0.0);
  vec3 col = mix(vec3(0.86,0.95,1.06), vec3(0.20,0.43,0.92), 1.0 - exp(-y*8.5));
#ifdef PASS_BG
  vec2 c = vec2(atan(rd.x, -rd.z), rd.y)*6.0;
  float f = fbm2_3(c*vec2(1.5, 2.3) + vec2(3.1, 1.7));                        // big billows
  f = 0.5 + 0.55*(f - 0.5) + 0.34*(abs(2.0*vnoise2(c*vec2(3.6, 5.2) + 2.0) - 1.0) - 0.42);
  float g = fbm2_3(c*vec2(6.3, 8.0) + vec2(7.0, 2.0));
  float s = cloudShape(c, f, g);
  float l = cloudShape(c + vec2(0.07, 0.10), f, g);          // toward the sun: thinner there means lit
  float lit = clamp(1.0 - 0.9*l + 0.35*(1.0 - s), 0.0, 1.0);
  vec3 cc = mix(vec3(0.66,0.74,0.92), vec3(1.75,1.70,1.58), lit*0.75 + 0.25);
  col = mix(col, cc, s*smoothstep(0.0, 0.05, rd.y));
#else
  col += vec3(0.20)*smoothstep(0.3, 1.0, sin(rd.x*4.0 + 1.0)*sin(rd.z*3.0 + 2.0) + rd.y);
#endif
  return col;
}

// ------------------------------------------------------------------ paint
float cloudShadow(vec2 xz){ return smoothstep(0.40, 0.60, fbm2_3(xz*vec2(0.017, 0.012) + vec2(2.0, 7.0))); }
vec3 flowerColour(float sp, float h){
  if(sp < 0.5) return vec3(0.92,0.90,0.82);
  if(sp < 1.5) return vec3(0.95,0.72,0.05);
  if(sp < 2.5) return vec3(0.62,0.012,0.008);
  if(sp < 3.5) return mix(vec3(0.10,0.20,0.85), vec3(0.42,0.16,0.78), h);
  if(sp < 4.5) return vec3(0.88,0.42,0.62);
  return vec3(0.30,0.29,0.10);
}
void paintGround(vec3 p, float t, inout Surf s){
  vec2 xz = p.xz;
  float n1 = fbm2(xz*0.021 + 3.0);
  float n2 = fbm2(xz*vec2(0.09, 0.05) + 9.0);
  vec3 a = mix(vec3(0.075,0.17,0.028), vec3(0.19,0.31,0.05), smoothstep(0.32, 0.68, n1));
  a = mix(a, vec3(0.38,0.42,0.09), 0.75*smoothstep(0.50, 0.70, n2));
  // flower speckles: anisotropic so they read as dots after the grazing foreshortening
  for(int i=0;i<3;i++){
    float fi = float(i), sc = 1.0 + fi*0.7;
    vec2 u = xz*vec2(0.75, 0.13)*sc + fi*11.3;
    vec2 ci = floor(u);
    vec3 h = hash33(vec3(ci, 31.0 + fi));
    vec2 b = (ci + 0.5 + (h.xy - 0.5)*0.7)/(vec2(0.75, 0.13)*sc);
    vec2 f = fract(u) - 0.5 - (h.xy - 0.5)*0.7;
    float keep = step(h.z, 1.15*flowerDensity(b*0.35));
    float sp = flowerSpecies(b*0.35, h.x - 0.5);
    a = mix(a, flowerColour(sp, h.y), keep*smoothstep(0.34, 0.14, length(f)));
  }
  a *= 0.78 + 0.34*vnoise2(xz*vec2(1.3, 0.4));
  float z = grassZone(xz);
  a = mix(a, vec3(0.05,0.075,0.02), z);                     // soil and old stems under the blades
  a *= mix(0.62, 1.0, cloudShadow(xz));
  a = mix(a, vec3(0.20,0.33,0.46), 0.55*smoothstep(-165.0, -225.0, xz.y));
  s.alb = a;
  s.wrap = 0.25; s.spec = 0.0; s.occ = mix(1.0, 0.45, z);
}
// thin leaves pass light: shade the far side as lit, tinted yellower and a bit darker
void thinLeaf(vec3 n, inout Surf s, float T){
  float nl = dot(n, uSunDir);
  if(nl < 0.0){
    s.n = normalize(n - 2.0*nl*uSunDir);
    s.alb *= vec3(1.0, 1.0, 0.55)*T;
  }
}
void setSurface(vec3 pw, vec3 rd, float t, float m, vec3 n, inout Surf s){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z);
  vec2 stash = setMap(pw, false);        // re-run at the true hit: fills the stash for this very point
  m = stash.y;
  bool outOfSteps = stash.x > 0.012*t;
  mdShade = true;
  WHEN(m == M_GROUND){
    paintGround(p, t, s);
    if(outOfSteps){ s.alb = vec3(0.17,0.30,0.045); s.occ = 0.8; }
  }
  WHEN(m == M_BLADE){
    float patchA = fbm2(p.xz*0.085 + 5.0);
    float patchB = fbm2(p.xz*0.24 + 1.0);
    vec3 c = mix(vec3(0.045,0.150,0.018), vec3(0.11,0.27,0.026), mdBH);
    c = mix(c, vec3(0.25,0.36,0.040), smoothstep(0.42, 0.66, patchA)*(0.5 + 0.7*mdBH));   // sunny yellow patches
    c = mix(c, vec3(0.035,0.125,0.028), 0.9*smoothstep(0.54, 0.34, patchB));              // deep green patches
    c = mix(c*0.55, c, smoothstep(0.1, 0.65, mdYn));
    c = mix(c, vec3(0.40,0.42,0.09), 0.50*smoothstep(0.72, 1.0, mdYn)*(0.4 + 0.6*mdBH));    // dry tips
    c = mix(c, c*1.35 + vec3(0.02,0.03,0.0), smoothstep(0.22, 0.05, abs(mdWn)));           // midrib
    s.alb = 0.88*c*mix(0.56, 1.0, cloudShadow(p.xz + vec2(40.0, 30.0)));
    s.wrap = 0.22; s.spec = 0.18; s.shin = 30.0; s.trans = 0.5; s.transCol = c;
    s.occ = mix(0.20, 0.82, smoothstep(0.15, 0.90, mdYn));
    thinLeaf(n, s, 0.85);
  }
  WHEN(m == M_MIDSTEM){
    s.alb = vec3(0.14,0.27,0.05); s.wrap = 0.3; s.occ = 0.75;
  }
  WHEN(m == M_MIDFL){
    float sp = mdFS, r = mdFV.x, y = mdFV.y, az = mdFV.z;
    vec3 c = flowerColour(sp, mdFH);
    if(sp == 0.0){
      c *= 0.90 + 0.10*cos(az*13.0);
      c = mix(c, vec3(0.95,0.66,0.05), smoothstep(0.40, 0.30, r));
    } else if(sp == 1.0){
      c = mix(c, vec3(0.65,0.55,0.05), smoothstep(0.30, 0.18, r));
    } else if(sp == 2.0){
      c = mix(c, vec3(0.03,0.02,0.03), smoothstep(0.36, 0.22, r)*smoothstep(-0.2, 0.2, y));
    } else if(sp == 3.0){
      c = mix(c, vec3(0.20,0.05,0.42), smoothstep(0.42, 0.25, r));
    } else if(sp == 4.0){
      c = mix(c, vec3(0.98,0.80,0.86), 0.6*smoothstep(0.1, 0.7, y));
    } else {
      c *= 0.8 + 0.3*sin(y*9.0 + az*2.0);
    }
    s.alb = c; s.wrap = 0.45; s.spec = 0.08; s.shin = 24.0;
    thinLeaf(n, s, 0.9);
  }
  WHEN(m == M_TREE){
    float n1 = fbm3(p*0.21 + 3.0), n2 = vnoise3(p*1.1);
    vec3 c = mix(vec3(0.030,0.085,0.030), vec3(0.11,0.20,0.055), smoothstep(0.30, 0.70, n1));
    c = mix(c, vec3(0.20,0.28,0.08), 0.45*smoothstep(0.55, 0.8, n2));
    c *= mix(0.55, 1.0, smoothstep(-0.6, 0.5, n.y));                  // shaded skirts
    c *= mix(0.72, 1.0, cloudShadow(p.xz));
    c = mix(c, vec3(0.17,0.27,0.31), 0.50*smoothstep(-90.0, -165.0, p.z));   // treeline stays pale and blue
    c = mix(c, vec3(0.12,0.21,0.30), 0.60*smoothstep(-175.0, -225.0, p.z));
    s.alb = c*(0.7 + 0.6*n2);
    s.n = normalize(n + 0.45*(vec3(vnoise3(p*0.9), vnoise3(p*0.9 + 5.0), vnoise3(p*0.9 + 9.0)) - 0.5));
    s.wrap = 0.30; s.occ = 0.55 + 0.45*n2;
  }
  WHEN(m == M_RIBBON){
    float w = abs(mdV), u = mdU, hv = fract(mdH*3.71);
    vec3 c = mix(vec3(0.10,0.27,0.035), vec3(0.24,0.43,0.055), hv);
    c = mix(c, vec3(0.42,0.50,0.10), 0.7*smoothstep(0.55, 1.0, u));
    c *= 0.90 + 0.10*sin(w*22.0)*sin(w*22.0);                       // parallel veins
    c = mix(c, c*1.5 + vec3(0.05,0.06,0.01), smoothstep(0.13, 0.04, w));   // pale midrib
    c = mix(c, c*0.82, smoothstep(0.85, 1.0, w));
    s.alb = c;
    s.wrap = 0.35; s.spec = 0.32; s.shin = 36.0; s.trans = 0.6; s.transCol = c;
    thinLeaf(n, s, 0.9);
  }
  WHEN(m == M_STEM){
    s.alb = mix(vec3(0.13,0.26,0.05), vec3(0.22,0.38,0.07), mdU);
    s.wrap = 0.3; s.spec = 0.12; s.shin = 24.0;
  }
  WHEN(m == M_DAISY){
    float u = mdU, v = mdV;
    vec3 c = vec3(0.93,0.92,0.85);
    c = mix(c, vec3(0.86,0.86,0.55), smoothstep(0.22, 0.02, u));    // greenish root of the petal
    c *= 0.93 + 0.07*cos(v*9.0 + mdH*4.0);                           // long ribs
    s.alb = c;
    s.n = normalize(n + 0.10*cross(n, vec3(0.0, 0.0, 1.0))*sin(v*9.0));
    s.wrap = 0.5; s.spec = 0.05; s.shin = 18.0; s.trans = 0.6; s.transCol = vec3(1.0,0.96,0.80);
    thinLeaf(n, s, 0.92);
  }
  WHEN(m == M_DISC){
    vec2 q = mdQ.xz/0.30;
    float r = length(q), a = atan(q.y, q.x);
    float K = 15.0*sqrt(r + 0.02);
    float f = cos(13.0*a + 6.0*K)*cos(21.0*a - 6.0*K);
    float dots = smoothstep(0.15, 0.75, f);
    vec3 c = mix(vec3(0.62,0.33,0.015), vec3(1.0,0.76,0.08), dots);
    c = mix(c, vec3(0.70,0.62,0.10), smoothstep(0.30, 0.05, r));
    s.alb = c;
    vec2 tilt = vec2(cos(a), sin(a))*(dots - 0.5)*0.35;
    s.n = normalize(n + vec3(tilt.x, 0.0, tilt.y));
    s.wrap = 0.3; s.spec = 0.25; s.shin = 30.0; s.occ = 0.75 + 0.25*dots;
  }
  WHEN(m == M_CALYX){
    vec3 c = vec3(0.16,0.30,0.06);
    if(mdU > 0.5){                                                    // cornflower bulb: overlapping scales
      vec2 q = vec2(atan(mdQ.z, mdQ.x)*2.0, mdQ.y*14.0);
      float sc = fract(q.y + 0.5*floor(q.x*1.2));
      c = mix(vec3(0.14,0.24,0.06), vec3(0.30,0.24,0.10), smoothstep(0.55, 0.95, sc));
    }
    s.alb = c; s.wrap = 0.3; s.spec = 0.1;
  }
  WHEN(m == M_POPPY){
    float r = mdU, ang = mdV;
    vec3 c = mix(vec3(0.29,0.022,0.012), vec3(0.42,0.052,0.020), smoothstep(0.30, 1.0, r));
    c *= 0.93 + 0.07*sin(ang*38.0 + 2.0*sin(ang*7.0));              // fine pleats
    float blotch = smoothstep(0.40, 0.26, r + 0.05*sin(ang*9.0 + mdH));
    c = mix(c, vec3(0.018,0.010,0.022), blotch);
    s.alb = c;
    s.wrap = 0.55; s.spec = 0.10; s.shin = 22.0; s.trans = 0.8; s.transCol = vec3(1.0,0.06,0.02);
    thinLeaf(n, s, 0.95);
  }
  WHEN(m == M_POPC){
    vec3 c = vec3(0.022,0.016,0.03);                                 // stamens
    if(mdU < 0.5) c = vec3(0.34,0.42,0.14);                           // capsule
    else if(mdU < 1.5){
      float a = atan(mdQ.z, mdQ.x);
      c = mix(vec3(0.38,0.44,0.16), vec3(0.05,0.04,0.03), smoothstep(0.3, 0.8, cos(a*11.0)));
    }
    s.alb = c; s.spec = 0.2; s.shin = 30.0;
  }
  WHEN(m == M_CORN){
    vec3 c = mix(vec3(0.05,0.14,0.75), vec3(0.11,0.30,0.98), smoothstep(0.2, 0.95, mdU));
    c *= 0.88 + 0.12*cos(mdV*8.0);
    if(mdH > 0.5) c = mix(vec3(0.20,0.03,0.42), vec3(0.42,0.08,0.60), mdU*2.0);
    s.alb = c; s.wrap = 0.45; s.spec = 0.08; s.shin = 20.0; s.trans = 0.6; s.transCol = c;
    thinLeaf(n, s, 0.95);
  }
  WHEN(m == M_BUTTER){
    vec3 c = vec3(0.98,0.70,0.03);
    c = mix(c, vec3(1.0,0.86,0.12), smoothstep(0.3, 0.9, mdU));
    if(mdH > 0.5) c = mix(vec3(0.58,0.62,0.08), vec3(0.95,0.74,0.08), smoothstep(0.05, 0.20, mdU));
    s.alb = c; s.wrap = 0.35; s.spec = 0.8; s.shin = 70.0; s.trans = 0.6; s.transCol = c;
    thinLeaf(n, s, 0.95);
  }
  WHEN(m == M_CLOVER){
    vec3 c = mix(vec3(0.80,0.30,0.52), vec3(0.97,0.74,0.84), smoothstep(0.1, 0.55, 0.6 - mdV + 0.25*mdH));
    c = mix(c, vec3(0.52,0.28,0.26), smoothstep(0.70, 0.95, mdU));   // old brown florets below
    c *= mix(0.55, 1.0, smoothstep(0.62, 0.20, mdV));
    s.alb = c; s.wrap = 0.45; s.spec = 0.1;
  }
  WHEN(m == M_SPIKE){
    float a = mdV;
    vec3 c = mix(vec3(0.20,0.30,0.08), vec3(0.58,0.56,0.24), 0.5 + 0.5*sin(mdU*420.0 + a*5.0));
    c = mix(c, vec3(0.30,0.16,0.26), 0.35*smoothstep(0.2, 0.8, sin(mdU*90.0 + a*13.0)));
    s.alb = c; s.wrap = 0.4; s.spec = 0.1;
  }
}

// ------------------------------------------------------------------ per-frame, cheap
vec3 setAmbient(vec3 pw, vec3 n){
  vec3 sky = SP_AMB_SKY*(0.57 + 0.43*n.y);
  vec3 bounce = SP_BOUNCE*0.95*clamp(0.55 - 0.45*n.y, 0.0, 1.0);
  return sky + bounce;
}
// soft slow cloud-shadow drift, nearly always bright
float setCookie(vec3 pw){
  vec2 q = pw.xz*0.06 + vec2(uTime*0.016, uTime*0.007);
  return 0.70 + 0.30*smoothstep(0.25, 0.70, vnoise2(q));
}
float setVol(vec3 pw){ return 0.02; }
float setFog(vec3 pw){
  float h = pw.y - G0;
  float s = smoothstep(-45.0, -205.0, pw.z);
  return 0.0125*s*s*smoothstep(32.0, 14.0, h);
}
vec3 setFogAmb(vec3 rd){ return vec3(0.76,0.87,1.04); }

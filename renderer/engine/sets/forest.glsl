// ---------------------------------------------------------------------------
//  engine/sets/forest.glsl — parameterizable forest clearing.
//  Bird's clearing, rescaled and recoloured; every look lives in `set_params` of the animal
//  JSON, which arrives as `#define SP_<NAME> ...` (see engine/render.py) and overrides the defaults.
//
//  SET API (every file in engine/sets must provide these):
//    void  setInit();                         per-fragment setup
//    vec2  setMap(vec3 p, bool shadow);       distance, material
//    vec3  setSky(vec3 rd);                   sky radiance
//    void  setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s);
//    vec3  setAmbient(vec3 p, vec3 n);        sky + bounce + fill irradiance
//    float setCookie(vec3 p);                 0..1 dappled light at a world point — TIME VARYING, CHEAP
//    float setVol(vec3 p);                    0..1 light reaching the air (god-ray shafts) — CHEAP
//    float setFog(vec3 p);                    fog density — CHEAP
// ---------------------------------------------------------------------------
#ifndef SP_SCALE
#define SP_SCALE 5.0                                // world units per bird-scene unit
#endif
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-2.7)
#endif
#ifndef SP_CANOPY_H
#define SP_CANOPY_H 2.6                             // canopy light plane (set units)
#endif
#ifndef SP_CANOPY_A
#define SP_CANOPY_A vec3(0.44,0.13,0.03)            // shadow side of the leaves
#endif
#ifndef SP_CANOPY_B
#define SP_CANOPY_B vec3(0.95,0.52,0.08)            // lit side of the leaves
#endif
#ifndef SP_GROUND_A
#define SP_GROUND_A vec3(0.14,0.065,0.025)            // ground base
#endif
#ifndef SP_GROUND_B
#define SP_GROUND_B vec3(0.40,0.145,0.03)            // ground highlight (leaf litter)
#endif
#ifndef SP_GROUND_C
#define SP_GROUND_C vec3(0.09,0.12,0.03)            // moss / grass patches
#endif
#ifndef SP_LEAF_A
#define SP_LEAF_A vec3(0.50,0.13,0.025)              // fallen-leaf colour 1
#endif
#ifndef SP_LEAF_B
#define SP_LEAF_B vec3(0.60,0.27,0.04)              // fallen-leaf colour 2
#endif
#ifndef SP_GRASS_A
#define SP_GRASS_A vec3(0.20,0.11,0.03)
#endif
#ifndef SP_GRASS_B
#define SP_GRASS_B vec3(0.46,0.24,0.06)
#endif
#ifndef SP_BARK_A
#define SP_BARK_A vec3(0.08,0.05,0.035)
#endif
#ifndef SP_BARK_B
#define SP_BARK_B vec3(0.22,0.145,0.09)
#endif
#ifndef SP_MOSS
#define SP_MOSS vec3(0.12,0.13,0.035)
#endif
#ifndef SP_SKY_LOW
#define SP_SKY_LOW vec3(1.00,0.76,0.44)*1.05         // horizon glow
#endif
#ifndef SP_SKY_HIGH
#define SP_SKY_HIGH vec3(0.95,0.82,0.62)*0.9
#endif
#ifndef SP_FOG_LO
#define SP_FOG_LO vec3(0.46,0.25,0.10)
#endif
#ifndef SP_FOG_HI
#define SP_FOG_HI vec3(1.00,0.68,0.32)
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#ifndef SP_AMB_SKY
#define SP_AMB_SKY vec3(0.52,0.54,0.58)              // open-sky ambient
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.46,0.30,0.12)              // light bounced off the ground
#endif
#ifndef SP_TREES
#define SP_TREES 1.0                                // 0 = no explicit big trunks
#endif
#ifndef SP_PINE
#define SP_PINE 0.0                                 // 1 = slim straight trunks
#endif
#ifndef SP_GRASS_AMT
#define SP_GRASS_AMT 0.35
#endif
#ifndef SP_CLEAR_C
#define SP_CLEAR_C vec2(0.6,-2.0)                    // clearing centre (set units)
#endif
#ifndef SP_CLEAR_R
#define SP_CLEAR_R 3.4
#endif
#ifndef SP_BUSHES
#define SP_BUSHES 0.55                              // 0 = no understory
#endif
#ifndef SP_BUSH_C
#define SP_BUSH_C vec3(0.46,0.07,0.03)              // third leaf colour (crimson)
#endif
#ifndef SP_DAPPLE_FREQ
#define SP_DAPPLE_FREQ 1.0                          // scale of the dapples on the character
#endif

const float SC = SP_SCALE;
const float G0 = SP_GROUND_Y;
const vec2  CLR = SP_CLEAR_C;
const float CLR_R = SP_CLEAR_R;
const float CANH = SP_CANOPY_H;

#define M_GROUND 1.0
#define M_BARK   2.0
#define M_CANOPY 9.0
#define M_LEAF   12.0
#define M_GRASS  17.0
#define M_STEM   11.0
#define M_BUSH   13.0

#define NT 11
const vec3 TRK[NT] = vec3[NT](
  vec3(-2.05,-7.20,0.30),   // frame left of the head
  vec3( 2.40,-8.40,0.28),   // frame right
  vec3(-4.90,-11.0,0.55),
  vec3( 5.30,-12.0,0.50),
  vec3( 0.55,-14.0,0.30),
  vec3( 3.60,-18.0,0.40),
  vec3(-3.00,-17.0,0.42),
  vec3(-8.40,-8.10,0.60),
  vec3( 8.60,-9.00,0.62),
  vec3(-10.5,-15.5,0.70),
  vec3(11.00,-16.0,0.70)
);

vec2  HOLE;
vec2  gWind;
float gGrassH;

void setInit(){
  HOLE = uSunDir.xz*((CANH - 0.0)/uSunDir.y);
  float T = uTime;
  gWind = 0.10*vec2(sin(T*0.80), cos(T*0.65)) + 0.035*vec2(sin(T*2.1+1.0), sin(T*1.7));
}

float groundHs(vec2 xz){       // set-space height above G0
  float h = 0.09*sin(xz.x*0.45+0.3)*sin(xz.y*0.35+1.1);
  h += 0.12*(vnoise2(xz*0.55)-0.5);
  return h;
}
float canopyPresence(vec2 c){
  float dc = length((c - CLR)*vec2(0.80,1.0));
  float pr = smoothstep(CLR_R, CLR_R+3.5, dc);
  pr *= smoothstep(0.30, 1.05, length(c - HOLE));
  pr *= smoothstep(-40.0, -36.0, c.y);
  return pr;
}
float sdTree(vec3 p, vec3 tr, float seed, float best){
  vec3 q = p - vec3(tr.x, 0.0, tr.y);
  float r0 = tr.z*mix(1.0, 0.55, SP_PINE);
  float l0 = length(q.xz);
  float bound = r0*(1.22 + 1.25*exp(-max(q.y,0.0)/(0.55*r0)));
  if((l0 - bound)*0.7 > best) return 1e9;
  float y = q.y;
  q.x += 0.14*r0*sin(y*0.8 + seed)*(1.0 - SP_PINE*0.7);
  q.z += 0.10*r0*sin(y*0.6 + seed*1.7)*(1.0 - SP_PINE*0.7);
  float a = atan(q.z, q.x);
  float l = length(q.xz);
  float yy = max(y, 0.0);
  float rad = r0*(1.0 + 0.6*exp(-yy/(0.55*r0)));
  rad *= 1.0 + 0.40*exp(-yy/(0.30*r0))*pow(0.5+0.5*cos(a*5.0+seed*3.0), 2.0);
  rad += r0*0.03*sin(a*13.0 + 2.0*sin(y*2.5+seed));
  float d = (l - rad)*0.7;
  return max(d, y - 7.0);
}
float sdForestTrunks(vec3 p){
  if(p.y > 5.1) return p.y - 5.0;
  vec2 g = p.xz/3.0 - 0.5; vec2 c0 = floor(g);
  float d = 1.4;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, 3.0));
    vec2 cpos = (ci + 0.5 + (h.xy-0.5)*0.40)*3.0;
    float pres = canopyPresence(cpos)*smoothstep(5.0, 6.0, length(cpos - vec2(0.0,-1.0)));
    if(h.z > pres*0.85) continue;
    float r = (0.30 + 0.22*h.y)*mix(1.0,0.55,SP_PINE);
    vec3 q = p - vec3(cpos.x, 0.0, cpos.y);
    q.x += 0.15*sin(q.y*0.5 + h.x*6.0);
    float rad = r*(1.0 + 0.6*exp(-max(q.y,0.0)/(0.6*r)));
    float dt = (length(q.xz) - rad)*0.8;
    d = min(d, max(dt, q.y - 5.0));
  }
  return d;
}
vec2 sdCanopy(vec3 p){
  if(p.y < 0.55) return vec2(0.70 - p.y, M_CANOPY);
  if(p.y > 6.6)  return vec2(p.y - 6.5, M_CANOPY);
  vec2 g = p.xz/2.2;
  vec2 ci = floor(g);
  float d = 1e9;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 cc = ci + vec2(float(i), float(j));
    vec3 h = hash33(vec3(cc, 7.0));
    vec2 cpos = (cc + 0.2 + 0.6*h.xy)*2.2;
    float pres = canopyPresence(cpos);
    if(h.z > pres*1.25) continue;
    float r = 0.90 + 0.70*h.x;
    vec3 c = vec3(cpos.x, 2.05 + 1.3*h.y, cpos.y);
    float e = sdEllipsoid(p - c, vec3(r, r*0.78, r));
    d = smin(d, e, 0.5);
  }
  if(d < 0.4) d += 0.55*(vnoise3(p*1.5) - 0.5) + 0.22*(vnoise3(p*4.0) - 0.5);    // leafy, lumpy, no clean blobs
  return vec2(d*0.8, M_CANOPY);
}
// grass / bracken tufts: a sparse band around the clearing
vec2 sdGrass(vec3 p, float gh, float best){
  float hy = p.y - gh;
  if(hy > 0.20 || hy - 0.2 > best) return vec2(1e9, M_GRASS);
  const float CS = 0.062;
  vec2 g = p.xz/CS - 0.5; vec2 c0 = floor(g);
  float d = 1e9; float hf = 0.0;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, 11.0));
    vec2 base = (ci + 0.5 + (h.xy - 0.5)*0.7)*CS;
    float pres = SP_GRASS_AMT*(0.30 + 0.70*smoothstep(0.40, 0.68, vnoise2(base*1.6)));
    if(h.z > pres) continue;
    float H = mix(0.040, 0.085, h.x);
    for(int k=0;k<4;k++){
      float a = h.y*6.283 + float(k)*1.7 + 0.6*float(k*k);
      vec2 lean = vec2(cos(a), sin(a));
      float bh = H*(0.65 + 0.35*fract(h.z*7.0 + float(k)*0.37));
      vec3 b0 = vec3(base.x + lean.x*0.006*float(k), gh - 0.01, base.y + lean.y*0.006*float(k));
      vec3 b1 = b0 + vec3(lean.x*bh*0.18, bh*0.55, lean.y*bh*0.18);
      vec3 b2 = b0 + vec3(lean.x*bh*0.55, bh, lean.y*bh*0.55);
      float db = min(sdTaper(p, b0, b1, 0.0034, 0.0024), sdTaper(p, b1, b2, 0.0024, 0.0004));
      if(db < d){ d = db; hf = clamp((p.y - gh)/bh, 0.0, 1.0); }
    }
  }
  if(d < best) gGrassH = hf;
  return vec2(d*0.9, M_GRASS);
}

// autumn understory: clumpy shrubs between the trunks (bake only, so noise is cheap here)
float sdBushes(vec3 p){
  if(p.y > 1.0) return p.y - 0.8;
  vec2 g = p.xz/0.8;
  vec2 ci = floor(g);
  float d = 1e9;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 cc = ci + vec2(float(i), float(j));
    vec3 h = hash33(vec3(cc, 23.0));
    vec2 cpos = (cc + 0.2 + 0.6*h.xy)*0.8;
    float back = smoothstep(-4.5, -7.0, cpos.y);          // only behind the subject
    if(h.z > back*SP_BUSHES) continue;
    float r = 0.16 + 0.24*h.x;
    vec3 c = vec3(cpos.x, r*0.55, cpos.y);
    d = smin(d, sdEllipsoid(p - c, vec3(r*1.2, r*0.9, r*1.1)), 0.14);
  }
  if(d < 0.2) d += 0.10*(vnoise3(p*6.0) - 0.5) + 0.05*(vnoise3(p*15.0) - 0.5);
  return d*0.8;
}

vec2 setMap(vec3 pw, bool shadow){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  float gh = groundHs(p.xz);
  float dg = (p.y - gh)*0.9;
  vec2 res = vec2(dg, M_GROUND);
  float dtr = 1e9;
  if(SP_TREES > 0.5){
    for(int i=0;i<NT;i++) dtr = min(dtr, sdTree(p, TRK[i], float(i)*1.37 + 0.5, res.x + 0.1));
  }
  if(dtr < res.x + 0.1){
    float bl = smin(res.x, dtr, 0.12);
    res = vec2(bl, dtr < res.x ? M_BARK : res.y);
  }
  if(!shadow || p.y - gh < 0.2) res = opU(res, sdGrass(p, gh, res.x));
  if(SP_BUSHES > 0.0) res = opU(res, vec2(sdBushes(p), M_BUSH));
  if(!shadow){
    res = opU(res, vec2(sdForestTrunks(p), M_BARK));
    res = opU(res, sdCanopy(p));
  }
  res.x *= SC;
  return res;
}

vec3 setSky(vec3 rd){
  float sd = max(dot(rd, uSunDir), 0.0);
  vec3 col = mix(SP_SKY_LOW, SP_SKY_HIGH, smoothstep(0.02, 0.60, rd.y));
  col += vec3(1.0,0.80,0.5)*(1.2*pow(sd,5.0) + 8.0*pow(sd,160.0));
  return col;
}

void setSurface(vec3 pw, vec3 rd, float t, float m, vec3 n, inout Surf s){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  vec2 xz = p.xz;
  if(m == M_GROUND){
    float n1 = fbm2(xz*1.3);
    float n2 = vnoise2(xz*9.0);
    vec3 a = mix(SP_GROUND_A, SP_GROUND_B, smoothstep(0.3,0.75,n1));
    a = mix(a, SP_GROUND_C, 0.75*smoothstep(0.52,0.72,fbm2(xz*0.6+9.0)));
    a *= 0.80 + 0.35*n2;
    // leaf litter, three scales
    for(int i=0;i<3;i++){
      float fi = float(i);
      float sc = 5.0 + fi*4.0;
      vec2 u = xz*sc + fi*13.1;
      vec2 lc = floor(u); vec2 lh = hash22(lc + fi);
      vec2 lf = fract(u) - 0.25 - 0.5*lh;
      float leaf = smoothstep(0.17, 0.12, length(rot(lh.x*6.0)*lf*vec2(1.0,1.9)));
      a = mix(a, mix(SP_LEAF_A, SP_LEAF_B, lh.y), leaf*step(0.42, lh.x));
    }
    s.alb = a;
    vec2 gn = vec2(vnoise2(xz*55.0), vnoise2(xz*55.0+31.0)) - 0.5;
    s.n = normalize(n + vec3(gn.x, 0.0, gn.y)*0.55*smoothstep(40.0, 5.0, t));
    s.occ = 0.62 + 0.38*vnoise2(xz*70.0);
    s.wrap = 0.2; s.sheen = 0.25;
  } else if(m == M_BARK){
    float bn = fbm3(vec3(p.x*6.0, p.y*1.6, p.z*6.0));
    vec3 a = mix(SP_BARK_A, SP_BARK_B, bn);
    a = mix(a, a*0.6, smoothstep(0.55,0.75, vnoise3(vec3(p.x*14.0,p.y*3.0,p.z*14.0))));
    float mb = smoothstep(0.9, 0.1, p.y + 0.4*fbm3(p*3.0));
    s.alb = mix(a, SP_MOSS, mb*0.6);
    s.wrap = 0.15; s.spec = 0.03; s.shin = 8.0;
  } else if(m == M_GRASS){
    vec2 cid = floor(xz/0.085);
    float v = hash12(cid);
    s.alb = mix(SP_GRASS_A*0.6, mix(SP_GRASS_A, SP_GRASS_B, v), smoothstep(0.0, 0.9, gGrassH));
    s.wrap = 0.6; s.trans = 0.8; s.transCol = s.alb; s.spec = 0.15; s.shin = 30.0; s.occ = 0.70 + 0.30*gGrassH;
  } else if(m == M_BUSH){
    float n1 = vnoise3(p*7.0);
    float n2 = vnoise3(p*19.0);
    float n3 = vnoise3(p*2.1 + 4.0);
    s.n = normalize(n + 0.7*(vec3(vnoise3(p*9.0), vnoise3(p*9.0+5.0), vnoise3(p*9.0+9.0)) - 0.5));
    s.alb = mix(mix(SP_CANOPY_A, SP_CANOPY_B, n1), SP_BUSH_C, smoothstep(0.42,0.62,n3))*(0.55 + 0.5*n2);
    s.occ = 0.5 + 0.5*n2;
    s.wrap = 0.5; s.trans = 0.7; s.transCol = s.alb; s.spec = 0.08; s.shin = 16.0;
  } else if(m == M_CANOPY){
    float n1 = vnoise3(p*5.0);
    float n2 = vnoise3(p*13.0);
    s.n = normalize(n + 0.55*(vec3(vnoise3(p*4.0), vnoise3(p*4.0+5.0), vnoise3(p*4.0+9.0)) - 0.5));
    float n3 = vnoise3(p*0.9 + 2.0);
    s.alb = mix(mix(SP_CANOPY_A, SP_CANOPY_B, n1), SP_BUSH_C, 0.7*smoothstep(0.50,0.68,n3))*(0.55 + 0.55*n2);
    s.occ = 0.45 + 0.55*n2;
    s.wrap = 0.5; s.trans = 0.5; s.transCol = s.alb; s.spec = 0.1; s.shin = 20.0;
  }
}

vec3 setAmbient(vec3 pw, vec3 n){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  float openSky = 1.0 - 0.75*canopyPresence(p.xz);
  vec3 sky = mix(vec3(0.20,0.20,0.14), SP_AMB_SKY, openSky)*(0.6 + 0.4*n.y);
  vec3 bounce = SP_BOUNCE*0.55*clamp(0.6 - 0.5*n.y, 0.0, 1.0);
  vec3 clearDir = normalize(vec3(0.15, 0.25, -1.0));
  vec3 fill = vec3(1.0,0.86,0.58)*0.55*pow(clamp(dot(n, clearDir)*0.5+0.5, 0.0, 1.0), 3.0);
  return sky + bounce + fill;
}

// art-directed canopy gaps: a pool on the subject + background shafts
float gapMask(vec2 q, vec2 w){
  float e = 0.30*(vnoise2(q*3.0 + w*2.0) - 0.5) + 0.12*(vnoise2(q*9.0 - w*3.0) - 0.5);
  float g = smoothstep(0.75, 0.38, length(q - HOLE) + e);
  g = max(g, smoothstep(0.55, 0.25, length(q - HOLE - vec2(-1.1,-1.9)) + e)*0.95);
  g = max(g, smoothstep(0.50, 0.22, length(q - HOLE - vec2( 1.4,-1.1)) + e)*0.85);
  g = max(g, smoothstep(0.60, 0.28, length(q - HOLE - vec2(-0.2,-3.3)) + e)*0.95);
  g = max(g, smoothstep(0.45, 0.20, length(q - HOLE - vec2( 2.1,-3.0)) + e)*0.8);
  return g;
}
float setCookie(vec3 pw){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  vec2 q = p.xz + uSunDir.xz*(max(CANH - p.y, 0.0)/uSunDir.y);
  vec2 w = gWind;
  float open = 1.0 - canopyPresence(q);
  float big  = fbm2_3(q*0.40 + 3.1 + w*0.3);
  float leaf = fbm2_3(q*2.0*SP_DAPPLE_FREQ + w) + 0.06*(vnoise2(q*9.0 + w*4.0) - 0.5);
  float gaps  = smoothstep(0.64, 0.86, big)*0.8;
  float fleck = smoothstep(0.52, 0.78, leaf);
  float l = max(max(gaps, fleck*0.85), gapMask(q, w));
  l = max(l, open*smoothstep(0.35, 0.55, leaf + open*0.5));
  // leaf-sized flecks: what you see crossing the character
  float fine = vnoise2(q*5.3*SP_DAPPLE_FREQ + w*2.0)*0.45 + vnoise2(q*11.0*SP_DAPPLE_FREQ - w*3.0)*0.55;
  l *= 0.52 + 0.48*smoothstep(0.26, 0.72, fine);
  return clamp(l, 0.0, 1.0);
}
float setVol(vec3 pw){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  vec2 q = p.xz + uSunDir.xz*(max(CANH - p.y, 0.0)/uSunDir.y);
  float open = 1.0 - canopyPresence(q);
  float beam = gapMask(q, gWind)*smoothstep(CANH + 0.6, CANH - 0.2, p.y);
  return max(open*0.06, beam*beam);
}
float setFog(vec3 pw){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  float forest = canopyPresence(p.xz);
  return (0.0070*exp(-max(p.y, 0.0)*0.30) + 0.010*forest*smoothstep(CANH+0.3, 0.8, p.y))/SC*SP_FOG_GAIN;
}
vec3 setFogAmb(vec3 rd){
  return mix(SP_FOG_LO, SP_FOG_HI, 0.5+0.5*dot(rd, uSunDir))*0.30;
}

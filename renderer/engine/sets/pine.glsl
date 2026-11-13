// ---------------------------------------------------------------------------
//  engine/sets/pine.glsl — a pine forest in cool early-morning light.
//  Low side sun from screen left and in front of the subject's plane, pale blue-green fog, plated
//  cinnamon trunks in depth layers, moss cushions in needle litter, near-focus fern + mossy log.
//  Works in world units with the ground at height 0 (p = world - SP_GROUND_Y); no set scale.
//
//  SET API: see the header of forest.glsl.
//  llvmpipe notes for this file: `if` never skips work, only a loop `break` does.  The fronds are in
//  loops that leave on the first iteration when every lane of the packet is outside the bound, and
//  all per-object detail (bark, boughs) is evaluated once, for the nearest object only.
//  setMap(p, true) folds to a much cheaper shadow scene: the bool is a constant at both call sites.
//
//  The sun is behind the camera, so the bake's back-lit term is always 0 here: translucency is done
//  in setSurface by giving the leaf's far side a share of the diffuse.
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-1.4)
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.27,0.36,0.19)              // green light bounced off the floor
#endif
#ifndef SP_AMB_SKY
#define SP_AMB_SKY vec3(0.43,0.58,0.72)             // cool sky fill
#endif
#ifndef SP_AMB_WARM
#define SP_AMB_WARM vec3(1.00,0.80,0.60)            // soft wrap from the sun side
#endif
#ifndef SP_SKY_LOW
#define SP_SKY_LOW vec3(0.86,0.97,0.98)
#endif
#ifndef SP_SKY_HIGH
#define SP_SKY_HIGH vec3(0.66,0.82,0.92)
#endif
#ifndef SP_FOG_LO
#define SP_FOG_LO vec3(0.34,0.52,0.58)              // fog away from the sun
#endif
#ifndef SP_FOG_HI
#define SP_FOG_HI vec3(0.62,0.78,0.76)              // fog toward the sun
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#ifndef SP_MOSS_A
#define SP_MOSS_A vec3(0.034,0.082,0.042)
#endif
#ifndef SP_MOSS_B
#define SP_MOSS_B vec3(0.100,0.185,0.068)
#endif
#ifndef SP_LITTER_A
#define SP_LITTER_A vec3(0.062,0.042,0.030)
#endif
#ifndef SP_LITTER_B
#define SP_LITTER_B vec3(0.270,0.160,0.095)
#endif
#ifndef SP_BARK_DEEP
#define SP_BARK_DEEP vec3(0.040,0.029,0.026)
#endif
#ifndef SP_BARK_A
#define SP_BARK_A vec3(0.105,0.068,0.052)
#endif
#ifndef SP_BARK_B
#define SP_BARK_B vec3(0.265,0.160,0.102)
#endif
#ifndef SP_FERN_A
#define SP_FERN_A vec3(0.030,0.086,0.046)
#endif
#ifndef SP_FERN_B
#define SP_FERN_B vec3(0.098,0.205,0.066)
#endif
#ifndef SP_BOUGH
#define SP_BOUGH vec3(0.014,0.030,0.028)
#endif
#ifndef SP_FOG_NEAR
#define SP_FOG_NEAR vec2(6.5,14.0)                  // the air between these distances from the camera clears
#endif
#ifndef SP_VOL_HOLE
#define SP_VOL_HOLE vec4(0.178,0.30,0.172,0.01)     // the shafts' hole around the subject, slopes: half width, flare below, height, centre
#endif

const float G0 = SP_GROUND_Y;

#define M_GROUND 1.0
#define M_BARK   2.0
#define M_ROCK   3.0
#define M_LOG    4.0
#define M_MUSH   5.0
#define M_FERN   6.0
#define M_BOUGH  7.0
#define M_STICK  8.0
#define M_CANOPY 9.0            // unused: this set has no canopy plane, its boughs take real shadow

// hand-placed trunks: x, z, radius, seed.  The first three compose the frame.
#define NT 15
const vec4 TRK[NT] = vec4[NT](
  vec4(-2.78, -4.50, 0.285, 1.3),
  vec4( 3.12, -6.10, 0.225, 2.7),
  vec4( 2.08, -8.30, 0.135, 4.1),
  vec4(-2.28, -9.60, 0.170, 5.9),
  vec4(-4.70,-10.50, 0.215, 7.3),
  vec4( 4.90, -9.80, 0.180, 8.8),
  vec4(-3.30,-13.50, 0.130, 9.4),
  vec4( 1.30,-12.80, 0.165,10.6),
  vec4( 3.70,-14.50, 0.230,11.8),
  vec4(-0.90,-15.50, 0.190,12.1),
  vec4(-6.20,-15.00, 0.260,13.7),
  vec4( 6.80,-13.80, 0.155,14.2),
  vec4(-1.90,-20.00, 0.210,15.5),
  vec4( 2.50,-22.00, 0.190,16.9),
  vec4( 7.60,-20.50, 0.240,17.4)
);
// rocks: x, z, radius, flatten
#define NR 7
const vec4 ROCK[NR] = vec4[NR](
  vec4(-3.42,-6.70,0.62,0.74),
  vec4(-2.72,-6.25,0.27,0.72),
  vec4( 3.92,-7.70,0.60,0.70),
  vec4( 3.20,-7.25,0.25,0.78),
  vec4(-1.15,-3.60,0.34,0.68),
  vec4( 1.32,-4.60,0.30,0.64),
  vec4(-5.60,-9.40,0.70,0.66)
);
// mushrooms: base x y z (y above ground), cap radius
#define NM 7
const vec4 MUSH[NM] = vec4[NM](
  vec4(2.26,0.585,-1.20,0.105),
  vec4(2.11,0.585,-1.11,0.070),
  vec4(2.41,0.560,-1.31,0.060),
  vec4(1.74,0.040,-0.70,0.085),
  vec4(1.64,0.030,-0.62,0.055),
  vec4(2.58,0.030,-0.72,0.100),
  vec4(2.70,0.030,-0.80,0.065)
);
// fallen sticks: a.xyz, r / b.xyz, r
#define NS 4
const vec4 STICK[NS*2] = vec4[NS*2](
  vec4(-2.30,0.05,-2.10,0.030), vec4(-0.90,0.09,-2.70,0.012),
  vec4(-1.70,0.07,-2.34,0.016), vec4(-1.30,0.13,-2.10,0.006),
  vec4( 1.30,0.06,-3.40,0.028), vec4( 2.60,0.08,-2.90,0.012),
  vec4(-3.00,0.08,-6.40,0.050), vec4(-1.20,0.10,-5.90,0.022)
);
// the log: near (broken) end -> far end, world minus ground
const vec3  LOG_A = vec3(1.84,0.30,-0.98);
const vec3  LOG_B = vec3(4.90,0.34,-2.90);
const float LOG_R = 0.33;

vec2  gWind;
vec3  gE1, gE2;                         // axes across the sun direction
// what the last setMap saw, read back by setSurface
vec2  gTrC;  float gTrR, gTrS;
float gFt, gFz, gFid, gFrib, gFv;
float gBt;
// bgAO strength, per material.  bgAO samples at 0.25, 1.75, 3.25 and 4.75 units, tuned for a much bigger
// scene: any upright surface near this floor reads occ = 1, ao = 0.1, and small changes in the far samples
// turn into black blotches.  The engine's only setMap(.., false) calls after setSurface are bgAO's, so
// setSurface leaves the hit point here and setMap reports distances blended toward "nothing there".
vec3  gAOp;  float gAOk = 0.0;

void setInit(){
  float T = uTime;
  gWind = 0.10*vec2(sin(T*0.80), cos(T*0.65)) + 0.035*vec2(sin(T*2.1+1.0), sin(T*1.7));
  gE1 = normalize(cross(uSunDir, vec3(0.0,1.0,0.0))); gE2 = cross(gE1, uSunDir);
}

// ------------------------------------------------------------------ the floor
float cushion(vec2 xz){                 // 0 = needle-litter hollow, 1 = top of a moss cushion
  float a = vnoise2(xz*2.3 + 3.1), b = vnoise2(xz*5.9 + 11.7);
  return smoothstep(0.47, 0.76, a*0.70 + b*0.30);
}
float groundLow(vec2 xz){
  float h = 0.10*sin(xz.x*0.31 + 0.7)*sin(xz.y*0.23 + 1.9) + 0.24*(vnoise2(xz*0.29 + 5.3) - 0.5);
  return h*smoothstep(1.0, 6.0, length(xz));          // level under the character
}
float groundH(vec2 xz){
  float c = cushion(xz);
  float h = groundLow(xz) + 0.15*c;
  h += (0.014 + 0.030*c)*vnoise2(xz*11.0 + 2.0);
  h += (0.006 + 0.012*c)*vnoise2(xz*33.0);
  return h;
}

// ------------------------------------------------------------------ trunks
vec2  sway(float y, float s){ return vec2(0.055*sin(y*0.34 + s), 0.045*sin(y*0.27 + s*1.7)); }
float trunkRad(float y, float r){ return r*max(1.0 - 0.030*y, 0.35) + 0.42*r*exp(-max(y, 0.0)/(1.1*r)); }
// bark height at b = (arc length around the trunk, height): 1 plate top, 0 furrow bottom
// stretched Voronoi: long plates, furrows at the cell edges.  gPlate = this plate's random id
float gPlate;
float barkH(vec2 b){
  vec2 q = vec2(b.x*10.5 + 0.7*sin(b.y*1.1), b.y*1.75);
  vec2 c = floor(q), f = q - c;
  float F1 = 9.0, F2 = 9.0; float id = 0.0;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 g = vec2(float(i), float(j));
    vec2 o = hash22(c + g);
    vec2 r = g + o - f;
    float d = dot(r, r);
    if(d < F1){ F2 = F1; F1 = d; id = o.x; } else if(d < F2) F2 = d;
  }
  gPlate = id;
  float e = sqrt(F2) - sqrt(F1);
  return smoothstep(0.015, 0.24, e)*(0.82 + 0.18*id);
}
// nearest trunk among the hand list and the hash grid.  Fills gTr*, returns the plain distance
float nearestTrunk(vec3 p, bool shadow){
  float d = 1e9; vec2 c = vec2(0.0); float R = 0.3, S = 0.0;
  for(int i=0;i<NT;i++){
    vec4 t = TRK[i];
    vec2 q = p.xz - t.xy + sway(p.y, t.w);
    float di = length(q) - trunkRad(p.y, t.z);
    if(di < d){ d = di; c = t.xy; R = t.z; S = t.w; }
  }
  const float CS = 2.7;
  vec2 c0 = floor(p.xz/CS - 0.5);
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, 3.0));
    vec2 cpos = (ci + 0.5 + (h.xy - 0.5)*0.72)*CS;
    bool keep = h.z < 0.66 && cpos.y < -17.0 && cpos.y > -115.0;
    if(!keep) continue;
    float s = h.x*40.0;
    float r = mix(0.12, 0.30, h.y*h.y);
    vec2 q = p.xz - cpos + sway(p.y, s);
    float di = length(q) - trunkRad(p.y, r);
    if(di < d){ d = di; c = cpos; R = r; S = s; }
  }
  gTrC = c; gTrR = R; gTrS = S;
  // the soft shadow reads h/t, so a capped h, or a trunk with no top, would dim the sun far along the ray
  return shadow ? max(d, p.y - 11.0) : min(d, 1.7);
}
// nearest-trunk detail: root lobes, bark relief
float trunkDetail(vec3 p, float d){
  vec2 q = p.xz - gTrC + sway(p.y, gTrS);
  float a = atan(q.x, q.y);                           // seam faces away from the camera
  float y = max(p.y, 0.0);
  float lob = exp(-y/(0.60*gTrR))*pow(0.5 + 0.5*cos(a*5.0 + gTrS*3.0), 2.0);
  d -= gTrR*0.62*lob;
  d -= 0.052*smoothstep(0.08, 0.30, gTrR)*(barkH(vec2(a*gTrR, p.y) + gTrS*3.1) - 0.6)*(1.0 - 0.6*exp(-y*2.0));
  return d;
}
// dead stub branches on the nearest trunk
float sdStubs(vec3 p){
  const float PH = 0.57;
  vec3 q = p - vec3(gTrC.x, 0.0, gTrC.y); q.xz += sway(p.y, gTrS);
  float k0 = floor(p.y/PH - 0.5);
  float d = 1e9;
  for(int j=0;j<2;j++){
    float k = k0 + float(j);
    vec3 h = hash31(k*3.17 + gTrS*11.0);
    if(h.x < 0.42 || k < 1.0) continue;
    float az = h.y*6.283;
    vec3 dir = normalize(vec3(cos(az), 0.10 + 0.42*h.z, sin(az)));
    vec3 a = vec3(dir.x, 0.0, dir.z)*gTrR*0.80 + vec3(0.0, (k + 0.5)*PH, 0.0);
    d = min(d, sdTaper(q, a, a + dir*(0.14 + 0.34*h.z), 0.040*(0.5 + gTrR/0.5), 0.007));
  }
  return d;
}
// ------------------------------------------------------------------ needle boughs
float sdBough(vec3 q, float L){            // +x along the branch, flat spray of twigs in x,z
  float x = q.x;
  float tt = clamp(x/L, 0.0, 1.0);
  float ys = -0.30*x - 0.16*x*x/L + 0.42*tt*tt*tt;
  float w = q.y - ys;
  float dsp = length(vec3(x - clamp(x, 0.0, L), w, q.z)) - mix(0.040, 0.010, tt);
  const float P = 0.105, SW = 0.70;
  float zs = abs(q.z);
  float xx = x - SW*zs + (q.z > 0.0 ? 0.0 : 0.5*P);
  float k = floor(xx/P + 0.5);
  float hk = hash11(k*1.73 + (q.z > 0.0 ? 3.1 : 8.2));
  float dx = (xx - k*P - (hk - 0.5)*0.05)*0.84;
  float tc = clamp(k*P/L, 0.0, 1.0);
  float Lp = (0.05 + 0.34*sin(PI*min(tc*0.80 + 0.20, 1.0)))*(0.45 + 0.85*hk)*step(0.16, fract(hk*5.3));
  float zt = clamp(zs/max(Lp, 1e-3), 0.0, 1.0);
  float w2 = (w + (0.10 + 0.5*hk)*zs + 1.1*zs*zs + 0.06*(hk - 0.5))*0.62;
  float saw = 0.030*abs(fract(zs*11.0 + hk*3.0) - 0.5);
  float dt = max(length(vec2(dx, w2)) - mix(0.046, 0.016, zt*zt) - saw, zs - Lp);
  dt = max(dt, max(x - L - 0.2, -x + 0.05));
  gBt = tt;
  return min(dsp, dt);
}
float sdBoughs(vec3 p){
  float H0 = 3.45 + 1.1*fract(gTrS*0.73);
  const float PH = 0.62;
  vec3 q = p - vec3(gTrC.x, 0.0, gTrC.y); q.xz += sway(p.y, gTrS);
  float k0 = max(floor((p.y - H0)/PH), 0.0);
  float d = 1e9;
  for(int j=0;j<2;j++){
    float k = k0 + float(j);
    vec3 h = hash31(k*2.31 + gTrS*7.0);
    vec2 qq = rot(h.x*6.283)*q.xz;
    float a = atan(qq.y, qq.x);
    const float SEC = 6.2832/6.0;
    float a2 = mod(a + 0.5*SEC, SEC) - 0.5*SEC;
    float rho = length(qq);
    float L = (0.82 + 0.62*h.y)*max(1.0 - 0.022*k, 0.55);
    vec3 b = vec3(rho*cos(a2) - gTrR*0.8, q.y - (H0 + k*PH + 0.3*(h.z - 0.5)), rho*sin(a2));
    d = min(d, sdBough(b, L));
  }
  return d*0.72;
}

// ------------------------------------------------------------------ ferns
// One frond: an arc in the vertical plane at azimuth `az`, starting at elevation `el` from the origin.
// Pinnae repeat along the arc in a sheared domain, alternating sides, tapering to the tip.
float sdFrond(vec3 p, float az, float el, float L, float R, float id, float fine){
  float ca = cos(az), sa = sin(az);
  float rho = ca*p.x + sa*p.z, zet = -sa*p.x + ca*p.z;
  vec2  T = vec2(cos(el), sin(el));
  vec2  v = vec2(rho, p.y) - vec2(T.y, -T.x)*R;
  float th0 = 0.5*L/R;
  float mid = el + 0.5*PI - th0;
  vec2  m = vec2(cos(mid), sin(mid));
  float r = length(v);
  float a = atan(m.x*v.y - m.y*v.x, dot(m, v));
  float u = (th0 - a)*R;
  float w = r - R;
  float t = clamp(u/L, 0.0, 1.0);
  float zs = abs(zet);
  float ww = w + 1.5*zs*zs/max(L, 0.4);               // the blade is a shallow roof
  float d = length(vec3(u - clamp(u, 0.0, L), w, zet)) - mix(0.015, 0.004, t)*fine;
  // pinnae
  float P = 0.052*fine;
  const float SWP = 0.50;
  float uu = u - SWP*zs - 0.8*zs*zs + (zet > 0.0 ? 0.0 : 0.5*P);       // pinnae curve toward the tip
  float k = floor(uu/P + 0.5);
  float hk = hash11(k*0.77 + id*1.31 + (zet > 0.0 ? 0.0 : 3.7));
  float du = abs(uu - k*P - 0.12*P*(hk - 0.5))*0.74;
  float cu = (k*P - (zet > 0.0 ? 0.0 : 0.5*P))/L;
  float sh = smoothstep(0.13, 0.36, cu)*pow(max(1.0 - cu, 0.0), 0.72)*step(cu, 1.0);
  float Lp = 0.165*L*sh*(0.84 + 0.30*hk);
  float x = zs/max(Lp, 1e-3);
  float hw = 0.30*P*pow(max(1.0 - x, 0.0), 0.62)*(1.0 + 0.25*x);     // pinnae end in a point, gaps between them
  hw *= 0.78 + 0.44*abs(fract(zs/(0.024*fine)) - 0.5);       // saw-toothed pinna edge
  float dp = max(du - hw, abs(ww + 0.05*(hk - 0.5)*zs) - 0.0055*fine);
  dp = max(dp, zs - Lp);
  dp = max(dp, max(u - L, -u));
  d = min(d, dp);
  if(d < gFrib){ gFrib = d; gFt = t; gFz = x; gFid = id; gFv = clamp(du/max(hw, 1e-3), 0.0, 1.0); }
  return d;
}
// the two hero clusters (near focus) and the hash clumps on the floor share one frond loop
float sdFerns(vec3 p, bool shadow){
  const vec3  CL = vec3(-2.05, 0.75, -0.55); const float RL = 2.25;
  const vec3  CR = vec3( 2.05, 0.65, -2.10); const float RR = 1.75;
  float dL = length(p - CL) - RL, dR = length(p - CR) - RR;
  // hash clumps, nearest cell only; every sphere stays inside its cell
  const float CS = 2.3;
  vec2 cell = floor(p.xz/CS);
  vec3 h = hash33(vec3(cell, 21.0));
  vec2 cp = (cell + 0.5 + (h.xy - 0.5)*0.40)*CS;
  float S = 0.42 + 0.34*h.z;
  bool here = h.z < 0.72 && cp.y < -2.8 && cp.y > -48.0 && vnoise2(cp*0.21 + 2.0) > 0.36 && !shadow;
  vec2 cl = abs(fract(p.xz/CS) - 0.5)*CS;
  float border = CS*0.5 - max(cl.x, cl.y);
  float dG = here ? length(p - vec3(cp.x, 0.42*S, cp.y)) - 1.05*S : 2.0;
  dG = shadow ? 1e3 : min(dG, border + 0.35);
  float db = min(min(dL, dR), dG);
  float d = db > 0.10 ? db : 2.0;                     // inside the bound the fronds alone give the distance
  int kind = dL < dR ? (dL < dG ? 0 : 2) : (dR < dG ? 1 : 2);
  int n = kind == 2 ? 7 : 19;
  gFrib = 1e9;
  for(int i=0;i<n;i++){
    if(db > 0.10) break;
    float fi = float(i);
    vec3 r = hash31(fi*1.37 + float(kind)*17.0 + (kind == 2 ? h.z*90.0 : 0.0));
    vec3 q; float L, Rc, el, az;
    az = fi*2.39996 + 2.4 + 0.5*r.x;
    if(kind == 0) az = 3.7*(fract(fi*0.618 + 0.31) - 0.5);        // left clump: a fan into the frame, none lost off the edge
    if(kind == 0){
      bool sec = i >= 13;                              // second, smaller clump behind the first
      q = p - (sec ? vec3(-2.45, 0.0, -1.55) : vec3(-1.90, 0.0, -0.30));
      float sc = sec ? 0.74 : 1.0;
      L = sc*mix(1.75, 2.45, r.y); Rc = L*mix(0.56, 0.78, r.z); el = mix(1.05, 1.42, fract(fi*0.618));
    } else if(kind == 1){
      bool sec = i >= 12;
      q = p - (sec ? vec3(1.40, 0.0, -2.35) : vec3(2.40, 0.0, -2.00));
      float sc = sec ? 0.66 : 0.84;
      L = sc*mix(1.45, 2.10, r.y); Rc = L*mix(0.56, 0.78, r.z); el = mix(1.05, 1.42, fract(fi*0.618));
    } else {
      q = p - vec3(cp.x, 0.0, cp.y);
      L = S*mix(1.3, 1.9, r.y); Rc = L*mix(0.55, 0.80, r.z); el = mix(0.95, 1.40, fract(fi*0.618));
    }
    float fine = kind == 2 ? 1.5 : 1.0;
    d = min(d, sdFrond(q, az, el, L, Rc, fi + float(kind)*20.0, fine));
  }
  // Real fronds shade the floor around the clump, but not each other: light goes through a frond, and
  // the bake has no term for that with the sun behind the camera.
  if(shadow) d = max(d, p.y - 0.50);
  return d*0.72;
}

// ------------------------------------------------------------------ the log
// axis frame: s along the log from the broken end, a around it (0 = top), rl radial, c = cross-section xy
void logFrame(vec3 p, out float s, out float a, out float rl, out vec2 c){
  vec3 ba = LOG_B - LOG_A; float L = length(ba); vec3 ax = ba/L;
  vec3 pa = p - LOG_A;
  s = dot(pa, ax);
  vec3 rad = pa - ax*s;
  rl = length(rad);
  vec3 e1 = normalize(cross(ax, vec3(0.0,1.0,0.0))), e2 = cross(e1, ax);
  c = vec2(dot(rad, e1), dot(rad, e2));
  a = atan(c.x, c.y);
}
// 1 where moss blankets the log
float logMoss(float s, float a){
  float cov = 1.25 - abs(a) + 0.95*(vnoise2(vec2(s*2.6, a*2.2) + 7.0) - 0.5)
            + 0.34*(vnoise2(vec2(s*8.0, a*7.0)) - 0.5);
  return smoothstep(0.02, 0.22, cov)*smoothstep(0.02, 0.26, s + 0.20*(vnoise2(vec2(a*3.0, 3.0)) - 0.5));
}
float sdLog(vec3 p){
  float s, a, rl; vec2 c;
  logFrame(p, s, a, rl, c);
  float R = LOG_R*(1.0 + 0.05*sin(s*2.3 + 1.0) + 0.04*sin(a*3.0 + s*0.7));
  float mo = logMoss(s, a);
  float b = barkH(vec2(a*LOG_R*1.3, s*0.9));
  float fuzz = vnoise2(vec2(s*19.0, a*16.0));
  float dr = rl - R - (1.0 - mo)*0.030*(b - 0.6) - mo*(0.040 + 0.040*vnoise2(vec2(s*6.0, a*5.5)) + 0.014*fuzz);
  // the end: an old break worn nearly flat, read in cross-section coordinates so it has no seam
  float e = 0.040*vnoise2(c*5.0 + 3.0) + 0.016*vnoise2(c*14.0 + 8.0);
  float d = smax(dr, e - s, 0.018);
  d = max(d, s - 3.8);
  return d*0.72;
}
int gMu;
float sdMushrooms(vec3 p){
  float d = 1e9;
  for(int i=0;i<NM;i++){
    vec4 m = MUSH[i];
    float h = m.w*(1.25 + 0.5*fract(float(i)*0.618));
    vec3 q = p - m.xyz;
    q.xz -= (q.y*q.y/max(h, 0.01))*0.16*vec2(cos(float(i)*2.0), sin(float(i)*2.0));   // a slight lean
    float stem = sdTaper(q, vec3(0.0), vec3(0.0, h, 0.0), m.w*0.27, m.w*0.20);
    vec3 c = q - vec3(0.0, h, 0.0);
    float cap = sdEllipsoid(c, vec3(m.w, m.w*0.66, m.w));
    cap = smax(cap, -c.y - 0.10*m.w, 0.012);
    float di = min(stem, cap);
    if(di < d){ d = di; gMu = i; }
  }
  return d;
}
float sdRocks(vec3 p){
  float d = 1e9; float best = 1e9; vec3 cq = p;
  for(int i=0;i<NR;i++){
    vec4 r = ROCK[i];
    vec3 q = p - vec3(r.x, r.z*r.w*0.45, r.y);
    float di = sdEllipsoid(q, vec3(r.z, r.z*r.w, r.z*0.86));
    if(di < d){ d = di; cq = q + float(i)*3.7; }
  }
  d += 0.075*(vnoise3(cq*4.3) - 0.5) + 0.026*(vnoise3(cq*11.0) - 0.5);
  return d*0.8;
}
float sdSticks(vec3 p){
  float d = 1e9;
  for(int i=0;i<NS;i++){
    vec4 a = STICK[i*2], b = STICK[i*2 + 1];
    d = min(d, sdTaper(p, a.xyz, b.xyz, a.w, b.w));
  }
  return d;
}

// What the forest upstream lets through.  A soft shadow here is 10*h/t, and a slim tree can never clear
// that at t = 40, so the light on the set is drawn instead: a function of the position ACROSS the sun
// direction, constant along each sun ray, added to the shadow field.  Long streaks on the floor, whole
// trunks lit or not, and a clear pool on the near props.  h = 7.5*gate gives sun = gate at the far end.
float sunGate(vec3 pw){
  vec2 q = vec2(dot(pw, gE1), dot(pw, gE2));
  float a = vnoise2(vec2(q.x*1.05, q.y*0.17) + 4.0), b = vnoise2(vec2(q.x*2.9, q.y*0.40) + 9.0);
  float g = smoothstep(0.43, 0.60, a*0.68 + b*0.32);
  vec3 c = vec3(0.0, G0, -1.0);                                   // the pool, on the props and the subject
  vec2 qc = vec2(dot(c, gE1), dot(c, gE2));
  float pool = smoothstep(2.6, 1.3, length((q - qc)*vec2(0.42, 0.95)));
  return max(max(g, pool), 0.012);
}
vec2 setMap(vec3 pw, bool shadow){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z);
  float gh = shadow ? groundLow(p.xz) + 0.15*cushion(p.xz) : groundH(p.xz);
  vec2 res = vec2((p.y - gh)*0.72, M_GROUND);
  float dt = nearestTrunk(p, shadow);
  if(!shadow) dt = trunkDetail(p, dt);
  dt *= 0.80;
  float bl = smin(res.x, dt, 0.16);
  res = vec2(bl, dt < res.x ? M_BARK : res.y);
  if(!shadow){
    res = opU(res, vec2(sdStubs(p), M_BARK));
    res = opU(res, vec2(sdBoughs(p), M_BOUGH));
    res = opU(res, vec2(sdSticks(p), M_STICK));
  }
  res = opU(res, vec2(sdRocks(p), M_ROCK));
  res = opU(res, vec2(sdLog(p), M_LOG));
  res = opU(res, vec2(sdMushrooms(p), M_MUSH));
  res = opU(res, vec2(sdFerns(p, shadow), M_FERN));
  if(shadow) res.x = min(res.x, 7.5*sunGate(pw));
  else if(gAOk > 0.0) res.x = mix(res.x, length(pw - gAOp), gAOk);
  return res;
}

// ------------------------------------------------------------------ light
vec3 setSky(vec3 rd){
  float sd = max(dot(rd, uSunDir), 0.0);
  vec3 col = mix(SP_SKY_LOW, SP_SKY_HIGH, smoothstep(0.02, 0.55, rd.y));
  col += vec3(1.0,0.80,0.55)*(0.9*pow(sd, 5.0) + 8.0*pow(sd, 160.0));
  return col;
}

// three scattered layers of pine needles, drawn into the albedo
float needles(vec2 xz, out float sel){
  float cov = 0.0; sel = 0.0;
  for(int i=0;i<4;i++){
    float fi = float(i);
    vec2 u = rot(fi*1.1 + 0.4)*xz*(7.0 + fi*3.0) + fi*13.1;
    vec2 c = floor(u); vec2 h = hash22(c + fi*7.0);
    vec2 f = fract(u) - 0.5 - 0.30*(h - 0.5);
    f = rot(h.x*6.283)*f;
    f.y += 0.10*sin(f.x*5.0 + h.y*6.0);
    float k = smoothstep(0.042, 0.016, abs(f.y))*smoothstep(0.50, 0.42, abs(f.x));
    if(k > cov){ cov = k; sel = h.y; }
  }
  return cov;
}
void setSurface(vec3 pw, vec3 rd, float t, float m, vec3 n, inout Surf s){
  if(!(abs(n.y) <= 1.0)){ n = vec3(0.0,1.0,0.0); s.n = n; }     // a NaN here would poison the whole FFT lens blur
  vec3 p = vec3(pw.x, pw.y - G0, pw.z);
  vec2 xz = p.xz;
  vec3 L = uSunDir;
  if(m == M_GROUND){
    float c = cushion(xz);
    float n1 = fbm2(xz*2.1 + 4.0), n2 = vnoise2(xz*24.0), n3 = vnoise2(xz*71.0), n4 = vnoise2(xz*190.0);
    float sel;
    float nd = needles(xz, sel);
    vec3 lit = mix(SP_LITTER_A, SP_LITTER_B, 0.15 + 0.5*n1)*(0.65 + 0.5*n2);
    lit = mix(lit, mix(vec3(0.30,0.17,0.085), vec3(0.50,0.32,0.17), sel), nd*0.85);
    vec3 mos = mix(SP_MOSS_A, SP_MOSS_B, smoothstep(0.25, 0.75, n1*0.5 + n2*0.25 + n3*0.25));
    mos *= 0.62 + 0.55*n3*(0.5 + n4);
    mos = mix(mos, vec3(0.21,0.25,0.09), 0.40*smoothstep(0.62, 0.86, vnoise2(xz*5.0 + 9.0)));
    float mk = smoothstep(0.10, 0.40, c + 0.30*(n2 - 0.5) + 0.30*(n1 - 0.5));
    s.alb = mix(lit, mos, mk);
    float bump = mix(0.42, 0.70, mk)*smoothstep(26.0, 7.0, t);
    vec2 gn = vec2(vnoise2(xz*55.0), vnoise2(xz*55.0 + 31.0)) - 0.5;
    vec2 gf = vec2(vnoise2(xz*150.0), vnoise2(xz*150.0 + 17.0)) - 0.5;
    s.n = normalize(n + vec3(gn.x + 0.6*gf.x, 0.0, gn.y + 0.6*gf.y)*bump);
    s.occ = mix(0.60 + 0.40*(1.0 - nd*0.0), 0.72 + 0.28*n3, mk);
    s.wrap = mix(0.15, 0.40, mk); s.sheen = mix(0.1, 0.55, mk); s.spec = 0.02;
  } else if(m == M_BARK){
    vec2 q = xz - gTrC + sway(p.y, gTrS);
    float a = atan(q.x, q.y);
    vec2 b = vec2(a*gTrR, p.y) + gTrS*3.1;
    float h = barkH(b);
    float pl = vnoise2(vec2(b.x*2.4, b.y*0.36));
    float fl = vnoise2(vec2(b.x*16.0, b.y*2.2));
    vec3 c = mix(SP_BARK_A, SP_BARK_B, smoothstep(0.28, 0.72, pl*0.40 + fl*0.20 + gPlate*0.40));
    c = mix(c, c*vec3(0.62,0.70,0.82), smoothstep(0.5, 0.8, fbm2(vec2(b.x*1.2, b.y*0.16))));   // grey scale patches
    c *= 0.72 + 0.52*vnoise2(vec2(b.x*26.0, b.y*5.0))*(0.6 + 0.8*vnoise2(vec2(b.x*7.0, b.y*14.0)));   // flaking scales on the plates
    c = mix(SP_BARK_DEEP, c, smoothstep(0.03, 0.42, h));
    // moss climbs out of the floor between the root lobes, higher on the shaded side
    float my = p.y + 0.55*(fbm2(vec2(b.x*3.5, b.y*1.4)) - 0.5) - 0.24*dot(normalize(q + 1e-4), normalize(vec2(0.6, -0.3)));
    float mo = smoothstep(0.62, 0.20, my);
    vec3 mc = mix(SP_MOSS_A, SP_MOSS_B, 0.45 + 0.6*vnoise2(b*vec2(26.0,17.0)))*(0.8 + 0.5*vnoise2(b*60.0));
    s.alb = mix(c, mc, mo);
    s.occ = mix(0.40 + 0.60*smoothstep(0.0, 0.40, h), 0.9, mo);
    s.wrap = 0.12; s.spec = 0.03; s.shin = 10.0; s.sheen = mix(0.10, 0.45, mo);
  } else if(m == M_ROCK){
    vec3 q = p*vec3(1.0, 1.4, 1.0);
    float n1 = fbm3(q*2.2), n2 = vnoise3(q*13.0);
    vec3 c = mix(vec3(0.15,0.16,0.17), vec3(0.34,0.36,0.36), n1)*(0.8 + 0.35*n2);
    float li = smoothstep(0.58, 0.68, vnoise3(q*7.0));                  // pale lichen flecks
    c = mix(c, vec3(0.42,0.46,0.34), li*0.75);
    float mo = smoothstep(0.30, 0.72, n.y + 0.9*(fbm3(p*3.4 + 3.0) - 0.5));
    vec3 mc = mix(SP_MOSS_A, SP_MOSS_B, 0.3 + 0.8*vnoise3(p*19.0))*(0.75 + 0.5*vnoise3(p*57.0));
    s.alb = mix(c, mc, mo);
    s.n = normalize(n + (vec3(vnoise3(p*31.0), vnoise3(p*31.0 + 5.0), vnoise3(p*31.0 + 9.0)) - 0.5)*0.35);
    s.wrap = mix(0.1, 0.4, mo); s.spec = 0.04; s.shin = 14.0; s.sheen = 0.35*mo;
  } else if(m == M_LOG){
    float ls, la, lr; vec2 lc;
    logFrame(p, ls, la, lr, lc);
    float mo = logMoss(ls, la);
    float b = barkH(vec2(la*LOG_R*1.3, ls*0.9));
    vec3 c = mix(vec3(0.030,0.022,0.018), mix(vec3(0.16,0.11,0.08), vec3(0.27,0.19,0.13), gPlate), smoothstep(0.05, 0.6, b));
    c *= 0.8 + 0.4*vnoise2(vec2(ls*9.0, la*14.0));
    // the end face: pale weathered wood, growth rings around an off-centre pith, a few long checks.
    // It sits in the character's cast shadow, so it is painted to read in shade.
    float face = smoothstep(0.105, 0.070, ls)*smoothstep(LOG_R*1.0, LOG_R*0.86, lr);
    vec2  cp = lc - vec2(0.045, -0.035);
    float lr2 = length(cp) + 0.030*vnoise2(lc*4.5 + 2.0);
    float ring = 0.5 + 0.5*sin(lr2*135.0);
    float chk = smoothstep(0.012, 0.003, abs(sin(atan(cp.x, cp.y)*1.5 + 0.8 + 1.1*vnoise2(lc*3.0 + 6.0)))*lr2)*smoothstep(0.015, 0.05, lr2);
    vec3 w = mix(vec3(0.27,0.17,0.10), vec3(0.52,0.37,0.23), ring*0.32 + 0.68*vnoise2(lc*vec2(9.0,22.0)));
    w = mix(w, vec3(0.08,0.055,0.045), chk*0.70);
    w = mix(w, w*vec3(0.60,0.52,0.46), smoothstep(0.075, 0.0, lr2));      // the old heart
    c = mix(c, w, face);
    float n3 = vnoise2(vec2(ls*44.0, la*38.0)), n4 = vnoise2(vec2(ls*130.0, la*120.0));
    vec3 mc = mix(SP_MOSS_A, SP_MOSS_B*1.25, 0.25 + 0.75*vnoise2(vec2(ls*7.0, la*6.0)))*(0.62 + 0.55*n3*(0.5 + n4));
    mo *= 1.0 - face;
    s.alb = mix(c, mc, mo);
    s.occ = mix(mix(0.55 + 0.45*smoothstep(0.0, 0.5, b), 1.0, face), 0.72 + 0.28*n3, mo);
    s.n = normalize(n + 0.30*face*(vec3(vnoise2(lc*38.0), vnoise2(lc*38.0 + 7.0), vnoise2(lc*38.0 + 13.0)) - 0.5));
    s.wrap = mix(0.10, 0.42, mo); s.sheen = 0.6*mo; s.spec = 0.02;
  } else if(m == M_MUSH){
    // cap above the gill line, pale stem and gills below
    vec4 mm = MUSH[gMu];
    float h = mm.w*(1.25 + 0.5*fract(float(gMu)*0.618));
    float y = (p.y - mm.y - h)/mm.w;
    vec2 rc = (xz - mm.xz)/mm.w;
    float capk = smoothstep(-0.04, 0.05, y);
    vec3 c = mix(vec3(0.74,0.34,0.075), vec3(0.42,0.13,0.035), smoothstep(0.1, 0.62, y));
    c = mix(c, vec3(0.92,0.76,0.50), smoothstep(0.80, 1.0, length(rc))*0.6);
    c *= 0.86 + 0.28*vnoise2(rc*9.0);
    vec3 st = mix(vec3(0.82,0.74,0.60), vec3(0.62,0.52,0.40), vnoise2(p.xy*60.0));
    float gill = 0.72 + 0.28*abs(sin(atan(rc.y, rc.x)*30.0));
    s.alb = mix(st*(y > -0.3 ? gill : 1.0), c, capk);
    s.wrap = 0.35; s.spec = 0.10; s.shin = 26.0; s.sheen = 0.5;
  } else if(m == M_FERN){
    float t0 = gFt, x0 = gFz, id = gFid, v0 = gFv;
    float hv = hash11(id*3.3 + 0.7);
    vec3 c = mix(SP_FERN_A, SP_FERN_B, 0.25 + 0.75*hv);
    c = mix(c, c*vec3(1.22,1.14,0.70), smoothstep(0.55, 1.0, t0));          // fresh tips, more yellow
    c *= 0.62 + 0.55*smoothstep(0.0, 0.9, x0);                               // darker toward the midrib
    c *= 0.78 + 0.34*smoothstep(0.05, 0.28, v0);                             // the pinna's own vein
    c *= 0.82 + 0.26*smoothstep(0.2, 0.9, v0)*(1.0 - v0*0.4);
    c = mix(c, vec3(0.05,0.10,0.03), smoothstep(0.03, 0.0, x0));
    c = mix(c, vec3(0.34,0.17,0.05), smoothstep(0.72, 1.0, hash11(id*7.1))*smoothstep(0.55, 1.0, x0)*0.8);
    s.alb = c;
    // both faces of a leaf take light.  s.trans is here for a sun behind the set; with this sun the
    // bake's back term is zero, so the far side of the leaf folds its normal toward the sun instead
    float nl = dot(s.n, L);
    if(nl < 0.0) s.n = normalize(s.n - L*nl*1.45);
    s.n = normalize(s.n + vec3(0.0,0.20,0.0));
    s.wrap = 0.38; s.spec = 0.10; s.shin = 28.0; s.sheen = 0.45;
    s.trans = 0.8; s.transCol = c*vec3(1.3,1.5,0.7);
    s.occ = 0.80 + 0.20*t0;
  } else if(m == M_BOUGH){
    float n1 = vnoise3(p*vec3(9.0,4.0,9.0));
    s.alb = SP_BOUGH*(0.75 + 0.7*n1);
    s.wrap = 0.22; s.spec = 0.02; s.shin = 12.0; s.sheen = 0.25;
  } else if(m == M_STICK){
    float n1 = vnoise3(p*vec3(9.0,30.0,9.0));
    s.alb = mix(vec3(0.05,0.035,0.028), vec3(0.24,0.17,0.12), n1);
    s.wrap = 0.1;
  }
  // contact shade: bgAO works at a very wide radius, this is the crease under everything small
  float ca = setMap(pw + n*0.055, false).x;
  if(m != M_FERN) s.occ *= 0.46 + 0.54*smoothstep(0.0, 0.052, ca);
  // fronds stacked toward the sun: the only self-shadow they get, a short probe, so the clump has layers
  float pr = setMap(pw + L*0.24 + s.n*0.015, false).x;
  if(m == M_FERN){ float k = smoothstep(0.0, 0.085, pr); s.alb *= 0.40 + 0.60*k; s.occ *= 0.70 + 0.30*k; }
  gAOp = pw;
  gAOk = m == M_GROUND ? 0.20 : (m == M_BOUGH ? 0.35 : (m == M_BARK ? 0.42 : (m == M_ROCK ? 0.55 : (m == M_FERN ? 0.48 : 0.80))));
}

vec3 setAmbient(vec3 pw, vec3 n){
  vec3 sky = SP_AMB_SKY*(0.60 + 0.40*n.y);
  vec3 bounce = SP_BOUNCE*0.62*clamp(0.6 - 0.5*n.y, 0.0, 1.0);
  vec3 wrap = SP_AMB_WARM*0.26*pow(clamp(dot(n, uSunDir)*0.5 + 0.5, 0.0, 1.0), 3.0);
  return sky + bounce + wrap;
}

// broad soft light with a gentle drift, never a pattern at face scale
float setCookie(vec3 pw){
  vec2 q = pw.xy*0.34 + pw.z*vec2(0.20, 0.11) + gWind*1.7 + vec2(uTime*0.055, -uTime*0.038);
  return 0.40 + 0.60*smoothstep(0.22, 0.78, vnoise2(q));
}
// Shafts: sheets of sunlight that all contain the sun axis through the camera, so each projects to
// one clean line converging on the antisolar point.  The volume pass only samples every 3.75 units;
// rods or free-standing bands would break into dotted repeats along the depth samples.
float setVol(vec3 pw){
  vec3 v = pw - uCamPos;
  float u = atan(dot(v, gE2), dot(v, gE1))*11.0;
  float s = dot(v, uSunDir);
  float a = vnoise2(vec2(u + uTime*0.012, 3.3 + gWind.x*0.4));
  float b = vnoise2(vec2(u*2.7 - uTime*0.020, 9.1));
  float beam = smoothstep(0.40, 0.76, a*0.76 + b*0.24);
  float along = 0.55 + 0.45*smoothstep(0.25, 0.70, vnoise2(vec2(u*0.45, s*0.085 + 2.0)));
  // No shafts along rays through the bear.  They are hidden there anyway, but the compositor keeps 15% of the
  // volume layer over the character, and that layer has trunk-shaped gaps: trunks ghost through its face.
  vec2 sc = v.xy/max(-v.z, 0.1);
  float cw = SP_VOL_HOLE.x + SP_VOL_HOLE.y*max(SP_VOL_HOLE.w - 0.03 - sc.y, 0.0);
  float hole = smoothstep(0.92, 1.14, length(vec2(sc.x/cw, max(sc.y - SP_VOL_HOLE.w, 0.0)/SP_VOL_HOLE.z)));
  return beam*along*hole;
}
float setFog(vec3 pw){
  float h = max(pw.y - G0, 0.0);
  float bank = vnoise2(pw.xz*0.085 + vec2(3.0, 1.0));
  float nearClear = smoothstep(SP_FOG_NEAR.x, SP_FOG_NEAR.y, length(pw - uCamPos));   // the air at the props is clear: their darks stay dark
  return (0.0085 + 0.0170*exp(-h*0.52)*(0.45 + 1.1*bank))*nearClear*SP_FOG_GAIN;
}
vec3 setFogAmb(vec3 rd){
  float k = 0.5 + 0.5*dot(rd, uSunDir);
  return mix(SP_FOG_LO, SP_FOG_HI, smoothstep(0.08, 0.42, k));
}

// ---------------------------------------------------------------------------
//  engine/sets/gallery.glsl — a sunlit gallery.  A pale terrazzo floor with brass strips, warm white plaster,
//  tall arched windows down the left wall throwing long crisp panes of light across floor and wall, an arch
//  through to a further room, a low plinth and potted plants at the frame edges.
//  Implements the SET API documented in engine/sets/autumn.glsl.
//
//  ROOM SPACE.  The building is turned PSI about y from the camera, so the camera looks into the far left corner
//  and sees the window wall as well as the back wall.  toRoom(): r.x along the back wall, r.y above the floor,
//  r.z away from the back wall.  The window wall is r.x = XW, the back wall r.z = ZB.
//
//  THE LIGHT.  scene_main's bake shadow is a soft-march (10*h/t); it cannot draw a glazing bar 8 units from the
//  window.  So the shell is left out of the shadow map (setMap(p, true) holds the furniture only) and direct
//  light is a PROJECTION: beamMask() slides a point along the beam to the window wall and asks whether it landed
//  on glass.  That is crisp at any distance, costs one arch and a few bars, and the same function drives the
//  light on the cube (setCookie), the air (setFog, setVol) and the reflections (roomEnv).
//  scene_main multiplies the sun term by s.alb, so the mask travels in s.alb and setAmbient() divides it back out
//  through gAmbMul.  Same carrier as lagoon.glsl.  gAdd is radiance that must not be lit at all (floor mirror).
//
//  The beam (SP_BEAM) is not quite uSunDir: the wall wants a raking light, the face wants a more frontal key.
//  The furniture's shadows follow uSunDir; the two stay close enough to read as one sun.
//
//  COST.  llvmpipe skips loops, not `if`: expensive parts sit in one-shot loops behind a bound, everything heavy
//  is #ifdef PASS_BG.  setCookie, setVol, setFog and roomEnv are plain arithmetic: they run every frame.
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-0.70)
#endif
#ifndef SP_ROOM_YAW
#define SP_ROOM_YAW 0.42
#endif
#ifndef SP_BEAM
#define SP_BEAM vec3(-0.88,0.105,0.46)                // toward the window's sun, ROOM axes
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.58,0.47,0.36)               // warm light off the floor
#endif
#ifndef SP_AMB_WALL
#define SP_AMB_WALL vec3(0.82,0.76,0.67)             // the room's own soft light
#endif
#ifndef SP_AMB_WIN
#define SP_AMB_WIN vec3(0.42,0.50,0.60)              // cool skylight from the window side
#endif
#ifndef SP_SKY
#define SP_SKY vec3(9.0,8.8,8.2)                     // the glass, blown out
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#ifndef SP_FLOOR_MIRROR
#define SP_FLOOR_MIRROR 1.0
#endif
#ifndef SP_SUBJECT
#define SP_SUBJECT vec4(0.0,0.0,0.70,-0.34)          // floor x, z, half width, yaw of the block standing here
#endif
#ifndef SP_SUBJECT_H
#define SP_SUBJECT_H 1.40
#endif
#ifndef SP_SUBJECT_C
#define SP_SUBJECT_C vec3(0.030,0.31,0.33)           // what the floor mirrors of it
#endif
#define SET_GALLERY 1

const float G0  = SP_GROUND_Y;
const float PSI = SP_ROOM_YAW;
const vec2  REX = vec2(cos(PSI), sin(PSI)), REZ = vec2(-sin(PSI), cos(PSI));
const vec3  BL  = normalize(SP_BEAM);

const float XW = -5.20, XR = 9.0, ZB = -5.0, ZN = 15.0, HC = 7.4;        // the room
const float TW = 0.36;                                                  // wall thickness
const float W_S0 = -3.05, W_SP = 2.52, W_A = 0.74, W_B = 0.30, W_H = 1.98;   // windows: first, spacing, half width, sill, spring
const float W_N = 4.0;
const float W_BAR = 0.017;
const float A_X = 1.30, A_W = 1.00, A_H = 1.78;                         // the arch: centre, half width, spring
const float F_X0 = -2.6, F_X1 = 6.4, F_Z = -11.5, F_H = 5.6;             // the further room

#define M_CANOPY 99.0     // unused: nothing here ignores direct shadowing
#define M_FLOOR   1.0
#define M_PLASTER 2.0
#define M_FAR     3.0
#define M_SKIRT   4.0
#define M_BAR     5.0
#define M_PLINTH  6.0
#define M_POT     7.0
#define M_SOIL    8.0
#define M_LEAF    9.0
#define M_STEM   10.0
#define M_FRAME  11.0
#define M_ART    12.0
#define M_VASE   13.0

vec2  gWind;
float gAmbMul = 1.0;     // PASS_BG: see the header
float gGI = 0.0;         // PASS_BG: how much sunlit surface this point can see, for the warm bounce
vec3  gAdd = vec3(0.0);

vec3 toRoom(vec3 p){ return vec3(dot(p.xz, REX), p.y - G0, dot(p.xz, REZ)); }
vec3 dirRoom(vec3 d){ return vec3(dot(d.xz, REX), d.y, dot(d.xz, REZ)); }
vec3 dirWorld(vec3 r){ return vec3(r.x*REX.x + r.z*REZ.x, r.y, r.x*REX.y + r.z*REZ.y); }

void setInit(){
  float T = uTime;
  gWind = 0.05*vec2(sin(T*0.13), cos(T*0.11)) + 0.02*vec2(sin(T*0.37 + 1.0), sin(T*0.29));
}

// ------------------------------------------------------------------ the windows
float arch2D(float c, float h, float A, float B, float H){
  return h < H ? max(abs(c) - A, B - h) : length(vec2(c, h - H)) - A;
}
// which bay; c = across it
float winCell(float s){
  float id = clamp(floor((s - W_S0)/W_SP + 0.5), 0.0, W_N - 1.0);
  return s - W_S0 - id*W_SP;
}
// distance to the glazing bars, in the plane of the window
float winBars(float c, float h){
  float ac = abs(c);
  float v = min(ac, abs(ac - 0.5*W_A*1.02));
  float y = h - W_B;
  float hb = abs(mod(y + 0.32, 0.64) - 0.32);               // panes 0.64 high
  float below = max(min(v, hb), h - W_H);
  below = min(below, abs(h - W_H));                         // the transom
  // the fan: a ring and five spokes
  vec2 f = vec2(c, h - W_H);
  float r = length(f), a = atan(f.x, f.y);
  float sec = 3.14159/5.0;
  float sp = abs(mod(a + sec*0.5, sec) - sec*0.5)*r;
  float fan = min(abs(r - 0.52*W_A), max(sp, 0.52*W_A - r));
  fan = max(fan, W_H - h);
  return min(below, fan) - W_BAR;
}
// 1 where the point can see the sun through the glass.  r = room space
float beamMask(vec3 r){
  float t = max((XW - TW*0.4 - r.x)/BL.x, 0.0);
  float s = r.z + BL.z*t, h = r.y + BL.y*t;
  float c = winCell(s);
  float pw = 0.003 + 0.0034*t;
  float o = arch2D(c, h, W_A, W_B, W_H);
  float m = smoothstep(pw, -pw, o)*smoothstep(-pw, pw, winBars(c, h));
  // behind the back wall the light has to come through the arch
  float tb = (ZB - TW*0.5 - r.z)/BL.z;
  float behind = step(r.z, ZB - TW*0.5);
  float door = arch2D(r.x + BL.x*tb - A_X, r.y + BL.y*tb, A_W, -1.0, A_H);
  return m*mix(1.0, smoothstep(0.03, -0.03, door), behind);
}
// the same with soft edges and no bars: air, not a surface
float beamSoft(vec3 r){
  float t = max((XW - TW*0.4 - r.x)/BL.x, 0.0);
  float s = r.z + BL.z*t, h = r.y + BL.y*t;
  float o = arch2D(winCell(s), h, W_A, W_B, W_H);
  return smoothstep(0.34, -0.26, o)*step(XW, r.x)*step(ZB, r.z);
}

// ------------------------------------------------------------------ what the glaze and the floor see
// one bounce, analytic: the ray from p leaves in `d` and lands on a wall of the room
vec3 roomEnv(vec3 pw, vec3 dw, float crisp){
  vec3 o = toRoom(pw), d = dirRoom(dw);
  float tx = d.x < 0.0 ? (XW - o.x)/min(d.x, -1e-4) : (XR - o.x)/max(d.x, 1e-4);
  float tz = d.z < 0.0 ? (ZB - o.z)/min(d.z, -1e-4) : (ZN - o.z)/max(d.z, 1e-4);
  float ty = d.y < 0.0 ? -o.y/min(d.y, -1e-4)       : (HC - o.y)/max(d.y, 1e-4);
  // the walls
  float t = min(tx, tz);
  vec3 h = o + d*t;
  vec3 wall = vec3(0.90,0.85,0.76);
  vec3 c = wall*0.46;
  if(tx > tz && d.z < 0.0){
    c = wall*(0.50 + 1.25*mix(beamSoft(h), beamMask(h), crisp));
    float door = arch2D(h.x - A_X, h.y, A_W, -1.0, A_H);
    c = mix(c, vec3(0.42,0.27,0.16), smoothstep(0.03, -0.03, door));
  } else if(tx <= tz && d.x < 0.0){
    c = wall*0.42;
    float cw = winCell(h.z);
    float o2 = arch2D(cw, h.y, W_A, W_B, W_H);
    c = mix(c, SP_SKY*(0.25 + 0.75*smoothstep(-0.03, 0.03, winBars(cw, h.y))), crisp*smoothstep(0.02, -0.02, o2));
  }
  // floor and ceiling.  The glaze sees them through a soft horizon: the one sharp thing on it is the window
  vec3 hy = o + d*ty;
  vec3 fl = d.y < 0.0 ? vec3(0.62,0.58,0.51)*(0.50 + 1.15*mix(beamSoft(hy), beamMask(hy), crisp))
                      : vec3(0.86,0.82,0.76)*0.62;
  float k = mix(d.y < 0.0 ? smoothstep(-0.03, -0.26, d.y) : smoothstep(0.30, 0.70, d.y), step(ty, t), crisp);
  return mix(c, fl, k);
}

#ifdef PASS_BG
// outside the glass: a bright sky, the top of a garden, all blown out
vec3 setSky(vec3 rd){
  float y = rd.y + 0.10*vnoise2(rd.xz*5.0)*smoothstep(0.30, 0.0, rd.y);
  vec3 c = mix(SP_SKY*vec3(0.96,0.92,0.82), SP_SKY*vec3(0.74,0.90,1.10), smoothstep(0.02, 0.55, y));
  float tr = smoothstep(0.10, 0.05, y + 0.12*(fbm2(rd.xz*vec2(6.0,3.0)) - 0.5));
  return mix(c, vec3(1.25,1.70,0.92), tr);
}
#else
vec3 setSky(vec3 rd){
  return mix(vec3(0.62,0.54,0.44), vec3(0.92,0.86,0.76), smoothstep(-0.30, 0.45, rd.y));
}
#endif

vec3 setAmbient(vec3 pw, vec3 n){
  vec3 r = toRoom(pw), nr = dirRoom(n);
  float toWin = clamp(0.5 - 0.5*nr.x, 0.0, 1.0);                       // faces the far side, sees the windows' wall glow
  float win = 0.5 + 0.5*nr.x;                                          // faces the windows
  float nearW = 0.45 + 0.75*exp(-max(r.x - XW, 0.0)*0.20);
  float up = 0.5 + 0.5*n.y;
  vec3 a = SP_AMB_WALL*(0.74 + 0.34*up) + SP_BOUNCE*(1.0 - up)*0.86 + SP_AMB_WIN*win*win*nearW*0.72 + vec3(0.13,0.11,0.09)*toWin;
#ifdef PASS_BG
  a += vec3(0.86,0.64,0.40)*0.42*gGI;
  return a*gAmbMul + gAdd;
#else
  return a;
#endif
}

// the pane light on the character.  A hair of drift: the sun does move
float setCookie(vec3 pw){
  vec3 r = toRoom(pw);
  // the block stands in a broad pool: bars across a face read as scars.  A long slow breath, and the far flank falls off
  float s = 0.5 + 0.5*sin(uTime*0.09 + r.z*0.5 + r.x*0.3);
  return 0.90 + 0.10*s*s;
}
float setVol(vec3 pw){
  vec3 r = toRoom(pw);
  return beamSoft(r)*(0.80 + 0.20*sin(uTime*0.17 + r.z*0.7 + r.x*0.4));
}
float setFog(vec3 pw){
  vec3 r = toRoom(pw);
  float inRoom = step(XW, r.x)*step(ZB - TW - 8.0, r.z)*step(r.y, HC);
  float d = 0.0022*exp(-max(r.y, 0.0)*0.05);
#ifdef PASS_BG
  // the shafts themselves are baked: 24 jittered samples per ray, per sample, crisp at the panes
  d += 0.022*beamMask(r)*step(ZB, r.z)*smoothstep(0.4, -1.8, r.z);
#endif
  return d*inRoom*SP_FOG_GAIN;
}
vec3 setFogAmb(vec3 rd){
  return mix(vec3(0.80,0.76,0.70), vec3(1.25,1.02,0.76), clamp(dot(rd, dirWorld(BL))*0.5 + 0.5, 0.0, 1.0));
}

// =========================================================================== bake only
#ifdef PASS_BG
float sdBox3(vec3 p, vec3 b){ vec3 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0); }

// -------- the free space of the two rooms, the arch and the windows.  Solid is what is left
float shell(vec3 r, out float room1, out float passD, out float passW){
  room1 = max(max(XW - r.x, r.x - XR), max(max(-r.y, r.y - HC), max(ZB - r.z, r.z - ZN)));
  float room2 = max(max(F_X0 - r.x, r.x - F_X1), max(max(-r.y, r.y - F_H), max(F_Z - r.z, r.z - (ZB - TW))));
  float a = arch2D(r.x - A_X, r.y, A_W, -1.0, A_H);
  passD = max(a, max(ZB - TW - 0.05 - r.z, r.z - ZB - 0.05));
  float w = arch2D(winCell(r.z), r.y, W_A, W_B, W_H);
  passW = max(w, max(XW - TW - 0.05 - r.x, r.x - XW - 0.05));
  float outside = r.x - (XW - TW);
  return -min(min(room1, room2), min(min(passD, passW), outside));
}

// -------- furniture, WORLD space.  Every one in a one-shot loop behind a bound
const vec3 PL_P  = vec3(-2.30, 0.0, -1.10);      // plinth (y = height of its top above the floor is PL_H)
const vec3 PL_B  = vec3(0.56, 0.0, 0.56);
const float PL_H = 0.74;
const vec3 POT_A = vec3( 2.66, 0.0, -1.25);      // the fig at the right edge
const vec3 POT_B = vec3(5.10, 0.0, -8.10);      // a palm-ish one in the far corner
const float POT_H = 0.78, POT_R = 0.46;

vec3  gNearQ; float gNearId;
vec3  gTmpQ;  float gTmpId;
vec2 opUn(vec2 a, vec2 b){ if(b.x < a.x){ gNearQ = gTmpQ; gNearId = gTmpId; return b; } return a; }

vec2 pot(vec3 p, float H, float R){
  vec2 q = vec2(length(p.xz), p.y);
  float t = clamp(q.y/H, 0.0, 1.0);
  float rad = mix(R*0.70, R, pow(t, 0.8));
  float d = (q.x - rad)*0.94;
  float rim = max(q.x - R - 0.035, abs(q.y - H + 0.075) - 0.075);
  d = smin(d, rim, 0.02);
  d = smax(d, q.y - H, 0.02);
  d = smax(d, -q.y, 0.02);
  float soil = max(q.x - R*0.93, q.y - H + 0.05);
  gTmpQ = vec3(q, t); gTmpId = 0.0;
  return (d < soil + 0.0 && q.y > H - 0.06 && q.x < R*0.93) ? vec2(max(soil, q.y - H), M_SOIL) : vec2(d, M_POT);
}
// a plant: a slim trunk and NL big glossy leaves spiralling up it
vec2 plant(vec3 p, vec3 base, float H, float S, float sd, int NL){
  float lean = 0.06*sin(sd);
  vec3 tip = base + vec3(lean, H, 0.05*cos(sd*1.7));
  float tr = sdTaper(p, base, tip, 0.040, 0.016);
  vec2 res = vec2(tr, M_STEM);
  gTmpQ = p; gTmpId = sd;
  for(int i=0;i<NL;i++){
    float fi = float(i), t = (fi + 0.5)/float(NL);
    vec3 h = hash31(fi*1.93 + sd), h2 = hash31(fi*4.37 + sd + 3.0);
    vec3 o = mix(base + vec3(0.0, 0.24*H, 0.0), tip, pow(t, 0.8));
    float az = fi*2.399 + sd + (h.x - 0.5)*0.7;
    vec3 dir = normalize(vec3(cos(az), mix(0.75, 0.05, t) - 0.25 + 0.5*h.y, sin(az)));
    vec3 nr = normalize(vec3(-dir.x*dir.y, 1.0 - dir.y*dir.y, -dir.z*dir.y) + 0.45*(h2 - 0.5));
    vec3 xa = normalize(cross(dir, nr)), za = cross(xa, dir);
    vec3 v = p - o;
    vec3 lq = vec3(dot(v, xa), dot(v, dir), dot(v, za));
    float Sz = S*mix(1.15, 0.72, t)*mix(0.85, 1.12, h.z), Wd = Sz*0.34;
    float stalk = sdTaper(lq, vec3(0.0), vec3(0.0, 0.08, 0.0), 0.010, 0.006);
    if(stalk < res.x) res = vec2(stalk, M_STEM);
    lq.y -= 0.075;
    lq.z += 0.55*lq.x*lq.x/Wd + 0.42*lq.y*lq.y/Sz - 0.05*sin(lq.y/Sz*3.14159);    // cupped, drooping
    float tl = clamp(lq.y/Sz, 0.0, 1.0);
    float wd = Wd*pow(sin(3.14159*pow(tl, 0.92)), 0.72)*(0.74 + 0.40*tl);
    wd *= 1.0 + 0.05*sin(tl*17.0 + sd);
    float d = max(abs(lq.x) - wd, max(-lq.y, lq.y - Sz));
    d = max(d*0.70, abs(lq.z) - 0.0055)*0.80;
    if(d < res.x){ res = vec2(d, M_LEAF); gTmpQ = vec3(lq.x/max(wd, 1e-3), tl, 0.0); gTmpId = h2.x; }
  }
  return res;
}
// two works on the wall, one in the further room.  Room space
const vec4 ART_A = vec4(-3.78, 1.86, 0.50, 0.62);     // along the wall, height, half width, half height
const vec4 ART_B = vec4( 1.75, 1.72, 1.05, 0.76);     // in the further room
vec2 art(vec3 r, vec4 a, float z, float id){
  vec3 q = vec3(r.x - a.x, r.y - a.y, r.z - z);
  float fr = sdBox3(q, vec3(a.z + 0.05, a.w + 0.05, 0.05));
  float cv = sdBox3(q - vec3(0.0, 0.0, 0.04), vec3(a.z, a.w, 0.024));
  gTmpQ = vec3(q.x/a.z, q.y/a.w, 0.0); gTmpId = id;
  return cv < fr ? vec2(cv, M_ART) : vec2(fr, M_FRAME);
}

vec2 nearMap(vec3 pw, bool shadow){
  const float G = 0.14;
  vec3 r = toRoom(pw);
  vec3 p = vec3(pw.x, pw.y - G0, pw.z);           // world, floor at 0
  vec2 res = vec2(G, M_PLINTH);

  vec3 qp = p - PL_P;
  float bP = length(qp - vec3(0.0, 0.6, 0.0)) - 1.25;
  res.x = max(bP, G);
  for(int i=0;i<1 && bP < G;i++){
    vec2 xz = rot(PSI)*qp.xz;
    float d = sdBox3(vec3(xz.x, qp.y - PL_H*0.5, xz.y), vec3(PL_B.x, PL_H*0.5, PL_B.z)) - 0.012;
    res = vec2(d, M_PLINTH);
    vec3 qv = qp - vec3(0.02, PL_H, 0.02);
    // a little glazed vase: a cousin
    float yy = qv.y;
    float yn = (yy - 0.20)/0.215;
    float rv = 0.255*sqrt(max(1.0 - yn*yn, 0.0));
    rv = max(rv, 0.075 + 0.04*smoothstep(0.34, 0.50, yy));
    float dv = max(length(qv.xz) - rv, max(-yy, yy - 0.50))*0.70;
    gTmpQ = qv; gTmpId = 0.0;
    res = opUn(res, vec2(dv, M_VASE));
  }

  vec3 qa = p - POT_A;
  float bA = sdCapsule(p, POT_A, POT_A + vec3(0.0, 3.1, 0.0), 1.05);
  res.x = min(res.x, max(bA, G));
  for(int i=0;i<1 && bA < G;i++){
    res = opU(res, pot(qa, POT_H, POT_R));
    res = opUn(res, plant(p, POT_A + vec3(0.0, POT_H - 0.05, 0.0), 2.35, 0.82, 2.2, 16));
  }

  vec3 qb = p - POT_B;
  float bB = sdCapsule(p, POT_B, POT_B + vec3(0.0, 2.9, 0.0), 1.15);
  res.x = min(res.x, max(bB, G));
  for(int i=0;i<1 && bB < G;i++){
    res = opU(res, pot(qb, POT_H, POT_R));
    res = opUn(res, plant(p, POT_B + vec3(0.0, POT_H - 0.05, 0.0), 2.05, 1.05, 5.1, 12));
  }
  return res;
}

vec2 setMap(vec3 pw, bool shadow){
  vec3 r = toRoom(pw);
  vec2 res = nearMap(pw, shadow);
  if(shadow) return res;                          // see the header: the shell does not shadow, beamMask does
  float room1, passD, passW;
  float s = shell(r, room1, passD, passW);
  res = opU(res, vec2(s, r.y < 0.03 ? M_FLOOR : (r.z < ZB - TW*0.98 ? M_FAR : M_PLASTER)));
  // skirting along the first room's walls, stopping at the openings
  float rw = max(max(XW - r.x, r.x - XR), max(ZB - r.z, r.z - ZN));
  float sk = max(max(r.y - 0.20, -rw - 0.022), -passD);
  sk = max(sk, -passW);
  res = opU(res, vec2(sk, M_SKIRT));
  // glazing bars, in the middle of the wall's thickness
  float c = winCell(r.z);
  float bx = abs(r.x - XW + TW*0.40) - 0.022;
  float bd = max(max(winBars(c, r.y), bx), arch2D(c, r.y, W_A + 0.03, W_B - 0.03, W_H));
  res = opU(res, vec2(bd, M_BAR));
  if(r.z < ZB + 0.3 && r.x < 0.0) res = opU(res, art(r, ART_A, ZB, 0.0));
  if(r.z < F_Z + 0.3) res = opU(res, art(r, ART_B, F_Z, 1.0));
  return res;
}

// -------- paint
vec3 stonePick(float h){
  if(h < 0.26) return vec3(0.60,0.57,0.52);        // warm grey
  if(h < 0.48) return vec3(0.92,0.88,0.80);        // marble
  if(h < 0.66) return vec3(0.68,0.49,0.31);        // ochre
  if(h < 0.80) return vec3(0.34,0.32,0.30);        // charcoal
  if(h < 0.92) return vec3(0.62,0.28,0.18);        // rust
  return vec3(0.42,0.52,0.47);                     // sea green
}
vec3 terrazzo(vec2 p){
  vec3 a = vec3(0.84,0.805,0.745)*(0.93 + 0.12*vnoise2(p*0.9))*(0.97 + 0.06*vnoise2(p*23.0));
  for(int i=0;i<3;i++){
    float fi = float(i);
    float sc = i == 0 ? 10.0 : (i == 1 ? 21.0 : 43.0);
    vec2 g = p*sc + fi*11.3, c = floor(g), f = fract(g);
    float F1 = 9.0, F2 = 9.0; vec2 id = c;
    for(int j=-1;j<=1;j++) for(int k=-1;k<=1;k++){
      vec2 o = vec2(float(k), float(j));
      vec2 rr = o + hash22(c + o + fi*7.0) - f;
      float d = dot(rr, rr);
      if(d < F1){ F2 = F1; F1 = d; id = c + o; } else if(d < F2) F2 = d;
    }
    float e = sqrt(F2) - sqrt(F1);
    vec3 h = hash33(vec3(id, fi + 3.0));
    float on = step(h.z, i == 0 ? 0.40 : (i == 1 ? 0.56 : 0.52));
    float chip = smoothstep(0.13, 0.20, e)*on;
    a = mix(a, mix(stonePick(h.x), vec3(0.80,0.765,0.705), 0.30)*(0.92 + 0.16*h.y), chip*0.92);
  }
  return a;
}
// ray from a point on the floor to the block: analytic, for the mirror.  Returns 0..1 hit, height of the hit
float mirrorBlock(vec3 ro, vec3 rd, out float hh, out float sideK){
  vec4 sb = SP_SUBJECT;
  vec2 c = vec2(cos(sb.w), sin(sb.w));
  mat2 R = mat2(c.x, c.y, -c.y, c.x);
  vec2 o = R*(ro.xz - sb.xy), d = R*rd.xz;
  vec3 oo = vec3(o.x, ro.y - G0 - 0.5*SP_SUBJECT_H, o.y), dd = vec3(d.x, rd.y, d.y);
  vec3 bb = vec3(sb.z, 0.5*SP_SUBJECT_H, sb.z);
  vec3 ia = (-bb - oo)/dd, ib = (bb - oo)/dd;
  vec3 tmin = min(ia, ib), tmax = max(ia, ib);
  float t0 = max(tmin.x, max(tmin.y, tmin.z)), t1 = min(tmax.x, min(tmax.y, tmax.z));
  float hit = (t1 > max(t0, 0.0)) ? 1.0 : 0.0;
  hh = oo.y + 0.5*SP_SUBJECT_H + dd.y*t0;
  sideK = tmin.x > tmin.z ? 1.0 : 0.0;
  return hit;
}

void setSurface(vec3 pw, vec3 rd, float t, float m, vec3 n, inout Surf s){
  vec3 r = toRoom(pw);
  gAdd = vec3(0.0);
  s.spec = 0.0; s.sheen = 0.0; s.wrap = 0.0; s.fur = 0.0;
  for(int i=0;i<1 && m == M_FLOOR;i++){
    vec2 xz = r.xz;
    s.alb = terrazzo(xz + vec2(40.0, 40.0));
    // brass strips, a grid of big panels
    vec2 gq = abs(fract(xz/2.6 + 0.27) - 0.5)*2.6;
    float strip = smoothstep(0.016, 0.009, min(gq.x, gq.y) - 0.0);
    s.alb = mix(s.alb, vec3(0.50,0.36,0.17), strip*0.85);
    if(r.z < ZB - TW*0.5) s.alb *= vec3(0.94,0.90,0.84);
    // the contact: a soft dark around the base of the block
    vec4 sb = SP_SUBJECT;
    vec2 c = vec2(cos(sb.w), sin(sb.w));
    vec2 bq = mat2(c.x, c.y, -c.y, c.x)*(pw.xz - sb.xy);
    // the block's corners are round: so is its contact.  x = distance outside the silhouette of the foot
    const float RF = 0.24;
    vec2 bd = abs(bq) - (sb.z - RF);
    float x = length(max(bd, 0.0)) + min(max(bd.x, bd.y), 0.0) - RF;
    x = max(x + 0.055, 0.0);
    // and the room's light comes from the left: a long soft shade trails away to the right
    vec2 tq = bq - vec2(0.32, -0.20);
    vec2 td = abs(tq) - (sb.z - RF);
    float xt = max(length(max(td, 0.0)) + min(max(td.x, td.y), 0.0) - RF, 0.0);
    s.occ *= max(1.0 - 0.62*exp(-x/0.040) - 0.40*exp(-x/0.26), 0.06)*(1.0 - 0.34*exp(-xt/0.34));
    // the furniture stands on the floor too
    vec2 fp = rot(PSI)*(pw.xz - PL_P.xz);
    vec2 fd = abs(fp) - PL_B.xz;
    float dp = length(max(fd, 0.0)) + min(max(fd.x, fd.y), 0.0);
    float da = min(length(pw.xz - POT_A.xz) - POT_R*0.70, length(pw.xz - POT_B.xz) - POT_R*0.70);
    s.occ *= (1.0 - 0.70*exp(-max(dp, 0.0)/0.05) - 0.24*exp(-max(dp, 0.0)/0.32))*(1.0 - 0.66*exp(-max(da, 0.0)/0.05) - 0.26*exp(-max(da, 0.0)/0.30));
    // polish: a fresnel mirror, blurred by a jittered normal and more so with the height of what it sees
    float jit = hash12(gl_FragCoord.xy*1.73 + uJitter*31.0), jit2 = hash12(gl_FragCoord.yx*2.11 + uJitter*17.0);
    vec3 nn = normalize(vec3(0.0, 1.0, 0.0) + vec3(jit - 0.5, 0.0, jit2 - 0.5)*0.085);
    vec3 d2 = reflect(rd, nn);
    float nv = clamp(-rd.y, 0.0, 1.0);
    float Fr = 0.030 + 0.50*pow(1.0 - nv, 5.0);
    vec3 env = roomEnv(pw + vec3(0.0, 0.003, 0.0), d2, 1.0);
    float hh, sk;
    float hit = mirrorBlock(pw, d2, hh, sk);
    vec3 blk = SP_SUBJECT_C*mix(1.15, 0.55, sk)*0.9*(1.0 - 0.35*smoothstep(0.0, 0.3, hh));
    env = mix(env, blk, 0.80*hit*exp(-max(hh, 0.0)*1.15));
    gAdd = env*Fr*SP_FLOOR_MIRROR*(1.0 - strip*0.5);
    s.wrap = 0.0;
  }
  for(int i=0;i<1 && (m == M_PLASTER || m == M_FAR || m == M_SKIRT);i++){
    float big = fbm3(pw*vec3(0.55,0.55,0.55));
    float trow = vnoise3(pw*vec3(2.6,1.3,2.6))*0.6 + vnoise3(pw*vec3(7.0,3.2,7.0))*0.4;
    vec3 pl = vec3(0.885,0.835,0.745)*(0.93 + 0.12*big)*(0.975 + 0.05*trow);
    if(m == M_FAR) pl = vec3(0.78,0.56,0.37)*(0.93 + 0.12*big)*(0.97 + 0.06*trow);
    if(m == M_SKIRT) pl = vec3(0.86,0.815,0.735);
    s.alb = pl;
    s.n = normalize(n + 0.012*(vec3(vnoise3(pw*5.0), vnoise3(pw*5.0 + 4.0), vnoise3(pw*5.0 + 9.0)) - 0.5));
    // dark where wall meets floor; a real corner is never lit evenly
    s.occ *= 1.50*(0.78 + 0.22*smoothstep(0.0, 0.70, r.y));
    float corner = min(r.x - XW, r.z - ZB);
    s.occ *= 0.74 + 0.26*smoothstep(0.0, 1.1, corner);
    // the reveal of the arch is a touch darker inside
    if(r.z < ZB + 0.01 && r.z > ZB - TW - 0.01) s.occ *= 0.82;
    s.wrap = 0.05;
  }
  if(m == M_BAR){ s.alb = vec3(0.17,0.145,0.12); s.spec = 0.10; s.shin = 30.0; }
  for(int i=0;i<1 && m >= M_PLINTH;i++){
    nearMap(pw, false);
    vec3 q = gNearQ;
    if(m == M_PLINTH){
      s.alb = vec3(0.86,0.83,0.77)*(0.96 + 0.08*vnoise3(pw*3.0));
      s.occ *= 0.70 + 0.30*smoothstep(0.0, 0.5, r.y);
    } else if(m == M_POT){
      float band = smoothstep(0.02, 0.0, abs(q.y - 0.64))*0.3;
      s.alb = vec3(0.62,0.27,0.135)*(0.82 + 0.34*fbm3(pw*vec3(5.0,14.0,5.0)))*(1.0 - band);
      s.spec = 0.05; s.shin = 18.0; s.wrap = 0.12; s.occ *= 1.5;
    } else if(m == M_SOIL){
      s.alb = vec3(0.09,0.065,0.05)*(0.7 + 0.6*vnoise3(pw*40.0));
    } else if(m == M_LEAF){
      float tone = fbm3(pw*6.0);
      vec3 c1 = mix(vec3(0.055,0.15,0.055), vec3(0.11,0.27,0.075), gNearId);
      vec3 ca = mix(c1, c1*vec3(1.35,1.25,0.85), smoothstep(0.30, 0.70, tone));
      float vein = smoothstep(0.050, 0.014, abs(q.x));
      float side = abs(fract(q.y*7.0 - abs(q.x)*2.0) - 0.5);
      vein = max(vein, 0.55*smoothstep(0.05, 0.015, side)*smoothstep(1.0, 0.3, abs(q.x)));
      s.alb = mix(ca, ca*vec3(1.7,1.8,1.25) + 0.015, vein*0.62);
      s.wrap = 0.35; s.trans = 0.7; s.transCol = s.alb*1.6; s.spec = 0.50; s.shin = 46.0;
    } else if(m == M_STEM){
      s.alb = mix(vec3(0.12,0.085,0.055), vec3(0.26,0.19,0.12), vnoise3(pw*vec3(40.0,9.0,40.0)));
      s.wrap = 0.2; s.spec = 0.04; s.shin = 14.0;
    } else if(m == M_VASE){
      float gl = smoothstep(0.062, 0.070, q.y + 0.010*sin(atan(q.z, q.x)*5.0));
      s.alb = mix(vec3(0.78,0.62,0.46), vec3(0.008,0.165,0.185), gl);
      s.spec = 1.4*gl; s.shin = 180.0; s.wrap = 0.08; s.occ *= 1.4;
    }
  }
  for(int i=0;i<1 && (m == M_ART || m == M_FRAME);i++){
    float pick = 0.0; vec2 uv = vec2(0.0);
    pick = r.z > ZB - 1.0 ? 0.0 : 1.0;
    uv = pick < 0.5 ? vec2(r.x - ART_A.x, r.y - ART_A.y)/ART_A.zw : vec2(r.x - ART_B.x, r.y - ART_B.y)/ART_B.zw;
    if(m == M_FRAME){
      s.alb = vec3(0.30,0.20,0.115)*(0.85 + 0.3*vnoise3(pw*vec3(30.0,30.0,30.0)));
      s.spec = 0.10; s.shin = 30.0;
    } else {
      float wob = 0.05*(vnoise2(uv*5.0) - 0.5) + 0.018*(vnoise2(uv*23.0) - 0.5);
      vec3 ca = vec3(0.84,0.79,0.68);
      if(pick < 0.5){
        // a sun over two hills
        ca = mix(vec3(0.82,0.68,0.46), vec3(0.76,0.40,0.27), smoothstep(-0.1, 0.8, uv.y + wob));
        float sun = smoothstep(0.03, -0.03, length(uv - vec2(0.22, 0.34)) - 0.33 + wob);
        ca = mix(ca, vec3(0.93,0.82,0.54), sun);
        float hill = smoothstep(0.02, -0.02, uv.y + 0.18 + 0.22*sin(uv.x*2.3 + 0.8) + wob);
        ca = mix(ca, vec3(0.05,0.28,0.31), hill);
        float h2 = smoothstep(0.02, -0.02, uv.y + 0.56 + 0.16*sin(uv.x*2.9 + 3.0) + wob);
        ca = mix(ca, vec3(0.03,0.13,0.16), h2);
      } else {
        // two fields of colour
        ca = mix(vec3(0.72,0.31,0.19), vec3(0.09,0.29,0.31), smoothstep(-0.04, 0.04, uv.y - 0.05 + wob));
        ca = mix(ca, vec3(0.90,0.82,0.62), smoothstep(0.03, -0.03, abs(uv.y - 0.05 + wob*1.3) - 0.05));
      }
      float weave = 0.97 + 0.05*vnoise2(uv*160.0);
      float mat_ = smoothstep(0.90, 0.88, max(abs(uv.x), abs(uv.y)));
      s.alb = mix(vec3(0.90,0.87,0.80), ca*weave, mat_);
    }
  }
  // the sun bounces: every surface near a pool of light glows warm.  Eight rays into the room, how many land in a beam
  vec3 nr = dirRoom(n);
  float gi = 0.0;
  for(int i=0;i<8;i++){
    vec3 d = hash31(float(i)*2.17 + 0.3)*2.0 - 1.0;
    d = normalize(d)*sign(dot(d, nr) + 1e-3) + nr*0.35;
    gi += beamSoft(r + d*1.25) + beamSoft(r + d*0.55);
  }
  gGI = gi/16.0;
  // the light goes through s.alb, see the header
  float M = m == M_PLINTH ? 0.86*beamSoft(r) : beamMask(r);       // pane bars on a near plinth blur into stains
  float k = max(M, 0.003);
  gAmbMul = 1.0/k;
  float kk = k;
  if(m == M_FLOOR) k *= 1.0 + 0.95*M;                // the sun rakes the floor: lift it to where the eye expects a pool of light
  gAdd /= kk*max(dot(s.alb, vec3(0.333)), 0.05);
  gAmbMul *= kk/k;
  gAdd *= kk/k;
  s.alb *= k; s.transCol *= k; s.spec *= M;
}
#endif

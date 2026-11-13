// ---------------------------------------------------------------------------
//  engine/sets/studio.glsl — a cozy studio corner in the late afternoon.  Warm oak planks, a faded wool rug under the
//  character, clay-coloured limewash walls with a cream skirting, a row of three tall steel-framed windows down the
//  left wall throwing warm panes of light across the floor and the back wall, a framed print and a fig tree in the
//  back-left corner, an oak shelf of books on the right.  The middle of the room is left empty: the character walks
//  there and the props (desk, chair) stand at the right.
//  Implements the SET API documented in engine/sets/autumn.glsl.  Started from gallery.glsl.
//
//  ROOM SPACE = WORLD SPACE (no yaw): floor y = SP_GROUND_Y, back wall z = ZB, window wall x = XL, right wall x = XR,
//  ceiling y = HC, front wall z = ZN (behind the camera, never seen).  The room encloses the camera, so the wide
//  panorama of the bake sees floor, three walls and the ceiling in every direction; only the glass looks out.
//
//  THE LIGHT is the gallery's.  The bake's shadow is a soft-march; it cannot draw a glazing bar 6 units from the
//  window, so the shell is left out of the shadow map (setMap(p, true) holds the furniture only) and the window
//  light is a PROJECTION: beamMask() slides a point along the beam to the window wall and asks whether it landed on
//  glass.  The same function drives the light on the character (setCookie), the air (setFog, setVol) and the
//  floor's reflections (roomEnv).  scene_main multiplies the sun term by s.alb, so the mask travels in s.alb and
//  setAmbient() divides it back out through gAmbMul.  gAdd is radiance that must not be lit at all (floor gloss).
//  The beam (SP_BEAM) is a little lower than uSunDir so that the panes reach the middle of the room; the character's
//  shadow follows uSunDir.  The two stay close enough to read as one sun.
//
//  COST.  llvmpipe skips loops, not `if`: expensive parts sit in one-shot loops behind a bound, everything heavy
//  is #ifdef PASS_BG.  setCookie, setVol, setFog and roomEnv are plain arithmetic: they run every frame.
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y 0.0
#endif
#ifndef SP_BEAM
#define SP_BEAM vec3(-0.74,0.30,0.60)                // toward the sun the windows see
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.46,0.31,0.19)               // warm light off the oak floor
#endif
#ifndef SP_AMB_WALL
#define SP_AMB_WALL vec3(0.44,0.36,0.29)             // the room's own soft light (clay walls)
#endif
#ifndef SP_AMB_WIN
#define SP_AMB_WIN vec3(0.46,0.47,0.49)              // skylight from the window side
#endif
#ifndef SP_SKY
#define SP_SKY vec3(6.0,5.5,4.6)                     // the glass, blown out
#endif
#ifndef SP_WALL_C
#define SP_WALL_C vec3(0.47,0.285,0.190)               // clay limewash
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#define SET_STUDIO 1

const float G0  = SP_GROUND_Y;
const vec3  BL  = normalize(SP_BEAM);

const float XL = -4.60, XR = 4.80, ZB = -3.40, ZN = 10.0, HC = 3.90;    // the room
const float TW = 0.30;                                                  // wall thickness
const float W_Z0 = -2.05, W_SP = 2.60, W_A = 1.00, W_B = 0.80, W_H = 3.30;   // windows: first centre, spacing, half width, sill, head
const float W_N = 5.0;                     // the last two are behind the wide views' edge: they light the middle of the room
const float W_BAR = 0.016;
const float SK_H = 0.22;                                                // skirting height
const float BM_Z0 = -1.60, BM_P = 2.80, BM_W = 0.13, BM_D = 0.20;      // oak beams across the ceiling: first, pitch, half width, depth

#define M_CANOPY 99.0     // unused: nothing here ignores direct shadowing
#define M_FLOOR   1.0
#define M_PLASTER 2.0
#define M_CEIL    3.0
#define M_SKIRT   4.0
#define M_BAR     5.0
#define M_RUG     6.0
#define M_POT     7.0
#define M_SOIL    8.0
#define M_LEAF    9.0
#define M_STEM   10.0
#define M_FRAME  11.0
#define M_ART    12.0
#define M_SHELF  13.0
#define M_BOOK   14.0
#define M_SILL   15.0
#define M_VASE   16.0
#define M_WOOD   17.0     // bookcase, bench
#define M_SOFA   18.0
#define M_CUSH   19.0     // cushions and pillows (q.z = which)
#define M_SHADE  20.0     // lamp shades: they glow
#define M_METAL  21.0
#define M_BEAM   22.0

vec2  gWind;
float gAmbMul = 1.0;     // PASS_BG: see the header
float gGI = 0.0;         // PASS_BG: how much sunlit surface this point can see, for the warm bounce
vec3  gAdd = vec3(0.0);
vec3  gLamp = vec3(0.0);  // PASS_BG: what the two lamps add (irradiance)
bool  gShaded = false;   // PASS_BG: set once setSurface ran.  The engine's AO taps (5 units, made for open sets)
                         // then see nothing: the occlusion of this low room is painted in setSurface instead

// room space: floor at y = 0
vec3 toRoom(vec3 p){ return vec3(p.x, p.y - G0, p.z); }

void setInit(){
  float T = uTime;
  gWind = 0.05*vec2(sin(T*0.13), cos(T*0.11)) + 0.02*vec2(sin(T*0.37 + 1.0), sin(T*0.29));
}

// ------------------------------------------------------------------ the windows
// which bay; c = across it (along z)
float winCell(float s){
  float id = clamp(floor((s - W_Z0)/W_SP + 0.5), 0.0, W_N - 1.0);
  return s - W_Z0 - id*W_SP;
}
// the two windows nearest the camera wear sheer linen: less sun on the floor in front of the character
float winTrans(float s){
  float id = clamp(floor((s - W_Z0)/W_SP + 0.5), 0.0, W_N - 1.0);
  return id > 3.5 ? 0.45 : (id > 2.5 ? 0.90 : 1.0);
}
// distance to the opening (negative inside), in the plane of the window
float winOpen(float c, float h){
  vec2 q = vec2(abs(c) - W_A, abs(h - 0.5*(W_B + W_H)) - 0.5*(W_H - W_B));
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}
// distance to the steel glazing bars: a 3 x 5 grid of panes and a frame
float winBars(float c, float h){
  float v = abs(abs(c) - W_A/3.0);
  float ph = (W_H - W_B)/5.0;
  float hb = abs(mod(h - W_B + 0.5*ph, ph) - 0.5*ph);
  float fr = -winOpen(c, h) - 0.035;                        // the frame around the edge
  return min(min(v, hb) - W_BAR, fr);
}
// 1 where the point can see the sun through the glass.  r = room space
float beamMask(vec3 r){
  float t = max((XL - TW*0.5 - r.x)/BL.x, 0.0);
  float s = r.z + BL.z*t, h = r.y + BL.y*t;
  float c = winCell(s);
  float pw = 0.006 + 0.0065*t;                             // soft: a big sky, not a point sun
  float o = winOpen(c, h);
  return smoothstep(pw, -pw, o)*smoothstep(-pw, pw, winBars(c, h))*winTrans(s);
}
// the same with soft edges and no bars: air, not a surface
float beamSoft(vec3 r){
  float t = max((XL - TW*0.5 - r.x)/BL.x, 0.0);
  float s = r.z + BL.z*t, h = r.y + BL.y*t;
  float o = winOpen(winCell(s), h);
  return smoothstep(0.34, -0.26, o)*step(XL, r.x)*step(ZB, r.z)*step(r.y, HC)*winTrans(s);
}

// ------------------------------------------------------------------ what the floor's gloss sees
// one bounce, analytic: the ray from p leaves in `d` and lands on a wall of the room
vec3 roomEnv(vec3 pw, vec3 d, float crisp){
  vec3 o = toRoom(pw);
  float tx = d.x < 0.0 ? (XL - o.x)/min(d.x, -1e-4) : (XR - o.x)/max(d.x, 1e-4);
  float tz = d.z < 0.0 ? (ZB - o.z)/min(d.z, -1e-4) : (ZN - o.z)/max(d.z, 1e-4);
  float ty = d.y < 0.0 ? -o.y/min(d.y, -1e-4)       : (HC - o.y)/max(d.y, 1e-4);
  float t = min(tx, tz);
  vec3 h = o + d*t;
  vec3 wall = SP_WALL_C;
  vec3 c = wall*0.42;
  if(tx > tz && d.z < 0.0){
    c = wall*(0.42 + 1.10*mix(beamSoft(h), beamMask(h), crisp));
  } else if(tx <= tz && d.x < 0.0){
    c = wall*0.36;
    float cw = winCell(h.z);
    float o2 = winOpen(cw, h.y);
    float bl = 0.06 + 0.10*t*(1.0 - crisp*0.5);           // the varnish blurs with distance
    float bars = mix(0.65, 0.30 + 0.70*smoothstep(-0.02, 0.02, winBars(cw, h.y)), crisp*smoothstep(0.3, 0.05, bl));
    c = mix(c, SP_SKY*bars, smoothstep(bl, -bl, o2));
  }
  // floor and ceiling.  The gloss sees them through a soft horizon: the one sharp thing in it is the window
  vec3 hy = o + d*ty;
  vec3 fl = d.y < 0.0 ? vec3(0.45,0.29,0.17)*(0.45 + 1.10*mix(beamSoft(hy), beamMask(hy), crisp))
                      : vec3(0.86,0.80,0.72)*0.50;
  float k = mix(d.y < 0.0 ? smoothstep(-0.03, -0.26, d.y) : smoothstep(0.30, 0.70, d.y), step(ty, t), crisp);
  return mix(c, fl, k);
}

#ifdef PASS_BG
// outside the glass: a bright warm sky over the top of a garden, all blown out
vec3 setSky(vec3 rd){
  float y = rd.y + 0.10*vnoise2(rd.yz*5.0)*smoothstep(0.30, 0.0, rd.y);
  vec3 c = mix(SP_SKY*vec3(1.00,0.93,0.80), SP_SKY*vec3(0.80,0.90,1.04), smoothstep(0.02, 0.60, y));
  float fo = fbm2(rd.yz*vec2(3.0,6.0));
  float tr = smoothstep(0.12, 0.05, y - 0.10*(fo - 0.5) - 0.05*sin(rd.z*9.0));
  // the tops of the garden's trees, sunlit and soft
  vec3 leaf = mix(vec3(0.16,0.30,0.07), vec3(0.70,0.85,0.30), smoothstep(0.35, 0.75, fbm2(rd.yz*vec2(9.0,14.0) + 3.0)))*1.2;
  return mix(c, leaf, tr);
}
#else
vec3 setSky(vec3 rd){
  return mix(vec3(0.55,0.42,0.32), vec3(0.88,0.80,0.70), smoothstep(-0.30, 0.45, rd.y));
}
#endif

vec3 setAmbient(vec3 pw, vec3 n){
  vec3 r = toRoom(pw);
  float win = clamp(0.5 - 0.5*n.x, 0.0, 1.0);                         // faces the windows
  float away = clamp(0.5 + 0.5*n.x, 0.0, 1.0);                        // faces the far wall: its glow
  float nearW = 0.55 + 0.65*exp(-max(r.x - XL, 0.0)*0.24);
  float up = 0.5 + 0.5*n.y;
  vec3 a = SP_AMB_WALL*(0.70 + 0.30*up) + SP_BOUNCE*(1.0 - up)*0.80 + SP_AMB_WIN*win*win*nearW*0.70
         + vec3(0.12,0.09,0.07)*away;
#ifdef PASS_BG
  a += vec3(0.86,0.60,0.38)*0.45*gGI + gLamp;
  return a*gAmbMul + gAdd;
#else
  return a*1.15;                                   // the character: a little more fill than the set's surfaces
#endif
}

// the window light on the character: a broad warm pool with a slow breath (the sun does move), falling off a little
// toward the far side of the room
float setCookie(vec3 pw){
  vec3 r = toRoom(pw);
  float s = 0.5 + 0.5*sin(uTime*0.09 + r.z*0.5 + r.x*0.3);
  // is the floor under this point in the pool?  (the windows' panes would be scars on a face: only the pool counts)
  float pool = beamSoft(vec3(r.x, 0.3, r.z));
  return (0.86 + 0.14*s*s)*mix(0.55, 1.0, pool);
}
float setVol(vec3 pw){
  vec3 r = toRoom(pw);
  return beamSoft(r)*(0.80 + 0.20*sin(uTime*0.17 + r.z*0.7 + r.x*0.4));
}
float setFog(vec3 pw){
  vec3 r = toRoom(pw);
  float inRoom = step(XL, r.x)*step(r.x, XR)*step(ZB, r.z)*step(r.y, HC);
  float d = 0.0020*exp(-max(r.y, 0.0)*0.05);
#ifdef PASS_BG
  // the shafts themselves are baked behind the character (24 jittered samples per ray), crisp at the panes
  d += 0.020*beamMask(r)*smoothstep(-0.6, -1.8, r.z);
#endif
  return d*inRoom*SP_FOG_GAIN;
}
vec3 setFogAmb(vec3 rd){
  return mix(vec3(0.80,0.72,0.64), vec3(1.25,1.00,0.74), clamp(dot(rd, BL)*0.5 + 0.5, 0.0, 1.0));
}

// =========================================================================== bake only
#ifdef PASS_BG
#ifndef BG_NO_CHAR
float charShadow(vec3 p);      // the character's shadow proxy (world space, scene_main): its contact on the floor
#endif
float sdBox3(vec3 p, vec3 b){ vec3 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0); }
float sdRect(vec2 p, vec2 b, float rr){ vec2 d = abs(p) - b + rr; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - rr; }

// -------- the free space of the room and the window openings.  Solid is what is left
float shell(vec3 r, out float room1, out float passW){
  room1 = max(max(XL - r.x, r.x - XR), max(max(-r.y, r.y - HC), max(ZB - r.z, r.z - ZN)));
  float w = winOpen(winCell(r.z), r.y);
  passW = max(w, max(XL - TW - 0.05 - r.x, r.x - XL - 0.05));
  float outside = r.x - (XL - TW);
  return -min(min(room1, passW), outside);
}

// -------- furniture, room space.  Every one in a one-shot loop behind a bound
const vec3 POT_A = vec3(-3.45, 0.0, -2.62);      // the fig in the back-left corner, by the windows
const float POT_H = 0.72, POT_R = 0.42;
const vec4 ART_A = vec4(-2.05, 2.02, 0.48, 0.60); // the print: along the wall, height, half width, half height
// scene: the wall TV hangs where the shelf of books was (renderer/animals/muse.props.json place "tv"): the shelf
// is gone (moved out of the room: its SDF and its shade on the wall never come near anything)
const vec3 SH_C = vec3(40.0, 2.00, ZB + 0.15);   // the shelf: centre of the board
const vec3 SH_B = vec3(1.05, 0.032, 0.15);

vec3  gNearQ; float gNearId;
vec3  gTmpQ;  float gTmpId;
vec2 opUn(vec2 a, vec2 b){ if(b.x < a.x){ gNearQ = gTmpQ; gNearId = gTmpId; return b; } return a; }

vec2 pot(vec3 p, float H, float R){
  vec2 q = vec2(length(p.xz), p.y);
  float t = clamp(q.y/H, 0.0, 1.0);
  float rad = mix(R*0.72, R, pow(t, 0.8));
  float d = (q.x - rad)*0.94;
  float rim = max(q.x - R - 0.03, abs(q.y - H + 0.065) - 0.065);
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
vec2 art(vec3 r, vec4 a, float z, float id){
  vec3 q = vec3(r.x - a.x, r.y - a.y, r.z - z);
  float fr = sdBox3(q, vec3(a.z + 0.05, a.w + 0.05, 0.045));
  float cv = sdBox3(q - vec3(0.0, 0.0, 0.035), vec3(a.z, a.w, 0.022));
  gTmpQ = vec3(q.x/a.z, q.y/a.w, 0.0); gTmpId = id;
  return cv < fr ? vec2(cv, M_ART) : vec2(fr, M_FRAME);
}
// the shelf: an oak board, a row of books, a little trailing plant, a stoneware jug
vec2 shelf(vec3 r){
  vec3 q = r - SH_C;
  vec2 res = vec2(sdBox3(q, SH_B) - 0.006, M_SHELF);
  // books: a fixed pitch, each a little different, standing on the board
  const float BP = 0.088, BX0 = -0.98;
  float fx = (q.x - BX0)/BP;
  float id = clamp(floor(fx), 0.0, 8.0);
  for(int k=0;k<2;k++){
    float ii = clamp(id + float(k) - 0.5 + 0.5*sign(fract(fx) - 0.5), 0.0, 8.0);
    vec3 h = hash31(ii*3.71 + 1.0);
    float bw = BP*mix(0.36, 0.47, h.x), bh = mix(0.30, 0.44, h.y), bd = mix(0.10, 0.13, h.z);
    vec3 bq = q - vec3(BX0 + (ii + 0.5)*BP, SH_B.y + bh, -SH_B.z + 0.02 + bd);
    float d = sdBox3(bq, vec3(bw, bh, bd)) - 0.004;
    gTmpQ = vec3(bq.y/bh, bq.x/bw, 0.0); gTmpId = ii;
    res = opUn(res, vec2(d, M_BOOK));
  }
  // a leaning book at the end of the row
  {
    vec3 bq = q - vec3(BX0 + 9.0*BP + 0.10, SH_B.y, -SH_B.z + 0.13);
    bq.xy = rot(-0.36)*bq.xy;
    bq.y -= 0.19;
    float d = sdBox3(bq, vec3(0.032, 0.19, 0.11)) - 0.004;
    gTmpQ = vec3(bq.y/0.19, bq.x/0.032, 0.0); gTmpId = 9.0;
    res = opUn(res, vec2(d, M_BOOK));
  }
  // the jug
  {
    vec3 vq = q - vec3(0.80, SH_B.y, -0.02);
    float yy = vq.y;
    float yn = (yy - 0.14)/0.15;
    float rv = 0.105*sqrt(max(1.0 - yn*yn, 0.0));
    rv = max(rv, 0.040 + 0.012*smoothstep(0.22, 0.30, yy));
    float dv = max(length(vq.xz) - rv, max(-yy, yy - 0.30))*0.75;
    gTmpQ = vq; gTmpId = 0.0;
    res = opUn(res, vec2(dv, M_VASE));
  }
  // a small pot with a round mound of leaves
  {
    vec3 pq = q - vec3(0.36, SH_B.y, 0.0);
    vec2 pr = pot(pq, 0.17, 0.085);
    res = opU(res, pr);
    vec3 lq = pq - vec3(0.0, 0.25, 0.0);
    float n = vnoise3(r*14.0)*0.05 + vnoise3(r*31.0)*0.02;
    float d = (length(lq*vec3(1.0, 1.25, 1.0)) - 0.13 + n)*0.6;
    // a few trailing strands over the front edge
    float tr = sdTaper(pq, vec3(0.05, 0.20, 0.06), vec3(0.10, -0.32, 0.17), 0.030, 0.016) + n*0.6;
    tr = min(tr, sdTaper(pq, vec3(-0.06, 0.20, 0.05), vec3(-0.12, -0.20, 0.16), 0.028, 0.014) + n*0.6);
    d = min(d, tr*0.8);
    gTmpQ = pq; gTmpId = 0.7;
    res = opUn(res, vec2(d, M_LEAF));
  }
  return res;
}

// -------- along the side walls: only the wide views see these
const vec3 BC_LO = vec3(XR - 0.46, 0.0, -3.36), BC_HI = vec3(XR, 3.10, -1.80);   // the bookcase, back-right corner
const float SO_Z = 2.95, SO_D = 1.10;                                            // the sofa against the right wall
const vec3 FL_P = vec3(XR - 0.50, 0.0, 0.75);                                    // a floor lamp between them
const float BN_Z0 = -1.45, BN_Z1 = 4.55, BN_D = 0.56, BN_H = 0.50;               // the window bench
const vec3 PD_P = vec3(0.90, 3.02, 2.70);                                        // the pendant's shade (its rim)
const vec4 ART_B = vec4(SO_Z, 2.25, 0.92, 0.50);                                 // the print over the sofa: z, y, half w, half h

vec2 bookcase(vec3 r){
  vec3 c = 0.5*(BC_LO + BC_HI), b = 0.5*(BC_HI - BC_LO);
  vec3 q = r - c;
  float outer = sdBox3(q, b);
  // hollow: four bays of shelves, open to the room (-x)
  const float SP = 0.70, Y0 = 0.16, TH = 0.035;
  float yy = r.y - Y0;
  float lv = clamp(floor(yy/SP), 0.0, 3.0);
  float iy = yy - lv*SP;
  vec3 iq = vec3(q.x + 0.035, r.y - (Y0 + lv*SP + 0.5*SP + 0.5*TH), q.z);
  float inner = sdBox3(iq, vec3(b.x - 0.002, 0.5*SP - 0.5*TH, b.z - TH));
  float d = max(outer, -inner);
  vec2 res = vec2(d, M_WOOD);
  gTmpQ = q; gTmpId = 0.0;
  // books along z on this shelf
  const float BP = 0.085;
  float fz = (q.z + b.z - TH)/BP;
  float id = clamp(floor(fz), 0.0, 16.0);
  float ii = id;
  vec3 h = hash31(ii*2.91 + lv*17.0 + 4.0);
  float gap = step(0.86, h.z) + step(lv, 0.5)*step(0.55, h.z);             // gaps; the bottom shelf holds a basket
  float bw = BP*mix(0.36, 0.47, h.x), bh = mix(0.22, 0.30, h.y)*(SP/0.70), bd = mix(0.13, 0.17, h.z);
  vec3 bq = vec3(r.x - (BC_HI.x - 0.04 - bd), r.y - (Y0 + lv*SP + 0.5*TH + bh), q.z + b.z - TH - (ii + 0.5)*BP);
  float bk = sdBox3(bq, vec3(bd, bh, bw)) - 0.004;
  bk = mix(bk, 1.0, gap);
  bk = max(bk, -0.5*(BP - 2.0*bw) + abs(q.z + b.z - TH - (ii + 0.5)*BP) - BP*0.5);   // stay in the cell
  gTmpQ = vec3(bq.y/bh, bq.z/bw, 0.0); gTmpId = ii + lv*20.0;
  res = opUn(res, vec2(bk, M_BOOK));
  // a woven basket on the bottom shelf
  vec3 kq = vec3(r.x - (BC_HI.x - 0.22), r.y - (Y0 + 0.5*TH + 0.20), q.z + 0.28);
  float bas = max(sdBox3(kq, vec3(0.17, 0.20, 0.30)) - 0.03, -sdBox3(kq - vec3(0.0, 0.08, 0.0), vec3(0.14, 0.20, 0.27)));
  gTmpQ = kq; gTmpId = 0.0;
  res = opUn(res, vec2(bas, M_CUSH));
  return res;
}
vec2 sofa(vec3 r){
  float x0 = XR - SO_D;
  vec3 q = vec3(r.x - (XR - 0.5*SO_D), r.y, r.z - SO_Z);
  float base = sdBox3(q - vec3(0.02, 0.24, 0.0), vec3(0.50, 0.19, 1.62)) - 0.05;
  float back = sdBox3(q - vec3(0.36, 0.78, 0.0), vec3(0.13, 0.42, 1.62)) - 0.07;
  float arm = sdBox3(vec3(q.x - 0.02, q.y - 0.52, abs(q.z) - 1.56), vec3(0.50, 0.24, 0.10)) - 0.07;
  float d = min(min(base, back), arm);
  vec2 res = vec2(d, M_SOFA);
  gTmpQ = q; gTmpId = 0.0;
  // seat and back cushions: two of each
  float sc = sdBox3(vec3(q.x + 0.03, q.y - 0.53, abs(q.z) - 0.73), vec3(0.42, 0.07, 0.70)) - 0.07;
  float bc = sdBox3(vec3(q.x - 0.17, q.y - 0.98, abs(q.z) - 0.73), vec3(0.08, 0.27, 0.70)) - 0.09;
  gTmpQ = q; gTmpId = 0.0;
  res = opUn(res, vec2(min(sc, bc), M_SOFA));
  // two pillows, leaning
  vec3 pq = q - vec3(-0.02, 0.92, -1.15);
  pq.xy = rot(0.30)*pq.xy; pq.xz = rot(0.35)*pq.xz;
  float p1 = sdBox3(pq, vec3(0.05, 0.20, 0.20)) - 0.07;
  vec3 pq2 = q - vec3(0.0, 0.90, 1.18);
  pq2.xy = rot(0.26)*pq2.xy; pq2.xz = rot(-0.30)*pq2.xz;
  float p2 = sdBox3(pq2, vec3(0.05, 0.18, 0.18)) - 0.07;
  gTmpQ = vec3(pq.yz, 1.0); gTmpId = 1.0;
  res = opUn(res, vec2(p1, M_CUSH));
  gTmpQ = vec3(pq2.yz, 2.0); gTmpId = 2.0;
  res = opUn(res, vec2(p2, M_CUSH));
  return res;
}
vec2 floorLamp(vec3 r){
  vec3 q = r - FL_P;
  float base = max(length(q.xz) - 0.20, abs(q.y - 0.02) - 0.02) - 0.01;
  float pole = sdCapsule(q, vec3(0.0), vec3(0.0, 2.55, 0.0), 0.016);
  vec2 res = vec2(min(base, pole), M_METAL);
  gTmpQ = q; gTmpId = 0.0;
  // a drum of linen
  vec3 sq = q - vec3(0.0, 2.62, 0.0);
  float rr = mix(0.27, 0.21, clamp((sq.y + 0.22)/0.44, 0.0, 1.0));
  float sh = max(abs(length(sq.xz) - rr) - 0.008, abs(sq.y) - 0.22);
  res = opUn(res, vec2(sh, M_SHADE));
  return res;
}
vec2 bench(vec3 r){
  vec3 q = vec3(r.x - (XL + 0.5*BN_D), r.y, r.z - 0.5*(BN_Z0 + BN_Z1));
  float hz = 0.5*(BN_Z1 - BN_Z0);
  float box = sdBox3(q - vec3(0.0, 0.5*(BN_H - 0.10), 0.0), vec3(0.5*BN_D, 0.5*(BN_H - 0.10), hz)) - 0.01;
  // a recessed plinth
  box = max(box, -sdBox3(q - vec3(-0.5*BN_D, 0.04, 0.0), vec3(0.04, 0.06, hz + 0.1)));
  vec2 res = vec2(box, M_WOOD);
  gTmpQ = q; gTmpId = 1.0;
  float cu = sdBox3(q - vec3(0.02, BN_H - 0.05, 0.0), vec3(0.5*BN_D - 0.06, 0.035, hz - 0.06)) - 0.05;
  gTmpQ = vec3(q.yz, 3.0); gTmpId = 3.0;
  res = opUn(res, vec2(cu, M_CUSH));
  // two pillows against the wall
  vec3 pq = q - vec3(0.16, BN_H + 0.20, 2.2);
  pq.xy = rot(-0.30)*pq.xy;
  float p1 = sdBox3(pq, vec3(0.05, 0.17, 0.19)) - 0.07;
  vec3 pq2 = q - vec3(0.16, BN_H + 0.19, 1.62);
  pq2.xy = rot(-0.34)*pq2.xy; pq2.yz = rot(0.25)*pq2.yz;
  float p2 = sdBox3(pq2, vec3(0.05, 0.15, 0.16)) - 0.07;
  gTmpQ = vec3(pq.yz, 4.0); gTmpId = 4.0;
  res = opUn(res, vec2(p1, M_CUSH));
  gTmpQ = vec3(pq2.yz, 1.0); gTmpId = 1.0;
  res = opUn(res, vec2(p2, M_CUSH));
  return res;
}
vec2 pendant(vec3 r){
  vec3 q = r - PD_P;
  float cord = sdCapsule(q, vec3(0.0, 0.25, 0.0), vec3(0.0, HC - PD_P.y + 0.1, 0.0), 0.008);
  vec2 res = vec2(cord, M_METAL);
  gTmpQ = q; gTmpId = 0.0;
  // a shallow dome
  float dm = abs(length(q - vec3(0.0, -0.05, 0.0)) - 0.32) - 0.010;
  dm = max(dm, -q.y);
  res = opUn(res, vec2(dm, M_SHADE));
  return res;
}
vec2 artX(vec3 r, vec4 a, float x, float id){
  vec3 q = vec3(r.z - a.x, r.y - a.y, x - r.x);
  float fr = sdBox3(q, vec3(a.z + 0.05, a.w + 0.05, 0.045));
  float cv = sdBox3(q - vec3(0.0, 0.0, 0.035), vec3(a.z, a.w, 0.022));
  gTmpQ = vec3(q.x/a.z, q.y/a.w, 0.0); gTmpId = id;
  return cv < fr ? vec2(cv, M_ART) : vec2(fr, M_FRAME);
}

vec2 nearMap(vec3 pw, bool shadow){
  const float G = 0.14;
  vec3 p = toRoom(pw);
  vec2 res = vec2(G, M_POT);

  vec3 qa = p - POT_A;
  float bA = sdCapsule(p, POT_A, POT_A + vec3(0.0, 3.0, 0.0), 1.05);
  res.x = max(bA, G);
  for(int i=0;i<1 && bA < G;i++){
    res = opU(res, pot(qa, POT_H, POT_R));
    res = opUn(res, plant(p, POT_A + vec3(0.0, POT_H - 0.05, 0.0), 2.25, 0.80, 2.2, 16));
  }

  float bS = sdBox3(p - SH_C - vec3(0.0, 0.32, 0.12), SH_B + vec3(0.10, 0.42, 0.18));
  res.x = min(res.x, max(bS, G));
  for(int i=0;i<1 && bS < G;i++){
    res = opU(res, shelf(p));
  }

  float bB = sdBox3(p - 0.5*(BC_LO + BC_HI), 0.5*(BC_HI - BC_LO) + 0.03);
  res.x = min(res.x, max(bB, G));
  for(int i=0;i<1 && bB < G;i++){
    res = opU(res, bookcase(p));
  }
  float bO = sdBox3(p - vec3(XR - 0.5*SO_D, 0.65, SO_Z), vec3(0.5*SO_D + 0.08, 0.70, 1.80));
  res.x = min(res.x, max(bO, G));
  for(int i=0;i<1 && bO < G;i++){
    res = opU(res, sofa(p));
  }
  float bL = sdCapsule(p, FL_P, FL_P + vec3(0.0, 2.85, 0.0), 0.32);
  res.x = min(res.x, max(bL, G));
  for(int i=0;i<1 && bL < G;i++){
    res = opU(res, floorLamp(p));
  }
  float bN = sdBox3(p - vec3(XL + 0.5*BN_D, 0.55, 0.5*(BN_Z0 + BN_Z1)), vec3(0.5*BN_D + 0.05, 0.60, 0.5*(BN_Z1 - BN_Z0) + 0.05));
  res.x = min(res.x, max(bN, G));
  for(int i=0;i<1 && bN < G;i++){
    res = opU(res, bench(p));
  }
  float bP = sdBox3(p - PD_P - vec3(0.0, 0.40, 0.0), vec3(0.36, 0.50, 0.36));
  res.x = min(res.x, max(bP, G));
  for(int i=0;i<1 && bP < G;i++){
    res = opU(res, pendant(p));
  }
  return res;
}

vec2 setMap(vec3 pw, bool shadow){
  if(gShaded && !shadow) return vec2(1e3, M_PLASTER);
  vec3 r = toRoom(pw);
  vec2 res = nearMap(pw, shadow);
  if(shadow) return res;                          // see the header: the shell does not shadow, beamMask does
  float room1, passW;
  float s = shell(r, room1, passW);
  res = opU(res, vec2(s, r.y < 0.03 ? M_FLOOR : (r.y > HC - 0.03 ? M_CEIL : M_PLASTER)));
  // oak beams across the ceiling
  float bz = mod(r.z - BM_Z0 + 0.5*BM_P, BM_P) - 0.5*BM_P;
  float bmd = max(max(abs(bz) - BM_W, HC - BM_D - r.y), max(XL - r.x, r.x - XR)) - 0.01;
  res = opU(res, vec2(bmd, M_BEAM));
  // the rug: a thin slab with a soft edge
  float rg = max(sdRect(r.xz - vec2(-0.25, 0.15), vec2(2.45, 1.75), 0.05), r.y - 0.016);
  res = opU(res, vec2(rg, M_RUG));
  // skirting along the walls, stopping at the openings
  float rw = max(max(XL - r.x, r.x - XR), max(ZB - r.z, r.z - ZN));
  float sk = max(max(r.y - SK_H, -rw - 0.022), -passW);
  res = opU(res, vec2(sk, M_SKIRT));
  // a plain cornice where the walls meet the ceiling: two steps
  float cn = min(max(HC - 0.13 - r.y, -rw - 0.045), max(HC - 0.05 - r.y, -rw - 0.085));
  cn = max(cn, -passW);
  res = opU(res, vec2(cn, M_SKIRT));
  // glazing bars, in the middle of the wall's thickness, and an oak sill inside each window
  float c = winCell(r.z);
  float bx = abs(r.x - XL + TW*0.50) - 0.022;
  float bd = max(max(winBars(c, r.y), bx), winOpen(c, r.y) - 0.03);
  res = opU(res, vec2(bd, M_BAR));
  float sl = sdBox3(vec3(r.x - XL + 0.02, r.y - W_B + 0.025, c), vec3(0.15, 0.025, W_A + 0.10)) - 0.006;
  res = opU(res, vec2(sl, M_SILL));
  if(r.z < ZB + 0.3 && r.x < 0.0) res = opU(res, art(r, ART_A, ZB, 0.0));
  if(r.x > XR - 0.3 && r.z > 1.0 && r.z < 5.0) res = opU(res, artX(r, ART_B, XR, 1.0));
  return res;
}

// -------- paint
// warm oak planks along z.  fw = the pixel's footprint on the floor (for filtering the joints and the grain)
vec3 planks(vec2 xz, float fw){
  const float PW = 0.31, PL = 2.70;
  float col = floor(xz.x/PW);
  float u = xz.x/PW - col;
  float off = hash11(col*7.13 + 0.3)*PL;
  float zz = (xz.y + off)/PL;
  float row = floor(zz), v = zz - row;
  vec3 h = hash33(vec3(col, row, 3.0));
  vec3 a = mix(vec3(0.215,0.112,0.055), vec3(0.335,0.185,0.092), h.x*0.8 + 0.1);
  a = mix(a, a*vec3(1.06,0.95,0.86), h.y*0.6);
  // the grain: long streaks along the plank, fading with the footprint
  float gk = smoothstep(0.03, 0.006, fw);
  float g = fbm2(vec2(u*5.0 + h.z*17.0, xz.y*1.6 + h.y*9.0));
  float fine = vnoise2(vec2(u*26.0 + h.x*31.0, xz.y*0.9 + h.z*5.0));
  a *= 0.88 + 0.22*mix(0.5, g, 0.8) + 0.10*(fine - 0.5)*gk;
  // knots, rarely
  vec2 kc = vec2((0.3 + 0.4*h.y)*PW, (0.2 + 0.6*h.z)*PL);
  vec2 kq = vec2(u*PW, v*PL) - kc;
  float kn = step(0.78, h.x)*exp(-dot(kq*vec2(1.0, 0.45), kq*vec2(1.0, 0.45))/0.0009);
  a *= 1.0 - 0.45*kn;
  // the joints: dark, filtered
  float dx = min(u, 1.0 - u)*PW, dz = min(v, 1.0 - v)*PL;
  float jw = max(fw*0.7, 0.0025);
  float jl = 1.0 - smoothstep(0.0025 - jw*0.5, 0.0025 + jw*0.5, min(dx, dz*1.6));
  a *= 1.0 - 0.55*jl*clamp(0.006/fw, 0.15, 1.0);
  return a;
}
// a faded wool rug: a rust field, an indigo border, cream lines, a lozenge medallion
vec3 rug(vec2 q, float fw, out float edge){
  vec2 hb = vec2(2.45, 1.75);
  float e = -sdRect(q, hb, 0.05);                  // inward from the edge
  edge = e;
  vec3 rust = vec3(0.34,0.095,0.058), indigo = vec3(0.055,0.075,0.135), cream = vec3(0.60,0.50,0.37), ochre = vec3(0.46,0.30,0.10);
  float aa = max(fw, 0.004);
  vec3 c = rust;
  // the field: a lattice of small lozenges
  vec2 g = q*vec2(1.0, 1.0)/0.36;
  vec2 gf = abs(fract(g) - 0.5);
  float lz = gf.x + gf.y;
  c = mix(c, ochre*0.9, smoothstep(0.20 + aa*3.0, 0.20 - aa*3.0, lz)*0.55);
  c = mix(c, rust*0.65, smoothstep(0.03 + aa*3.0, 0.0, abs(lz - 0.42))*0.6);
  // the medallion
  float md = abs(q.x)/1.15 + abs(q.y)/0.80;
  c = mix(c, indigo*1.2, smoothstep(1.0 + aa, 1.0 - aa, md));
  c = mix(c, cream, smoothstep(0.025 + aa, 0.0, abs(md - 0.86))*0.9);
  c = mix(c, rust*1.1, smoothstep(0.5 + aa, 0.5 - aa, md));
  c = mix(c, cream*0.95, smoothstep(0.18 + aa, 0.18 - aa, md));
  // the borders
  c = mix(c, cream, smoothstep(0.012 + aa, 0.0, abs(e - 0.40)));
  float bord = smoothstep(0.38 + aa, 0.38 - aa, e);
  vec3 bc = indigo;
  vec2 bg = vec2(abs(q.x) > hb.x - 0.40 ? q.y : q.x, 0.0)/0.22;
  float zig = abs(fract(bg.x) - 0.5)*2.0;
  bc = mix(bc, ochre*0.8, smoothstep(0.05 + aa*6.0, 0.0, abs(e - 0.12 - 0.12*zig))*0.85);
  c = mix(c, bc, bord);
  c = mix(c, cream*0.85, smoothstep(0.010 + aa, 0.0, abs(e - 0.06)));
  c = mix(c, indigo*0.7, smoothstep(0.035 + aa, 0.035 - aa, e));
  return c;
}

void setSurface(vec3 pw, vec3 rd, float t, float m, vec3 n, inout Surf s){
  vec3 r = toRoom(pw);
  gAdd = vec3(0.0);
  gShaded = true;
  s.spec = 0.0; s.sheen = 0.0; s.wrap = 0.0; s.fur = 0.0;
  // the pixel's footprint on a horizontal surface
  float fw = t*2.0/(uRes.y*uFl)/max(abs(rd.y), 0.12);
  for(int i=0;i<1 && m == M_FLOOR;i++){
    s.alb = planks(r.xz, fw);
    // a soft darkening where the floor meets the walls and under the rug's edge
    float wd = min(min(r.x - XL, XR - r.x), r.z - ZB);
    s.occ *= 0.72 + 0.28*smoothstep(0.0, 0.55, wd);
    float re = sdRect(r.xz - vec2(-0.25, 0.15), vec2(2.45, 1.75), 0.05);
    s.occ *= 1.0 - 0.35*exp(-max(re, 0.0)/0.025);
    // the fig's pot
    float da = length(r.xz - POT_A.xz) - POT_R*0.72;
    s.occ *= 1.0 - 0.66*exp(-max(da, 0.0)/0.05) - 0.26*exp(-max(da, 0.0)/0.30);
    // varnish: a blurred fresnel gloss of the room (the window, the pools of light)
    float jit = hash12(gl_FragCoord.xy*1.73 + uJitter*31.0), jit2 = hash12(gl_FragCoord.yx*2.11 + uJitter*17.0);
    vec3 nn = normalize(vec3(0.0, 1.0, 0.0) + vec3(jit - 0.5, 0.0, jit2 - 0.5)*0.02);
    vec3 d2 = reflect(rd, nn);
    float nv = clamp(-rd.y, 0.0, 1.0);
    float Fr = 0.025 + 0.40*pow(1.0 - nv, 5.0);
    vec3 env = roomEnv(pw + vec3(0.0, 0.003, 0.0), d2, 0.3);
    gAdd = env*Fr*0.50;
  }
  for(int i=0;i<1 && m == M_RUG;i++){
    vec2 q = r.xz - vec2(-0.25, 0.15);
    float e;
    vec3 c = rug(q, fw, e);
    // wool: a fine nap and a slow fading, worn lighter in the middle
    float nap = vnoise2(q*90.0)*0.6 + vnoise2(q*37.0)*0.4;
    float nk = smoothstep(0.02, 0.005, fw);
    c *= (0.90 + 0.20*mix(0.5, nap, nk))*(0.94 + 0.12*fbm2(q*1.7));
    c = mix(c, c*1.25 + 0.02, 0.35*exp(-dot(q, q)/2.5));
    s.alb = c;
    s.wrap = 0.25; s.sheen = 0.25;
    s.occ *= 0.80 + 0.20*smoothstep(0.0, 0.10, e);
    s.n = normalize(vec3(0.0, 1.0, 0.0) + 0.05*(vec3(vnoise2(q*60.0), 0.0, vnoise2(q*60.0 + 7.0)) - 0.5)*nk);
  }
  for(int i=0;i<1 && (m == M_PLASTER || m == M_CEIL || m == M_SKIRT);i++){
    float big = fbm3(pw*vec3(0.55,0.55,0.55));
    float trow = vnoise3(pw*vec3(2.6,1.3,2.6))*0.6 + vnoise3(pw*vec3(7.0,3.2,7.0))*0.4;
    // limewash: clouds of slightly different tone
    vec3 pl = SP_WALL_C*(0.90 + 0.16*big)*(0.97 + 0.06*trow);
    pl = mix(pl, pl*vec3(1.08,1.02,0.94), smoothstep(0.45, 0.70, fbm3(pw*vec3(1.3,0.9,1.3) + 3.0))*0.6);
    if(m == M_CEIL) pl = vec3(0.70,0.62,0.53)*(0.95 + 0.08*big)*(0.70 + 0.30*smoothstep(2.0, -4.0, r.x));
    if(m == M_SKIRT) pl = vec3(0.84,0.78,0.67);
    s.alb = pl;
    s.n = normalize(n + 0.014*(vec3(vnoise3(pw*5.0), vnoise3(pw*5.0 + 4.0), vnoise3(pw*5.0 + 9.0)) - 0.5));
    // dark where wall meets floor and ceiling; a real corner is never lit evenly
    s.occ *= 1.05*(0.78 + 0.22*smoothstep(0.0, 0.80, r.y))*(0.80 + 0.20*smoothstep(0.0, 0.70, HC - r.y));
    float corner = min(min(r.x - XL, XR - r.x), r.z - ZB);
    s.occ *= 0.72 + 0.28*smoothstep(0.0, 1.0, corner);
    // the reveals of the windows are a touch darker inside
    if(r.x < XL + 0.01) s.occ *= 0.85;
    s.wrap = 0.05;
  }
  if(m == M_BEAM){
    s.alb = vec3(0.24,0.145,0.080)*(0.80 + 0.30*vnoise3(pw*vec3(14.0,3.0,2.0)) + 0.10*vnoise3(pw*vec3(40.0,9.0,5.0)));
    s.occ *= 0.72 + 0.28*smoothstep(HC, HC - BM_D, r.y);              // darker where it meets the ceiling
    s.spec = 0.06; s.shin = 20.0;
  }
  if(m == M_BAR){ s.alb = vec3(0.075,0.065,0.055); s.spec = 0.15; s.shin = 30.0; }
  if(m == M_SILL){
    s.alb = vec3(0.52,0.33,0.18)*(0.85 + 0.25*vnoise3(pw*vec3(2.0,30.0,2.0)));
    s.spec = 0.20; s.shin = 40.0;
  }
  for(int i=0;i<1 && m >= M_POT && m != M_FRAME && m != M_ART && m != M_SILL;i++){
    nearMap(pw, false);
    vec3 q = gNearQ;
    if(m == M_POT){
      float band = smoothstep(0.02, 0.0, abs(q.y - 0.60))*0.3;
      s.alb = vec3(0.60,0.30,0.16)*(0.82 + 0.34*fbm3(pw*vec3(5.0,14.0,5.0)))*(1.0 - band);
      s.spec = 0.05; s.shin = 18.0; s.wrap = 0.12;
      s.occ *= 0.80 + 0.20*smoothstep(0.0, 0.25, r.y);
    } else if(m == M_SOIL){
      s.alb = vec3(0.09,0.065,0.05)*(0.7 + 0.6*vnoise3(pw*40.0));
    } else if(m == M_LEAF){
      float tone = fbm3(pw*6.0);
      vec3 c1 = mix(vec3(0.050,0.13,0.045), vec3(0.10,0.24,0.065), gNearId);
      vec3 ca = mix(c1, c1*vec3(1.35,1.25,0.85), smoothstep(0.30, 0.70, tone));
      float vein = smoothstep(0.050, 0.014, abs(q.x));
      float side = abs(fract(q.y*7.0 - abs(q.x)*2.0) - 0.5);
      vein = max(vein, 0.55*smoothstep(0.05, 0.015, side)*smoothstep(1.0, 0.3, abs(q.x)));
      s.alb = mix(ca, ca*vec3(1.7,1.8,1.25) + 0.015, vein*0.62*step(gNearId, 0.69));
      s.wrap = 0.35; s.trans = 0.7; s.transCol = s.alb*1.6; s.spec = 0.45; s.shin = 46.0;
    } else if(m == M_STEM){
      s.alb = mix(vec3(0.12,0.085,0.055), vec3(0.26,0.19,0.12), vnoise3(pw*vec3(40.0,9.0,40.0)));
      s.wrap = 0.2; s.spec = 0.04; s.shin = 14.0;
    } else if(m == M_SHELF){
      s.alb = vec3(0.46,0.28,0.15)*(0.85 + 0.25*vnoise3(pw*vec3(1.5,25.0,6.0)));
      s.spec = 0.12; s.shin = 30.0;
    } else if(m == M_BOOK){
      vec3 h = hash31(gNearId*5.17 + 2.0);
      vec3 bc = h.x < 0.25 ? vec3(0.45,0.13,0.08) : (h.x < 0.45 ? vec3(0.10,0.20,0.24) : (h.x < 0.62 ? vec3(0.62,0.52,0.36)
              : (h.x < 0.80 ? vec3(0.22,0.28,0.14) : vec3(0.58,0.38,0.12))));
      bc *= 0.80 + 0.35*h.y;
      // the pages show on top, a band or two on the spine
      float top = smoothstep(0.90, 0.97, q.x);
      float bands = smoothstep(0.03, 0.0, abs(q.x - 0.62 + 0.2*h.z) - 0.03)*step(0.5, h.y);
      s.alb = mix(mix(bc, vec3(0.80,0.70,0.48), bands*0.7), vec3(0.78,0.72,0.60), top*step(0.0, n.y - 0.5));
      s.spec = 0.06; s.shin = 20.0; s.wrap = 0.1;
    } else if(m == M_VASE){
      float gl = smoothstep(0.10, 0.12, q.y + 0.010*sin(atan(q.z, q.x)*5.0));
      s.alb = mix(vec3(0.66,0.52,0.38), vec3(0.80,0.74,0.62), gl);
      s.spec = 0.8*gl; s.shin = 90.0; s.wrap = 0.08; s.occ *= 1.3;
    } else if(m == M_WOOD){
      if(gNearId < 0.5){
        // the bookcase: oak, darker inside the bays
        s.alb = vec3(0.40,0.245,0.13)*(0.82 + 0.30*vnoise3(pw*vec3(2.0,22.0,3.0)));
        s.occ *= 0.55 + 0.45*smoothstep(BC_HI.x - 0.36, BC_HI.x - 0.46, r.x);
        s.spec = 0.10; s.shin = 30.0;
      } else {
        // the bench: painted, the colour of the skirting
        s.alb = vec3(0.80,0.74,0.63)*(0.96 + 0.05*vnoise3(pw*6.0));
        s.spec = 0.08; s.shin = 24.0;
        s.occ *= 0.75 + 0.25*smoothstep(0.0, 0.30, r.y);
      }
    } else if(m == M_SOFA){
      // deep green velvet: a sheen at grazing angles, a soft crush in the pile
      s.alb = vec3(0.050,0.085,0.060)*(0.88 + 0.22*fbm3(pw*vec3(3.0,3.0,3.0)));
      s.sheen = 0.45; s.wrap = 0.25;
      s.occ *= 0.70 + 0.30*smoothstep(0.0, 0.5, r.y);
    } else if(m == M_CUSH){
      float k = gNearId;
      vec2 cq = q.xy;
      if(k < 0.5){
        float wv = abs(sin(r.y*80.0))*abs(sin(r.z*60.0 + r.x*60.0));
        s.alb = vec3(0.46,0.31,0.16)*(0.75 + 0.35*wv);                   // rattan
      } else if(k < 1.5){
        s.alb = vec3(0.80,0.74,0.62)*(0.94 + 0.08*vnoise2(cq*40.0));     // cream boucle
        s.sheen = 0.3;
      } else if(k < 2.5){
        s.alb = vec3(0.52,0.19,0.09)*(0.92 + 0.10*vnoise2(cq*40.0));     // rust
      } else if(k < 3.5){
        float st = smoothstep(0.30, 0.42, abs(fract(cq.y*5.0) - 0.5)*2.0); // ticking stripes along the bench
        s.alb = mix(vec3(0.74,0.66,0.52), vec3(0.20,0.24,0.32), st*0.0 + 0.0) * 1.0;
        s.alb = mix(vec3(0.62,0.45,0.20), vec3(0.70,0.55,0.30), st);       // mustard
      } else {
        s.alb = vec3(0.12,0.16,0.26)*(0.92 + 0.10*vnoise2(cq*40.0));     // indigo
      }
      s.wrap = max(s.wrap, 0.25);
    } else if(m == M_SHADE){
      s.alb = vec3(0.86,0.80,0.68);
      s.wrap = 0.5;
      // lit from inside: bright where the inner face shows, the linen glowing through outside
      vec3 lc = r.y > 2.95 && r.x < 3.0 ? PD_P : FL_P + vec3(0.0, 2.62, 0.0);
      vec3 toC = normalize(lc - r);
      float inner = smoothstep(-0.1, 0.3, dot(n, toC));
      gAdd = vec3(1.45,0.98,0.56)*mix(0.55, 2.4, inner);
    } else if(m == M_METAL){
      s.alb = vec3(0.075,0.060,0.045); s.spec = 0.6; s.shin = 60.0;
    }
  }
  for(int i=0;i<1 && (m == M_ART || m == M_FRAME);i++){
    float pick = r.x > 0.0 ? 1.0 : 0.0;
    vec2 uv = pick < 0.5 ? vec2(r.x - ART_A.x, r.y - ART_A.y)/ART_A.zw : vec2(r.z - ART_B.x, r.y - ART_B.y)/ART_B.zw;
    if(m == M_FRAME){
      s.alb = vec3(0.30,0.19,0.11)*(0.85 + 0.3*vnoise3(pw*vec3(30.0,30.0,30.0)));
      s.spec = 0.10; s.shin = 30.0;
    } else {
      float wob = 0.05*(vnoise2(uv*5.0) - 0.5) + 0.018*(vnoise2(uv*23.0) - 0.5);
      vec3 ca;
      if(pick > 0.5){
        // a wide one: arches and a moon, block colours
        vec2 w = uv*vec2(ART_B.z/ART_B.w, 1.0);
        ca = vec3(0.84,0.76,0.60);
        float a1 = max(length(w - vec2(-0.85, -1.0)) - 0.95, -w.y - 2.0);
        ca = mix(ca, vec3(0.70,0.34,0.20), smoothstep(0.02, -0.02, a1 + wob));
        float a2 = length(w - vec2(0.55, -1.0)) - 0.80;
        ca = mix(ca, vec3(0.13,0.27,0.27), smoothstep(0.02, -0.02, a2 + wob));
        float a3 = length(w - vec2(0.55, -1.0)) - 0.45;
        ca = mix(ca, vec3(0.86,0.66,0.38), smoothstep(0.02, -0.02, a3 + wob));
        float mo = length(w - vec2(-0.10, 0.45)) - 0.22;
        ca = mix(ca, vec3(0.90,0.70,0.40), smoothstep(0.02, -0.02, mo + wob));
        float weave = 0.97 + 0.05*vnoise2(uv*160.0);
        float mat_ = smoothstep(0.92, 0.90, max(abs(uv.x), abs(uv.y)));
        s.alb = mix(vec3(0.90,0.86,0.78), ca*weave, mat_);
        break;
      }
      // a sun over two hills
      ca = mix(vec3(0.86,0.72,0.50), vec3(0.80,0.48,0.32), smoothstep(-0.1, 0.8, uv.y + wob));
      float sun = smoothstep(0.03, -0.03, length(uv - vec2(0.22, 0.34)) - 0.30 + wob);
      ca = mix(ca, vec3(0.94,0.80,0.50), sun);
      float hill = smoothstep(0.02, -0.02, uv.y + 0.18 + 0.22*sin(uv.x*2.3 + 0.8) + wob);
      ca = mix(ca, vec3(0.12,0.30,0.30), hill);
      float h2 = smoothstep(0.02, -0.02, uv.y + 0.56 + 0.16*sin(uv.x*2.9 + 3.0) + wob);
      ca = mix(ca, vec3(0.06,0.15,0.17), h2);
      float weave = 0.97 + 0.05*vnoise2(uv*160.0);
      float mat_ = smoothstep(0.86, 0.84, max(abs(uv.x), abs(uv.y)));
      s.alb = mix(vec3(0.90,0.86,0.78), ca*weave, mat_);
    }
  }
  // the two lamps: a soft warm light without shadows (the floor lamp sends it up and down out of its drum, the
  // pendant down)
  {
    vec3 l1 = FL_P + vec3(0.0, 2.62, 0.0) - r;
    float d1 = dot(l1, l1);
    vec3 L1 = l1*inversesqrt(d1);
    float lobe1 = 0.35 + 0.65*abs(L1.y);
    vec3 l2 = PD_P + vec3(0.0, -0.05, 0.0) - r;
    float d2 = dot(l2, l2);
    vec3 L2 = l2*inversesqrt(d2);
    float lobe2 = smoothstep(-0.2, -0.8, -L2.y);          // only below the dome
    gLamp = vec3(1.0,0.66,0.36)*(0.55*clamp(dot(n, L1)*0.8 + 0.2, 0.0, 1.0)*lobe1/(1.0 + 1.1*d1)
                               + 0.45*clamp(dot(n, L2)*0.8 + 0.2, 0.0, 1.0)*lobe2/(1.0 + 0.9*d2));
    if(m == M_SHADE) gLamp = vec3(0.0);
  }
  // contact shadows of the furniture on the floor, and of the shelf on the wall
  if(m == M_FLOOR || m == M_RUG){
#ifndef BG_NO_CHAR
    // the character's own contact, from its shadow proxy (the bake without it leaves this out: under root motion
    // the live shadow draws the character, engine/scene_main.glsl)
    float c0 = charShadow(pw + vec3(0.0, 0.05, 0.0)), c1 = charShadow(pw + vec3(0.0, 0.16, 0.0)), c2 = charShadow(pw + vec3(0.0, 0.38, 0.0));
    float co = max(0.05 - c0, 0.0)*2.2 + max(0.16 - c1, 0.0)*1.3 + max(0.38 - c2, 0.0)*0.55;
    s.occ *= clamp(1.0 - 2.2*co, 0.22, 1.0);
#endif
    float ds = sdRect(r.xz - vec2(XR - 0.5*SO_D + 0.02, SO_Z), vec2(0.55, 1.70), 0.05);
    float db = r.x - (XL + BN_D);
    float dk = max(BC_LO.x - r.x, r.z - BC_HI.z);
    float dl = length(r.xz - FL_P.xz) - 0.20;
    float dmin = min(min(max(ds, 0.0), max(db, 0.0)*step(BN_Z0, r.z)*step(r.z, BN_Z1) + 9.0*(1.0 - step(BN_Z0, r.z)*step(r.z, BN_Z1))),
                     min(max(dk, 0.0), max(dl, 0.0)));
    s.occ *= 1.0 - 0.62*exp(-dmin/0.05) - 0.22*exp(-dmin/0.30);
  }
  if(m == M_PLASTER && r.z < ZB + 0.02){
    float inX = smoothstep(SH_B.x + 0.10, SH_B.x - 0.10, abs(r.x - SH_C.x));
    float below = SH_C.y - SH_B.y - r.y;
    s.occ *= 1.0 - 0.40*inX*exp(-max(below, 0.0)/0.10)*step(-0.01, below);
    float above = r.y - SH_C.y - SH_B.y;
    s.occ *= 1.0 - 0.22*inX*exp(-max(above, 0.0)/0.20)*step(0.0, above)*step(above, 0.6);
  }
  if(m == M_PLASTER && r.x > XR - 0.02){
    float ds = max(abs(r.z - SO_Z) - 1.7, 0.0) + max(r.y - 1.25, 0.0);
    s.occ *= 1.0 - 0.45*exp(-ds/0.22);
    float dk = max(r.z - BC_HI.z, 0.0) + max(r.y - BC_HI.y, 0.0);
    s.occ *= 1.0 - 0.40*exp(-dk/0.15);
  }
  if(m == M_PLASTER && r.x < XL + 0.02){
    float dn = max(r.y - BN_H, 0.0) + max(max(BN_Z0 - r.z, r.z - BN_Z1), 0.0);
    s.occ *= 1.0 - 0.35*exp(-dn/0.18);
  }
  // the sun bounces: every surface near a pool of light glows warm.  Eight rays into the room, how many land in a beam
  float gi = 0.0;
  for(int i=0;i<8;i++){
    vec3 d = hash31(float(i)*2.17 + 0.3)*2.0 - 1.0;
    d = normalize(d)*sign(dot(d, n) + 1e-3) + n*0.35;
    gi += beamSoft(r + d*1.25) + beamSoft(r + d*0.55);
  }
  gGI = gi/16.0;
  // the light goes through s.alb, see the header
  float M = beamMask(r);
  float k = max(M, 0.003);
  gAmbMul = 1.0/k;
  float kk = k;
  if(m == M_FLOOR || m == M_RUG) k *= 1.0 + 0.25*M;   // the sun rakes the floor: lift it to where the eye expects a pool
  gAdd /= kk*max(dot(s.alb, vec3(0.333)), 0.05);
  gAmbMul *= kk/k;
  gAdd *= kk/k;
  s.alb *= k; s.transCol *= k; s.spec *= M;
}
#endif

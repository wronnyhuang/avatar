// ---------------------------------------------------------------------------
//  engine/places.glsl — props: the places props rest on at home (a stump, a flat rock, a side table, a gallery
//  plinth, a crate, a log), part of the SET (baked once into the background, lit and shadowed with it — the
//  character's rest shadow falls on them).  World units.  Generated defines (engine/props_defs.py, from the
//  character's renderer/props/<c>.json): PLACES_N, PLACEi_KIND, PLACEi_C (centre of the top face), PLACEi_SIZE (width x,
//  depth z), PLACEi_YAW, PLACEi_FLOOR (where it stands), PLACEi_COL (its main colour), PLACEi_SEED.
//  A set calls placesMap(pw) in setMap (union, material M_PLACE) and placesSurface() for that material.
// ---------------------------------------------------------------------------
#ifndef PLACES_N
#define PLACES_N 0
#endif
#define M_PLACE 61.0
// scene: the chalkboard's slate albedo at board-local (x right, y up from the writable centre; scene units) —
// a dark green-charcoal with old chalk dust: soft clouds of lighter tone, swipe marks along x, a fine speckle.  Used by
// the bake (placesSurface) AND the live surface pass (scene_main.glsl), which divides the baked slate back out to light
// the chalk the way the bake lit the slate.
vec3 chalkSlateAlb(vec2 xy){
  float cloud = fbm2(xy*vec2(1.6, 2.4) + 3.1);
  float swipe = vnoise2(vec2(xy.x*0.9, xy.y*7.0) + 11.0);
  float dust = 0.55*smoothstep(0.45, 0.85, cloud) + 0.35*smoothstep(0.55, 0.90, swipe);
  float speck = vnoise2(xy*90.0);
  vec3 a = vec3(0.040, 0.056, 0.048)*(0.90 + 0.18*speck);
  return a + vec3(0.050, 0.054, 0.050)*dust;
}
#if PLACES_N > 0
float gPlId;            // which place was nearest (painter)
vec3  gPlQ;             // its local point (x right, y up from the top face, z front)

float plBox(vec3 p, vec3 b, float r){ vec3 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r; }
float gPlStand;         // distance to the stand (kind 8), for the painter
float gPlPart;          // scene: which part of a board / TV (0 slate / screen, 1 oak frame / tray / posts, 2 bezel)
float placeSDF3(vec3 q, float w, float d, float hgt){
  float hw = 0.5*w, hd = 0.5*d;
  float slab = plBox(q - vec3(0.0, -0.035*w, 0.0), vec3(hw, 0.035*w, hd), 0.012*w);
  vec3 l = vec3(abs(q.x) - hw*0.80, q.y, abs(q.z) - hd*0.78);
  float leg = length(l.xz) - 0.035*w*(1.0 - 0.25*smoothstep(-0.1*hgt, -0.9*hgt, q.y));
  leg = max(leg, max(q.y + 0.05*w, -q.y - hgt));
  float apron = plBox(q - vec3(0.0, -0.10*w, 0.0), vec3(hw*0.82, 0.035*w, hd*0.80), 0.006*w);
  return min(min(slab, leg), apron);
}

// one place in its local frame: q.y = 0 on the top face, the floor at y = -hgt; w, d = width, depth
float placeSDF(int kind, vec3 q, float w, float d, float hgt, float seed, float post){
  float hw = 0.5*w, hd = 0.5*d;
  if(kind == 8){                       // a small side table (kind 3) with a headphone stand on it: a post, a padded saddle
    float tb = placeSDF3(q, w, d, hgt);
    float base = max(length(q.xz) - 0.11, abs(q.y - 0.012) - 0.012) - 0.006;
    float pst = sdCapsule(q, vec3(0.0, 0.0, 0.0), vec3(0.0, post - 0.05, 0.0), 0.022);
    vec3 sq = q - vec3(0.0, post - 0.04, 0.0);
    float sad = length(vec2(length(sq.xy) - 0.10, sq.z)) - 0.030;      // a short arc the band hangs on
    sad = max(sad, -sq.y - 0.02);
    gPlStand = min(min(base, pst), sad);
    return min(tb, gPlStand);
  }
  if(kind == 9){                       // scene: a free-standing chalkboard.  Local frame: origin = the centre of the
    // WRITABLE area on the slate's front face, x along the board, y up, z out of the slate; w x d = the writable area
    float hw = 0.5*w, hh = 0.5*d;
    const float SM = 0.04, FW = 0.05;                                   // slate margin round the writable area, frame width
    float slate = plBox(q - vec3(0.0, 0.0, -0.016), vec3(hw + SM + 0.004, hh + SM + 0.004, 0.016), 0.002);
    vec3 fo = vec3(hw + SM + FW, hh + SM + FW, 0.028);
    float frame = plBox(q - vec3(0.0, 0.0, -0.016), fo, 0.010);
    frame = max(frame, -plBox(q - vec3(0.0, 0.0, 0.0), vec3(hw + SM, hh + SM, 0.08), 0.004));
    // the tray: a ledge along the bottom of the frame, a small lip at its front edge
    float ty = -(hh + SM + FW);
    float tray = plBox(q - vec3(0.0, ty - 0.0125, 0.034), vec3(fo.x, 0.0125, 0.066), 0.004);
    tray = min(tray, plBox(q - vec3(0.0, ty + 0.006, 0.094), vec3(fo.x, 0.008, 0.006), 0.003));
    // two posts past the frame's ends, from the floor to above the frame, with splayed feet along z
    vec3 pq = vec3(abs(q.x) - (fo.x + 0.035), q.y, q.z + 0.016);
    float pst = plBox(pq - vec3(0.0, 0.5*(post - hgt), 0.0), vec3(0.030, 0.5*(hgt + post), 0.030), 0.008);
    float foot = plBox(pq - vec3(0.0, -hgt + 0.030, 0.0), vec3(0.034, 0.030, 0.30), 0.012);
    float pin = plBox(pq - vec3(-0.032, 0.0, 0.0), vec3(0.012, 0.03, 0.03), 0.004);      // where the board hangs on the post
    float wood = min(min(frame, tray), min(min(pst, foot), pin));
    gPlPart = wood < slate ? 1.0 : 0.0;
    return min(wood, slate);
  }
  if(kind == 10){                      // scene: a wall-mounted TV.  Local: origin = the screen's centre on its front,
    // x right, y up, z out of the screen (toward the room); w x d = the visible picture
    float hw = 0.5*w, hh = 0.5*d;
    float body = plBox(q - vec3(0.0, 0.0, -0.032), vec3(hw + 0.022, hh + 0.022, 0.030), 0.006);
    float glass = plBox(q - vec3(0.0, 0.0, -0.004), vec3(hw, hh, 0.004), 0.001);
    float b = max(body, -plBox(q, vec3(hw, hh, 0.006), 0.001));
    float back = plBox(q - vec3(0.0, 0.0, -0.075), vec3(0.6*hw, 0.6*hh, 0.020), 0.01);     // the wall mount behind it
    float d0 = min(min(b, back), glass);
    gPlPart = glass <= d0 + 1e-5 ? 0.0 : 2.0;
    return d0;
  }
  if(kind == 1){                       // stump: a sawn trunk, the cut on top, roots flaring into the floor
    float r = length(q.xz/vec2(hw, hd));
    float a = atan(q.z, q.x);
    float bark = 0.035*w*(vnoise2(vec2(a*7.0, q.y*3.0/w) + seed) - 0.5) + 0.012*w*sin(a*23.0 + 3.0*vnoise2(vec2(q.y*6.0/w, a)));
    float flare = 0.20*smoothstep(-0.35*hgt, -hgt, q.y)*(0.7 + 0.6*vnoise2(vec2(a*2.5 + seed, 0.0)));
    float side = (r - 1.0 - flare)*min(hw, hd) - bark;
    float top = q.y + 0.010*w*smoothstep(0.85, 1.0, r);
    return max(max(side, top), -q.y - hgt - 0.3);
  } else if(kind == 2){                // a flat-topped boulder
    vec3 p = q + vec3(0.0, 0.42*hgt, 0.0);
    float e = sdEllipsoid(p, vec3(hw*1.05, 0.62*hgt + 0.05, hd*1.05));
    float n = fbm3(q*3.0/w + seed);
    e -= 0.06*w*(n - 0.5);
    float top = q.y - 0.02*w*(vnoise2(q.xz*4.0/w + seed) - 0.5);
    return smax(e, top, 0.10*w)*0.85;
  } else if(kind == 3){                // a small wooden side table: a slab and four turned legs
    return placeSDF3(q, w, d, hgt);
  } else if(kind == 4){                // a gallery plinth: a rounded block with a recessed foot
    float b = plBox(q - vec3(0.0, -0.5*hgt, 0.0), vec3(hw, 0.5*hgt, hd), 0.015*w);
    float foot = plBox(q - vec3(0.0, -hgt + 0.03*w, 0.0), vec3(hw*1.04, 0.03*w, hd*1.04), 0.008*w);
    return min(b, foot);
  } else if(kind == 5){                // a wooden crate
    float b = plBox(q - vec3(0.0, -0.5*hgt, 0.0), vec3(hw, 0.5*hgt, hd), 0.01*w);
    // plank grooves on the sides
    float gr = abs(fract((q.y + hgt)/(0.22*w)) - 0.5)*0.22*w;
    b += 0.006*w*smoothstep(0.012*w, 0.0, gr)*step(q.y, -0.02*w);
    return b;
  } else if(kind == 7){                // a log lying across (axis x), its top at y 0
    float r = 0.5*d;
    float dd = length(vec2(q.y + r, q.z)) - r - 0.03*r*(vnoise2(vec2(q.x*3.0/w, atan(q.z, q.y + r)*3.0) + seed) - 0.5);
    return max(dd, abs(q.x) - hw);
  }
  return 1e3;
}

vec2 placesMap(vec3 pw){
  float best = 1e9; gPlId = -1.0;
  for(int i=0;i<PLACES_N;i++){
    vec3 c; vec2 sz; float yaw, fl, sd, po; int kind;
    if(i == 0){ c = PLACE0_C; sz = PLACE0_SIZE; yaw = PLACE0_YAW; fl = PLACE0_FLOOR; kind = PLACE0_KIND; sd = PLACE0_SEED; po = PLACE0_POST; }
#if PLACES_N > 1
    else if(i == 1){ c = PLACE1_C; sz = PLACE1_SIZE; yaw = PLACE1_YAW; fl = PLACE1_FLOOR; kind = PLACE1_KIND; sd = PLACE1_SEED; po = PLACE1_POST; }
#endif
#if PLACES_N > 3
    else if(i == 2){ c = PLACE2_C; sz = PLACE2_SIZE; yaw = PLACE2_YAW; fl = PLACE2_FLOOR; kind = PLACE2_KIND; sd = PLACE2_SEED; po = PLACE2_POST; }
    else { c = PLACE3_C; sz = PLACE3_SIZE; yaw = PLACE3_YAW; fl = PLACE3_FLOOR; kind = PLACE3_KIND; sd = PLACE3_SEED; po = PLACE3_POST; }
#elif PLACES_N > 2
    else { c = PLACE2_C; sz = PLACE2_SIZE; yaw = PLACE2_YAW; fl = PLACE2_FLOOR; kind = PLACE2_KIND; sd = PLACE2_SEED; po = PLACE2_POST; }
#endif
    vec3 q = pw - c;
    float hgt = c.y - fl;
    // bound: a capsule around the whole place
    float bnd = length(max(abs(q - vec3(0.0, 0.5*(po - hgt), 0.0)) - vec3(0.65*max(sz.x, sz.y), 0.5*(hgt + po) + 0.1, 0.65*max(sz.x, sz.y)), 0.0));
    if(bnd > 0.3){ best = min(best, bnd); continue; }
    q.xz = rot(yaw)*q.xz;
    float d = placeSDF(kind, q, sz.x, sz.y, hgt, sd, po);
    if(d < best){ best = d; gPlId = float(i); gPlQ = q; }
  }
  return vec2(best, M_PLACE);
}

void placesSurface(vec3 pw, vec3 n, inout Surf s){
  placesMap(pw);
  int i = int(gPlId + 0.5);
  int kind; vec3 col; vec2 sz; float sd;
  if(i <= 0){ kind = PLACE0_KIND; col = PLACE0_COL; sz = PLACE0_SIZE; sd = PLACE0_SEED; }
#if PLACES_N > 1
  else if(i == 1){ kind = PLACE1_KIND; col = PLACE1_COL; sz = PLACE1_SIZE; sd = PLACE1_SEED; }
#endif
#if PLACES_N > 3
  else if(i == 2){ kind = PLACE2_KIND; col = PLACE2_COL; sz = PLACE2_SIZE; sd = PLACE2_SEED; }
  else { kind = PLACE3_KIND; col = PLACE3_COL; sz = PLACE3_SIZE; sd = PLACE3_SEED; }
#elif PLACES_N > 2
  else { kind = PLACE2_KIND; col = PLACE2_COL; sz = PLACE2_SIZE; sd = PLACE2_SEED; }
#endif
  vec3 q = gPlQ;
  float w = sz.x;
  s.fur = 0.0; s.trans = 0.0; s.mirror = 0.0; s.emit = vec3(0.0);
  float top = smoothstep(0.55, 0.85, n.y)*step(-0.03*w, q.y);
  if(kind == 1 || kind == 7){
    // bark on the sides, sawn wood with rings on the cut
    float a = atan(q.z, q.x);
    float bk = fbm3(vec3(a*3.0, q.y*4.0/w, sd));
    vec3 bark = col*mix(0.45, 1.05, bk)*(0.8 + 0.3*vnoise2(vec2(a*20.0, q.y*30.0/w)));
    float r = length(q.xz/(0.5*sz));
    float rings = 0.5 + 0.5*sin(r*26.0 + 2.0*vnoise2(q.xz*8.0/w + sd));
    vec3 wood = mix(vec3(0.62, 0.45, 0.28), vec3(0.78, 0.60, 0.40), rings)*mix(0.75, 1.0, vnoise2(q.xz*30.0/w));
    wood = mix(wood, bark*0.8, smoothstep(0.84, 0.95, r));
    if(kind == 7) top = 0.0;
    s.alb = mix(bark, wood, top);
    s.spec = 0.04; s.shin = 10.0; s.wrap = 0.25; s.sheen = 0.2;
    s.occ = mix(0.75, 1.0, bk);
  } else if(kind == 2){
    float n1 = fbm3(pw*3.0 + sd);
    vec3 a = col*mix(0.70, 1.15, n1);
    // moss and lichen on the upper faces
    float moss = smoothstep(0.55, 0.9, n.y + 0.5*(vnoise3(pw*5.0) - 0.5))*(1.0 - top*0.6);
    a = mix(a, vec3(0.16, 0.20, 0.06), 0.55*moss);
    s.alb = a; s.spec = 0.08; s.shin = 16.0; s.wrap = 0.20; s.sheen = 0.15; s.occ = 0.85 + 0.15*n1;
  } else if(kind == 8 && gPlStand < 0.004){      // the stand: brushed dark metal, a soft grey saddle
    s.alb = vec3(0.10, 0.10, 0.11)*(0.9 + 0.2*vnoise3(pw*40.0)); s.spec = 0.8; s.shin = 70.0; s.wrap = 0.1; s.sheen = 0.3;
  } else if(kind == 9 && gPlPart < 0.5){        // scene: the slate (the live pass puts the chalk on it)
    s.alb = chalkSlateAlb(q.xy);
    s.spec = 0.0; s.shin = 8.0; s.wrap = 0.10; s.sheen = 0.05;
  } else if(kind == 9){                         // its oak frame, tray and posts
    vec3 g3 = abs(q.y) > 0.5*sz.y + 0.06 && abs(q.x) < 0.5*sz.x + 0.10 ? q.xzy : (abs(q.x) > 0.5*sz.x + 0.06 ? q.yxz : q);
    float grain = vnoise2(vec2(g3.x*3.0, g3.y*40.0))*0.6 + vnoise2(vec2(g3.x*9.0, g3.z*60.0))*0.4;
    s.alb = col*(0.70 + 0.42*grain);
    s.spec = 0.16; s.shin = 30.0; s.wrap = 0.20; s.sheen = 0.2;
    s.occ *= q.y < -0.5*sz.y - 0.08 && q.z > 0.0 ? 0.92 : 1.0;
  } else if(kind == 10 && gPlPart < 0.5){       // the screen when off: dark glass (the live pass draws the picture)
    s.alb = vec3(0.010, 0.011, 0.013); s.spec = 1.2; s.shin = 140.0; s.wrap = 0.0; s.sheen = 0.3;
  } else if(kind == 10){                        // bezel + mount: satin black
    s.alb = vec3(0.022, 0.022, 0.024)*(0.95 + 0.1*vnoise3(pw*30.0)); s.spec = 0.35; s.shin = 40.0; s.wrap = 0.05; s.sheen = 0.2;
  } else if(kind == 3 || kind == 5 || kind == 8){
    float grain = vnoise2(vec2(q.x*3.0/w, q.z*40.0/w)) *0.6 + vnoise2(vec2(q.x*9.0/w, q.y*60.0/w + q.z*60.0/w))*0.4;
    s.alb = col*(0.72 + 0.45*grain);
    s.spec = 0.18; s.shin = 30.0; s.wrap = 0.20; s.sheen = 0.2;
  } else if(kind == 4){
    s.alb = col*(0.94 + 0.06*vnoise3(pw*20.0));
    s.spec = 0.10; s.shin = 20.0; s.wrap = 0.15; s.sheen = 0.1;
  }
}
#else
vec2 placesMap(vec3 pw){ return vec2(1e9, M_PLACE); }
void placesSurface(vec3 pw, vec3 n, inout Surf s){}
#endif

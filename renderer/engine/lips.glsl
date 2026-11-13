// ---------------------------------------------------------------------------
//  engine/lips.glsl — real lips for code-rendered characters (docs/LIPS.md, docs/SHAPE_GUIDE.md).
//
//  The mouth lives in a MOUTH FRAME: origin LIP_O on the host's surface, y up, x right, z out of the face.
//  (x, y) locate a point in the plane of the face; DEPTH COMES FROM THE HOST'S OWN DISTANCE, so the lips
//  ride a sphere, a cube face or a blob without being modelled twice.
//
//  Everything is built on one curve, the APERTURE: a lens made of two circular arcs through the two mouth
//  corners, an upper arc rising gLAu and a lower arc falling gLAl, mixed with a rounded box of the same corners
//  and heights (gLSq).  With both heights at 0 both are the same line segment; a shallow lens has pointed
//  corners and the box rounds them; heights equal to the half width make a true circle in either, which is
//  what the pucker wants.  `e` is the signed distance to that curve, on both of its sides:
//     e < 0 inside the opening; e = 0 on the inner edge of the lips; e > 0 out through the lips onto skin.
//  Upper and lower lip are two volumes around that curve: each a soft section (height along e, thickness
//  along the depth) with its own height, fullness, depth, and taper to the corners.  They overlap a little
//  at the line, which makes the sealed pose one fine crease rather than two sausages.  The corners are the
//  ends of the same curve, so they cannot come apart.
//
//  Host carve: a column through the lens, wider under the lips and behind them, closed outright at rest.
//  A small opening is a dark slit first (dark-slit discipline: gLDark, gLShow); colour and teeth come up
//  with room.  There is no hinge: the host never splits.
//
//  Required defines, before `#pragma include lips`
//    LIP_O     vec3   mouth centre on the host surface, head space
//    LIP_TILT  float  radians: the frame's z leans down by this.  Aim z at the CAMERA, not along the host normal:
//                     the carve runs along z, and a pucker seen off axis shows its tunnel from the side
//    LIP_W     float  half width of the resting mouth
//  Everything else has a default below, in proportion to LIP_W.  A character's mouth is these defines plus what
//  it paints after lipsPaint, from gPLip, gPLine, gPCave, gPTeeth, gPTongue.
//
//  API
//    void  lipsInit();                     from charInit, reads uP[]
//    float lipsEval(vec3 q, float host);   q head space, host = host distance at q.  Returns the skin:
//                                          host, soft face moves, carve, lips.  Sets gLipBare (teeth, tongue)
//    void  lipsPaint(vec3 n, inout Surf s);  after your own paint, over it.  Uses the last lipsEval
// ---------------------------------------------------------------------------
#define LIPU (LIP_W/0.27)                // every length below is proportional to the mouth
#ifndef LIP_HU
#define LIP_HU (0.074*LIPU)              // upper lip height at the centre
#endif
#ifndef LIP_HL
#define LIP_HL (0.098*LIPU)              // lower lip height: fuller
#endif
#ifndef LIP_TU
#define LIP_TU (0.028*LIPU)              // how far the upper lip stands off the host
#endif
#ifndef LIP_TL
#define LIP_TL (0.036*LIPU)              // and the lower
#endif
#ifndef LIP_PUSH
#define LIP_PUSH (0.105*LIPU)            // forward travel of the mouth at lip_protrude = 1
#endif
#ifndef LIP_DEPTH
#define LIP_DEPTH (0.42*LIPU)            // how far the cavity reaches behind LIP_O
#endif
#ifndef LIP_GAPY
#define LIP_GAPY (0.262*LIPU)            // lower teeth drop at teeth_gap = 1; the lower lip hides them unless drawn down
#endif
#ifndef LIP_TOOTH_Z
#define LIP_TOOTH_Z (-0.040*LIPU)        // front of the teeth, in the mouth frame.  Behind the lips, inside the host
#endif
#ifndef LIP_TEETH_H
#define LIP_TEETH_H (0.070*LIPU)         // how long the teeth are, from the gum to the biting edge
#endif
#ifndef LIP_ARCH_X
#define LIP_ARCH_X (0.292*LIPU)          // half width of the upper row
#endif
#ifndef LIP_SCALLOP
#define LIP_SCALLOP 1.0                  // how far the biting edge is scalloped.  0 = one straight soft band
#endif
#ifndef LIP_GROOVE
#define LIP_GROOVE 0.09                  // how dark the lines between teeth are.  A whisper, or they read as dentures
#endif
#ifndef LIP_SQUARE
#define LIP_SQUARE 0.22                  // 0 the pointed lens .. 1 a rounded box.  A little of it softens every corner
#endif
#ifndef LIP_BOXR
#define LIP_BOXR 0.80                    // the box's corner radius, in half heights of the opening
#endif
#ifndef LIP_RIM
#define LIP_RIM 0.0                      // 1 = an open mouth keeps its lips all the way round the corners, like a rolled rim
#endif
#ifndef LIP_BOW
#define LIP_BOW 1.0                      // cupid's bow and philtrum.  0 = a plain arc, nobody's upper lip
#endif
#ifndef LIP_OPEN_Y
#define LIP_OPEN_Y 1.0                   // height of the opening at lip_open = 1, against the default
#endif
#ifndef LIP_LIFT
#define LIP_LIFT 0.27                    // lip_upper, lip_lower: travel at 1, in LIP_W
#endif
#ifndef LIP_SPREAD
#define LIP_SPREAD 0.30                  // lip_width = 1: corners out, in LIP_W
#endif
#ifndef LIP_PUCKER
#define LIP_PUCKER 0.80                  // lip_width = -1: corners in, in LIP_W, at no opening
#endif
#ifndef LIP_SKIN
#define LIP_SKIN 1.0                     // gain of the soft moves around the mouth: chin pad, bunching, dimples, cheeks
#endif
#ifndef LIP_BLEND
#define LIP_BLEND (0.017*LIPU)           // how softly the lips run into the skin
#endif
#ifndef LIP_ROLL
#define LIP_ROLL 0.60                    // lip_press rolls the lips in: slope of the lip section
#endif
#ifndef LIP_FLARE
#define LIP_FLARE 0.55                   // lip_protrude turns them out
#endif
#ifndef LIP_BEND
#define LIP_BEND 1.0                     // how far a smile or a spread bends the mouth line
#endif
#ifndef LIP_AO
#define LIP_AO 0.30                      // the soft shade the lips throw on the skin
#endif
#ifndef LIP_SPEC
#define LIP_SPEC 0.55
#endif
#ifndef LIP_SHIN
#define LIP_SHIN 62.0
#endif
#ifndef LIP_MIRROR
#define LIP_MIRROR 0.20
#endif
#ifndef LIP_SHEEN
#define LIP_SHEEN 0.20
#endif
#ifndef LIP_WRAP
#define LIP_WRAP 0.45
#endif
#ifndef LIP_EMIT
#define LIP_EMIT vec3(0.12,0.060,0.050)  // how much the lip colour glows by itself
#endif
#ifndef LIP_PEEL
#define LIP_PEEL (0.120*LIPU)            // how much opening it takes for the lips to come unstuck all the way to the corners
#endif
#ifndef LIP_ZIP
#define LIP_ZIP 0.18                     // how far from the corners the lips stay sealed as a mouth begins to part
#endif
#ifndef LIP_LINE
#define LIP_LINE 1.0                     // width of the closed line
#endif
#ifndef LIP_LINE_SOFT
#define LIP_LINE_SOFT 0.35               // the soft shade the lips carry beside a closed line: what a phone can see
#endif
#ifndef LIP_FLOW
#define LIP_FLOW 1.0                     // how far the skin's paint follows the jaw and the corners
#endif
#ifndef LIP_SMIRK
#define LIP_SMIRK 0.15                   // how far smirk = 1 slides the mouth, in LIP_W
#endif
#ifndef LIP_TOOTH_GAIN
#define LIP_TOOTH_GAIN 1.0               // how bright the teeth are, on top of their colour
#endif
#ifndef LIP_COL_U
#define LIP_COL_U vec3(0.66,0.20,0.19)   // upper lip
#endif
#ifndef LIP_COL_L
#define LIP_COL_L vec3(0.78,0.27,0.25)   // lower lip, a touch lighter: it faces the sky
#endif
#ifndef LIP_COL_LINE
#define LIP_COL_LINE vec3(0.20,0.050,0.050)  // the closed line
#endif
#ifndef LIP_COL_CAVE
#define LIP_COL_CAVE vec3(0.42,0.105,0.105)  // cavity, near the lips
#endif
#ifndef LIP_COL_DEEP
#define LIP_COL_DEEP vec3(0.100,0.026,0.032) // cavity, deep
#endif
#ifndef LIP_COL_TOOTH
#define LIP_COL_TOOTH vec3(0.84,0.785,0.690)
#endif
#ifndef LIP_COL_GUM
#define LIP_COL_GUM vec3(0.52,0.16,0.16)
#endif
#ifndef LIP_COL_TONGUE
#define LIP_COL_TONGUE vec3(0.72,0.24,0.25)
#endif

// ---- pose, filled once per fragment by lipsInit
float gLc, gLs;                    // frame tilt
float gLW, gLAu, gLAl;             // aperture: half width, rise, fall
float gLRu, gLCu, gLRl, gLCl;      // the two arcs: radius, centre's distance from the chord
float gLYC, gLBend, gLTub;         // mid line: height at the centre, corner lift, the small dip at the middle
float gLCornMin, gLCapU, gLPad;
float gLRnd, gLCornT;              // 1 = a round O; thickness left at the corners (0 pointed .. 1 a ring)
float gLHU, gLHL, gLTU, gLTL, gLDU, gLDL;   // lip height, fullness, depth of the section centre
float gLHole;                      // host carve margin under the lips; negative = closed
float gLSeal, gLDark, gLShow;      // 1 sealed .. 0 | 1 dark slit .. 0 | 0 teeth hidden .. 1 out
float gLTunnel;
float gLPro, gLPress, gLTuck, gLSp, gLPk, gLSmile, gLCheek, gLGap;
float gLTipU, gLTipL, gLToothZ;    // upper and lower tooth tips (y), tooth front (z)
float gLTgY, gLTgZ, gLTgTip, gLTgBack, gLTgFlat, gLTgDim, gLTgOut;   // tongue (gLTgOut: stuck out past the lips)
float gLSq, gLBc, gLBr;            // aperture: how much of the box, the box's centre above the mid line, its corner radius
vec2  gLBb;                        // and its half size less that radius
float gLFlare, gLCz;               // lip section: slope, + turned out, - rolled in; how far the corners sit back
float gLWo, gLZipK;                // half width of the part that has come unstuck; softness of the join with the seal
float gLSm, gLSmX, gLTgWide;
#ifdef LIP_TONGUE_FREE
float gLTgFree;                    // 1 where the stuck-out blade (not the tongue inside the mouth) is the tongue surface
#endif

float lipsSq(float x){ return x*x; }
float lipsBell(float x){ x = clamp(1.0 - x*x, 0.0, 1.0); return x*x; }

void lipsInit(){
  gLc = cos(LIP_TILT); gLs = sin(LIP_TILT);
  float mo    = clamp(uP[P_MOUTH_OPEN], 0.0, 1.0), to = clamp(uP[P_TONGUE_OUT], 0.0, 1.0);
  gLTgOut = to;
  float press = clamp(uP[P_LIP_PRESS], 0.0, 1.0)*(1.0 - max(mo, to));
  float keep  = 1.0 - press;
  float open  = max(clamp(uP[P_LIP_OPEN], 0.0, 1.0)*keep, max(mo, 0.42*to));     // commanded opening; the tongue parts the lips
  float up    = clamp(uP[P_LIP_UPPER], 0.0, 1.0)*keep;
  float lo    = clamp(uP[P_LIP_LOWER], 0.0, 1.0)*keep;
  float wid   = clamp(uP[P_LIP_WIDTH], -1.0, 1.0);
  float pro   = clamp(uP[P_LIP_PROTRUDE], 0.0, 1.0);
  float tuck  = clamp(uP[P_LIP_TUCK], 0.0, 1.0)*keep;
  float gap   = max(max(clamp(uP[P_TEETH_GAP], 0.0, 1.0), open - 0.30), max(mo, 0.36*to));   // lips can part that far only with the jaw down
  float tip   = clamp(uP[P_TONGUE_TIP], 0.0, 1.0);
  float dent  = clamp(uP[P_TONGUE_DENTAL], 0.0, 1.0);
  float back  = clamp(uP[P_TONGUE_BACK], 0.0, 1.0);
  float smile = clamp(uP[P_SMILE], 0.0, 1.0);
  float cheek = clamp(uP[P_CHEEK], 0.0, 1.0);
  gLSm = clamp(uP[P_SMIRK], -1.0, 1.0); gLSmX = gLSm*LIP_SMIRK*LIP_W;
  float pk = max(-wid, 0.0), sp = max(wid, 0.0);
  gLPro = pro; gLPress = press; gLTuck = tuck; gLSp = sp; gLPk = pk; gLSmile = smile; gLCheek = cheek; gLGap = gap;

  // corners in x.  A full pucker draws them in to the rim of a small opening, and the opening sets the size of that rim
  gLW = LIP_W*(1.0 + LIP_SPREAD*sp - (LIP_PUCKER - 0.30*open)*pk + 0.05*smile + 0.03*cheek)*(1.0 - 0.10*open*(1.0 - pk))*(1.0 - 0.04*press);

  // the opening.  A round O needs height as well as narrowness, so the pucker pulls both heights to the width
  float au = LIP_W*(0.26*LIP_OPEN_Y*open + LIP_LIFT*up);
  float al = LIP_W*(0.80*LIP_OPEN_Y*open + LIP_LIFT*lo);
  float amt = au + al;
  gLRnd = pk*pk*smoothstep(0.0, 0.11, open + 0.5*(up + lo));
  au = mix(au, 0.92*gLW*(1.0 + 0.3*open), gLRnd);
  al = mix(al, 1.00*gLW*(1.0 + 0.5*open), gLRnd);
  au += LIP_W*0.17*tuck;                               // the upper lip clears the teeth ...
  al *= 1.0 - 0.85*tuck;                               // ... and the lower lip comes up to them
  // the corners ride down with the lower lip, about a fifth of its travel
  float drop = 0.20*al*(1.0 - gLRnd);
  gLYC = -drop;
  gLAu = max(au + drop, 0.0009);
  gLAl = max(al - drop, 0.0009);
  // Lips part from the middle: toward the corners they stay stuck until there is room, and a press holds them longer.
  // So a mouth starting to open is a small lens on one closed line, and press first, peel on release needs no history
  float hc = 0.5*(gLAu + gLAl);
  float peel = smoothstep(0.0, LIP_PEEL, hc*2.0*(1.0 - 0.55*press));
  gLWo = gLW*mix(1.0 - LIP_ZIP*(1.0 - smoothstep(0.25, 0.75, pk))*(1.0 - LIP_RIM), 1.0, peel);   // a rim has no seal to run on past it
  gLZipK = max(0.9*hc*(1.0 - peel), 1e-4);                 // once unstuck the join must not bulge the corners
  gLRu = 0.5*(gLWo*gLWo + gLAu*gLAu)/gLAu; gLCu = gLRu - gLAu;
  gLRl = 0.5*(gLWo*gLWo + gLAl*gLAl)/gLAl; gLCl = gLRl - gLAl;

  // how much of the box: every mouth a little, the funnel of CH and a lip drawn off the teeth more.  A round mouth is
  // the same curve either way
  float hb = 0.5*(gLAu + gLAl);
  float fun = smoothstep(0.08, 0.36, pk)*smoothstep(0.92, 0.55, pk)*smoothstep(0.20, 0.60, up + lo);
  gLSq = clamp(LIP_SQUARE + 0.62*fun + 0.42*sp*smoothstep(0.0, 0.30, up + lo), 0.0, 1.0);
  gLBr = min(LIP_BOXR*hb, 0.96*gLWo);
  gLBb = vec2(gLWo, hb) - gLBr;
  gLBc = 0.5*(gLAu - gLAl);

  float o = smoothstep(0.0, 0.6, open);
  gLBend = LIP_BEND*LIP_W*(0.024 + 0.29*smile + 0.09*cheek + 0.105*sp - 0.07*pk)*(1.0 - 0.55*o)*(1.0 - 0.45*press)*(1.0 - gLRnd);   // the bend works in units of the corner: a ring must not sag
  gLSeal = 1.0 - smoothstep(0.0, 0.012*LIPU, amt);
  gLTub  = 0.0045*LIPU*gLSeal*(1.0 - pk);

  float mid = smoothstep(0.15, 0.85, pk);
  gLCornT = max(mid, LIP_RIM*smoothstep(0.015*LIPU, 0.085*LIPU, hb));
  gLCornMin = mix(0.27 - 0.07*sp, 1.0, gLCornT);
  gLCapU = 0.5*(LIP_HU + LIP_HL)*gLCornMin*0.80/gLW;
  gLPad = lipsSq(1.0 - gLCornT)*(1.0 - smoothstep(0.004*LIPU, 0.042*LIPU, hb));   // only a shut mouth needs it: open, it bulges into the corner
  // a lip thins as it stretches: the upper one most when lifted off the teeth, the lower when the jaw drops
  float thinU = 1.0 - 0.30*sp - 0.14*open - 0.30*up, thinL = 1.0 - 0.28*sp - 0.24*open - 0.16*lo;
  float hm = 0.5*(LIP_HU + LIP_HL);
  gLHU = mix(LIP_HU, hm*0.84, pk)*(thinU + 0.24*pk)*(1.0 - 0.46*press);
  gLHL = mix(LIP_HL, hm, pk)*(thinL + 0.24*pk)*(1.0 - 0.42*press)*(1.0 - 0.22*tuck);
  float tm = 0.5*(LIP_TU + LIP_TL);
  gLTU = mix(LIP_TU, tm, pk)*(1.0 - 0.22*press + 0.34*pk + 0.25*pro);
  gLTL = mix(LIP_TL, tm, pk)*(1.0 - 0.22*press + 0.34*pk + 0.25*pro)*(1.0 - 0.45*tuck);
  gLDU = (0.004*LIPU)*(1.0 - press) - 0.002*LIPU*press + 0.016*LIPU*pro;
  gLDL = gLDU - 0.030*LIPU*tuck;
  gLFlare = LIP_FLARE*pro*(0.45 + 0.55*smoothstep(0.02, 0.30, up + lo)) - LIP_ROLL*press;
  gLCz = 0.016*LIPU*sp - 0.010*LIPU*pk*(1.0 - gLCornT);

  gLHole = mix(-0.024*LIPU, 0.030*LIPU, smoothstep(0.0, 0.020*LIPU, amt));
  // Short on purpose.  While a brightness depends on the opening every pixel of the interior changes with every move of
  // the lips, and that costs far more than one band arriving: settle early, then only the lip edges move
  gLDark = 1.0 - smoothstep(0.012*LIPU, 0.050*LIPU, amt);
  gLShow = smoothstep(0.012*LIPU, 0.032*LIPU, amt + 0.03*LIPU*tuck);
  gLTunnel = pk*pk*smoothstep(0.1, 0.9, pro)*smoothstep(0.42, 0.16, open);   // deep behind a tight pucker the teeth fall into the dark

  gLTipU = -0.007*LIPU + 0.030*LIPU*(1.0 - gLShow) + 0.002*LIPU*tuck;   // the teeth wait behind the lip until there is room
  gLTipL = gLTipU + 0.012*LIPU - LIP_GAPY*gap;           // shut, the lower row stands behind the upper: one clean seam
  gLTipL = mix(gLTipL, min(gLTipL, gLYC - gLAl - 0.040*LIPU), smoothstep(0.40, 0.66, pk));   // only a true round hides the lower row: a funnel keeps its teeth together
  gLToothZ = LIP_TOOTH_Z + 0.006*LIPU*tuck;

  gLTgY = max(gLTipL - 0.060*LIPU, gLYC - 0.60*gLAl - 0.034*LIPU);
  gLTgDim = 1.0 - smoothstep(0.035*LIPU, 0.120*LIPU, amt);   // no tongue colour in a narrow opening: a dark mouth, then a mouth
  gLTgTip = tip;
  gLTgBack = back;
  gLTgWide = tip*back;                                       // tip and back together: the whole blade held up, not a flick
  gLTgFlat = dent;
  gLTgZ = LIP_TOOTH_Z - 0.308*LIPU + 0.128*LIPU*dent + 0.80*LIPU*gLTgOut;
  gLTgY = mix(gLTgY, gLYC - gLAl + 0.034*LIPU, gLTgOut);       // stuck out, it lies on the lower lip
  gLTgDim *= 1.0 - gLTgOut;
}

// ---- outputs of lipsEval
vec3  gLQ;                         // the point in the mouth frame
float gLDepthS, gLTgX;
float gLE, gLV, gLU, gLRing, gLHoleD, gLUpper, gLH, gLTeeth, gLTongue, gLipBare, gLSkin, gLTgB;
vec3  gLSt;
float gLToothR;                    // 0 upper row, 1 lower row

// A row of teeth as ONE soft shape: a band around the dental arch (a half ellipse in plan: half axes arch.xy, centre
// z arch.z), its biting edge softly scalloped.  w = height above the gum line, tips toward w < 0.  The band is
// rigid: it never follows lip_width, a narrow mouth just shows less of it.  st = (0 middle .. 1 end of the row,
// 0 tooth centre .. 1 between teeth, w).
float lipsToothRow(float x, float z, float w, vec3 arch, float tw, float n, float thick, float len, float scal, out vec3 st){
  vec2 e = vec2(x/arch.x, (z - arch.z)/arch.y);
  float R = 0.5*(arch.x + arch.y);
  float a = atan(e.x, e.y)*R/tw;
  float f = abs(fract(a) - 0.5)*2.0;
  st = vec3(a/n, f, w);
  float band = abs((length(e) - 1.0)*R) - thick;
  float tip = -w - len*(1.0 - 0.14*smoothstep(0.3, 1.0, a/n)) + scal*0.0065*LIPU*f*f*f;
  float d = smax(band, tip, 0.010*LIPU);
  d = max(d, w - 0.100*LIPU);                            // the gum runs on, so the band never ends in the open
  return max(d, (a - n)*tw);
}

float lipsEval(vec3 q, float host){
  vec3 pq = q - LIP_O;
  vec3 m = vec3(pq.x, gLc*pq.y + gLs*pq.z, -gLs*pq.y + gLc*pq.z);
  gLQ = m;
  float x = m.x - gLSmX, y = m.y, ax = abs(x);              // smirk slides the lips and the skin, never the teeth
  float u = ax/gLW, uu = u*u;
  float sx = clamp(x/gLW, -1.0, 1.0);

  // ---- the mid line, and where it sits in y.  It levels off past the corner, with no kink.  A smirk lifts one corner
  float v = y - gLYC - gLBend*(1.0 + 0.85*gLSm*sx)*uu/sqrt(1.0 + 0.39*uu*uu) + gLTub*lipsBell(x/(0.032*LIPU)) - gLSm*0.095*LIP_W*sx;
  gLV = v; gLU = u;

  // ---- signed distance to the lens, with its gradient
  float eU = length(vec2(ax, v + gLCu)) - gLRu;
  float eL = length(vec2(ax, v - gLCl)) - gLRl;
  float e = max(eU, eL);
  vec2 gl = eU > eL ? vec2(ax, v + gLCu)/max(eU + gLRu, 1e-5) : vec2(ax, v - gLCl)/max(eL + gLRl, 1e-5);
  float cc = v > 0.0 ? gLCu : gLCl;
  // past a corner the corner itself is nearest.  Only for an arc shallower than a semicircle (cc > 0): a rounder
  // arc's centre is on the far side of the chord, and the same test would claim the middle of the opening
  if(cc > 0.0 && gLWo*(abs(v) + cc) < cc*ax){ e = length(vec2(ax - gLWo, v)); gl = vec2(ax - gLWo, v)/max(e, 1e-5); }
  // and to the rounded box of the same corners and heights.  At a corner the two fields point different ways, so
  // their mix runs slow there and the lips would swell: divide by the length of the mixed gradient
  vec2 bq = vec2(ax, abs(v - gLBc)) - gLBb;
  vec2 bo = max(bq, 0.0);
  float bl = length(bo);
  float eB = bl + min(max(bq.x, bq.y), 0.0) - gLBr;
  vec2 gb = bl > 0.0 ? bo/bl : (bq.x > bq.y ? vec2(1.0,0.0) : vec2(0.0,1.0));
  gb.y *= v > gLBc ? 1.0 : -1.0;
  e = mix(e, eB, gLSq)/max(length(mix(gl, gb, gLSq)), 0.45);
  // the seal: the line runs on from the lens to the true corners, softly joined.  Once unstuck it lies inside the lens
  e = smin(e, length(vec2(max(ax - gLW, 0.0), v)), gLZipK);
  gLE = e;

  // ---- the skin around the mouth moves first
  float D = host;
  float my = y + 0.02*LIPU;
  D -= LIP_PUSH*gLPro*lipsBell(length(vec2(x/(1.30*LIP_W), my/(1.12*LIP_W))));         // pucker: the whole mouth comes forward
  // tuck: the skin under the mouth line folds in, and the lower lip goes with it
  D += 0.046*LIPU*gLTuck*lipsBell(length(vec2(x/(1.25*LIP_W), (y + 0.080*LIPU)/(0.165*LIPU))));
  // chin: a soft pad that drops with the teeth, and the crease under the lower lip
  float lipBot = -gLAl + gLYC - gLHL;
  // (never out at the host's silhouette: it would dent the outline)
  float chinY = max(lipBot - 0.115*LIPU - 0.075*LIPU*gLGap + 0.030*LIPU*gLPress, -1.30*LIP_W);
  D -= LIP_SKIN*(0.010*LIPU + 0.006*LIPU*gLGap + 0.006*LIPU*gLPress)*lipsBell(length(vec2(x/(0.62*LIP_W), (y - chinY)/(0.105*LIPU))));
  D += LIP_SKIN*(0.0060*LIPU + 0.0050*LIPU*gLPress)*lipsBell(length(vec2(x/(0.45*LIP_W), (y - lipBot + 0.018*LIPU)/(0.024*LIPU))));
  // the jaw takes the cheeks with it: they fall in beside an open mouth, and fall in further round a pucker
  D += LIP_SKIN*(0.012*LIPU*gLGap + 0.010*LIPU*gLPk)*lipsBell(length(vec2((ax - 1.30*LIP_W)/(0.52*LIP_W), (y + 0.16*LIP_W)/(0.62*LIP_W))));
  // press: the skin bunches softly above and below the sealed lips
  float lipTop = gLAu + gLYC + gLHU;
  D -= LIP_SKIN*0.0100*LIPU*gLPress*lipsBell(length(vec2(x/(0.84*LIP_W), (y - lipBot + 0.040*LIPU)/(0.052*LIPU))));
  D -= LIP_SKIN*0.0075*LIPU*gLPress*lipsBell(length(vec2(x/(0.70*LIP_W), (y - lipTop - 0.032*LIPU)/(0.046*LIPU))));
  // philtrum, a soft dip above the bow
  D += LIP_BOW*0.0034*LIPU*(1.0 - gLPk)*lipsBell(x/(0.046*LIPU))*smoothstep(lipTop - 0.012*LIPU, lipTop + 0.020*LIPU, y)*smoothstep(lipTop + 0.17*LIPU, lipTop + 0.06*LIPU, y);
  // corner dimples: deeper when the corners are pulled or pressed
  float dm = 0.0035*LIPU + 0.0065*LIPU*(0.6*gLSmile + gLSp + 1.5*gLPress + 0.9*gLSm*sign(x));
  D += LIP_SKIN*dm*(1.0 - gLPk)*lipsBell(length(vec2((ax - gLW - 0.012*LIPU)/(0.040*LIPU), v/(0.052*LIPU))));
  // cheeks: lift with cheek and the smile
  float smk = 1.0 + 0.8*gLSm*sign(x);                      // the cheek on the lifted side rises with its corner
  D -= LIP_SKIN*(0.008*LIPU + 0.030*LIPU*gLCheek + 0.022*LIPU*gLSmile + 0.012*LIPU*gLSp)*smk*lipsBell(length(vec2((ax - 1.42*LIP_W)/(0.62*LIP_W), (y - 0.20*LIP_W)/(0.56*LIP_W))));
  // and a soft fold runs up from the corner, away from it, when the corners are drawn back
  float fold = ax - gLW*1.10 - 0.42*max(y - gLYC, 0.0);
  D += LIP_SKIN*0.0060*LIPU*(0.5*gLSmile + gLSp)*smk*lipsBell(fold/(0.042*LIPU))*smoothstep(-0.03*LIPU, 0.05*LIPU, y - gLYC)*smoothstep(0.34*LIPU, 0.12*LIPU, y - gLYC);

  float dd = D;

  // ---- thickness along the curve
  float T = pow(max(1.0 - uu, 0.0) + 1e-4, 0.60);         // tapers toward the corner, but stops at a soft rounded end ...
  T = gLCornMin + (1.0 - gLCornMin)*T;                    // ... or does not taper at all when the mouth is a ring
  float cap = clamp((u - 1.0)/(gLCapU + 1e-3), 0.0, 1.0);
  T *= mix(sqrt(max(1.0 - cap*cap, 0.0)), 1.0, gLCornT*gLCornT);   // past the corner: a round end, not a cut
  // a pucker is not a donut: the lips stay fullest top and bottom, and thin where they turn round the sides
  T *= 1.0 - 0.26*gLRnd*ax*ax/(ax*ax + v*v + 1e-5);
  float ws = mix(0.0045*LIPU, 0.80*(gLW + LIP_HU), gLCornT);   // a ring blends upper into lower around its sides
  float wu = smoothstep(-ws, ws, v);
  gLUpper = wu;
  float rx = x*LIP_W/gLW;
  float bow = 1.0 + LIP_BOW*(1.0 - gLPk)*(0.12*lipsBell((abs(rx) - 0.062*LIPU)/(0.085*LIPU)) - 0.16*lipsBell(rx/(0.034*LIPU)));   // cupid's bow, philtrum dip
  float hAvg = 0.5*(gLHU + gLHL);                         // both lips arrive at the corner equal, so the end is one soft shape
  float kc = smoothstep(0.55, 0.98, u)*(1.0 - gLCornT);
  float hu = mix(gLHU, hAvg, kc)*T*bow, hl = mix(gLHL, hAvg, kc)*T;
  float h = mix(hl, hu, wu);
  gLH = h;
  float dc = mix(gLDL, gLDU, wu) - gLCz*uu/(1.0 + 0.35*uu);   // the corners sit back in a spread
  float t = mix(gLTL, gLTU, wu)*sqrt(T + 1e-3);
  t = mix(t, 0.5*(gLTL + gLTU)*sqrt(T + 1e-3), kc);

  // ---- the section: soft, and overlapping the other lip a little at the line.  Its slope turns the inner edge out
  // in a funnel and rolls it in under a press
  const float OV = 0.26;
  float hh = max(0.5*h, 0.0030), tt = max(t, 0.0030);
  float ec = e - hh*(1.0 - OV);
  vec2 tp = vec2(ec, dd - dc + gLFlare*ec);
  vec2 tr = vec2(hh, tt);
  float k0 = length(tp/tr), k1 = length(tp/(tr*tr));
  float ring = k0*(k0 - 1.0)/max(k1, 1e-4);
  // the corner: a small soft pad where the lips meet, so the end is round and never a knife point
  vec3 pr = vec3(0.034, 0.021, 0.017)*LIPU*gLPad*(1.0 - 0.62*gLPress) + 0.001;   // a press squeezes it too, or the corners grow knobs
  float pad = sdEllipsoid(vec3(ax - gLW + 0.008*LIPU, v, dd - 0.5*(gLDU + gLDL)), pr);
  ring = smin(ring, pad, 0.014*LIPU);
  ring = max(ring, -(m.z + LIP_DEPTH));                   // the far side of the host carries no lips
  gLRing = ring;
  gLDepthS = tp.y/tt;
  gFOffLips = smoothstep(0.03*LIPU, 0.17*LIPU, e - h);

  // ---- the carve: closed at rest, wider under the lips and behind them
  float mg = gLHole*max(T, 0.0);
  float ce = e - mg - 0.42*clamp(-dd - 0.020*LIPU, 0.0, 0.17*LIPU)*smoothstep(0.0, 0.012*LIPU, gLHole);
  float hole = smax(ce, -(m.z + LIP_DEPTH), 0.05*LIPU);
  gLHoleD = hole;
  float body = smax(D, -hole, 0.008*LIPU);
  float skin = smin(body, ring, LIP_BLEND);
  gLSkin = skin;

  // ---- teeth, rigid on their arch, in the mouth frame
  vec3 st, stL;
  vec3 archU = vec3(LIP_ARCH_X, 0.32*LIPU, gLToothZ - 0.32*LIPU);
  float tx = abs(m.x);
  float tU = lipsToothRow(tx, m.z, y - gLTipU - LIP_TEETH_H, archU, 0.052*LIPU, 8.5, 0.015*LIPU, LIP_TEETH_H, LIP_SCALLOP, st);
  vec3 archL = vec3(0.885*LIP_ARCH_X, 0.30*LIPU, gLToothZ - 0.016*LIPU - 0.30*LIPU);
  float tL = lipsToothRow(tx, m.z, gLTipL - 0.92*LIP_TEETH_H - y, archL, 0.046*LIPU, 8.5, 0.014*LIPU, 0.92*LIP_TEETH_H, 0.4*LIP_SCALLOP, stL);
  gLToothR = tL < tU ? 1.0 : 0.0;
  gLSt = tL < tU ? stL : st;
  // only inside the carve, clear of its wall, and always under the skin: where the host curves or leans away from the
  // frame a rigid row would come out through the surface, a lower row under the lip or an upper one up a sloped face
  gLTeeth = max(max(min(tU, tL), hole + 0.007*LIPU), dd + 0.010*LIPU);

  // ---- tongue.  A body that lies low, and a blade that narrows to a round tip.  The tip rises to touch behind the upper
  // teeth, comes forward between the teeth, or the body humps at the back; tip and back together hold the whole blade up.
  // Never a change of topology: one warped solid
  vec3 tg = vec3(m.x, y - gLTgY, m.z - gLTgZ);
  float tz = clamp((tg.z + 0.22*LIPU)/(0.44*LIPU), 0.0, 1.0);     // 0 back .. 1 tip
  float toTip = gLTipU + 0.022*LIPU - gLTgY;                      // from its bed to behind the upper teeth
  float toGap = 0.5*(gLTipU + gLTipL) - gLTgY;                    // or to the middle of the gap
  float act = max(gLTgFlat, gLTgTip);
  tg.y += 0.15*(tg.z < 0.0 ? tg.z : tg.z*(1.0 - act));            // its back rises, its idle tip dips behind the lower lip
  float blade = smoothstep(0.34, 1.0, tz);
  tg.y -= gLTgTip*toTip*smoothstep(0.26, 0.86, tz) + gLTgFlat*toGap*smoothstep(0.30, 0.95, tz);
  tg.y -= gLTgBack*(0.190 + 0.080*gLTgWide)*LIPU*lipsBell((tz - 0.26 - 0.20*gLTgWide)/(0.34 + 0.20*gLTgWide));
  float nar = act*(1.0 - 0.72*gLTgWide);
  tg.x *= (1.0 + 0.50*nar)*(1.0 + 0.95*nar*smoothstep(0.10, 0.76, tz));   // the working blade is narrow: that is what reads
  gLTgX = tg.x/(0.195*LIPU);
  gLTgB = blade;
  float tong = sdEllipsoid(tg, vec3(0.195*LIPU, (0.052 + 0.012*act)*LIPU, 0.255*LIPU))*0.72;
  tong += 0.0016*LIPU*smoothstep(0.022*LIPU, 0.0, abs(tg.x));     // the soft centre groove
  tong = max(max(tong, hole + 0.007*LIPU), dd + 0.010*LIPU - 0.30*LIPU*gLTgOut);   // stuck out, it passes the lips
#ifdef LIP_TONGUE_FREE
  // muse (optional, LIP_TONGUE_FREE = how far below the lower lip the tip hangs, in LIPU): stuck out, a soft rounded
  // blade comes out over the lower lip and lies on the chin, NOT clipped to the opening (inside the lens alone a small
  // mouth never shows a tongue out).  It grows from just inside the lower lip; at tongue_out 0 it is not there
  gLTgFree = 0.0;
  if(gLTgOut > 0.001){
    float tk = gLTgOut;
    float y0 = gLYC - gLAl;                                            // the lower lip's inner edge
    vec3 ta = vec3(0.0, y0 + 0.012*LIPU, -0.06*LIPU);
    vec3 tb = mix(ta, vec3(0.0, y0 - LIP_TONGUE_FREE*LIPU, 0.08*LIPU), tk);
    vec3 tv = tb - ta; float tl = max(length(tv), 1e-4); vec3 td = tv/tl;
    vec3 pa = m - ta;
    float th = clamp(dot(pa, td)/tl, 0.0, 1.0);
    vec3 lc = pa - td*(th*tl);
    vec3 tn = normalize(cross(vec3(1.0, 0.0, 0.0), td));
    float wx = (0.175 - 0.020*th*th)*LIPU*mix(0.55, 1.0, tk), wn = 0.050*LIPU;    // wide and round: a cartoon tongue
    vec3 ts = vec3(lc.x/wx, dot(lc, tn)/wn, dot(lc, td)/(0.70*wx));                     // a rounded, not pointed, tip
    float blade = (length(ts) - 1.0)*min(wx, wn);
    blade += 0.0016*LIPU*smoothstep(0.022*LIPU, 0.0, abs(lc.x));      // the same soft centre groove
    if(blade < tong){ gLTgFree = 1.0; gLTgX = 0.85*lc.x/wx; gLTgB = 1.0; }
    tong = min(tong, blade);
  }
#endif
  gLTongue = tong;

  gLipBare = min(gLTeeth, tong);
  return skin;
}

// How far the mouth has carried the skin at this point, in the plane of the mouth.  A character looks its paint up at
// q - this, so beds, cracks, nap and blush move with the jaw and the corners instead of the lips sliding over a fixed
// surface.  Nothing moves alone
vec2 lipsFlow(){
  vec2 p = vec2(gLQ.x - gLSmX, gLQ.y);
  float a = abs(p.x), wx = exp(-lipsSq(p.x/(1.65*LIP_W)));
  float fl = (gLYC - gLAl)*exp(-lipsSq(min(p.y, 0.0)/(1.15*LIP_W)));          // everything under the line follows the jaw
  float fu = (gLYC + gLAu)*exp(-lipsSq(max(p.y, 0.0)/(0.70*LIP_W)));          // the skin over it follows the upper lip, less far
  float fy = mix(fl, fu, smoothstep(-0.45*LIP_W, 0.45*LIP_W, p.y))*wx;
  float fx = (gLW - LIP_W)*sign(p.x)*exp(-lipsSq((a - LIP_W)/(0.95*LIP_W)))*exp(-lipsSq(p.y/(1.05*LIP_W)));
  return LIP_FLOW*(vec2(fx, fy) + vec2(gLSmX, 0.0)*exp(-dot(p, p)/lipsSq(1.5*LIP_W)));
}

// ---- outputs of lipsPaint, for the character's own material after it
float gPLip, gPLine, gPCave, gPTeeth, gPTongue;

void lipsPaint(vec3 n, inout Surf s){
  vec3 m = gLQ;
  float e = gLE, v = gLV;
  bool isTeeth = gLTeeth < 0.003 && gLTeeth <= min(gLSkin, gLTongue) + 0.003;
  bool isTongue = gLTongue < 0.003 && gLTongue <= gLSkin + 0.003;
  float lipM = smoothstep(0.0075*LIPU, 0.0015*LIPU, gLRing);
  float wall = smoothstep(0.008*LIPU, 0.0, gLHoleD)*(1.0 - lipM);          // cavity skin; never the lips
  gPLip = lipM; gPLine = 0.0; gPCave = 0.0; gPTeeth = 0.0; gPTongue = 0.0;

  // ---- dimples, and the soft shade the lips throw on the skin around them
  float dimple = exp(-lipsSq(length(vec2((abs(m.x) - gLW*1.03)/(0.024*LIPU), v/(0.030*LIPU)))))*(1.0 - gLPk);
  s.alb *= mix(vec3(1.0), vec3(0.80,0.66,0.64), LIP_SKIN*(0.40 + 0.5*gLPress + 0.25*gLSp)*dimple);
  float shade = smoothstep(0.075*LIPU, 0.0, gLRing)*(1.0 - lipM);
  s.alb *= 1.0 - LIP_AO*(0.34 + 0.66*gLPro)*shade*shade;

  // ---- lips
  vec3 lipC = mix(LIP_COL_L, LIP_COL_U, gLUpper);
  float wet = smoothstep(0.28, -0.02, e/max(gLH, 1e-3));   // the inner edge deepens into the mouth
  lipC = mix(lipC, lipC*vec3(0.72,0.54,0.54), 0.50*wet);
  float under = smoothstep(0.0, -0.9, gLDepthS)*smoothstep(0.30, -0.10, e/max(gLH, 1e-3));
  lipC = mix(lipC, mix(LIP_COL_CAVE, LIP_COL_DEEP, 0.4), under);
  // pressed lips lose their colour where they are squeezed, and fold darkly into the seam
  lipC = mix(lipC, lipC*vec3(1.10,0.86,0.86) + 0.05, 0.55*gLPress*smoothstep(0.75, 0.10, abs(e/max(gLH, 1e-3) - 0.45)));
  lipC *= 1.0 - LIP_LINE_SOFT*(0.75 + 0.5*gLPress)*(1.0 - gLShow)*smoothstep(0.55, 0.02, e/max(gLH, 1e-3));
  s.alb = mix(s.alb, lipC, lipM);
  s.spec = mix(s.spec, LIP_SPEC, lipM); s.shin = mix(s.shin, LIP_SHIN, lipM);
  s.mirror = mix(s.mirror, LIP_MIRROR, lipM);
  s.sheen = mix(s.sheen, LIP_SHEEN, lipM); s.wrap = mix(s.wrap, LIP_WRAP, lipM);
  s.emit = mix(s.emit, lipC*LIP_EMIT, lipM);
  // a pucker gathers the lip into soft radial folds
  if(gLRnd > 0.0 && lipM > 0.0){
    float fa = atan(m.x, v + 1e-4);
    vec3 fold = vec3(cos(fa), -sin(fa), 0.0)*sin(fa*13.0 + 0.7*sin(fa*5.0));
    s.n = normalize(s.n + 0.11*gLRnd*lipM*fold*smoothstep(0.15, 0.9, e/max(gLH, 1e-3)));
  }

  // ---- the line: at rest one fine crease; opening, the same paint is the inner edge
  float line = smoothstep((0.0048 + 0.0030*gLPress)*LIPU*LIP_LINE, 0.0008*LIPU, e)*max(smoothstep(1.08, 0.97, gLU), gLCornT);
  s.alb = mix(s.alb, LIP_COL_LINE, 0.92*line);
  s.spec *= 1.0 - line; s.mirror *= 1.0 - line;
  gPLine = line;

  // ---- the cavity
  if(wall > 0.0 || e < 0.0){
    float cm = max(wall, smoothstep(0.0, -0.002*LIPU, e)*(1.0 - lipM));
    float deep = max(smoothstep(-0.015*LIPU, -0.20*LIPU, m.z), 0.92*gLTunnel);
    vec3 c = mix(LIP_COL_CAVE, LIP_COL_DEEP, deep);
    c = mix(c, LIP_COL_DEEP*0.7, max(gLDark, gLTgDim*0.8));
    s.alb = mix(s.alb, c, cm);
    s.occ = mix(s.occ, 0.55, cm);
    s.spec *= 1.0 - cm; s.mirror *= 1.0 - cm; s.sheen *= 1.0 - cm;
    s.emit = mix(s.emit, c*0.34*(1.0 - gLDark), cm);
#ifdef LOOK_V2
    // look: the cavity has depth — its walls by the lips glow warm, the back falls away into the dark
    s.emit = mix(s.emit, c*mix(0.46, 0.13, deep)*(1.0 - gLDark), cm);
#endif
    s.wrap = mix(s.wrap, 0.3, cm);
    gPCave = cm;
  }
  // the interior is lit mostly by emit: the engine's 4-step AO draws hard edges in here.  What form it has comes
  // from this soft top-and-front light on the true normal
  float form = 0.70 + 0.30*smoothstep(-0.3, 0.8, n.z + 0.5*n.y);
  if(isTongue){
    vec3 c = LIP_COL_TONGUE;
    float deep = smoothstep(-0.16*LIPU, -0.46*LIPU, m.z);
#ifdef LIP_TONGUE_FREE
    deep *= 1.0 - gLTgFree;                        // the blade out over the lip is in the light
#endif
    c = mix(c, c*0.42, max(deep, 0.8*gLTunnel));
    float act = max(gLTgTip, gLTgFlat);
    c = mix(c, mix(LIP_COL_DEEP, c*0.62, act), smoothstep(mix(0.50, 0.72, act), 1.0, abs(gLTgX)));   // a working tip keeps its outline, a resting body sinks away
    // the raised tip shows its underside: the tip itself lit, the blade behind it falling into the shade, the seam down the middle
    c = mix(c, mix(LIP_COL_CAVE, LIP_COL_DEEP, 0.40), 0.62*act*(1.0 - gLTgWide)*smoothstep(0.56, 0.12, gLTgB));
    c *= 1.0 - 0.30*gLTgTip*(1.0 - gLTgWide)*smoothstep(0.09, 0.02, abs(gLTgX))*smoothstep(0.95, 0.4, gLTgB);
    float tgDark = max(gLDark*(1.0 - act), gLTgDim*(1.0 - max(act, gLTgBack)));
#ifdef LIP_TONGUE_FREE
    tgDark *= 1.0 - gLTgFree;
#endif
    c = mix(c, LIP_COL_DEEP, tgDark);
    s.alb = c*0.10;
    s.n = n; s.spec = 0.12; s.shin = 28.0; s.wrap = 0.35; s.sheen = 0.0; s.mirror = 0.0; s.occ = 1.0; s.trans = 0.0;
    s.emit = c*0.62*form*(1.0 - 0.7*tgDark);
#ifdef LOOK_V2
    // look: a moist tongue — a little fuller in colour, a soft wet highlight on its upper face
    float wetT = pow(clamp(dot(n, normalize(vec3(-0.25, 0.80, 0.55))), 0.0, 1.0), 10.0);
    s.emit = (c*vec3(1.06,0.98,0.98)*0.74*form + vec3(1.0,0.90,0.88)*0.20*wetT)*(1.0 - 0.7*tgDark);
#endif
    gPTongue = 1.0;
  }
  if(isTeeth){
    // one soft shape: a whisper of a groove near the biting edge, falling into shadow toward the corners
    float groove = smoothstep(0.80, 1.0, gLSt.y)*smoothstep(-0.020*LIPU, -0.055*LIPU, gLSt.z);
    vec3 c = LIP_COL_TOOTH*(1.0 - LIP_GROOVE*groove)*mix(1.0, 0.76, gLToothR);
    float side = smoothstep(0.50, 1.00, gLSt.x);
    c = mix(c, LIP_COL_CAVE*0.40, side);
    c *= mix(0.60, 1.0, smoothstep(0.0, 0.042*LIPU, -e));     // the lips' shade: even along their edge, whatever the curve
    c *= mix(1.0, 0.56, gLToothR*smoothstep(0.016*LIPU, 0.0, gLTipU - m.y));   // the upper row's edge shades the row behind it
    float gum = smoothstep(-0.006*LIPU, 0.006*LIPU, gLSt.z);
    c = mix(c, mix(LIP_COL_GUM*(1.0 - 0.3*side), LIP_COL_CAVE*0.6, smoothstep(0.0, 0.03*LIPU, gLSt.z)), gum);
    float dk = max(max(1.0 - gLShow, 0.985*gLTunnel), smoothstep(-0.011*LIPU, -0.0065*LIPU, gLHoleD));
    c = mix(c, LIP_COL_DEEP, dk);
    s.alb = c*0.08;                                // as the tongue: AO must not cut the row, and no sun may find it
    s.n = n; s.spec = 0.07*(1.0 - gum)*(1.0 - dk); s.shin = 40.0; s.wrap = 0.3; s.sheen = 0.0; s.mirror = 0.0; s.occ = 1.0; s.trans = 0.0;
    s.emit = c*vec3(0.72,0.69,0.66)*LIP_TOOTH_GAIN*(1.0 - 0.7*dk);
#ifdef LOOK_V2
    s.emit += vec3(1.0,0.97,0.92)*0.10*(1.0 - gum)*(1.0 - dk)*pow(clamp(dot(n, normalize(vec3(-0.25, 0.55, 0.80))), 0.0, 1.0), 14.0);   // enamel
#endif
    gPTeeth = 1.0;
  }
}

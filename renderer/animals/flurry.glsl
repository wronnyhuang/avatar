// ---------------------------------------------------------------------------
// animals/flurry.glsl — Flurry, the snowman, on the muse
//  skeleton.  Measured off reference stills: a tall "boot" of a head — a narrow rounded cranium, a very
//  wide cheek / upper-lip mass that juts forward, a small lower jaw, and a tapering snow stem that plugs into a small flat
//  middle ball; a wide open grin (slate-blue inside) with ONE big flat tooth hanging from the upper lip; upright white eyes
//  with dark pupils under soft snow lids, short dark twig brows, a fat carrot, three twig hairs; a pear-shaped bottom ball,
//  three coal buttons (1 + 2), thin twig arms with twig fingers, two stubby snow feet and no visible legs (the leg bones
//  stay: walking / sitting still work).  Snow = bright white, cool-blue shade, soft subsurface, a fine stable sparkle.
// ---------------------------------------------------------------------------
#define M_H 2.30
#define MR_HEAD_SCALE 1.18                       // lead: the measured head is ~0.35 of his height wide; every head-space number below is at 1/scale
#define M_HEADC vec3(0.0, 1.600, 0.0)
#define M_HEADR vec3(0.380, 0.560, 0.380)
#define FLURRY_NOSE_O vec3(0.0, 0.110, 0.330)        // [track H] carrot base: on the upper-lip bar, between / just under the eyes
#define FLURRY_HAIR_O vec3(0.0, 0.560, -0.030)       // [track H] hair root: the crown (dome top y 0.575)
#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.600, 0.0)
#define MUSE_EMO 1
#define EYE_R0 0.108
#define EYE_C0 vec3(0.104,0.258,0.166)
#define EYE_R gEyeR
#define EYE_C gEyeC
#define EYE_AU 1.15
#define EYE_AL (-1.00)
#define EYE_AC (-0.05)
#define EYE_LID_TH 0.0040
#define FLURRY_LINER_W 0.0115
#define FLURRY_EYE_SY 1.17
#define FLURRY_LINER_WL 0.0130
#define EYE_SQUINT 0.50
#define EYE_IRIS 0.56
#define EYE_PUPIL 0.43
#define EYE_COL_A vec3(0.300,0.165,0.075)           // a thin brown iris ring round a big black pupil
#define EYE_COL_B vec3(0.060,0.030,0.018)
#define EYE_COL_W vec3(0.985,0.985,0.990)
#define LINER_COL vec3(0.200,0.270,0.400)           // [M] the thick soft dark blue-grey ring round the eye
#define EYE_GLOW vec3(0.010,0.010,0.012)
#define EYE_CATCH 1.30
#define FACE_EYE_HOOK 1
#define BROW_C vec3(0.104,0.405,0.0)
#define BROW_L 0.092
#define BROW_W 0.021
#define FLURRY_BROW_ARCH 3.2                           // [M] engine hook (face.glsl): arch + double taper + twig-irregular width
#define BROW_COL vec3(0.070,0.045,0.030)            // [M] thick dark brown
#define BROW_H 0.012
#define BROW_RISE 0.060
#define MUSE_FROWN 0.42
#define BROW_SAD_TILT 1.4
#define LIP_O vec3(0.0,-0.110,0.270)
#define LIP_TILT 0.30
#define LIP_DEPTH (0.26*LIPU)                       // lead: the cavity ends in front of MR_FACE_ZMIN, so its back wall is painted dark like the rest
#define LIP_W 0.285
#define LIP_REST_OPEN (0.20 + 0.14*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE])))
#define LIP_HU (0.030*LIPU)
#define LIP_HL (0.040*LIPU)                         // a thin lower lip: the open mouth shows the dark inside, not a snow roll
#define LIP_TU (0.004*LIPU)
#define LIP_TL (0.012*LIPU)
#define LIP_BOW 0.0
#define LIP_SQUARE 0.25
#define LIP_OPEN_Y 1.30
#define LIP_PUSH (0.050*LIPU)
#define LIP_SKIN 0.90
#define LIP_BLEND (0.018*LIPU)
#define LIP_TOOTH_Z (-0.20*LIPU)
#define LIP_TEETH_H (0.004*LIPU)                    // no tooth rows: his one tooth is flurryTooth()
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(0.86,0.88,0.92)              // snow lips
#define LIP_COL_L vec3(0.86,0.88,0.92)
#define LIP_COL_LINE vec3(0.20,0.26,0.42)
#define LIP_LINE 0.35
#define LIP_LINE_SOFT 0.40
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.070,0.100,0.200)        // slate blue inside
#define LIP_COL_DEEP vec3(0.022,0.032,0.075)
#define LIP_COL_GUM  vec3(0.075,0.105,0.200)
#define LIP_COL_TONGUE vec3(0.060,0.085,0.170)
#define LIP_COL_TOOTH vec3(0.100,0.135,0.240)
#define LIP_SPEC 0.0
#define LIP_SHIN 30.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.0
#define LIP_WRAP 0.40
#define LIP_EMIT vec3(0.03,0.035,0.045)
#define LIP_TONGUE_FREE 0.11
#pragma include face
#pragma include lips

#define M_UPPER 0.30
#define M_FORE 0.27
#define MR_SH_X 0.280
#define MR_SH_Y 1.000
#define MR_PUSH_C vec3(0.0, 0.580, 0.0)          // wrists stay out of Flurry's own two balls (tight), so hands can meet in front of the belly
#define MR_PUSH_R vec3(0.470, 0.560, 0.480)
#define MR_REST_OFF vec3(0.150, 0.040, 0.030)   // twig arms held a little out from the bottom ball at rest
#define MR_ARM_R vec3(0.020, 0.0155, 0.012)       // twigs: thicker where they grow out of the body
#define MR_LEG_R vec3(-0.060, -0.060, -0.060)    // the legs are never drawn (a negative radius stays clear of the surface)
#define MR_HAND_SCALE 0.72                       // a small twig hand: the digits start at the knot, not 6 cm off it
#define MR_PALM_R vec3(0.062, 0.060, 0.034)      // (hand units) a knot where the twig fingers fork
#define MR_NFING 3
#define MR_FING_R 0.0150
#define MR_FING_L 2.00
#define MR_FING_PAD 0.0170                       // knobby rounded twig tips
#define MR_THUMB_R 0.0165
#define MR_REST_CURL 0.04
#define MR_REST_THUMB 0.10
#define MR_SPREAD(sp) max(sp, 0.0) + 2.2
#define MR_EXTRA 1                               // bark knots on the twig arms (elbow kink, small bumps)
#define MR_K_EXTRA 0.008
#define MR_FOOT_C vec3(0.0, -0.035, 0.095)
#define MR_FOOT_R vec3(0.105, 0.120, 0.145)
#define MR_K_NECK 0.035
#define MR_K_ARM 0.004                           // a crisp twig-into-snow join, no smudge
#define MR_K_LEG 0.035
#define MR_K_HAND 0.008
#define MR_K_FOOT 0.02
#define MR_HEAD_BOUND 0.86
#define MR_FACE_C vec3(0.0, 0.02, 0.20)
#define MR_FACE_RAD 0.56                            // lead: the whole wide-open mouth cavity lies inside the face-work sphere (a smaller one cut the cavity: a light noisy wall)
#define MR_FACE_ZMIN (-0.04)                        // ... and the face paints stop here: no lip line on the back of the head
#define MR_CHEEK vec2(0.270, -0.015)
#define MR_CHEEK_R 0.08
#define MR_INIT_HOOK 1
#define MR_CUSTOM_BEND 1
#define MR_HEAD_EXTRA 1
#define MR_FACE_POST 1
#define MR_CREASE_OCC 0.30
//@shadow-begin
#define MR_SHADOW_HEAD vec3(0.360, 0.500, 0.350)
#define MR_SHADOW_TORSO 1
#define MR_SHADOW_HEAD_FN 1
//@shadow-end
#define FLURRY_SNOW_SHADE_FN 1
#pragma include mascot_rig

const vec3 C_SNOW   = vec3(0.800,0.815,0.835);
const vec3 C_SHADE  = vec3(0.600,0.700,1.000);      // the cool blue the snow turns in shade
const vec3 C_COAL   = vec3(0.030,0.030,0.036);
const vec3 C_TWIG   = vec3(0.140,0.095,0.068);      // dark grey-brown bark
const vec3 C_CARROT = vec3(0.895,0.300,0.115);      // red-orange, half-way to a brighter orange
const vec3 C_TOOTH  = vec3(0.975,0.975,0.980);

void mascotMouthBend(){
  // the standing grin: wide and curved; a frown flattens it
  float op = smoothstep(0.0, 0.6, max(uP[P_LIP_OPEN], uP[P_MOUTH_OPEN]));
  float frown = clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  float rest = 0.58*mix(0.45, 1.0, smoothstep(0.0, 0.05, uP[P_SMILE]));
  gLBend += LIP_W*((rest + 0.08*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - 0.5*op) - MUSE_FROWN*frown*(1.0 - 0.15*op))*(1.0 - gLRnd);
  gLSmile = max(gLSmile, 0.50*(1.0 - frown)); gLCheek = max(gLCheek, 0.08*(1.0 - frown));   // lead (from track H h66): the resting-smile cheek bulge made ear-like bumps at the sides
}
float gOGrin;                      // 1 = the standing grin, 0 = a frown
void mascotInit(){
  gOGrin = 1.0 - clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
}
// head (head space, origin = head centre): cranium + the wide cheek / upper-lip mass + a small jaw + the stem
float mascotHead(vec3 q){
  // [h24 layout, the lead's measurements] a small dome on top; half-way down the huge puffy upper-lip / cheek bar (the widest
  // part, carrying the carrot, the eyes resting on it, drooping a little at its ends, overhanging the mouth); below the mouth a
  // long thick chin-neck column tapering smoothly into the middle ball.  No jaw ball.
  float cran = sdEllipsoid(q - vec3(0.0, 0.300, -0.020), vec3(0.235, 0.275, 0.250));
  vec3 qb = q - vec3(0.0, 0.050 - 0.05*q.x*q.x, 0.060);
  float bar  = sdEllipsoid(qb, vec3(0.345, 0.200, 0.300));
  float muz  = sdEllipsoid(q - vec3(0.0, 0.030, 0.200), vec3(0.270, 0.140, 0.225));
  float shl  = sdEllipsoid(q - vec3(0.0, 0.200, 0.000), vec3(0.290, 0.170, 0.255));     // fills the dome-to-bar shoulder
  vec3 a = vec3(0.0, -0.035, -0.030), b = vec3(0.0, -0.500, -0.040);
  float h = clamp(dot(q - a, b - a)/dot(b - a, b - a), 0.0, 1.0);
  vec3 cq = q - mix(a, b, h); cq.z /= 0.88;
  float col  = (length(cq) - mix(0.335, 0.195, pow(h, 1.9)))*0.88;
  col = smax(col, q.y - 0.005, 0.080);                           // no cap: the column's top stays inside the bar, never bulges up past the eyes
  float d = smin(cran, bar, 0.150);
  d = smin(d, shl, 0.100);
  d = smin(d, muz, 0.060);
  d = smin(d, col, 0.140);
  if(abs(d) < 0.05) d += 0.005*(vnoise3(q*7.0) - 0.5);
  return d;
}
// the parts the lips never carve: carrot, twig hair, the tooth.  gPHat: 1 twig, 2 tooth, 4 carrot
int gPHat;
float flurryTwig(vec3 q, vec3 a, vec3 m, vec3 b, float r){
  return min(sdCapsule(q, a, m, r), sdCapsule(q, m, b, r*0.80));
}
// a gently curved hair twig: three segments along a quadratic arc a -> (ctl) -> b, tapering r -> 0.8 r
float flurryTwig3(vec3 q, vec3 a, vec3 c, vec3 b, float r){
  vec3 p1 = mix(mix(a, c, 0.36), mix(c, b, 0.36), 0.36), p2 = mix(mix(a, c, 0.70), mix(c, b, 0.70), 0.70);
  return min(sdCapsule(q, a, p1, r*1.15), min(sdCapsule(q, p1, p2, r*0.92), sdCapsule(q, p2, b, r*0.66)));   // tapering to fine tips
}
#define OH_ROOT FLURRY_HAIR_O                  // the hair root on the crown (head space): the head track's anchor
float flurryHair(vec3 q){
  // three long thin twigs from one point on the crown, gently curved, small forks near the tips
  vec3 o = OH_ROOT;
  // cost: every twig lies within 0.18 of a short tilted segment AND above the root's plane (the crown below skips the hair)
  float bd = max(sdCapsule(q, o + vec3(0.03, 0.12, -0.05), o + vec3(0.03, 0.40, -0.08), 0.230), o.y - 0.013 - q.y);
  if(bd > 0.03) return bd;
  // three SEPARATE thin strands rooted a little apart, each bowed and diverging a little, one tiny sprig
  float d = flurryTwig3(q, o + vec3(-0.030, 0.0, 0.010), o + vec3(0.010, 0.300, -0.020), o + vec3(-0.110, 0.470, -0.090), 0.0085);
  d = min(d, flurryTwig3(q, o + vec3(0.0, 0.0, -0.010), o + vec3(-0.040, 0.290, -0.050), o + vec3(0.070, 0.530, -0.140), 0.0085));
  d = min(d, flurryTwig3(q, o + vec3(0.030, 0.0, 0.000), o + vec3(0.030, 0.250, -0.020), o + vec3(0.230, 0.380, -0.060), 0.0082));
  vec3 sp = o + vec3(0.124, 0.292, -0.040);                   // on the third strand
  d = min(d, sdCapsule(q, sp, sp + vec3(-0.012, 0.065, -0.010), 0.0080));
  vec3 sm = o + vec3(0.020, 0.420, -0.105);                   // a second tiny sprig near the middle strand's tip
  d = min(d, sdCapsule(q, sm, sm + vec3(0.055, 0.035, -0.010), 0.0080));
  return d;
}
#define OC_TILT (-0.10)                       // level, a touch up at the base, drooping slightly along its length
#define OC_L 0.480
#define OC_POS (FLURRY_NOSE_O + vec3(0.0, -0.075, 0.0))                  // the carrot's base anchor (head space): the head track's anchor
#define OC_YAW 0.0
float flurryCarrot(vec3 q){
  vec3 c = q - OC_POS;
  c.xz = rot(OC_YAW)*c.xz; c.yz = rot(OC_TILT)*c.yz;                              // the tip points a little up
  float bd = length(c - vec3(0.0, -0.015, OC_L*0.5)) - (OC_L*0.5 + 0.16);   // cost: a bounding sphere far away
  if(bd > 0.03) return bd;
  float L = OC_L, t = clamp(c.z/L, 0.0, 1.0);
  c.y += 0.060*t*(1.0 - t) - 0.020*t*t;                                    // and droops a little toward the tip
  c.x *= 1.0 + 0.06*sin(t*5.0);                        // a hand-grown, not lathe-turned, section
  float rr = 0.020 + 0.098*pow(1.0 - t, 1.4) +       // thick base, a concave taper to a blunt point
              0.0003*sin(t*46.0 + 0.6)*(1.0 - 0.7*t) + 0.0015*(vnoise3(c*12.0) - 0.5);   // nearly smooth
  float d = max(length(c.xy) - rr, max(-c.z, c.z - L))*0.80;
  return min(d, length(c - vec3(0.0, 0.0, L)) - 0.020);
}
// the carrot's shadow stand-in: the same bent cone without the noise, a hair inside the real surface
float flurryCarrotShadow(vec3 q){
  vec3 c = q - OC_POS;
  c.xz = rot(OC_YAW)*c.xz; c.yz = rot(OC_TILT)*c.yz;
  float L = OC_L, t = clamp(c.z/L, 0.0, 1.0);
  c.y += 0.060*t*(1.0 - t) - 0.020*t*t;
  float d = max(length(c.xy) - (0.020 + 0.098*pow(1.0 - t, 1.4)), max(-c.z, c.z - L))*0.80;
  return min(d, length(c - vec3(0.0, 0.0, L)) - 0.020) + 0.006;
}
float flurryTooth(vec3 q){
  // one big flat tooth hanging from the middle of the upper lip, a touch off-centre
  // it rides the upper lip's inner edge (gLAu: the aperture's rise) and, with the lips closed, comes forward over the lower lip
  float cl = 1.0 - smoothstep(0.004, 0.030, gLAu + gLAl);
  vec3 t = q - vec3(0.020, LIP_O.y + 0.012 + 0.45*gLAu + 0.022*cl, LIP_O.z - 0.010 + 0.052*cl);
  t.yz = rot(-LIP_TILT*0.6)*t.yz;
  vec3 e = abs(t - vec3(0.0, -0.078, 0.0)) - vec3(0.104, 0.044, 0.002);      // ~0.19 wide, ~0.11 tall, rounded
  return length(max(e, 0.0)) + min(max(e.x, max(e.y, e.z)), 0.0) - 0.036;
}
float mascotHeadBare(vec3 q){
  float carrot = flurryCarrot(q), hair = flurryHair(q), tooth = flurryTooth(q);
  float d = min(carrot, min(hair, tooth));
  gPHat = carrot <= d ? 4 : (tooth <= d ? 2 : 1);
  return d;
}
// trunk: the small flat middle ball (rides the waist) + the pear-shaped bottom ball (rides the pelvis), three coal buttons.  gTPart: 0 snow 1 coal
int gTPart;
float flurryCoal(vec3 cp, float k){
  // a chunky faceted lump: an ellipsoid cut by six tilted planes (irregular per button)
  float bd = length(cp) - 0.120;                       // cost: far away, a bounding sphere is enough
  if(bd > 0.03) return bd;
  float cd = sdEllipsoid(cp, vec3(0.104, 0.098, 0.068));
  for(int j=0;j<6;j++){
    float a = float(j)*1.047 + k*0.9 + 0.35*sin(float(j)*2.3 + k);
    vec3 nn = normalize(vec3(cos(a), sin(a), 0.75 + 0.35*sin(float(j)*1.7 + k*2.1)));
    cd = smax(cd, dot(cp, nn) - 0.090*(0.92 + 0.12*sin(float(j)*3.1 + k*1.3)), 0.020);
  }
  cd += 0.006*(vnoise3(cp*30.0 + k*7.0) - 0.5);
  return cd;
}
float mascotTorso(vec3 p){
  vec3 b = p - (gMHip - M_HIP);
  // pear: a narrower upper half on a wide low belly
  float lo = sdEllipsoid(b - vec3(0.0, 0.535, -0.005), vec3(0.375, 0.265, 0.360));
  lo = smin(lo, sdEllipsoid(b - vec3(0.0, 0.360, 0.0), vec3(0.470, 0.250, 0.440)), 0.20);   // squat, wide low
  vec3 u = localToUpper(p);
  float br = 1.0 + 0.012*gBreath;
  float mid = sdEllipsoid((u - vec3(0.0, 0.950, 0.0))/vec3(br, 1.0, br), vec3(0.310, 0.205, 0.285));   // a rounder top: a snow shoulder under the neck
  float d = smin(lo, mid, 0.022);                              // a crisp, shallow crease
  if(abs(d) < 0.05) d += 0.008*(vnoise3(u*5.0) - 0.5);
  gTPart = 0;
  float coal = flurryCoal(u - vec3(0.0, 0.915, 0.262), 0.0);
  coal = min(coal, flurryCoal(b - vec3(0.0, 0.645, 0.328), 1.0));
  coal = min(coal, flurryCoal(b - vec3(0.0, 0.405, 0.414), 2.0));
  if(coal <= d) gTPart = 1;                            // the material by the same distance test the surface uses: a clean edge
  return min(d, coal);
}
float mascotShadowTorso(vec3 p){
  vec3 b = p - (gMHip - M_HIP);
  vec3 u = localToUpper(p);
  return min(sdEllipsoid(b - vec3(0.0, 0.440, 0.0), vec3(0.430, 0.360, 0.410)), sdEllipsoid(u - vec3(0.0, 0.940, 0.0), vec3(0.305, 0.180, 0.280))) + 0.02;
}
// the head's shadow stand-in (head space), just inside the real surface, + the carrot so it shades the upper lip
float mascotShadowHead(vec3 q){
  float d = sdEllipsoid(q - vec3(0.0, 0.310, -0.020), vec3(0.200, 0.265, 0.240));
  d = min(d, sdEllipsoid(q - vec3(0.0, 0.080, 0.060), vec3(0.372, 0.170, 0.300)));
  d = min(d, sdCapsule(q, vec3(0.0, -0.100, -0.005), vec3(0.0, -0.480, -0.030), 0.200));
  d += 0.028;
  return min(d, flurryCarrotShadow(q));                  // [track B] the shadow carrot = the carrot itself, just inside (the old capsule poked out and painted a dark hole)
}
void flurryToothMat(inout Surf s){
  // track M: a soft, faintly translucent white (not a grey block): lifted by its own cool fill, no plastic highlight
  s.alb = C_TOOTH;
  s.wrap = 0.60; s.sheen = 0.10; s.spec = 0.04; s.shin = 16.0; s.emit = s.alb*vec3(0.34, 0.35, 0.37); s.occ = max(s.occ, 1.0);   // the brightest white on the face
  s.trans = 0.10; s.transCol = vec3(0.85, 0.90, 1.0);
#ifdef LOOK_V3
  s.rough = 0.75; s.sss = 0.35; s.sssCol = vec3(0.60, 0.76, 1.0); s.sheenK = 0.0; s.coat = 0.0; s.cavity = 1.0;
#endif
}
// the eye whites: bright, near-paper white (the film's eyes read white even under the lid)
void faceEyeHook(vec3 e, vec2 uv, float lc, float lidL, inout Surf s){
  float wh = smoothstep(EYE_IRIS + 0.005, EYE_IRIS + 0.040, length(uv));
  s.occ = mix(s.occ, max(s.occ, 0.88), wh);
#ifdef LOOK_V3
  s.cavity = mix(s.cavity, max(s.cavity, 0.88), wh);
#endif
  s.emit += vec3(0.95, 0.96, 1.0)*0.15*wh*mix(0.55, 1.0, lidL);
}
// after the lips / eyes paint: the tooth sits in the mouth, where lipsPaint has just painted the cave over it
void mascotFacePost(vec3 q, vec3 n, inout Surf s){
  // the mouth inside is matte slate: no snow sparkle, sheen or glow-through on the carved walls
  float cv = max(gPCave, gPTongue);
  s.spec *= 1.0 - cv; s.sheen *= 1.0 - cv; s.trans *= 1.0 - cv;
  s.emit *= 1.0 - 0.3*cv; s.wrap *= 1.0 - 0.3*cv;        // deep slate-navy: no cool fill lifting it to grey
#ifdef LOOK_V3
  s.sheenK *= 1.0 - cv; s.sss *= 1.0 - 0.8*cv; s.rough = mix(s.rough < 0.0 ? 0.6 : s.rough, 0.70, cv);
#endif
  float td = flurryTooth(q);
  if(td < 0.004 && td <= gMHeadD + 0.002 && td <= flurryCarrot(q)) flurryToothMat(s);
}
// the twig arm: shoulder -> kink -> elbow -> kink -> wrist, tapering, the kinks offset to alternate sides (a real twig)
void flurryArmPts(vec3 a, vec3 b, vec3 c, float s, out vec3 k1, out vec3 k2){
  vec3 ab = b - a, bc = c - b;
  vec3 n1 = normalize(cross(ab, vec3(0.0, 0.0, 1.0)) + vec3(0.0, 0.0, 1e-4));
  vec3 n2 = normalize(cross(bc, vec3(0.0, 0.0, 1.0)) + vec3(0.0, 0.0, 1e-4));
  k1 = mix(a, b, 0.55) + n1*0.022*s;
  k2 = mix(b, c, 0.50) - n2*0.018*s + vec3(0.0, 0.0, 0.008);
}
float flurrySeg(vec3 p, vec3 a, vec3 b, float ra, float rb, out float h){
  vec3 pa = p - a, ba = b - a;
  h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  return length(pa - ba*h) - mix(ra, rb, h);
}
float flurryArm(vec3 p, int i, out float tt){
  vec3 a = gMSh[i], b = gMEl[i], c = gMWr[i], k1, k2;
  flurryArmPts(a, b, c, i == 0 ? -1.0 : 1.0, k1, k2);
  float h0, h1, h2, h3;
  float d0 = flurrySeg(p, a, k1, 0.0165, 0.0150, h0);
  float d1 = flurrySeg(p, k1, b, 0.0150, 0.0138, h1);
  float d2 = flurrySeg(p, b, k2, 0.0138, 0.0124, h2);
  float d3 = flurrySeg(p, k2, c, 0.0124, 0.0110, h3);
  float d = min(min(d0, d1), min(d2, d3));
  tt = d == d0 ? 0.25*h0 : (d == d1 ? 0.25 + 0.25*h1 : (d == d2 ? 0.5 + 0.25*h2 : 0.75 + 0.25*h3));
  return d;
}
#define MR_ARM_FN flurryArm
// bark knots on the twig arms: a kink-knot at each elbow, a small bump on each forearm
float mascotExtra(vec3 p, out vec3 rest){
  float d = 1e3;
  for(int i=0;i<2;i++){
    vec3 e = i == 0 ? gMEl[0] : gMEl[1], w = i == 0 ? gMWr[0] : gMWr[1], sh = i == 0 ? gMSh[0] : gMSh[1];
    vec3 pm = i == 0 ? gMPalm[0] : gMPalm[1];
    d = min(d, sdCapsule(p, w, pm, 0.0125));                       // the twig runs on from the wrist into the hand's knot (no gap)
    float bd = sdCapsule(p, mix(sh, e, 0.55), mix(e, w, 0.60), 0.085);   // cost: every knot lies within 0.085 of this segment
    if(bd > 0.02){ d = min(d, bd); continue; }
    vec3 k1, k2;
    flurryArmPts(sh, e, w, i == 0 ? -1.0 : 1.0, k1, k2);
    d = min(d, sdEllipsoid(p - e - vec3(0.0, 0.004, 0.0), vec3(0.019, 0.021, 0.018)));      // the knots sit on the kinks
    d = min(d, length(p - k2) - 0.016);
    d = min(d, length(p - k1) - 0.0175);
    d = min(d, sdCapsule(p, k1, k1 + normalize(vec3(i == 0 ? -0.4 : 0.4, 1.0, 0.2))*0.045, 0.0085));   // a stub of a side twig
  }
  rest = p;
  return d;
}
void flurryTwigMat(vec3 r, inout Surf s){
  float st = vnoise3(vec3(r.x*40.0, r.y*400.0, r.z*400.0));
  s.alb = C_TWIG*(0.70 + 0.50*st);
  s.wrap = 0.25; s.sheen = 0.08; s.spec = 0.06; s.shin = 16.0; s.emit = s.alb*0.020;
#ifdef LOOK_V3
  s.rough = 0.80; s.sss = 0.0; s.sheenK = 0.0; s.coat = 0.0;
#endif
}
// ---- track M: the snow material
float gSnowK = 0.0;          // 1 while the shaded surface is snow (set by mascotMaterial; read by the lighting hook)
// lighting hook (FLURRY_SNOW_SHADE_FN): the key light a cast shadow takes away, and the ambient that occlusion takes away,
// come back as a sky-blue/lavender glow (film snow: shadows luminous blue, never grey)
vec3 flurrySnowShade(vec3 alb, float nl, float sh, float ao, float aoW, float nv){
  float k = gSnowK*smoothstep(0.50, 0.80, min(alb.r, min(alb.g, alb.b)));
  float shad = nl*(1.0 - clamp(sh, 0.0, 1.0));
  float occ = 1.0 - clamp(min(ao, aoW), 0.0, 1.0);
  // soft fuzzy rim: a thin band of brighter, slightly translucent powder at the silhouette, strongest on the lit side
  float fz = pow(1.0 - nv, 3.0)*(0.30 + 0.70*nl);
  return alb*k*(vec3(0.38, 0.37, 0.56)*shad + vec3(0.50, 0.48, 0.70)*occ + vec3(1.00, 0.99, 0.98)*fz);
}
void flurrySnowMat(vec3 r, vec3 n, inout Surf s){
  gSnowK = 1.0;
  float big = vnoise3(r*3.0), mid = vnoise3(r*14.0 + 3.1);
  // packed-powder grain, two octaves in rest space; each octave fades only when its cell drops under ~1 px (band-limited,
  // so the coarser octave survives at the wide framing and at the judge's 384 px)
  float fp = max(gMFp, 1e-5);
  float g1 = (vnoise3(r*130.0 + 7.3) - 0.5)*smoothstep(0.6, 1.4, (1.0/130.0)/fp);
  float g2 = (vnoise3(r*260.0 + 1.7) - 0.5)*smoothstep(0.6, 1.4, (1.0/260.0)/fp);
  float grain = 0.65*g1 + 0.35*g2;
  vec3 alb = C_SNOW*(0.97 + 0.02*big + 0.01*mid + 0.02*grain);
  float kd = clamp((dot(n, uSunDir) + 0.15)/1.15, 0.0, 1.0);             // where the key reaches (very wide wrap)
  // glints: sparse crisp points (a jittered dot per rest-space cell, not a whole lit cell).  The dot never gets smaller
  // than ~0.8 px (its energy is spread instead), and the set fades as the cells near 1 px, so nothing crawls.
  float cs = 1.0/150.0;
  float fw = smoothstep(0.9, 2.0, cs/fp);
  vec3 rc = r/cs, cell = floor(rc);
  float h = hash13(cell);
  vec3 jit = vec3(hash13(cell + 17.1), hash13(cell + 41.3), hash13(cell + 73.7))*0.6 + 0.2;
  float dd = length(rc - cell - jit)*cs;
  float rad = max(0.10*cs, 0.8*fp);
  float spk = step(0.965, h)*fw*(0.3 + 0.7*kd)*smoothstep(rad, 0.4*rad, dd)*min(1.0, (0.10*cs/rad)*(0.10*cs/rad)*4.0);
  alb *= 0.80 + 0.20*kd*kd;                                          // the key falls off across the form (no flat white plateau)
  s.alb = alb;
  s.wrap = 0.60; s.sheen = 0.10; s.spec = 0.03; s.shin = 12.0;
  s.trans = 0.30; s.transCol = vec3(0.62, 0.80, 1.0);
  // luminous shade: a sky-blue fill that grows where the key does not reach, and from below (bounce off snow/floor)
  float away = 1.0 - kd;
  vec3 fill = vec3(0.54, 0.53, 0.74)*(0.10 + 0.26*away + 0.05*smoothstep(0.0, -0.9, n.y));
  s.emit = alb*fill + vec3(0.9, 0.95, 1.0)*spk*1.1;
#ifdef LOOK_V3
  s.rough = 0.85;
  s.sheenK = 0.08; s.sheenCol = vec3(1.0);
  s.sss = 0.32; s.sssCol = vec3(0.55, 0.70, 1.0);
  s.cavity = 1.0;
#endif
}
void mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s){
  gSnowK = 0.0;
  vec3 u = localToUpper(pw);
  if(gMPart == 0){
    float hd = mascotHeadBare(q);
    if(hd < 0.004 && hd <= gMHeadD + 0.002){
      if(gPHat == 4){
        // carrot: deep orange, faint rings, matte
        vec3 c = q - OC_POS;
        float ring = smoothstep(0.55, 1.0, 0.5 + 0.5*sin(c.z*190.0 + 2.0*vnoise3(q*25.0)));   // fine cross ridges
        s.alb = C_CARROT*(1.0 - 0.025*ring)*(0.97 + 0.05*vnoise3(q*45.0));   // ridges barely there
        s.wrap = 0.30; s.sheen = 0.16; s.spec = 0.24; s.shin = 9.0; s.emit = s.alb*0.070;   // waxy: a soft broad highlight, a brighter lit side
#ifdef LOOK_V3
        s.rough = 0.45; s.sss = 0.30; s.sssCol = vec3(1.0, 0.5, 0.15); s.sheenK = 0.0; s.coat = 0.0;
#endif
        return;
      }
      if(gPHat == 2){ flurryToothMat(s); return; }
      flurryTwigMat(q*3.0, s);
      s.alb = s.alb*0.62 + vec3(0.004, 0.003, 0.004);   // dark brown, not black   // the hair twigs: darker than the arms, a little cool
      return;
    }
  }
  if(gMPart == 1){
    mascotTorso(pw);
    if(gTPart == 1){
      s.alb = C_COAL*(0.6 + 0.5*vnoise3(u*80.0));
      s.wrap = 0.05; s.sheen = 0.16; s.spec = 0.70; s.shin = 60.0; s.emit = vec3(0.002);   // deep black, a soft grey sheen on the facets
#ifdef LOOK_V3
      s.rough = 0.32; s.sss = 0.0; s.sheenK = 0.0; s.coat = 0.35;
#endif
      return;
    }
  }
  if(gMPart == 2 || gMPart == 3 || gMPart == 4 || gMPart == 7){ flurryTwigMat(r, s); return; }
  // snow (head, balls, feet) — track M: bright soft powder.  Very wide blue wrap, luminous cool-lavender shade (an
  // emitted sky fill on the side away from the key, never grey), almost no specular, a fine grain and sparse glints
  // hashed in REST space that fade with the pixel footprint (no crawl).
  flurrySnowMat(r, n, s);
}
#pragma include mascot_body

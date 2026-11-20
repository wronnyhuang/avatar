// ---------------------------------------------------------------------------
// animals/doughlad.glsl — the little boy made of fresh dough, in the muse's classroom.
//  A soft pale dough body: a round head, a plump pear trunk, stubby arms and legs with mitten hands; a tall pleated
//  chef's hat with a blue badge on its band, a white neckerchief knotted at the throat; small blue eyes, thin brows,
//  a dimpled smile.  Dough = a soft matte skin that glows a little where the light passes through it.
//  Skeleton / hand map / grasp = the muse's (engine/mascot_rig.glsl).
// ---------------------------------------------------------------------------
#define M_H 2.30
#define M_HEADC vec3(0.0, 1.56, 0.03)
#define M_HEADR vec3(0.56, 0.52, 0.52)
#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.56, 0.03)
#define MUSE_EMO 1
#define EYE_R0 0.052
#define EYE_C0 vec3(0.165,0.090,0.455)
#define EYE_R gEyeR
#define EYE_C gEyeC
#define EYE_AU 0.95
#define EYE_AL (-0.95)
#define EYE_AC (-0.05)
#define EYE_LID_TH 0.0075
#define EYE_SQUINT 0.50
#define EYE_IRIS 0.97
#define EYE_PUPIL 0.30
#define EYE_COL_A vec3(0.070,0.200,0.850)           // bright blue eyes
#define EYE_COL_B vec3(0.020,0.070,0.480)
#define EYE_COL_W vec3(0.030,0.080,0.480)
#define LINER_COL vec3(0.56,0.47,0.36)
#define EYE_GLOW vec3(0.010,0.025,0.110)
#define EYE_CATCH 0.930
#define BROW_C vec3(0.172,0.215,0.0)
#define BROW_L 0.070
#define BROW_W 0.008
#define BROW_COL vec3(0.66,0.56,0.42)
#define BROW_H 0.010
#define BROW_RISE 0.060
#define MUSE_FROWN 0.42
#define BROW_SAD_TILT 1.4
#define LIP_O vec3(0.0,-0.150,0.492)
#define LIP_TILT 0.10
#define LIP_W 0.130
#define LIP_HU (0.030*LIPU)
#define LIP_HL (0.040*LIPU)
#define LIP_TU (0.012*LIPU)
#define LIP_TL (0.016*LIPU)
#define LIP_BOW 0.0
#define LIP_SQUARE 0.30
#define LIP_OPEN_Y 1.40
#define LIP_PUSH (0.070*LIPU)
#define LIP_SKIN 0.90
#define LIP_BLEND (0.014*LIPU)
#define LIP_TOOTH_Z (-0.20*LIPU)
#define LIP_TEETH_H ((0.008 + 0.050*gEmTeeth)*LIPU)
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(0.78,0.62,0.44)              // dough lips: the skin, a shade warmer
#define LIP_COL_L vec3(0.80,0.64,0.46)
#define LIP_COL_LINE vec3(0.40,0.25,0.15)
#define LIP_LINE 2.2
#define LIP_LINE_SOFT 0.40
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.40,0.20,0.12)
#define LIP_COL_DEEP vec3(0.12,0.05,0.03)
#define LIP_COL_GUM  vec3(0.60,0.36,0.24)
#define LIP_COL_TONGUE vec3(0.88,0.50,0.42)
#define LIP_COL_TOOTH vec3(0.60,0.36,0.24)
#define LIP_SPEC 0.10
#define LIP_SHIN 30.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.05
#define LIP_WRAP 0.40
#define LIP_EMIT vec3(0.06,0.04,0.03)
#define LIP_TONGUE_FREE 0.11
#pragma include face
#pragma include lips

#define MR_ARM_R vec3(0.125, 0.112, 0.100)
#define MR_LEG_R vec3(0.165, 0.150, 0.135)
#define MR_PALM_R vec3(0.108, 0.100, 0.072)
#define MR_FING_R 0.034
#define MR_FING_L 0.80
#define MR_THUMB_R 0.040
#define MR_FOOT_C vec3(0.0, -0.045, 0.060)
#define MR_FOOT_R vec3(0.150, 0.105, 0.195)
#define MR_K_NECK 0.14
#define MR_K_ARM 0.10
#define MR_K_LEG 0.13
#define MR_K_HAND 0.07
#define MR_K_FOOT 0.09
#define MR_HEAD_BOUND 1.22
#define MR_FACE_C vec3(0.0, -0.03, 0.40)
#define MR_FACE_RAD 0.42
#define MR_CHEEK vec2(0.300, -0.090)
#define MR_PUSH_R vec3(0.60, 0.56, 0.54)
#define MR_INIT_HOOK 1
#define MR_CUSTOM_BEND 1
//@shadow-begin
#define MR_SHADOW_HEAD vec3(0.54, 0.66, 0.50)
//@shadow-end
#pragma include mascot_rig

const vec3 C_DOUGH = vec3(0.925,0.860,0.715);       // fresh dough: a warm cream
const vec3 C_CLOTH = vec3(0.900,0.900,0.910);       // hat and neckerchief: a cooler white
const vec3 C_BADGE = vec3(0.030,0.150,0.600);

void mascotMouthBend(){
  // a wide happy smile at rest; a knit brow with no smile turns it down
  float op = smoothstep(0.0, 0.6, max(uP[P_LIP_OPEN], uP[P_MOUTH_OPEN]));
  float frown = clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  float rest = 0.24*mix(0.30, 1.0, smoothstep(0.0, 0.05, uP[P_SMILE]));
  gLBend += LIP_W*((rest + 0.08*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - 0.6*op) - MUSE_FROWN*frown*(1.0 - 0.15*op))*(1.0 - gLRnd);
}
mat3 gHat; float gHatSq;
void mascotInit(){
  // the hat's puffy crown leans and squashes with the ear dials (it flops when the head moves fast)
  float a = 0.5*(clamp(uP[P_EAR_R], -1.0, 1.0) - clamp(uP[P_EAR_L], -1.0, 1.0));
  float f = 0.5*(clamp(uP[P_EAR_L], -1.0, 1.0) + clamp(uP[P_EAR_R], -1.0, 1.0));
  float fl = clamp(uP[P_EAR_FLICK_L], 0.0, 1.0) - clamp(uP[P_EAR_FLICK_R], 0.0, 1.0);
  gHat = rotZ(0.22*a + 0.16*fl)*rotX(-0.10*f);
  gHatSq = 1.0 + 0.10*f;
}
int gDPart;            // head sub-part: 0 dough, 1 hat band, 2 hat crown
float gHatU;           // angle round the hat, for the pleats
float mascotHead(vec3 q){
  // the head: a soft ball, a touch fuller at the cheeks
  float d = sdEllipsoid(q, vec3(0.540, 0.500, 0.500));
  d -= 0.020*smoothstep(0.15, -0.40, q.y);
  gDPart = 0;
  // the hat: a band round the crown, a tall pleated puff above it that leans a little to one side
  vec3 h = q - vec3(0.0, 0.330, -0.020);
  h.xy = rot(-0.06)*h.xy;
  float band = max(length(h.xz) - (0.385 + 0.035*h.y), abs(h.y - 0.115) - 0.125) - 0.022;
  vec3 c = transpose(gHat)*(h - vec3(0.0, 0.24, 0.0)) + vec3(0.0, 0.24, 0.0);
  c.y = (c.y - 0.24)/gHatSq + 0.24;
  gHatU = atan(c.x, c.z);
  float pl = 0.5 + 0.5*cos(gHatU*13.0 + 1.5*sin(c.y*7.0));
  float lean = 0.10*smoothstep(0.25, 0.85, c.y);
  float crown = sdEllipsoid(c - vec3(lean, 0.560, 0.0), vec3(0.500, 0.340, 0.470)) - 0.030*pl*smoothstep(0.26, 0.45, c.y);
  crown = smin(crown, sdEllipsoid(c - vec3(0.5*lean, 0.330, 0.0), vec3(0.400, 0.150, 0.385)), 0.10);
  float hat = smin(band, crown, 0.035);
  if(hat < d){ gDPart = band < crown ? 1 : 2; }
  return smin(d, hat, 0.018);
}
int gTPart;            // trunk sub-part: 0 dough, 1 neckerchief
float mascotTorso(vec3 p){
  vec3 b = p - (gMHip - M_HIP);
  float lo = sdEllipsoid(b - vec3(0.0, 0.650, 0.000), vec3(0.535, 0.380, 0.465));          // a plump tummy
  vec3 u = localToUpper(p);
  float br = 1.0 + 0.014*gBreath;
  float hi = sdEllipsoid((u - vec3(0.0, 0.930, 0.0))/vec3(br, 1.0, br), vec3(0.465, 0.330, 0.405));
  float d = smin(lo, hi, 0.18);
  d = smin(d, sdEllipsoid(u - vec3(0.0, 1.090, -0.010), vec3(0.330, 0.130, 0.270)), 0.12);
  gTPart = 0;
  // the neckerchief: a rolled band round the neck, a knot in front, two pointed tails hanging on the chest
  vec3 k = u - vec3(0.0, 1.130, 0.0);
  float roll = length(vec2(length(k.xz*vec2(1.0, 1.08)) - 0.335, k.y + 0.05*smoothstep(0.0, 0.35, k.z))) - 0.062;
  float knot = sdEllipsoid(u - vec3(0.0, 1.050, 0.392), vec3(0.085, 0.070, 0.058));
  float tails = 1e3;
  for(int i=0;i<2;i++){
    float sg = i == 0 ? -1.0 : 1.0;
    // a flat pointed tongue of cloth hanging from the knot, fanned out a little
    vec3 t = u - vec3(sg*0.105, 0.860, 0.430 - 0.012*gBreath);
    t.xy = rot(sg*0.46)*t.xy;
    tails = min(tails, sdEllipsoid(t, vec3(0.085, 0.225, 0.026)));
  }
  float ker = smin(smin(roll, knot, 0.03), tails, 0.025);
  if(ker < d) gTPart = 1;
  return smin(d, ker, 0.012);
}
void mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s){
  vec3 u = localToUpper(pw);
  bool cloth = false; float badge = 0.0, band = 0.0;
  if(gMPart == 0){ mascotHead(q); cloth = gDPart > 0; band = float(gDPart == 1); }
  if(gMPart == 1){ mascotTorso(pw); cloth = gTPart == 1; }
  vec3 alb;
  if(cloth){
    // starched cotton: cool white, a fine weave, soft shading in the pleats
    alb = C_CLOTH;
    vec3 w = gMPart == 0 ? q : u;
    float wv = smoothstep(1.4, 3.0, (1.0/170.0)/max(gMFp, 1e-5));
    float g = sin(w.x*520.0)*sin(w.y*520.0 + w.z*310.0);
    alb *= 1.0 + 0.030*wv*g;
    if(gDPart == 2 && gMPart == 0){
      float pl = 0.5 + 0.5*cos(gHatU*13.0 + 1.5*sin((q.y - 0.33)*7.0));
      s.occ *= 0.66 + 0.34*pl;
      alb *= 0.93 + 0.07*pl;
    }
    if(band > 0.5){
      // the badge on the band: a blue disc with a white ring and a row of white dots (no lettering)
      vec2 bc = vec2(q.x, q.y - 0.445);
      float rr = length(bc);
      badge = smoothstep(0.090, 0.080, rr)*smoothstep(0.30, 0.42, q.z);
      vec3 bcol = C_BADGE;
      bcol = mix(bcol, vec3(0.92), smoothstep(0.006, 0.0, abs(rr - 0.066) - 0.005));
      float da = atan(bc.y, bc.x);
      bcol = mix(bcol, vec3(0.92), smoothstep(0.011, 0.006, length(vec2(rr - 0.040, 0.040*(fract(da*8.0/6.2832) - 0.5)*0.785))));
      alb = mix(alb, bcol, badge);
    }
    s.alb = alb;
    s.wrap = 0.35; s.sheen = 0.22; s.spec = mix(0.05, 0.30, badge); s.shin = mix(14.0, 60.0, badge);
    s.emit = alb*0.055;
#ifdef LOOK_V3
    s.rough = mix(0.78, 0.35, badge); s.sheenK = 0.35*(1.0 - badge); s.sheenCol = vec3(1.0);
    s.sss = 0.22*(1.0 - badge); s.sssCol = vec3(1.0, 0.92, 0.82); s.coat = 0.35*badge;
#endif
    return;
  }
  // dough: smooth, matte, soft — light sinks in and comes back warm; a faint floury mottling, tiny pores up close
  float big = vnoise3(r*5.0), mid = vnoise3(r*22.0 + 3.1);
  alb = C_DOUGH*(0.96 + 0.07*big + 0.03*mid);
  if(gMPart == 0){
    // warm cheeks (stronger with a smile)
    vec2 ck = vec2(abs(q.x), q.y) - MR_CHEEK;
    float bl = exp(-dot(ck, ck)/0.012)*smoothstep(0.05, 0.30, q.z);
    alb = mix(alb, vec3(0.930, 0.620, 0.540), (0.26 + 0.14*gLSmile)*bl);
  }
  float wg = smoothstep(1.5, 3.2, (1.0/140.0)/max(gMFp, 1e-5));
  if(wg > 0.0){
    float g = vnoise3(r*140.0) - 0.5;
    vec3 bt = normalize(cross(vec3(0.0, 1.0, 0.0), n) + vec3(1e-4));
    s.n = normalize(n + (bt*g + cross(n, bt)*(vnoise3(r*140.0 + 9.1) - 0.5))*0.035*wg);
  }
  s.alb = alb;
  s.wrap = 0.36; s.sheen = 0.18; s.spec = 0.07; s.shin = 18.0;
  s.trans = 0.18*float(gMPart == 3 || gMPart == 2); s.transCol = vec3(1.0, 0.72, 0.48);
  s.emit = alb*vec3(0.036, 0.032, 0.026);
#ifdef LOOK_V3
  s.rough = 0.62;
  s.sheenK = 0.20; s.sheenCol = vec3(1.0, 0.95, 0.86);
  s.sss = 0.46; s.sssCol = vec3(1.0, 0.76, 0.56);
#endif
}
#pragma include mascot_body

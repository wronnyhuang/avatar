// ---------------------------------------------------------------------------
// animals/hooty.glsl — the round bright-green language-app owl in the muse's classroom.
//  One soft body: a wide flat-topped head with two feather tufts on its corners, huge white eyes in a lighter mask,
//  a small golden beak that talks, stubby wings with three feather fingers, three pale chest feathers, orange feet.
//  Soft matte vinyl-felt with a faint grain.  Skeleton / hand map / grasp = the muse's (engine/mascot_rig.glsl).
// ---------------------------------------------------------------------------
#define M_H 2.30
#define M_HEADC vec3(0.0, 1.58, 0.03)
#define M_HEADR vec3(0.82, 0.62, 0.62)
#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.58, 0.03)
#define MUSE_EMO 1
#define EYE_R0 0.300
#define EYE_C0 vec3(0.288,0.085,0.335)
#define EYE_R gEyeR
#define EYE_C gEyeC
#define EYE_AU 1.05
#define EYE_AL (-1.05)
#define EYE_AC (-0.05)
#define EYE_LID_TH 0.012
#define EYE_SQUINT 0.55
#define EYE_IRIS 0.430
#define EYE_PUPIL 0.418
#define EYE_COL_A vec3(0.085,0.085,0.090)
#define EYE_COL_B vec3(0.060,0.060,0.065)
#define EYE_COL_W vec3(0.930,0.930,0.920)
#define LINER_COL vec3(0.130,0.420,0.010)
#define EYE_CATCH 0.972
#define EYE_GLOSS 0.20
#define BROW_C vec3(0.310,0.470,0.0)
#define BROW_L 0.120
#define BROW_W 0.022
#define BROW_COL vec3(0.300,0.760,0.020)
#define BROW_H 0.014
#define BROW_RISE 0.060
#define MUSE_FROWN 0.42
#define BROW_SAD_TILT 1.4
#define LIP_O vec3(0.0,-0.160,0.690)
#define LIP_TILT 0.05
#define LIP_W 0.120
#define LIP_HU (0.090*LIPU)
#define LIP_HL (0.075*LIPU)
#define LIP_TU (0.030*LIPU)
#define LIP_TL (0.024*LIPU)
#define LIP_BOW 0.0
#define LIP_SQUARE 0.35
#define LIP_OPEN_Y 1.30
#define LIP_PUSH (0.050*LIPU)
#define LIP_SKIN 0.0
#define LIP_BLEND (0.016*LIPU)
#define LIP_TOOTH_Z (-0.30*LIPU)
#define LIP_TEETH_H (0.004*LIPU)
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(1.00,0.700,0.015)             // the beak: yellow above, deeper orange below
#define LIP_COL_L vec3(0.920,0.340,0.004)
#define LIP_COL_LINE vec3(0.420,0.130,0.010)
#define LIP_LINE 1.6
#define LIP_LINE_SOFT 0.40
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.42,0.08,0.06)
#define LIP_COL_DEEP vec3(0.09,0.016,0.016)
#define LIP_COL_GUM  vec3(0.60,0.16,0.12)
#define LIP_COL_TONGUE vec3(0.95,0.42,0.40)
#define LIP_COL_TOOTH vec3(0.60,0.16,0.12)
#define LIP_SPEC 0.30
#define LIP_SHIN 70.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.05
#define LIP_WRAP 0.30
#define LIP_EMIT vec3(0.05,0.02,0.0)
#define LIP_TONGUE_FREE 0.11
#pragma include face
#pragma include lips

#define M_EAR_PIV vec3(0.470, 0.400, -0.02)
#define MR_ARM_R vec3(0.125, 0.115, 0.105)
#define MR_LEG_R vec3(0.070, 0.058, 0.055)
#define MR_PALM_R vec3(0.165, 0.130, 0.050)
#define MR_CREASE_OCC 0.15
#define MR_NFING 3
#define MR_FING_R 0.045
#define MR_FING_L 1.05
#define MR_THUMB_R 0.040
#define MR_FOOT_C vec3(0.0, -0.085, 0.075)
#define MR_FOOT_R vec3(0.150, 0.062, 0.205)
#define MR_K_NECK 0.30
#define MR_K_ARM 0.08
#define MR_K_LEG 0.03
#define MR_K_FOOT 0.05
#define MR_HEAD_BOUND 0.98
#define MR_FACE_C vec3(0.0, 0.02, 0.36)
#define MR_FACE_RAD 0.74
#define MR_PUSH_C vec3(0.0, 0.86, 0.0)
#define MR_PUSH_R vec3(0.88, 0.62, 0.64)
#define MR_CHEEK vec2(0.50, -0.14)
//@shadow-begin
#define MR_SHADOW_HEAD vec3(0.74, 0.60, 0.60)
#define MR_SH_X 0.640
//@shadow-end
#pragma include mascot_rig

const vec3 C_BODY  = vec3(0.105,0.500,0.003);       // the green
const vec3 C_MASK  = vec3(0.270,0.720,0.015);       // the lighter mask round the eyes, the chest feathers
const vec3 C_WING  = vec3(0.085,0.430,0.003);
const vec3 C_FOOT  = vec3(0.900,0.330,0.004);

vec3 duoEar(vec3 q){
  vec3 qa = vec3(abs(q.x), q.yz);
  bool el = q.x < 0.0;
  return (el ? gMEar0 : gMEar1)*(qa - M_EAR_PIV) + M_EAR_PIV - (el ? gMEarL0 : gMEarL1)*vec3(0.45, 0.89, 0.0);
}
float mascotHead(vec3 q){
  // a wide head, flat on top, dipping a little in the middle between the two tufts
  float d = sdEllipsoid(q, vec3(0.850, 0.640, 0.615));
  float top = q.y - (0.455 - 0.10*smoothstep(0.50, 0.0, abs(q.x)));
  d = smax(d, top, 0.16);
  vec3 e = duoEar(q);
  float tuft = sdTaper(e, vec3(0.500, 0.330, -0.02), vec3(0.640, 0.570, -0.04), 0.210, 0.062);
  d = smin(d, tuft, 0.10);
  // the beak's root: a small mound the lips (the beak) stand on
  d = smin(d, sdEllipsoid(q - vec3(0.0, -0.160, 0.560), vec3(0.172, 0.126, 0.138)), 0.04);
  return d;
}
float mascotTorso(vec3 p){
  vec3 b = p - (gMHip - M_HIP);
  float lo = sdEllipsoid(b - vec3(0.0, 0.640, 0.000), vec3(0.730, 0.455, 0.560));
  vec3 u = localToUpper(p);
  float br = 1.0 + 0.012*gBreath;
  float hi = sdEllipsoid((u - vec3(0.0, 0.960, 0.0))/vec3(br, 1.0, br), vec3(0.790, 0.430, 0.585));
  return smin(lo, hi, 0.20);
}
void mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s){
  vec3 u = localToUpper(pw);
  vec3 alb = C_BODY;
  float soft = 1.0;
  if(gMPart == 0){
    // the mask: two big light discs round the eyes that meet over the beak
    vec2 m = vec2(abs(q.x) - 0.292, q.y - 0.085);
    float mask = smoothstep(0.392, 0.380, length(m*vec2(1.0, 0.95)))*smoothstep(0.10, 0.25, q.z);
    alb = mix(alb, C_MASK, mask);
    // the beak's mound takes the beak's colour (the lips paint the rest)
    float bk = smoothstep(0.016, 0.004, sdEllipsoid(q - vec3(0.0, -0.160, 0.560), vec3(0.172, 0.126, 0.138)));
    alb = mix(alb, mix(vec3(0.920, 0.340, 0.004), vec3(1.00, 0.700, 0.015), smoothstep(-0.172, -0.158, q.y)), bk);
    soft = 1.0 - bk;
  } else if(gMPart == 1){
    // three pale chest feathers: two above, one below (downward half discs)
    vec3 un = transpose(gMW)*n;
    vec2 c = vec2(abs(u.x) - 0.165, u.y - 0.800);
    float f = smoothstep(0.012, -0.004, max(length(c*vec2(1.0, 1.55)) - 0.105, c.y));
    vec2 c2 = vec2(u.x, u.y - 0.640);
    f = max(f, smoothstep(0.012, -0.004, max(length(c2*vec2(1.0, 1.55)) - 0.105, c2.y)));
    alb = mix(alb, C_MASK, f*smoothstep(0.1, 0.45, un.z));
  } else if(gMPart == 2 || gMPart == 3){
    alb = C_WING;
  } else if(gMPart == 4 || gMPart == 5){
    alb = C_FOOT; soft = 0.3;
  }
  // feathers: rows of soft scallops over the body and the back of the head (shade + a little relief), none on the face
  if(gMPart <= 1){
    vec3 w = gMPart == 0 ? q + vec3(0.0, 1.58, 0.03) : u;
    float sz = 0.085;
    float wF = smoothstep(2.5, 5.0, sz/max(gMFp, 1e-5))*(gMPart == 0 ? smoothstep(0.25, 0.05, q.z) : 1.0);
    if(wF > 0.0){
      float row = floor(w.y/sz);
      float a = atan(w.x, w.z)*0.62/sz + 0.5*row;
      vec2 c = vec2(fract(a) - 0.5, fract(w.y/sz));
      float arc = length(vec2(c.x, c.y - 1.0)*vec2(1.0, 0.9));          // a U hanging from the row above
      float edge = smoothstep(0.10, 0.0, abs(arc - 0.52));
      alb *= 1.0 - 0.10*edge*wF;
      alb *= 1.0 + 0.05*smoothstep(0.52, 0.2, arc)*wF;
      s.occ *= 1.0 - 0.10*edge*wF;
    }
  }
  // a soft toy's surface: broad gentle tone variation + a fine felt grain where a pixel can hold it
  float big = vnoise3(r*4.0);
  alb *= 0.95 + 0.10*big;
  float wg = smoothstep(1.5, 3.2, (1.0/150.0)/max(gMFp, 1e-5))*soft;
  if(wg > 0.0){
    float g = vnoise3(r*vec3(150.0, 190.0, 150.0)) + vnoise3(r*vec3(210.0, 140.0, 210.0) + 5.3) - 1.0;
    vec3 bt = normalize(cross(vec3(0.0, 1.0, 0.0), n) + vec3(1e-4));
    s.n = normalize(n + (bt*g + cross(n, bt)*g*0.6)*0.075*wg);
    alb *= 1.0 + 0.060*wg*g;
  }
  s.alb = alb;
  s.wrap = 0.30; s.sheen = 0.30*soft; s.spec = mix(0.22, 0.05, soft); s.shin = mix(60.0, 14.0, soft);
  s.emit = alb*0.028;
#ifdef LOOK_V3
  s.rough = mix(0.34, 0.74, soft);
  s.sheenK = 0.42*soft; s.sheenCol = vec3(0.80, 1.0, 0.55);
  s.sss = mix(0.20, 0.30, soft); s.sssCol = gMPart >= 4 || soft < 0.5 ? vec3(1.0, 0.55, 0.20) : vec3(0.60, 1.0, 0.25);
  s.coat = 0.20*(1.0 - soft);
#endif
}
#pragma include mascot_body

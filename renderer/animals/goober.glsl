// ---------------------------------------------------------------------------
// animals/goober.glsl — the dapper goober gentleman in the muse's classroom.
//  A goober in its shell — two lobes with a waist, the upper one the face — under a black top hat, a monocle on one
//  eye; thin black arms and legs, white gloves, white spats over black shoes.  Bead eyes, thin brows, a neat smile.
//  The shell is a dry tan husk with its lengthwise ridges and criss-cross net.  Skeleton / hand map / grasp = the muse's.
// ---------------------------------------------------------------------------
#define M_H 2.30
#define M_HEADC vec3(0.0, 1.54, 0.02)
#define M_HEADR vec3(0.50, 0.52, 0.47)
#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.54, 0.02)
#define MUSE_EMO 1
#define EYE_R0 0.052
#define EYE_C0 vec3(0.150,0.120,0.420)
#define EYE_R gEyeR
#define EYE_C gEyeC
#define EYE_AU 0.95
#define EYE_AL (-0.95)
#define EYE_AC (-0.05)
#define EYE_LID_TH 0.0075
#define EYE_SQUINT 0.50
#define EYE_IRIS 0.97
#define EYE_PUPIL 0.62
#define EYE_COL_A vec3(0.030,0.022,0.020)
#define EYE_COL_B vec3(0.010,0.008,0.008)
#define EYE_COL_W vec3(0.020,0.016,0.015)
#define LINER_COL vec3(0.16,0.09,0.03)
#define EYE_CATCH 0.930
#define BROW_C vec3(0.155,0.232,0.0)
#define BROW_L 0.070
#define BROW_W 0.012
#define BROW_COL vec3(0.060,0.035,0.015)
#define BROW_H 0.010
#define BROW_RISE 0.060
#define MUSE_FROWN 0.42
#define BROW_SAD_TILT 1.4
#define LIP_O vec3(0.0,-0.120,0.455)
#define LIP_TILT 0.08
#define LIP_W 0.110
#define LIP_HU (0.024*LIPU)
#define LIP_HL (0.030*LIPU)
#define LIP_TU (0.008*LIPU)
#define LIP_TL (0.010*LIPU)
#define LIP_BOW 0.0
#define LIP_SQUARE 0.30
#define LIP_OPEN_Y 1.40
#define LIP_PUSH (0.060*LIPU)
#define LIP_SKIN 0.90
#define LIP_BLEND (0.012*LIPU)
#define LIP_TOOTH_Z (-0.20*LIPU)
#define LIP_TEETH_H ((0.010 + 0.050*gEmTeeth)*LIPU)
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(0.52,0.33,0.13)
#define LIP_COL_L vec3(0.54,0.35,0.14)
#define LIP_COL_LINE vec3(0.090,0.040,0.012)
#define LIP_LINE 2.4
#define LIP_LINE_SOFT 0.40
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.22,0.07,0.05)
#define LIP_COL_DEEP vec3(0.045,0.014,0.012)
#define LIP_COL_GUM  vec3(0.36,0.14,0.10)
#define LIP_COL_TONGUE vec3(0.84,0.40,0.38)
#define LIP_COL_TOOTH vec3(0.90,0.88,0.82)
#define LIP_SPEC 0.10
#define LIP_SHIN 30.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.05
#define LIP_WRAP 0.30
#define LIP_EMIT vec3(0.03,0.02,0.01)
#define LIP_TONGUE_FREE 0.11
#pragma include face
#pragma include lips

#define MR_ARM_R vec3(0.042, 0.038, 0.036)
#define MR_LEG_R vec3(0.050, 0.044, 0.042)
#define MR_PALM_R vec3(0.105, 0.095, 0.060)
#define MR_FING_R 0.030
#define MR_FING_L 1.0
#define MR_THUMB_R 0.036
#define MR_FOOT_C vec3(0.0, -0.055, 0.075)
#define MR_FOOT_R vec3(0.110, 0.092, 0.215)
#define MR_K_NECK 0.34
#define MR_K_ARM 0.012
#define MR_K_LEG 0.012
#define MR_K_HAND 0.030
#define MR_K_FOOT 0.030
#define MR_CREASE_OCC 0.0
#define MR_HEAD_BOUND 1.20
#define MR_FACE_C vec3(0.0, 0.0, 0.38)
#define MR_FACE_RAD 0.40
#define MR_CHEEK vec2(0.28, -0.07)
#define MR_INIT_HOOK 1
#define MR_HEAD_EXTRA 1
#define MR_PUSH_C vec3(0.0, 0.84, 0.0)
#define MR_PUSH_R vec3(0.60, 0.58, 0.54)
//@shadow-begin
#define MR_SHADOW_HEAD vec3(0.48, 0.66, 0.45)
//@shadow-end
#pragma include mascot_rig

const vec3 C_SHELL = vec3(0.700,0.470,0.110);
const vec3 C_SHELL2 = vec3(0.900,0.670,0.190);
const vec3 C_BLACK = vec3(0.010,0.010,0.013);
const vec3 C_WHITE = vec3(0.900,0.900,0.900);

mat3 gHat;
void mascotInit(){
  // the top hat tips with the ear dials (a doff of the hat = ears forward)
  float a = 0.5*(clamp(uP[P_EAR_R], -1.0, 1.0) - clamp(uP[P_EAR_L], -1.0, 1.0));
  float f = 0.5*(clamp(uP[P_EAR_L], -1.0, 1.0) + clamp(uP[P_EAR_R], -1.0, 1.0));
  float fl = clamp(uP[P_EAR_FLICK_L], 0.0, 1.0) - clamp(uP[P_EAR_FLICK_R], 0.0, 1.0);
  gHat = rotZ(0.20*a + 0.14*fl - 0.09)*rotX(-0.16*f);
}
float mascotHead(vec3 q){
  // the upper lobe of the shell
  float d = sdEllipsoid(q - vec3(0.0, 0.030, 0.0), vec3(0.470, 0.500, 0.440));
  return d;
}
int gPHead;        // 1 hat, 2 hat band, 3 monocle rim, 4 monocle glass
float mascotHeadBare(vec3 q){
  // the top hat: a brim and a tall crown that flares a little, sitting on the crown of the shell
  vec3 h = transpose(gHat)*(q - vec3(0.0, 0.440, 0.0));
  float rr = length(h.xz);
  float brim = max(rr - 0.560, abs(h.y - 0.020) - 0.014) - 0.012;
  brim += 0.05*smoothstep(0.30, 0.56, abs(h.x))*smoothstep(0.0, 0.3, 1.0)*0.0;
  float cr = max(rr - (0.335 + 0.060*smoothstep(0.05, 0.62, h.y)), abs(h.y - 0.330) - 0.310) - 0.016;
  float hat = min(brim, cr);
  gPHead = h.y > 0.045 && h.y < 0.150 && cr < brim ? 2 : 1;
  // the monocle: a thin dark ring standing just off the screen-right eye
  vec3 m = q - vec3(0.150, 0.120, 0.487);
  float ring = length(vec2(length(m.xy) - 0.098, m.z)) - 0.0085;
  if(ring < hat){ gPHead = 3; }
  return min(hat, ring);
}
float mascotTorso(vec3 p){
  // the lower lobe: fuller than the upper; the waist between them comes from the smooth join
  vec3 b = p - (gMHip - M_HIP);
  vec3 u = localToUpper(p);
  float br = 1.0 + 0.010*gBreath;
  float lo = sdEllipsoid(b - vec3(0.0, 0.700, 0.0), vec3(0.500, 0.400, 0.450));
  float hi = sdEllipsoid((u - vec3(0.0, 0.900, 0.0))/vec3(br, 1.0, br), vec3(0.505, 0.340, 0.450));
  return smin(lo, hi, 0.12);
}
void mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s){
  vec3 u = localToUpper(pw);
  if(gMPart == 0 && gMBare < gMHeadD + 0.002){
    mascotHeadBare(q);
    if(gPHead == 3){        // the monocle's rim: dark polished metal
      s.alb = vec3(0.70, 0.48, 0.12); s.spec = 0.9; s.shin = 120.0; s.mirror = 0.25; s.wrap = 0.0; s.sheen = 0.0;
#ifdef LOOK_V3
      s.rough = 0.22; s.coat = 0.5;
#endif
      return;
    }
    // the silk hat: black with a soft sheen that runs round it; a grey ribbon at the foot of the crown
    vec3 alb = gPHead == 2 ? vec3(0.780, 0.780, 0.800) : C_BLACK;
    s.alb = alb; s.wrap = 0.10; s.sheen = 0.45; s.spec = 0.85; s.shin = 90.0; s.emit = vec3(0.0); s.mirror = 0.06;
    s.T = normalize(cross(vec3(0.0, 1.0, 0.0), n) + vec3(1e-4));
#ifdef LOOK_V3
    s.rough = 0.26; s.sheenK = 0.40; s.sheenCol = vec3(0.55, 0.58, 0.70); s.coat = 0.55;
#endif
    return;
  }
  if(gMPart == 0 || gMPart == 1){
    // the husk: lengthwise ridges crossed by shorter ones (a net of shallow pits), paler on the ridges, fibrous
    vec3 w = gMPart == 0 ? mix(u, q + vec3(0.0, 1.54, 0.02), smoothstep(-0.42, -0.12, q.y)) : u;      // one pattern across the waist: the head's own frame only above it
    float ang = atan(w.x, w.z);
    float wav = 0.9*vnoise3(w*2.6) + 0.35*vnoise3(w*7.0 + 3.0);
    float lon = abs(sin(ang*8.0 + wav*2.0));                    // 16 ridges round
    float lat = abs(sin((w.y + 0.03*sin(ang*8.0))*24.0 + wav*3.0));
    float ridge = max(smoothstep(0.55, 1.0, lon), 0.60*smoothstep(0.70, 1.0, lat)*smoothstep(0.35, 0.65, vnoise3(w*vec3(9.0, 4.0, 9.0) + 1.7)));
    float face = gMPart == 0 ? smoothstep(0.36, 0.22, length((q.xy - vec2(0.0, 0.01))*vec2(1.0, 1.15)))*smoothstep(0.10, 0.30, q.z) : 0.0;
    ridge *= 1.0 - 0.80*face;                                    // the face is smooth shell
    float fib = vnoise3(w*vec3(60.0, 14.0, 60.0));
    vec3 alb = mix(C_SHELL, C_SHELL2, clamp(0.25 + 0.70*ridge + 0.20*(fib - 0.5) + 0.25*face, 0.0, 1.0));
    alb *= 0.94 + 0.10*vnoise3(w*5.0);
    s.alb = alb;
    vec3 e1 = normalize(cross(vec3(0.0, 1.0, 0.0), n) + vec3(1e-4)), e2 = cross(n, e1);
    float dl = cos(ang*8.0 + wav*2.0)*sign(sin(ang*8.0 + wav*2.0)), dt = cos((w.y)*24.0 + wav*3.0)*sign(sin(w.y*24.0 + wav*3.0));
    float wR = smoothstep(1.2, 2.6, 0.05/max(gMFp, 1e-5));
    s.n = normalize(n + (e1*dl*(1.0 - smoothstep(0.55, 1.0, lon))*0.30 + e2*dt*(1.0 - smoothstep(0.70, 1.0, lat))*0.20)*(1.0 - 0.8*face)*wR);
    s.occ = 1.0 - 0.22*(1.0 - ridge)*(1.0 - face);
    s.wrap = 0.30; s.sheen = 0.12; s.spec = 0.12; s.shin = 22.0;
    s.emit = alb*0.030;
#ifdef LOOK_V3
    s.rough = 0.62; s.sss = 0.20; s.sssCol = vec3(1.0, 0.62, 0.30); s.sheenK = 0.16; s.sheenCol = vec3(1.0, 0.86, 0.60);
    s.cavity = 1.0 - 0.25*(1.0 - ridge)*(1.0 - face);
#endif
    return;
  }
  bool glove = gMPart == 3;
  // spats: white over the ankle and the top of a black shoe
  float spat = gMPart == 4 ? smoothstep(0.80, 0.86, gMLimbT) : 0.0;
  if(gMPart == 5){ vec3 fq = r - vec3(9.0 + 2.0*float(gMSide), 0.0, 0.0); spat = smoothstep(0.06, 0.02, fq.z)*smoothstep(-0.085, -0.060, fq.y); }
  vec3 alb = glove ? C_WHITE : mix(C_BLACK, C_WHITE, spat);
  float gloss = (gMPart == 5 ? 1.0 : 0.25)*(1.0 - spat)*float(!glove);
  s.alb = alb*(0.97 + 0.05*vnoise3(r*9.0));
  s.wrap = 0.30; s.sheen = 0.18; s.spec = mix(0.08, 0.80, gloss); s.shin = mix(18.0, 140.0, gloss); s.mirror = 0.10*gloss;
  s.emit = s.alb*0.035;
#ifdef LOOK_V3
  s.rough = mix(0.70, 0.18, gloss); s.coat = 0.7*gloss; s.sheenK = 0.25*(1.0 - gloss); s.sheenCol = vec3(1.0);
  s.sss = 0.15*float(glove); s.sssCol = vec3(1.0, 0.9, 0.8);
#endif
}
#pragma include mascot_body

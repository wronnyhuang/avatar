// ---------------------------------------------------------------------------
// animals/pneu.glsl — the white man made of stacked tires, in the muse's classroom.
//  Everything is a stack of soft white rings: four on the head (the face sits on the middle two), three big ones for
//  the trunk, rings down the arms and legs, mitt hands, round boots.  Round eyes with black pupils behind a dark
//  rim, thin arched brows, a wide friendly smile.  White rubber-vinyl: a soft gloss, grey shade deep in the creases.
//  Skeleton / hand map / grasp = the muse's (engine/mascot_rig.glsl).
// ---------------------------------------------------------------------------
#define M_H 2.30
#define M_HEADC vec3(0.0, 1.60, 0.03)
#define M_HEADR vec3(0.50, 0.52, 0.48)
#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.60, 0.03)
#define MUSE_EMO 1
#define EYE_R0 0.108
#define EYE_C0 vec3(0.172,0.125,0.372)
#define EYE_R gEyeR
#define EYE_C gEyeC
#define EYE_AU 0.95
#define EYE_AL (-0.95)
#define EYE_AC (-0.05)
#define EYE_LID_TH 0.012
#define EYE_SQUINT 0.50
#define EYE_IRIS 0.470
#define EYE_PUPIL 0.458
#define EYE_COL_A vec3(0.004,0.004,0.005)
#define EYE_GLOSS 0.25
#define EYE_COL_B vec3(0.003,0.003,0.004)
#define EYE_COL_W vec3(0.930,0.930,0.930)
#define LINER_COL vec3(0.030,0.030,0.040)
#define EYE_CATCH 0.975
#define BROW_C vec3(0.160,0.290,0.0)
#define BROW_L 0.085
#define BROW_W 0.006
#define BROW_COL vec3(0.700,0.700,0.720)
#define BROW_H 0.010
#define BROW_RISE 0.060
#define MUSE_FROWN 0.42
#define BROW_SAD_TILT 1.4
#define LIP_O vec3(0.0,-0.115,0.515)
#define LIP_TILT 0.08
#define LIP_W 0.195
#define LIP_HU (0.024*LIPU)
#define LIP_HL (0.028*LIPU)
#define LIP_TU (0.008*LIPU)
#define LIP_TL (0.010*LIPU)
#define LIP_BOW 0.0
#define LIP_SQUARE 0.25
#define LIP_OPEN_Y 1.30
#define LIP_PUSH (0.050*LIPU)
#define LIP_SKIN 0.25
#define LIP_BLEND (0.012*LIPU)
#define LIP_TOOTH_Z (-0.20*LIPU)
#define LIP_TEETH_H ((0.008 + 0.040*gEmTeeth)*LIPU)
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(0.015,0.015,0.020)
#define LIP_COL_L vec3(0.015,0.015,0.020)
#define LIP_COL_LINE vec3(0.012,0.012,0.016)
#define LIP_LINE 3.6
#define LIP_LINE_SOFT 0.40
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.020,0.012,0.014)
#define LIP_COL_DEEP vec3(0.020,0.015,0.022)
#define LIP_COL_GUM  vec3(0.30,0.20,0.24)
#define LIP_COL_TONGUE vec3(0.55,0.12,0.12)
#define LIP_COL_TOOTH vec3(0.30,0.20,0.24)
#define LIP_SPEC 0.20
#define LIP_SHIN 50.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.05
#define LIP_WRAP 0.30
#define LIP_EMIT vec3(0.02)
#define LIP_TONGUE_FREE 0.11
#pragma include face
#pragma include lips

#define MR_ARM_R vec3(0.136, 0.126, 0.114)
#define MR_CUSTOM_BEND 1
#define MR_LEG_R vec3(0.168, 0.152, 0.138)
#define MR_RING_ARM(h) (0.040*mrRing((h)*3.0) - 0.018)
#define MR_RING_LEG(h) (0.040*mrRing((h)*2.5) - 0.018)
#define MR_PALM_R vec3(0.130, 0.120, 0.075)
#define MR_REST_CURL 0.12
#define MR_CREASE_OCC 0.12
#define MR_FING_R 0.040
#define MR_FING_L 1.10
#define MR_THUMB_R 0.046
#define MR_FOOT_C vec3(0.0, -0.040, 0.065)
#define MR_FOOT_R vec3(0.172, 0.122, 0.245)
#define MR_K_NECK 0.030
#define MR_K_ARM 0.050
#define MR_K_LEG 0.050
#define MR_K_HAND 0.030
#define MR_K_FOOT 0.030
#define MR_HEAD_BOUND 0.90
#define MR_FACE_C vec3(0.0, 0.0, 0.42)
#define MR_FACE_RAD 0.44
#define MR_CHEEK vec2(0.30, -0.06)
//@shadow-begin
#define MR_SHADOW_HEAD vec3(0.46, 0.52, 0.44)
//@shadow-end
#pragma include mascot_rig

void mascotMouthBend(){
  // a broad jolly grin at rest
  float op = smoothstep(0.0, 0.6, max(uP[P_LIP_OPEN], uP[P_MOUTH_OPEN]));
  float frown = clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  float rest = 0.33*mix(0.30, 1.0, smoothstep(0.0, 0.05, uP[P_SMILE]));
  gLBend += LIP_W*((rest + 0.08*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - 0.6*op) - MUSE_FROWN*frown*(1.0 - 0.15*op))*(1.0 - gLRnd);
}
const vec3 C_TIRE = vec3(0.930,0.950,1.000);

float gRingCrease;      // 1 deep in a crease between two rings (set by the last head / trunk evaluation)
float mascotHead(vec3 q){
  // a round ball with two faint ring lines (the old stack of rings, all but smoothed away), the face on its middle
  float d = 1e3, d2 = 1e3, st = 1e3;
  for(int k=0;k<3;k++){
    float y = -0.250 + 0.250*float(k);
    float R = k == 1 ? 0.495 : 0.415;
    float e = sdEllipsoid(q - vec3(0.0, y, 0.0), vec3(R, 0.250, R*0.96));
    if(e < d){ d2 = d; d = e; } else if(e < d2) d2 = e;
    st = smin(st, e, 0.20);
  }
  float face = sdEllipsoid(q - vec3(0.0, 0.010, 0.020), vec3(0.455, 0.340, 0.475));
  float fw = smoothstep(0.40, 0.24, length(q.xy*vec2(1.0, 1.20)))*smoothstep(0.10, 0.30, q.z);
  gRingCrease = clamp(1.0 - (d2 - d)/0.05, 0.0, 1.0)*(1.0 - fw)*0.6;
  return mix(st, smin(st, face, 0.06), fw);
}
float mascotTorso(vec3 p){
  // four fat tire rings, crisp creases between them
  vec3 b = p - (gMHip - M_HIP);
  vec3 u = localToUpper(p);
  float br = 1.0 + 0.012*gBreath;
  float r0 = sdEllipsoid(b - vec3(0.0, 0.575, 0.0), vec3(0.455, 0.150, 0.410));                          // hips
  float r1 = sdEllipsoid(b - vec3(0.0, 0.750, 0.0), vec3(0.555, 0.160, 0.495));                          // belly
  float r2 = sdEllipsoid((u - vec3(0.0, 0.925, 0.0))/vec3(br, 1.0, br), vec3(0.545, 0.160, 0.480));     // ribs
  float r3 = sdEllipsoid((u - vec3(0.0, 1.085, 0.0))/vec3(br, 1.0, br), vec3(0.455, 0.145, 0.400));     // chest
  float a = min(min(r0, r1), min(r2, r3));
  float second = min(min(max(r0, r1), max(r2, r3)), max(min(r0, r1), min(r2, r3)));
  gRingCrease = clamp(1.0 - (second - a)/0.05, 0.0, 1.0);
  return smin(smin(smin(r0, r1, 0.040), r2, 0.040), r3, 0.040);
}
#define MR_SHADOW_TORSO 1
float mascotShadowTorso(vec3 p){
  vec3 b = p - (gMHip - M_HIP);
  return sdEllipsoid(b - vec3(0.0, 0.800, 0.0), vec3(0.500, 0.430, 0.440));
}
void mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s){
  float crease = 0.0;
  if(gMPart == 0){ mascotHead(q); crease = gRingCrease; }
  else if(gMPart == 1){ mascotTorso(pw); crease = gRingCrease; }
  else if(gMPart == 2 || gMPart == 4){
    float h = gMLimbT < 0.5 ? gMLimbT*2.0 : gMLimbT*2.0 - 1.0;
    crease = smoothstep(0.35, 0.0, mrRing(h*2.0));
  }
  vec3 alb = C_TIRE*(0.97 + 0.05*vnoise3(r*6.0));
  alb *= 1.0 - 0.03*crease;                       // road dust settles in the creases
  alb = mix(alb, vec3(0.42, 0.42, 0.45), 0.85*gMSoleW);
  if(gMPart == 0){
    // a dark rim round each eye (he wore pince-nez once)
    vec2 e = vec2(abs(q.x), q.y) - vec2(gEyeC.x, gEyeC.y);
    float rim = smoothstep(0.011, 0.005, abs(length(e) - (0.86*gEyeR + 0.004)))*smoothstep(0.25, 0.40, q.z);
    alb = mix(alb, vec3(0.02, 0.02, 0.03), 0.0*rim);
  }
  s.alb = alb;
  s.occ = 1.0 - 0.14*crease;
  s.wrap = 0.34; s.sheen = 0.25; s.spec = 0.50; s.shin = 90.0;
  s.emit = alb*0.060;
#ifdef LOOK_V3
  s.rough = 0.30; s.coat = 0.40; s.sss = 0.22; s.sssCol = vec3(1.0, 0.94, 0.88);
  s.sheenK = 0.14; s.sheenCol = vec3(1.0);
  s.cavity = 1.0 - 0.20*crease;
#endif
}
#pragma include mascot_body

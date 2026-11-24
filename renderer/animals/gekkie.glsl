// ---------------------------------------------------------------------------
// animals/gekkie.glsl — a green day-gekkie spokes-lizard in the muse's classroom.
//  A wide flat head with big dark glossy eyes high on its corners, an orange-red blaze between them, a long wide smile
//  over a cream jaw, a slim neck and body with a pale scaled belly, thin arms with padded fingers, a long tail that
//  sweeps out on the floor.  Pebbled scales, a soft skin gloss.
//  The skeleton, the hand map and the grasp are the muse's (engine/mascot_rig.glsl): it shares muse.props.json.
// ---------------------------------------------------------------------------
#define M_H 2.30
// the real one's build — a small head on a long neck, a long slender trunk, thin limbs.  The skeleton's joints are
// the muse's; the shoulders sit higher (MR_SH_Y) and the hand map moves up with them (MR_MAP_OFF); the head is modelled at
// 1/MR_HEAD_SCALE (every face define below is in that scaled head space)
#define MR_HEAD_SCALE 0.400
#define M_HEADC vec3(0.0, 1.800, 0.130)
#define M_HEADR vec3(0.235, 0.144, 0.200)
#define M_UPPER 0.285
#define M_FORE 0.265
#define MR_SH_X 0.140
#define MR_SH_Y 1.405
#define MR_MAP_OFF vec3(-0.085, 0.37, 0.0)
#define MR_PUSH_C vec3(0.0, 1.05, 0.01)
#define MR_PUSH_R vec3(0.25, 0.55, 0.23)
#define MR_HIP_X 0.075
#define MR_THIGH 0.200                              // geo: shorter leg bones = a straighter, slimmer stance (engine/mascot_rig.glsl)
#define MR_SHIN 0.200
#define MR_ANK_X 0.125
#define MR_REST_OFF vec3(-0.300, 0.130, 0.11)
#define MR_REST_ASYM vec3(0.02, 0.07, 0.05)                 // engine/mascot_rig.glsl
#define MR_STANCE_ASYM vec3(-0.03, 0.0, 0.07)               // engine/mascot_rig.glsl
#define MR_REST_POLE vec3(0.30, -0.35, -1.0)
#define MR_WRIST_UP_OFF 0.30                         // engine/mascot_rig.glsl: relaxed hands at waist height, not fingers-up
#define MR_PALM_FWD 0.75
#define MR_TOE_OUT 0.50
#define MR_HAND_THICK 0.035
#define MR_HAND_SCALE 0.62
#define MR_SPREAD(sp) (0.55 + 1.6*(sp))

#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.800, 0.130)
#define MUSE_EMO 1
#define EYE_R0 0.168
#define EYE_C0 vec3(0.432,0.118,0.070)
#define EYE_R gEyeR
#define EYE_C gEyeC
#define EYE_AU 0.82
#define EYE_AL (-0.92)
#define EYE_AC (-0.08)
#define EYE_LID_TH 0.007
#define EYE_LID_YAW 0.60
#define EYE_LID_SOLID 1                            // hd: engine/face.glsl — solid lids, no ragged slits in profile
#define EYE_SQUINT 0.50
#define EYE_IRIS 0.950
#define EYE_PUPIL 0.700
#define EYE_COL_A vec3(0.70,0.30,0.060)            // amber glowing low in the iris
#define EYE_COL_B vec3(0.34,0.110,0.028)           // a deep maroon-brown ring and top
#define EYE_COL_W vec3(0.10,0.028,0.016)
#define LINER_COL vec3(0.045,0.050,0.026)          // hd: the lid's edge: a thin dark line where it meets the eye
#define EYE_GLOW vec3(0.160,0.050,0.012)
#define EYE_CATCH 0.950
#define EYE_CATCH2 0.0
#define EYE_V3_FIB 0.12
#define GK_PUPIL 0.620                              // hd: a visible warm iris ring round the big pupil
// hi: the V3_HI (cloud / eval) skin, see mascotMaterial (each overridable for tuning)
#ifndef GK_HI_BEAD
#define GK_HI_BEAD 0.60
#endif
#ifndef CHAR_KEY_TOP
#define CHAR_KEY_TOP 0.15                          // engine/scene_main.glsl (V3_HI): key gradient by height on each form, gekkie only
#endif
#ifndef CHAR_BOUNCE
#define CHAR_BOUNCE 0.35                           // engine/scene_main.glsl (V3_HI): warm bounce from below, gekkie only
#define CHAR_BOUNCE_COL vec3(1.0, 0.90, 0.74)
#endif
#ifndef GK_JW0
#define GK_JW0 0.03
#endif
#ifndef GK_JW1
#define GK_JW1 0.12
#endif
#ifndef GK_JWZ
#define GK_JWZ 0.55
#endif
#ifndef GK_HI_DIGIT
#define GK_HI_DIGIT 0.30
#endif
#ifndef GK_HI_BEAD0
#define GK_HI_BEAD0 1.0
#endif
#ifndef GK_HI_MINPX0
#define GK_HI_MINPX0 2.4
#endif
#ifndef GK_HI_MINPX
#define GK_HI_MINPX 2.2
#endif
#ifndef GK_HI_OCT0
#define GK_HI_OCT0 0.45
#endif
#ifndef GK_HI_OCT1
#define GK_HI_OCT1 0.40
#endif
#ifndef GK_HI_HK0
#define GK_HI_HK0 0.40
#endif
#ifndef GK_HI_HK1
#define GK_HI_HK1 0.32
#endif
#ifndef GK_HI_RD
#define GK_HI_RD 0.30
#endif
#ifndef GK_HI_COAT
#define GK_HI_COAT 0.18
#endif
#ifndef GK_HI_THW
#define GK_HI_THW 0.20
#endif
#ifndef GK_HI_CHIN
#define GK_HI_CHIN 0.62
#endif
#ifndef GK_ULIP_H
#define GK_ULIP_H 0.120                            // hd: height of the upper lip's cream fade above the mouth line (head units)
#endif
#ifndef GK_ULIP_K
#define GK_ULIP_K 0.90
#endif
#ifndef GK_ULIP_COL
#define GK_ULIP_COL vec3(0.420, 0.420, 0.315)
#endif
#define GK_NOS vec3(0.132, -0.145, 0.630)               // hd: the nostril pits (head space, +x side)
#define GK_BLZ_Z0 (-0.02)                               // hd: crown patch: back end ...
#define GK_BLZ_Z1 0.36                                  // ... where the forehead turns into the crown
#define GK_BLZ_WB 0.200                                 // half width at the back ...
#define GK_BLZ_WF 0.060                                 // ... and at the front tip
#define GK_BLZ_Y0 0.10                                  // ... its tip's height between the eyes
#define GK_BLZ_COL vec3(0.330, 0.112, 0.108)            // salmon-red under this light (was 0.44,0.14,0.14: too orange, too bright)
#define GK_BLZ_K 0.72
#define GK_EYE_WIN 0.72                                 // hd: the window's reflection (was 1.15: a pasted white blob)
#define GK_EYE_COAT 0.28                                // the key's crisp glint (was 0.45: a white dot)
#ifndef CHAR_SKY_AO
#define CHAR_SKY_AO 0.55                           // engine/scene_main.glsl (V3_HI): bent-normal sky occlusion, gekkie only
#endif
#ifndef GK_HI_MOT
#define GK_HI_MOT 0.16
#endif
#ifndef GK_HI_BTONE
#define GK_HI_BTONE 0.16
#endif
#ifndef GK_HI_LIFT
#define GK_HI_LIFT 0.12
#endif
#ifndef GK_HI_CAV
#define GK_HI_CAV 0.32
#endif
#define BROW_C vec3(0.400,0.303,0.0)
#define BROW_L 0.110
#define BROW_W 0.010
#define BROW_COL vec3(0.170,0.360,0.070)           // a faint ridge over each eye (it has no hair)
#define BROW_H 0.006
#define BROW_RISE 0.060
#define MUSE_FROWN 0.42
#define BROW_SAD_TILT 1.4
#define LIP_O vec3(0.0,-0.305,0.535)
#define LIP_TILT 0.16
#define LIP_W 0.440
#define LIP_REACH 1.08                              // how far a spread or a grin may carry the corners
#define LIP_HU (0.020*LIPU)
#define LIP_HL (0.018*LIPU)
#define LIP_TU (0.005*LIPU)                          // hd: the lips lie flush (no applied roll)
#define LIP_TL (0.005*LIPU)
#define LIP_NO_BANDS 1
#define LIP_PAD_K 0.0                                 // hd: engine/lips.glsl — a small corner pad (no knob where the line ends)
#define LIP_DEPTH (0.11*LIPU)
#define LIP_BOW 0.0
#define LIP_SQUARE 0.0
#define LIP_OPEN_Y 0.38
#define LIP_PUSH (0.045*LIPU)
#define LIP_SKIN 0.10                                 // hd: barely any corner pit (it drew a notch where the line ends)
#define LIP_BLEND (0.032*LIPU)
#define LIP_TOOTH_Z (-0.20*LIPU)
#define LIP_TEETH_H ((0.002 + 0.010*gEmTeeth)*LIPU)
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(0.55,0.55,0.45)             // the upper lip: a thick pale cream-yellow band
#define LIP_COL_L vec3(0.55,0.54,0.46)             // the lower lip: pinkish cream
#define LIP_COL_LINE vec3(0.040,0.026,0.020)
#define LIP_LINE 1.30
#define LIP_LINE_SOFT 0.22
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.080,0.030,0.030)
#define LIP_COL_DEEP vec3(0.045,0.014,0.018)
#define LIP_COL_GUM  vec3(0.20,0.06,0.06)
#define LIP_COL_TONGUE vec3(0.58,0.27,0.27)
#define LIP_COL_TOOTH vec3(0.20,0.06,0.06)
#define LIP_SPEC 0.12
#define LIP_AO 0.10
#define LIP_SHIN 60.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.05
#define LIP_WRAP 0.30
#define LIP_EMIT vec3(0.04,0.02,0.02)
#define LIP_TONGUE_FREE 0.11
#define GK_REST_OPEN_UNUSED (0.075 + 0.09*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE])))
#pragma include face
#pragma include lips

#define MR_ARM_R vec3(0.046, 0.032, 0.024)
// geo: the arm's own shape (engine/mascot_body.glsl MR_ARM_FN): a soft upper arm that grows out of the shoulder, a
// small elbow point, a forearm that swells a little below the elbow and tapers to a thin wrist
#define MR_ARM_FN gkArm
#define MR_FOOT_C vec3(0.0, -0.090, 0.050)              // the foot's SHADOW stand-in only (MR_FOOT_CUSTOM draws the foot): small, no dark disc on the floor
#define MR_FOOT_R vec3(0.060, 0.035, 0.080)
#define MR_REST_CURL 0.38
#define MR_LEG_R vec3(0.054, 0.033, 0.022)
#define MR_PALM_R vec3(0.078, 0.078, 0.032)
#define MR_FING_R 0.0140
#define MR_FING_L 1.60
#define MR_FING_PAD 0.0172
#define MR_THUMB_R 0.0180
#define MR_K_NECK 0.12
#define MR_CREASE_OCC 0.15
#define MR_K_ARM 0.065
#define MR_K_LEG 0.08
#define MR_K_HAND 0.030
#define MR_K_FOOT 0.035
#define MR_K_EXTRA 0.10
#define MR_HEAD_BOUND 0.72
#define MR_FACE_C vec3(0.0, 0.0, 0.25)
#define MR_FACE_RAD 0.66
#define MR_EXTRA 1
#define MR_INIT_HOOK 1
#define MR_FOOT_CUSTOM 1
#define MR_CUSTOM_BEND 1
#define MR_BARE_EXTRA 1
#define MR_FACE_POST 1
#define MR_BOUND_EXTRA PB_(gTail[1], 0.28) PB_(gTail[2], 0.28) PB_(gTail[3], 0.22) PB_(gTail[4], 0.19) PB_(gTail[5], 0.16)
//@shadow-begin
#define MR_SHADOW_HEAD vec3(0.225, 0.134, 0.190)
//@shadow-end
#define MR_SHADOW_ARM_R vec3(-0.25, -0.25, -0.25)     // no arm / palm shadow stand-ins (engine/mascot_body.glsl): even thin ones smudge the trunk through the soft-shadow penumbra
#define MR_AO_ARM_PAD 0.10                            // the AO pass sees the arms 0.10 thinner: no black armpits
#pragma include mascot_rig
float gkArm(vec3 p, int i, out float tt){
  vec3 a = gMSh[i], b = gMEl[i], c = gMWr[i];
  vec3 pa = p - a, ba = b - a;
  float h1 = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  float d1 = length(pa - ba*h1) - (mix(0.046, 0.029, h1) + 0.004*sin(3.1416*h1));
  vec3 pb = p - b, cb = c - b;
  float h2 = clamp(dot(pb, cb)/dot(cb, cb), 0.0, 1.0);
  float d2 = length(pb - cb*h2) - (0.030 + 0.005*exp(-(h2 - 0.22)*(h2 - 0.22)*25.0) - 0.008*h2);
  tt = d1 < d2 ? 0.5*h1 : 0.5 + 0.5*h2;
  vec3 eo = normalize(ba) - normalize(cb);
  float el = length(p - b - (dot(eo, eo) > 1e-4 ? normalize(eo) : vec3(0.0))*0.008) - 0.026;   // the elbow's point
  return smin(smin(d1, d2, 0.035), el, 0.025);
}

// colours (linear): a natural leaf green, a pale grey-cream belly, cream-yellow jaw and hands, a soft orange crown patch
const vec3 C_GREEN  = vec3(0.088,0.174,0.090);      // the back and the sides
const vec3 C_GREEND = vec3(0.046,0.086,0.040);      // darker flecks
const vec3 C_GREENL = vec3(0.172,0.262,0.148);      // lighter flecks, lower flanks
const vec3 C_BELLY  = vec3(0.205,0.262,0.262);      // belly and throat: pale grey-cream
const vec3 C_CREAM  = vec3(0.262,0.310,0.235);      // jaw, lip rim
const vec3 C_HAND   = vec3(0.340,0.400,0.300);      // fingers and toes: pale cream-yellow
const vec3 C_BLAZE  = vec3(0.400,0.120,0.075);      // the salmon-red crown patch
const vec3 C_BOW    = vec3(0.300,0.012,0.022);      // the bow tie: deep red

vec3 gTail[6];
float gTailR[6];
vec3 gTailS[12]; float gTailSR[12]; vec4 gTailB;   // geo: the tail path rounded once (corner cutting), and its bounding sphere
vec3 gNeckA, gNeckB;
vec3 gBowO; mat3 gBowF;
vec2 gGkHeadFw;      // the head's forward in the upper body's frame (xz): the throat's cream turns with it
void mascotInit(){
  // the tail: from the back of the pelvis down to the floor and out to screen-left in a long S, where it shows beside the
  // feet.  tail_wag swings it (more toward the tip), tail_raise lifts it off the floor, tail_curl winds the tip in.
  float wag = clamp(uP[P_TAIL_WAG], -1.0, 1.0), rs = clamp(uP[P_TAIL_RAISE], 0.0, 1.0), cu = clamp(uP[P_TAIL_CURL], 0.0, 1.0);
  vec3 b = gMHip - M_HIP;
  // geo: the tail continues the trunk — thick at its root (near the pelvis' width), leaving the body low and
  // tapering smoothly along a relaxed S on the floor
  gTail[0] = vec3(0.00, 0.50, -0.08); gTail[1] = vec3(-0.01, 0.27, -0.19); gTail[2] = vec3(-0.06, 0.08, -0.30);
  gTail[3] = vec3(-0.28, 0.05, -0.42); gTail[4] = vec3(-0.56, 0.04, -0.38); gTail[5] = vec3(-0.80, 0.03, -0.18);
  gTailR[0] = 0.090; gTailR[1] = 0.074; gTailR[2] = 0.061; gTailR[3] = 0.048; gTailR[4] = 0.034; gTailR[5] = 0.012;
  for(int k=1;k<6;k++){
    float w = float(k)/5.0;
    vec3 v = gTail[k] - gTail[0];
    v.xz = rot(0.40*wag*w*w + 0.9*cu*w*w*w)*v.xz;
    v.y += rs*(0.10 + 0.55*w*w) + 0.010*sin(uTime*1.3 + 2.0*w)*w;
    gTail[k] = gTail[0] + v;
    gTail[k].y = max(gTail[k].y, gTailR[k] + 0.004 - b.y);        // it lies ON the floor, never in it
  }
  for(int k=0;k<6;k++) gTail[k] += b;
  gTailS[0] = gTail[0]; gTailSR[0] = gTailR[0]; gTailS[11] = gTail[5]; gTailSR[11] = gTailR[5];
  for(int k=0;k<5;k++){
    gTailS[2*k + 1] = mix(gTail[k], gTail[k+1], 0.25); gTailSR[2*k + 1] = mix(gTailR[k], gTailR[k+1], 0.25);
    gTailS[2*k + 2] = mix(gTail[k], gTail[k+1], 0.75); gTailSR[2*k + 2] = mix(gTailR[k], gTailR[k+1], 0.75);
  }
  vec3 lo = gTail[0], hi = gTail[0];
  for(int k=1;k<6;k++){ lo = min(lo, gTail[k]); hi = max(hi, gTail[k]); }
  gTailB = vec4(0.5*(lo + hi), 0.5*length(hi - lo) + 0.10);
  // the neck: from the top of the chest into the back of the head (it follows the head)
  gNeckA = upperToLocal(vec3(0.0, 1.320, -0.005));
  gNeckB = gHO + transpose(gHR)*vec3(0.0, -0.085, -0.035);   // low and back: never seen through the open mouth
  // the bow tie: at the base of the throat, on the neck's front, facing between the chest and the face
  vec3 ax = normalize(gNeckB - gNeckA);
  vec3 fw = normalize(gMW*vec3(0.0, 0.0, 1.0) + transpose(gHR)*vec3(0.0, 0.0, 0.6));
  fw = normalize(fw - ax*dot(fw, ax));
  vec3 bx = normalize(cross(ax, fw));
  gBowF = mat3(bx, ax, fw);
  gBowO = mix(gNeckA, gNeckB, 0.30) + fw*0.148;
  vec3 hfw = localToUpper(gHO + transpose(gHR)*vec3(0.0, 0.0, 1.0)) - localToUpper(gHO);
  gGkHeadFw = normalize(hfw.xz + vec2(0.0, 1e-3));
}
void mascotMouthBend(){
  // the long lizard smile: the line turns up toward its corners at rest; a knit brow with no smile turns it down
  float op = smoothstep(0.0, 0.6, max(uP[P_LIP_OPEN], uP[P_MOUTH_OPEN]));
  float frown = clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  float rest = 0.150*mix(0.30, 1.0, smoothstep(0.0, 0.05, uP[P_SMILE]));
  gLBend += LIP_W*((rest + 0.07*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - 0.6*op) - MUSE_FROWN*frown*(1.0 - 0.15*op))*(1.0 - gLRnd);
}

// ---- the head (scaled head space): a tapered wedge — a flat crown, the eyes bulging at its top corners, a long snout
// that narrows to a round tip, a defined lower jaw set a little back under it
float mascotHead(vec3 q){
  float d = sdEllipsoid(q - vec3(0.0, 0.010, -0.100), vec3(0.440, 0.230, 0.420));
  d = smin(d, sdEllipsoid(q - vec3(0.0, 0.150, 0.050), vec3(0.310, 0.165, 0.260)), 0.12);   // the crown rounds up between the eyes            // cranium
  d = smin(d, sdEllipsoid(q - vec3(0.0, -0.160, 0.080), vec3(0.505, 0.265, 0.400)), 0.14);   // cheeks / upper jaw
  d = smin(d, sdEllipsoid(q - vec3(0.0, -0.140, 0.380), vec3(0.300, 0.185, 0.285)), 0.14);   // the snout's short round front
  d = smin(d, sdEllipsoid(q - vec3(0.0, -0.310, 0.080), vec3(0.440, 0.135, 0.430)), 0.10);   // lower jaw: deep, a round chin
  vec3 qa = vec3(abs(q.x), q.yz);
  d = smin(d, length(qa - vec3(0.407, 0.113, 0.005)) - 0.145, 0.050);   // the eye's socket mound: the ball domes out of it
  d = smin(d, sdEllipsoid(qa - vec3(0.355, 0.250, 0.090), vec3(0.160, 0.040, 0.110)), 0.075);  // the brow ridge over the eye                       // the eye's mound (lids)
  // hd: the nostrils are two small sculpted pits on the front of the snout, a soft rim round each
  d = smax(d, -sdEllipsoid(qa - GK_NOS, vec3(0.016, 0.011, 0.030)), 0.012);
  // a shallow dip between the eye mounds and down the snout's bridge
  d += 0.0*smoothstep(0.16, 0.0, qa.x)*smoothstep(-0.05, 0.25, q.y)*smoothstep(-0.10, 0.25, q.z);
  return d;
}
// geo: the trunk is ONE swept body along a curved spine (upper-body space): a varying elliptical section — the belly
// wide and deep and hanging forward low, the chest receding, narrowing up into the neck with no shoulder shelf
float gkTrunk(vec3 u){
  float t = (u.y - 0.58)/0.82;
  t = 0.5*(t + sqrt(t*t + 0.01)); t = 1.0 - 0.5*((1.0 - t) + sqrt((1.0 - t)*(1.0 - t) + 0.01));   // a SMOOTH clamp: no shading line where the profile stops
  float bl = exp(-(t - 0.28)*(t - 0.28)*9.2);                   // the belly's swell
  float cz = 0.010 + 0.050*bl - 0.016*t;
  float rx = 0.105 + 0.050*exp(-(t - 0.35)*(t - 0.35)*5.0);   // wide across the front ...
  float rz = 0.086 + 0.040*bl;                                  // ... and shallow, wider than deep
  // below the hips it rounds off like the bottom of an egg (no cap, no seam with the pelvis)
  float yb = (0.64 - u.y)/0.29;
  float fb = yb > 0.0 ? sqrt(max(1.0 - yb*yb, 0.0025)) : 1.0;
  float br = (1.0 + 0.014*gBreath)*fb;
  rx *= br; rz *= br;
  float e = (length(vec2(u.x/rx, (u.z - cz)/rz)) - 1.0)*min(rx, rz)*0.80;
  float dy = max(0.35 - u.y, u.y - 1.40);
  return dy > 0.0 ? length(vec2(max(e, 0.0), dy)) + min(e, 0.0) : e;
}
// the neck: an elliptical section (narrower across than deep), slim at the chest, flaring to the jaw at the head
float gkNeck(vec3 p){
  vec3 pa = p - gNeckA, ba = gNeckB - gNeckA;
  float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  vec3 v = pa - ba*h;
  vec3 X = gMW[0]; vec3 bn = normalize(ba);
  vec3 sx = normalize(X - bn*dot(X, bn));
  float vx = dot(v, sx); float vz = length(v - sx*vx);
  float rx = mix(0.115, 0.155, h*h), rz = mix(0.105, 0.150, smoothstep(0.0, 0.8, h));
  float d = (length(vec2(vx/rx, vz/rz)) - 1.0)*min(rx, rz);
  // a soft fold across the throat just under the jaw
  vec3 fz = normalize(cross(sx, bn));
  float fr = dot(v, fz)/max(length(v), 1e-4);
  return d + 0.006*exp(-(h - 0.80)*(h - 0.80)*220.0)*smoothstep(0.2, 0.8, fr);
}
float mascotTorso(vec3 p){
  vec3 b = p - (gMHip - M_HIP);
  float lo = sdEllipsoid(b - vec3(0.0, 0.630, -0.010), vec3(0.125, 0.185, 0.105));          // pelvis: low, the long trunk runs down between the thighs
  if(b.y < 0.62 && abs(b.x) < 0.22) lo = smin(lo, sdEllipsoid(b - vec3(0.0, 0.470, 0.000), vec3(0.066, 0.080, 0.085)), 0.05);   // the belly's low V between the thighs (no dark gap); only evaluated near it
  vec3 u = localToUpper(p);
  float d = smin(gkTrunk(u), lo, 0.10);
  d = smin(d, sdEllipsoid(u - vec3(0.0, 1.345, -0.012), vec3(0.108, 0.075, 0.082)), 0.07);  // narrow sloping shoulders
  d = smin(d, gkNeck(p), 0.12);
  return d;
}
float mascotExtra(vec3 p, out vec3 rest){
  float bd = length(p - gTailB.xyz) - gTailB.w;
  if(bd > 0.12){ rest = vec3(0.0); return bd; }                // far from the tail (beyond MR_K_EXTRA, so the smooth join never sees the stand-in)
  float d = 1e3; float along = 0.0, acc = 0.0; vec3 ax = vec3(0.0, 0.0, -1.0), base = gTailS[0];
  for(int k=0;k<11;k++){
    vec3 a = gTailS[k], c = gTailS[k+1];
    vec3 pa = p - a, ba = c - a;
    float L = length(ba);
    float h = clamp(dot(pa, ba)/(L*L), 0.0, 1.0);
    float dk = length(pa - ba*h) - mix(gTailSR[k], gTailSR[k+1], h);
    if(dk < d){ along = acc + h*L; ax = ba/L; base = a + ba*h; }
    d = smin(d, dk, 0.015);
    acc += L;
  }
  // one continuous frame along the tail (up-referenced): the scales do not jump at the segment joints (no banding)
  vec3 e1 = normalize(cross(ax, vec3(0.0, 1.0, 0.0)) + vec3(1e-4)), e2 = cross(e1, ax);
  rest = vec3(along, dot(p - base, e1), dot(p - base, e2));
  return d;
}
// the bow tie: two pinched wings round a small knot, in its own frame (x across, y along the neck, z out)
float bowTie(vec3 b){
  float a = abs(b.x);
  // the knot: a small soft block
  vec3 kq = abs(b - vec3(0.0, 0.0, 0.005)) - vec3(0.010, 0.013, 0.007);
  float k = length(max(kq, 0.0)) + min(max(kq.x, max(kq.y, kq.z)), 0.0) - 0.005;
  // the wings: flat, flaring out from the knot to square-ish ends, a little puffed in the middle, the ends swept back
  float ha = 0.009 + 0.34*clamp(a - 0.008, 0.0, 0.065);
  float dx = a - 0.066;
  float dy = (abs(b.y) - ha)*0.947;
  float d2 = max(dx, dy);
  float th = 0.0045 + 0.0045*smoothstep(0.0, 0.03, a)*smoothstep(0.075, 0.03, a)*(1.0 - abs(b.y)/0.035);
  float dz = abs(b.z + 0.006*smoothstep(0.02, 0.07, a)) - th;
  vec2 w = vec2(d2 + 0.004, dz);
  float dw = length(max(w, 0.0)) + min(max(w.x, w.y), 0.0) - 0.004;
  return smin(dw, k, 0.005);
}
float mascotBare(vec3 p){
#ifdef GK_NO_BOW
  return 1e3;                                      // eval only (AVATAR_RQP_DEFS=GK_NO_BOW=1): the 2AFC test hides the one deliberate difference
#endif
  vec3 b = transpose(gBowF)*(p - gBowO);
  if(dot(b, b) > 0.0144) return length(b) - 0.09;
  return bowTie(b);
}
// the foot: a small pad with five long splayed toes that end in round pads
float museFoot(vec3 p, int i, out float sole){
  vec3 q = transpose(gMFoot[i])*(p - gMAnk[i]);
  float d = sdEllipsoid(q - vec3(0.0, -0.098, 0.030), vec3(0.048, 0.040, 0.060));
  d = smin(d, sdCapsule(q, vec3(0.0, 0.010, -0.005), vec3(0.0, -0.085, 0.020), 0.034), 0.030);   // the ankle runs down into the foot
  float s = i == 0 ? -1.0 : 1.0;
  for(int k=0;k<5;k++){
    float a = (float(k) - 2.0)*0.52 + s*0.12;
    vec3 dir = vec3(sin(a), 0.0, cos(a));
    float len = 0.155 - 0.018*abs(float(k) - 2.0);          // geo: shorter, a touch fuller toes (the real foot is small)
    vec3 t0 = vec3(0.0, -0.108, 0.040) + dir*0.02, t1 = vec3(0.0, -0.128, 0.040) + dir*len;
    float f = sdCapsule(q, t0, t1, 0.0130);
    f = smin(f, sdEllipsoid(q - t1 - vec3(0.0, -0.002, 0.004), vec3(0.019, 0.011, 0.020)), 0.010);   // a flat, slightly wider tip pad
    d = smin(d, f, 0.018);
  }
  d = smax(d, -(q.y + M_ANK_Y - 0.002), 0.008);        // flat on the floor
  sole = smoothstep(-0.128, -0.142, q.y);
  return d;
}

// ---- scales: a jittered cell pattern in the rest frame (it rides the skin).  x = distance to the cell centre 0..1,
// y = a random value per cell.  8 cells: evaluated once per pixel, in the shading only.
vec3 gkOff;          // offset from the nearest cell centre (cell units), set by gkCells
vec2 gkCells(vec3 x){
  vec3 ip = floor(x - 0.5), fp = x - ip;
  float best = 9.0, id = 0.0; vec3 off = vec3(0.0);
  for(int k=0;k<8;k++){
    vec3 g = vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1));
    vec3 h = hash33(ip + g);
    vec3 r = g + 0.27 + 0.46*h - fp;                 // close-packed beads: a little more regular than a free jitter
    float dd = dot(r, r);
    if(dd < best){ best = dd; id = h.x; off = r; }
  }
  gkOff = -off;
  return vec2(sqrt(best), id);
}

#ifdef V3_HI
// hi: the cloud / eval bead cells: a full 27-cell search with free jitter (organic, no grid squares), and the
// bead's normalised radius F1 / ((F1 + F2)/2) (0 at the bead's centre, 1 on the groove midway to its neighbour) so every
// bead fills its own cell however it is shaped.  Returns x = that radius scaled to gkCells' range (0.72 on the groove
// line), y = a random value per bead; gkOff as gkCells.
vec2 gkCellsHi(vec3 x){
  vec3 ip = floor(x), fp = x - ip;
  float b1 = 9.0, b2 = 9.0, id = 0.0; vec3 off = vec3(0.0);
  for(int k=0;k<27;k++){
    vec3 g = vec3(float(k % 3), float((k / 3) % 3), float(k / 9)) - 1.0;
    vec3 h = hash33(ip + g);
    vec3 r = g + 0.12 + 0.76*h - fp;
    float dd = dot(r, r);
    if(dd < b1){ b2 = b1; b1 = dd; id = h.x; off = r; } else if(dd < b2){ b2 = dd; }
  }
  gkOff = -off;
  float f1 = sqrt(b1), f2 = sqrt(b2);
  return vec2(0.72*f1/(0.5*(f1 + f2)), id);
}
#endif
// the trunk and neck's cream front by angle round the body's axis (u = upper-body rest space); hk: also used by the
// head's underside next to the neck, so the jaw hands over to the neck's layout with no seam
float gkTrunkBelly(vec3 u){
    vec3 c = vec3(0.0, u.y, 0.0);
    float nk = smoothstep(1.30, 1.55, u.y);
    vec3 nA = localToUpper(gNeckA), nB = localToUpper(gNeckB);
    float hh = clamp((u.y - nA.y)/max(nB.y - nA.y, 1e-3), 0.0, 1.0);
    c = mix(c, mix(nA, nB, hh), nk);
    vec2 rr = u.xz - c.xz;
    float ang = atan(abs(rr.x), rr.y);                       // 0 straight ahead .. pi straight back
    // up the neck the throat turns with the head (a turned head keeps its cream under the chin, green down the sides)
    vec2 f2 = normalize(mix(vec2(0.0, 1.0), gGkHeadFw, nk*hh));
    vec2 rn = normalize(rr + vec2(0.0, 1e-4));
    ang = mix(ang, acos(clamp(dot(rn, f2), -1.0, 1.0)), nk);
#ifdef V3_HI
    float edge = mix(0.66, 0.50, nk) + 0.08*(vnoise3(u*7.0) - 0.5);       // hk: a calm round-noise edge (the vertically stretched noise drew streaks down the neck)
#else
    float edge = mix(0.66, 0.50, nk) + 0.16*(vnoise3(u*vec3(9.0, 4.0, 9.0)) - 0.5) + 0.06*(vnoise3(u*31.0) - 0.5);
#endif
    return smoothstep(edge + 0.42, edge - 0.34, ang);
}
#ifdef GK_DBG_JW
float gDbgJw = 0.0;
#endif
vec3 gGkHeadAlb;      // the head's own paint before the lips (mascotFacePost restores it where the mouth's paint leaks)
void mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s){
  // the bow tie (a hard worn solid): deep red satin, a fold either side of the knot
  float bd = mascotBare(pw);
  if(bd < 0.003 && bd < gMSkin){
    vec3 b = transpose(gBowF)*(pw - gBowO);
    float a = abs(b.x);
    float fold = smoothstep(0.012, 0.030, a)*smoothstep(0.050, 0.030, a)*smoothstep(0.004, 0.020, abs(b.y));
    float knot = smoothstep(0.020, 0.012, a);
    s.alb = C_BOW*(1.0 - 0.30*fold)*(1.0 + 0.15*knot);
    s.spec = 0.30; s.shin = 40.0; s.wrap = 0.10; s.sheen = 0.25; s.occ = 1.0 - 0.25*smoothstep(0.022, 0.012, a)*(1.0 - knot);
    s.emit = vec3(0.0);
#ifdef LOOK_V3
    s.rough = 0.38; s.coat = 0.0; s.sss = 0.0; s.sheenK = 0.45; s.sheenCol = vec3(1.0, 0.55, 0.55); s.cavity = 1.0 - 0.3*fold;
#endif
    s.trans = 0.0;
    return;
  }
  vec3 u = localToUpper(pw);
  float belly = 0.0, hand = 0.0, cream = 0.0;
  vec3 rs = r;                                   // the scales' coordinate (world units)
  float rsk = 1.0;                               // world units per rest unit
  float mid = 0.0;
  if(gMPart == 0){
    rsk = MR_HEAD_SCALE;
    // under the jaw: cream.  The line runs just under the mouth, rises a little with the smile toward its corners, and
    // back along the jaw into the throat
    float ax = abs(q.x);
    float jl = LIP_O.y - 0.030 + 0.050*smoothstep(0.10, 0.41, ax)*smoothstep(-0.10, 0.25, q.z)
             - 0.080*smoothstep(0.10, -0.30, q.z);
    belly = smoothstep(jl + 0.070, jl - 0.160, q.y + 0.03*(mid - 0.5));
    belly *= 1.0 - smoothstep(0.10, 0.34, ax)*smoothstep(0.10, -0.30, q.z);
#ifdef V3_HI
    // hi: below the mouth the chin's cream narrows toward the throat's stripe (the pale jaw bottom spanned the head's
    // whole width and met the green-sided neck on a hard horizontal seam)
    belly *= 1.0 - smoothstep(jl - 0.04, jl - 0.18, q.y)*smoothstep(GK_HI_THW, GK_HI_THW + 0.14, ax);
#endif
    // hk: toward the neck the jaw's underside takes the neck's own cream / green layout (the hand-over was a seam)
    float jw = smoothstep(jl - GK_JW0, jl - GK_JW1, q.y)*smoothstep(GK_JWZ, GK_JWZ - 0.30, q.z);
    belly = mix(belly, gkTrunkBelly(u), jw);
#ifdef GK_DBG_JW
    gDbgJw = jw;
#endif
    // the upper lip's rim: a thin cream band right over the mouth line, round the snout's tip
    float ul = LIP_O.y + 0.030 + 0.050*smoothstep(0.10, 0.41, ax);
    cream = 0.70*smoothstep(ul + 0.050, ul + 0.000, q.y)*smoothstep(jl - 0.01, jl + 0.03, q.y)*smoothstep(-0.05, 0.25, q.z);
    if(gMFace > 0.5){
      // hd: the upper lip is the snout's own lower edge — the green fades softly to a pale cream over a broad zone
      // above the mouth line (it follows the line's curve up to the corners); below the line the jaw is cream
      float lat = smoothstep(1.85, 1.00, gLU)*smoothstep(-0.12, 0.20, q.z);
      float vu = gLV - gLAu;
      cream = GK_ULIP_K*smoothstep(GK_ULIP_H, 0.30*GK_ULIP_H, vu)*smoothstep(-0.02, 0.01, gLV)*lat;
      float jadd = smoothstep(0.0, -0.035, gLV + gLAl)*smoothstep(-0.30, -0.06, gLV + gLAl)*smoothstep(MR_FACE_RAD, MR_FACE_RAD - 0.16, length(q - MR_FACE_C))*smoothstep(1.75, 0.95, gLU)*smoothstep(-0.05, 0.20, q.z);
#ifdef V3_HI
      jadd *= 1.0 - smoothstep(jl - 0.04, jl - 0.18, q.y)*smoothstep(GK_HI_THW, GK_HI_THW + 0.14, ax);   // the same narrowing toward the throat's stripe
#endif
      belly = max(belly, jadd);
    }
  } else if(gMPart == 1){
    // the trunk and the neck: pale over the front, the line down the flanks (by position round the body's axis, so
    // it is a clean soft line, not a normal-driven smear)
    belly = gkTrunkBelly(u);
    rs = u;
  } else if(gMPart == 2){
    hand = smoothstep(0.70, 1.0, gMLimbT)*0.55;
  } else if(gMPart == 3){
    hand = 0.45; rsk = 1.0/1.6;
  } else if(gMPart == 4){
    hand = smoothstep(0.80, 1.0, gMLimbT)*0.35;
    belly = 0.35*smoothstep(0.3, 0.8, n.z)*smoothstep(0.45, 0.15, gMLimbT);
  } else if(gMPart == 5){
    // the feet: green over the top, the long toes go pale toward their pads
    vec3 fq = r - vec3(9.0 + 2.0*float(gMSide), 0.0, 0.0);
    hand = 0.30 + 0.30*smoothstep(0.08, 0.14, length(fq.xz - vec2(0.0, 0.04)));   // green feet and toes, a little paler toward the toe pads
  } else if(gMPart == 7){
    belly = smoothstep(-0.45, -0.85, n.y)*0.70*smoothstep(0.25, 0.60, r.x);   // the tail's underside (not its root between the legs)
  }
  // scales: tiny, flatter and larger on the belly; faded where a pixel cannot hold them
  float cellW = mix(0.0088, 0.0115, belly);                  // world units
  if(gMPart == 3 || gMPart == 5) cellW = 0.0055;
  if(gMPart == 0){
    // the head (world units after the head's scale): beads largest on the snout and brow, finer on the cheeks and chin,
    // tiny round the eyes and toward the lips
    float rE0 = length(vec3(abs(q.x), q.yz) - gEyeC);
    cellW = mix(0.0064, 0.0080, smoothstep(0.05, 0.35, q.z)*smoothstep(-0.25, 0.05, q.y));
    cellW = mix(cellW, 0.0060, belly);
    cellW *= mix(0.62, 1.0, smoothstep(gEyeR + 0.02, gEyeR + 0.12, rE0));
  }
#ifdef V3_HI
  // hi: finer beads on the cloud / eval sizes (the real head carries ~90 across), resolved down to ~2 internal px
  // A bead smaller than ~GK_HI_MINPX internal pixels is never drawn (it would alias into moss); instead the cells step up
  // by octaves (x2 each, cross-faded, the octave fixed per frame by the camera distance: no swimming) and their relief
  // fades per octave: far away the skin keeps a fine granular sparkle at the pixel's own scale, never flat plastic.
  cellW *= gMPart == 0 ? GK_HI_BEAD0 : GK_HI_BEAD;
  float lvF = clamp(log2((gMPart == 0 ? GK_HI_MINPX0 : GK_HI_MINPX)*max(gMFp, 1e-5)/cellW), 0.0, 1.0), lv0 = floor(lvF), lvT = lvF - lv0;
  float cA = cellW*exp2(lv0);
  vec2 ceA = gkCellsHi(rs*(rsk/cA)); vec3 offA = gkOff;
  vec2 ceB = gkCellsHi(rs*(0.5*rsk/cA) + 17.31); vec3 offB = gkOff;
  vec2 ce = mix(ceA, ceB, smoothstep(0.0, 1.0, lvT)); gkOff = mix(offA, offB, smoothstep(0.0, 1.0, lvT));
  float wS = pow(gMPart == 0 ? GK_HI_OCT0 : GK_HI_OCT1, lvF)*(1.0 - 0.30*belly);
  if(gMPart == 3 || gMPart == 5) wS *= GK_HI_DIGIT;      // hk: hands and feet a calm fine skin (the beads striped the thin digits)
#else
  float wS = smoothstep(gMPart == 0 ? 0.8 : 1.6, gMPart == 0 ? 2.0 : 3.4, cellW/max(gMFp, 1e-5))*(1.0 - 0.40*belly);
#endif
  if(gMPart == 0){
    float rE = length(vec3(abs(q.x), q.yz) - gEyeC);
    wS *= mix(0.15, 1.0, smoothstep(0.0, 0.9, gFOffLips))*smoothstep(gEyeR + 0.030, gEyeR + 0.060, rE);   // hd: smoother skin on and by the lips (no stretched bead streaks)     // smooth skin on the lips and the lids
  }
#ifndef V3_HI
  vec2 ce = gkCells(rs*(rsk/cellW));
#endif
  float dome = 1.0 - smoothstep(0.0, 0.62, ce.x);
  float groove = smoothstep(mix(0.56, 0.62, belly), mix(0.74, 0.80, belly), ce.x);
  // soft large mottling so no area is one flat green; fine flecks that survive at a distance
  vec3 wp = rs*rsk;
  float big = vnoise3(wp*5.0), fine = vnoise3(wp*38.0 + 3.7);
  mid = vnoise3(wp*13.0 + 1.3);
#ifdef V3_HI
  // hi: the colour varies bead by bead (each its own shade), the broad mottling only faint (it read as blotches)
  float tone = clamp(0.52 + GK_HI_MOT*(big - 0.5) + 0.03*(fine - 0.5) + GK_HI_BTONE*(ce.y - 0.5)*wS + 0.06*n.y, 0.0, 1.0);
#else
  float tone = clamp(0.52 + 0.34*(big - 0.5) + 0.03*(fine - 0.5) + 0.10*(ce.y - 0.5)*wS + 0.06*n.y, 0.0, 1.0);
#endif
  vec3 green = mix(C_GREEND, C_GREEN, smoothstep(0.0, 0.55, tone));
  green = mix(green, C_GREENL, smoothstep(0.55, 1.0, tone)*0.8);
  // mottling: soft lighter, yellower spots over the green; the back along the spine a little deeper
  green = mix(green, green*vec3(1.12, 1.10, 0.92), 0.35*smoothstep(0.58, 0.85, big));
  if(gMPart == 1) green *= mix(1.0, 0.86, smoothstep(2.2, 2.9, atan(abs(u.x), u.z - 0.0)));
  vec3 pale = C_BELLY*(0.99 + 0.025*(big - 0.5) + 0.015*(ce.y - 0.5)*wS);
  // the flank: a narrow lighter yellow-green band where green meets the belly
  float flank = belly*(1.0 - belly)*4.0;
  green = mix(green, C_GREENL*vec3(1.12, 1.08, 0.92)*(0.95 + 0.08*mid), 0.60*smoothstep(0.0, 0.45, belly)*step(0.5, float(gMPart == 1 || gMPart == 7)));   // the green brightens to yellow-green toward the cream
  // limbs and tail: the undersides go a lighter yellow-green (soft, by the normal)
  if(gMPart == 2 || gMPart == 4 || gMPart == 7) green = mix(green, C_GREENL*vec3(1.08, 1.06, 0.92), 0.55*smoothstep(0.25, -0.65, n.y));
  // the real one's form light: lit tops a fresh lighter green, the lower sides a deeper olive (a soft gradient by the normal)
  green *= mix(vec3(0.66, 0.68, 0.62), vec3(1.12, 1.20, 1.12), smoothstep(-0.65, 0.75, n.y));
  vec3 alb = mix(green, pale, belly);
  alb = mix(alb, mix(C_GREENL*vec3(1.10, 1.06, 0.92), C_BELLY, 0.40), 0.40*flank);   // a yellow-green light along the boundary
  alb = mix(alb, GK_ULIP_COL*(0.95 + 0.08*mid), cream);   // hd: a warmer, paler cream on the lip
  if(gMPart == 0){
    // the head's zones: a fresh light green over the top, yellower down the snout toward the lips (soft, faded at the back
    // of the head so it meets the neck's green without a seam)
    float hf = smoothstep(-0.40, -0.05, q.z);
    alb = mix(alb, alb*vec3(1.06, 1.0, 0.92), hf*smoothstep(-0.15, 0.20, q.y));
    float yl = hf*smoothstep(0.10, -0.20, q.y)*smoothstep(0.05, 0.35, q.z)*(1.0 - cream);
    alb = mix(alb, vec3(0.200, 0.250, 0.140)*(0.96 + 0.06*mid), 0.45*yl);
    alb = mix(alb, mix(C_BELLY, C_CREAM*1.12, 0.65*smoothstep(0.15, 0.45, q.z)), belly);   // the chin's cream only toward the lips: under the jaw it is the throat's colour (no seam)   // the chin: a warmer cream, the throat as the belly
  }   // the chin: a warmer cream, the throat as the belly
  alb = mix(alb, C_HAND*(0.94 + 0.10*mid), hand);
  if(gMPalmW > 0.0 || gMSoleW > 0.0) alb = mix(alb, C_HAND*vec3(0.98, 0.95, 0.90), 0.7*max(gMPalmW, gMSoleW));
  // the crown patch: a soft orange oval between and a little behind the eyes, fading out at its edges
  float blaze = 0.0;
  if(gMPart == 0){
    // c: on the forehead between and just behind the eyes, seen from the front; its edge breaks up scale by scale
    // hd: a broad V — wide over the crown, narrowing forward between the eyes toward the snout
    // (seen from the front: its tip at GK_BLZ_Y0 between the eyes, widening up over the brow; then back over the crown)
    float ty = q.y - GK_BLZ_Y0 + max(GK_BLZ_Z1 - q.z, 0.0)*0.55;   // distance up from the tip, continuing back over the top
    float bw = mix(GK_BLZ_WF, GK_BLZ_WB, smoothstep(0.0, 0.26, ty));
    float e2 = lipsSq(q.x/bw) + lipsSq(max(ty - 0.30, 0.0)/0.16) + 0.45*(ce.y - 0.5)*wS + 0.25*(mid - 0.5);
    blaze = smoothstep(0.95, 0.30, e2)*smoothstep(0.0, 0.07, ty)*smoothstep(GK_BLZ_Z0, GK_BLZ_Z0 + 0.12, q.z);
    alb = mix(alb, GK_BLZ_COL*(0.94 + 0.06*ce.y*wS + 0.03*fine), GK_BLZ_K*blaze);
  }
  // the grooves between scales are darker, the domes catch light
  alb *= (1.0 - (0.14 - 0.06*belly)*groove*wS)*(1.0 + (0.08 - 0.03*belly)*dome*wS);
#ifdef V3_HI
  alb *= 1.0 + GK_HI_LIFT*wS;                        // hi: the grooves' extra darkening (cavity) taken back on average
#endif
  float lid = 0.0;
  if(gMPart == 0){
    // the lids: a pale cream-pink ring of fine skin round each eye
    vec3 ev = vec3(abs(q.x), q.yz) - gEyeC;
    float re = length(ev);
    lid = smoothstep(gEyeR + 0.026, gEyeR + 0.014, re)*smoothstep(-0.05, 0.05, ev.z + 0.07);
    alb = mix(alb, alb*vec3(1.10, 1.08, 1.0), 0.50*lid*(1.0 - blaze));   // smooth green lids, a faintly lighter edge
    // hd: the lower lid's thin pale rim (pale yellow-green skin right at the eye, seen on every close-up)
    float lrim = smoothstep(gEyeR + 0.036, gEyeR + 0.012, re)*smoothstep(0.06, -0.08, ev.y);
    alb = mix(alb, GK_ULIP_COL*vec3(0.95, 0.97, 0.95), 0.75*lrim*(1.0 - blaze));
    // nostrils: two small dark slits on top of the snout's tip
    vec2 nv = vec2(abs(q.x) - GK_NOS.x, (q.y - GK_NOS.y)*1.25);
    float no = smoothstep(0.019, 0.006, length(nv))*smoothstep(0.30, 0.42, q.z);
    alb = mix(alb, vec3(0.030, 0.040, 0.016), 0.80*no);
    s.occ *= 1.0 - 0.5*no;
  }
  s.alb = alb;
#ifdef GK_DBG_JW
  if(gMPart == 0) s.alb = mix(s.alb, vec3(0.8, 0.0, 0.0), gDbgJw);
#endif
  gGkHeadAlb = alb;
  // relief: each scale is a small dome (the normal leans away from the cell centre toward the groove)
  vec3 lean = gkOff - n*dot(gkOff, n);
  lean *= smoothstep(0.10, 0.45, abs(dot(n, normalize(uCamPos - pw))));   // no relief on the silhouette (jagged edge)
#ifdef V3_HI
  // hi: each bead a true rounded cap (a sphere-cap height field: the slope grows toward the bead's edge and
  // turns over into the groove), so every bead catches its own highlight and its own shadowed side
  float rn = ce.x/0.72;                               // 0 the bead's centre .. 1 the groove line
  float slope = rn/sqrt(max(1.0 - rn*rn, 0.09))*smoothstep(1.0, 0.86, rn);
  float hk = gMPart == 0 ? GK_HI_HK0*(1.0 - 0.80*belly) : GK_HI_HK1*(1.0 - 0.55*belly);   // the chin and throat: smooth fine skin
  vec3 ldir = lean/max(length(gkOff), 1e-3);
  s.n = normalize(n + hk*min(slope, 2.2)*ldir*wS);
#else
  s.n = normalize(n + (gMPart == 0 ? 1.0 : 0.5)*lean*(1.00 - 0.55*belly)*smoothstep(0.10, 0.62, ce.x)*wS);
#endif
  s.occ = (1.0 - 0.14*groove*wS)*mix(0.50 + 0.45*belly, 1.0, smoothstep(-0.75, 0.55, n.y));   // soft top light: the undersides take less of the room
#ifdef V3_HI
  if(gMPart == 0) s.occ *= mix(GK_HI_CHIN, 1.0, smoothstep(-0.85, 0.05, n.y));   // hi: the jaw's underside turns away from the room (the cream chin was as bright as a lit top: a pasted-on band over the throat)
#endif
  s.wrap = 0.26; s.sheen = 0.10;
  s.spec = mix(0.42, 0.20, belly)*(0.55 + 0.60*dome*wS + 0.45*(1.0 - wS)); s.shin = 40.0;
  s.emit = alb*vec3(0.018, 0.020, 0.014);
#ifdef LOOK_V3
  s.rough = mix(0.40, 0.62, groove*wS) + 0.06*belly + 0.06*(1.0 - wS);
  s.coat = 0.0;
  s.sss = mix(0.38, 0.44, max(belly, hand)); s.sssCol = mix(vec3(0.70, 0.85, 0.35), vec3(1.0, 0.78, 0.55), max(belly, hand));
  s.sheenK = 0.30; s.sheenCol = vec3(0.88, 1.0, 0.82);
  s.cavity = 1.0 - 0.22*groove*wS;
#ifdef V3_HI
  // hi: a wet tight lobe on each bead's crown over the broad skin lobe; the grooves dry, rough and dark;
  // beads too small for the pixel widen the lobe instead (their slopes' spread), never vanish into flat plastic
  float crown = (1.0 - smoothstep(0.0, 0.45, ce.x))*wS;
  s.rough = mix(GK_HI_RD, 0.62, groove*wS) + 0.05*belly + 0.10*(1.0 - wS);
  s.coat = GK_HI_COAT*crown*(1.0 - 0.6*belly)*(0.6 + 0.8*ce.y);
  s.cavity = 1.0 - GK_HI_CAV*groove*wS;
#endif
#endif
  // thin parts glow a little with the light behind them: fingers, toes, the tail's tip
  float thin = max(hand, gMPart == 7 ? smoothstep(1.2, 2.2, r.x - 17.0) : 0.0);
  s.trans = 0.10 + 0.16*thin; s.transCol = vec3(0.75, 0.85, 0.40);
  // contact: the feet and the tail sit in a soft dark core where they meet the floor
  if(gMPart == 5 || gMPart == 7){
    float cg = smoothstep(0.003, 0.030, pw.y);
    s.occ *= mix(0.70, 1.0, cg);
  }
}
// after the face's own paint: the eye is a dark wet bead — the room's reflection on it stays soft (a sharp mirror of the
// set drew olive swirls across the iris), one crisp catch-light (face.glsl) on top
void mascotFacePost(vec3 q, vec3 n, inout Surf s){
  // the mouth's opening is painted by its 2D outline: well behind the corners it would mark the cheeks (a dark gash on
  // the side of the head in profile) — give that skin back its own colour
  float bk = smoothstep(LIP_O.z - 0.36, LIP_O.z - 0.48, q.z)*step(gMSkin + 0.002, gFEye);
  if(bk > 0.0){
    s.alb = mix(s.alb, gGkHeadAlb, bk); s.emit = mix(s.emit, gGkHeadAlb*vec3(0.018, 0.020, 0.014), bk);
    s.occ = mix(s.occ, max(s.occ, 0.85), bk);
  }
  // c: the lips are one soft pale cream roll — calm the engine's wet / rolled-under bands (they read as stripes)
  float lipK = gPLip*(1.0 - gPLine)*(1.0 - gPCave);
  // a soft cream edge that fades out into the skin (the engine's lip mask is a hard edge: an outlined, pasted-on oval)
  // hd: the lips ARE the snout's and the jaw's cream (the material's gradient), a touch deeper toward the inner edge
  float lipIn = smoothstep(0.55, 0.0, gLE/max(gLH, 1e-3));
  s.alb = mix(s.alb, gGkHeadAlb*mix(1.0, 0.80, lipIn), 0.95*lipK);
  s.n = normalize(mix(s.n, n, lipK));                                   // no scale relief on the lip roll
  // hd: the mouth line is a real crease: a soft shade either side of it (the upper lip overhangs a little)
  float crz = exp(-lipsSq(gLE/(0.014*LIPU)))*(1.0 - gPCave)*(1.0 - gPTongue)*smoothstep(1.12, 0.92, gLU);
  s.alb *= 1.0 - 0.38*crz*mix(0.6, 1.0, smoothstep(0.0, 0.01, -gLV));
  // past each corner the crease turns up into a small soft cheek dimple (the line never just stops)
  vec2 dq = vec2((gLU - 1.04)/0.10, (gLV - 0.030*LIPU)/(0.040*LIPU));
  dq = vec2(dq.x - 0.5*dq.y, dq.y);
  s.alb *= 1.0 - 0.22*exp(-dot(dq, dq))*(1.0 - gPCave)*smoothstep(0.004*LIPU, 0.0, gLAu + gLAl - 0.004*LIPU);
  float halo = max(lipK, smoothstep(0.75, 0.0, gFOffLips)*(1.0 - gPCave)*(1.0 - gPTongue));   // no scale grooves on or right by the lips
  s.n = normalize(mix(s.n, n, halo)); s.occ = mix(s.occ, max(s.occ, 0.92), halo);
  // hd: round an open mouth the carved field is no true distance and its marched normal comes out in onion rings
  // (streaks over the upper lip): there the skin takes the plain host's smooth normal instead
  float hz = smoothstep(0.004*LIPU, 0.020*LIPU, gLAu + gLAl)*(1.0 - gPCave)*(1.0 - gPTongue)*smoothstep(0.12*LIPU, 0.06*LIPU, gLE)*smoothstep(-0.01*LIPU, 0.01*LIPU, gLV)*smoothstep(1.25, 1.0, gLU);
  if(hz > 0.0){
    const vec2 k = vec2(1.0, -1.0)*0.006;
    vec3 nh = normalize(k.xyy*mascotHead(q + k.xyy) + k.yyx*mascotHead(q + k.yyx) + k.yxy*mascotHead(q + k.yxy) + k.xxx*mascotHead(q + k.xxx));
    s.n = normalize(mix(s.n, transpose(gHR)*nh, hz));
    s.occ = mix(s.occ, max(s.occ, 0.90), hz);
#ifdef LOOK_V3
    s.cavity = mix(s.cavity, 1.0, hz);
#endif
  }
#ifdef LOOK_V3
  s.sssCol = mix(s.sssCol, vec3(0.95, 0.90, 0.60), gPLip); s.sss = mix(s.sss, 0.30, gPLip);   // cream, not an orange-red glow
#endif
#ifdef LOOK_V3
  s.cavity = mix(s.cavity, 1.0, halo);
#endif
  // the open mouth: dark, with a soft pink tongue filling its floor (the real one always shows it when talking)
  // the hole's wall outside the lip ring is skin, not the mouth's dark red (it drew a maroon outline round the lips)
  float wo = gPCave*smoothstep(0.0, 0.004*LIPU, gLE)*(1.0 - gPLip);
  s.alb = mix(s.alb, gGkHeadAlb, wo); s.emit = mix(s.emit, gGkHeadAlb*vec3(0.018, 0.020, 0.014), wo); s.occ = mix(s.occ, 0.85, wo);
  float tg = gPCave*(1.0 - wo)*(1.0 - gPLip)*smoothstep(-0.05*gLAl, -0.55*gLAl, gLQ.y)*smoothstep(0.95*gLW, 0.45*gLW, abs(gLQ.x));
  s.alb = mix(s.alb, LIP_COL_TONGUE*0.12, tg);
  s.emit = mix(s.emit, LIP_COL_TONGUE*mix(0.40, 0.16, smoothstep(-0.30*gLAl, -0.9*gLAl, gLQ.y)), tg);
  if(gFEye < 0.002 && gFEye <= gMSkin + 0.002){
    // the real one's eye: a wet glass dome over a warm amber iris with radial depth, a huge round pupil, a dark limbus
    // (no white), and the room's windows mirrored sharply on top
    float sx = q.x < 0.0 ? -1.0 : 1.0;
    vec3 ce = vec3(sx*EYE_C.x, EYE_C.y, EYE_C.z);
    vec3 toCam = rootGaze(gHR*(uCamPos - gHO)) + SHAPE_PIVOT - ce;
    vec3 eyeF = normalize(normalize(toCam) + vec3(gGaze.x*0.28, gGaze.y*0.22, 0.0));
    vec3 ex = normalize(cross(vec3(0,1,0), eyeF)), ey = cross(eyeF, ex);
    vec3 ee = (q - ce)/EYE_R;
    vec2 uv = vec2(dot(ee, ex), dot(ee, ey));
    float fz = dot(ee, eyeF);
    float r = length(uv);
    // c: a very large pupil inside a NARROW warm amber ring with radial fibres, a dark limbus
    const float IR = 0.93, PR = GK_PUPIL;
    float rn = clamp((r - PR)/(IR - PR), 0.0, 1.0);
    float ang = atan(uv.y, uv.x);
    float fib = vnoise2(vec2(ang*18.0, r*3.0)) + 0.5*vnoise2(vec2(ang*47.0, r*9.0 + 2.0));
    vec3 iris = mix(vec3(0.16, 0.050, 0.016), vec3(0.50, 0.200, 0.060), clamp(0.05 + 0.70*fib*fib, 0.0, 1.0));   // warm amber with radial streaks
    iris *= (0.50 + 0.60*smoothstep(0.55, -0.75, uv.y))*(0.80 + 0.40*vnoise2(vec2(ang*5.0, 1.7)));   // lit low, shaded under the lid
    iris *= mix(1.0, 0.18, smoothstep(0.55, 1.0, rn));                         // the dark limbal edge
    iris = mix(iris, vec3(0.16, 0.045, 0.012), smoothstep(0.18, 0.0, rn)*0.8); // a dark collar round the pupil
    float irisM = smoothstep(IR + 0.03, IR - 0.04, r)*step(0.0, fz);
    float pup = smoothstep(PR + 0.020, PR - 0.020, r);
    vec3 c = mix(vec3(0.040, 0.014, 0.008)*(0.6 + 0.5*fib), iris, irisM);   // past the iris: a deep brown ball, never white
    c = mix(c, vec3(0.012, 0.006, 0.006), pup*irisM);
    s.alb = c;
    // the room mirrored on a wet dome: reflect the view about the ball's own normal and look that direction up in a
    // simple soft environment (a big bright window up and to the key side with a gradient, a sky sheet, a faint horizon,
    // a warm floor, a small second window) — it slides over the eye as the camera or the head moves
    float lidL = smoothstep(0.95, 0.60, ee.y)*smoothstep(0.0, 0.025, gFLid);
    vec3 Vh = normalize(rootGaze(gHR*(uCamPos - gHO)) + SHAPE_PIVOT - q);
    vec3 Nh = normalize(ee);
    vec3 Rh = reflect(-Vh, Nh);
    float nv = clamp(dot(Nh, Vh), 0.0, 1.0);
    vec3 wd = normalize(vec3(-0.45, 0.42, 0.79));
    vec3 wx = normalize(cross(vec3(0.0, 1.0, 0.0), wd)), wy = cross(wd, wx);
    vec2 wa = vec2(dot(Rh, wx), dot(Rh, wy))/max(dot(Rh, wd), 0.05);
    vec2 wq = abs(wa - vec2(0.0, 0.10)) - vec2(0.70, 0.42);
    float wdist = length(max(wq, 0.0)) + min(max(wq.x, wq.y), 0.0);
    float win = smoothstep(0.30, -0.20, wdist)*step(0.0, dot(Rh, wd));
    win *= 0.30 + 0.70*smoothstep(-0.45, 0.40, wa.y);                              // brighter toward its top
    win *= 0.62 + 0.38*smoothstep(-0.10, 0.25, sin(wa.x*7.0 + 2.0*wa.y + 1.0));   // what stands outside it: darker shapes across the panes
    win *= mix(1.0, 0.55, step(0.0, wa.x - 0.05));                                // the two panes are lit differently
    win *= 1.0 - 0.60*smoothstep(0.060, 0.025, abs(wa.x + 0.05))*step(abs(wa.y), 0.38);   // a soft mullion
    vec3 wd2 = normalize(vec3(0.62, 0.20, 0.76));
    float win2 = smoothstep(0.9965, 0.9990, dot(Rh, wd2));
    float sky = smoothstep(-0.05, 0.60, Rh.y);
    float hor = smoothstep(0.06, 0.0, abs(Rh.y + 0.08));
    float flo = smoothstep(-0.10, -0.60, Rh.y);
    float fres = 0.10 + 0.90*pow(1.0 - nv, 4.0);
    vec3 env = vec3(0.84, 0.88, 0.95)*GK_EYE_WIN*win + vec3(0.90, 0.92, 0.95)*0.90*win2
             + vec3(0.42, 0.46, 0.50)*0.30*sky + vec3(0.40, 0.33, 0.26)*0.08*hor + vec3(0.30, 0.17, 0.09)*0.05*flo;
    s.emit = env*mix(0.55, 1.0, fres)*lidL + s.alb*0.05;
#ifdef LOOK_V3
    s.rough = 0.06; s.coat = GK_EYE_COAT*smoothstep(0.98, 0.70, ee.y);
    s.cavity = mix(0.40, 1.0, smoothstep(0.0, 0.035, gFLid))*mix(0.75, 1.0, smoothstep(0.85, 0.20, ee.y));
#endif
  }
}
#pragma include mascot_body

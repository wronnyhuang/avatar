// ---------------------------------------------------------------------------
//  animals/muse.glsl — "Muse" (body, look): a plush mascot after Meta's Muse "Jolly" — one soft rounded
//  toy figure (a big round head on a soft pear body), a warm oatmeal shaggy pile hooding a smooth bare peach-cream
//  face panel, black bead eyes, dark brows (engine/face.glsl), soft salmon blush, a dark line mouth that lip-syncs
//  (engine/lips.glsl), little rounded
//  ears, mitten hands with a thumb and four short fingers, two short legs with round feet.  It STANDS.
//
//  CHARACTER-LOCAL space (engine/scene_main.glsl's root transform maps the world into it): origin on the floor
//  under the character, +x screen right, +y up, +z toward the camera; scene units, H = 2.30 = floor to the top of
//  the head (ears excluded).  Joints (renderer/animals/muse.body.json has the numbers):
//    pelvis  M_HIP (+ body_bounce)            the legs hang from it; two-bone IK to the ankle targets
//    waist   M_WAIST, pelvis space            body_yaw / body_pitch / body_lean bend the upper body over the hips
//    neck    M_NECK, upper-body space         head_* turn the head
//    arms    shoulders in upper-body space    two-bone IK to the hand targets (contract docs/RIG.md hand_*)
//    feet    ankle = rest ankle + foot_{l,r}_{x,y,z} (WORLD axes, H; mapped into local through the root)
//  For the props agent (char-local): museGrasp(side, pos, frame), museHeadFrame(pos, frame); side 0 = screen-left.
// ---------------------------------------------------------------------------
#define M_H 2.30
const vec3 M_HIP   = vec3(0.0, 0.54, 0.0);       // pelvis origin (rest)
const vec3 M_WAIST = vec3(0.0, 0.70, 0.0);       // waist pivot
const vec3 M_NECK  = vec3(0.0, 1.12, 0.0);       // neck pivot, upper-body space
const vec3 M_HEADC = vec3(0.0, 1.64, 0.03);      // head centre, upper-body space at rest
const vec3 M_HEADR = vec3(0.70, 0.64, 0.62);     // head ellipsoid radii (head space)
const float M_UPPER = 0.30, M_FORE = 0.27;       // arm bones
const float M_THIGH = 0.215, M_SHIN = 0.215;     // leg bones
const float M_ANK_Y = 0.145;                     // ankle height at rest (sole on the floor)

// face.glsl / lips.glsl: head space = origin at the head centre (SHAPE_PIVOT 0), +z out of the face
#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.64, 0.03)
#define EYE_R 0.074
#define EYE_C vec3(0.228,0.060,0.565)
#define EYE_AU 0.95
#define EYE_AL (-0.95)
#define EYE_AC (-0.05)
#define EYE_LID_TH 0.0085
#define EYE_SQUINT 0.50
#define EYE_IRIS 0.97
#define EYE_PUPIL 0.62
#define EYE_COL_A vec3(0.030,0.022,0.020)
#define EYE_COL_B vec3(0.010,0.008,0.008)
#define EYE_COL_W vec3(0.020,0.016,0.015)
#define LINER_COL vec3(0.13,0.085,0.07)            // style: a dark lash line on the pale face panel
#define BROW_C vec3(0.235,0.245,0.0)
#ifdef LOOK_V2            // look: a glossy bead eye (a clear catch-light, a little warm depth low in the iris)
#define EYE_GLOW vec3(0.050,0.026,0.012)
#define EYE_CATCH 0.930
#endif
#define BROW_L 0.098                  // style: longer, bolder, dark brows on the bare face panel (read at 270p)
#define BROW_W 0.027
#define BROW_COL vec3(0.075,0.045,0.035)
#define BROW_H 0.016
#define BROW_RISE 0.075
#define MUSE_FROWN 0.42             // r2-joy: corners down per unit knit (smile-less)
#define BROW_SAD_TILT 1.4          // style: knit + raised brows = worried / sad (inner ends up)
#define LIP_O vec3(0.0,-0.205,0.578)
#define LIP_TILT 0.10
#define LIP_W 0.125
#define LIP_HU (0.040*LIPU)
#define LIP_HL (0.048*LIPU)
#define LIP_TU (0.012*LIPU)
#define LIP_TL (0.014*LIPU)
#define LIP_BOW 0.0
#define LIP_SQUARE 0.30
#define LIP_OPEN_Y 1.45
#define LIP_PUSH (0.070*LIPU)
#define LIP_SKIN 0.55
#define LIP_BLEND (0.012*LIPU)
#define LIP_TOOTH_Z (-0.20*LIPU)
#define LIP_TEETH_H (0.010*LIPU)
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(0.15,0.055,0.050)          // style: soft rose lips close to the panel, a dark closed-mouth line
#define LIP_COL_L vec3(0.19,0.075,0.065)
#define LIP_COL_LINE vec3(0.035,0.015,0.015)
#define LIP_LINE 2.3
#define LIP_LINE_SOFT 0.40
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.22,0.05,0.06)
#define LIP_COL_DEEP vec3(0.045,0.012,0.018)
#define LIP_COL_GUM  vec3(0.34,0.10,0.12)
#define LIP_COL_TONGUE vec3(0.86,0.42,0.46)
#define LIP_COL_TOOTH vec3(0.34,0.10,0.12)
#define LIP_SPEC 0.18
#define LIP_SHIN 60.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.05
#define LIP_WRAP 0.30
#define LIP_EMIT vec3(0.10,0.05,0.05)
#define LIP_TONGUE_FREE 0.11            // tongue_out hangs a wide rounded tip ~0.07 below the lower lip (lips.glsl)
#pragma include face
#pragma include lips

// style (after the real Jolly): a warm oatmeal-beige shaggy pile (darker at the roots,
// pale at the tips) hooding a smooth bare peach-cream FACE PANEL; black bead eyes, dark brows and mouth line, soft
// salmon blush; taupe felt pads on palms and soles.
const vec3 C_PLUSH = vec3(0.52,0.39,0.27);       // pile roots / the coat in its own shade
const vec3 C_PLUSH2 = vec3(0.74,0.60,0.45);      // the tips catch light paler
const vec3 C_FACE = vec3(0.84,0.64,0.47);          // the bare face panel (peach cream, smooth)
const vec3 C_BLUSH = vec3(0.92,0.36,0.36);
const vec3 C_SOLE = vec3(0.42,0.30,0.25);          // a taupe felt pad under each foot / on each palm

// ---- pose (charInit)
mat3 gMW;            // waist: upper-body space -> local rotation
vec3 gMHip;          // pelvis origin, local
vec3 gMSh[2], gMEl[2], gMWr[2];      // arm joints, local (0 = screen-left)
vec3 gMHipJ[2], gMKn[2], gMAnk[2];   // leg joints, local
mat3 gMFoot[2];                       // foot frame (local): x across, y up, z toward the toes
mat3 gMHand[2];                       // hand frame: x across the palm, y toward the fingertips, z out of the palm
vec3 gMPalm[2];                       // palm centre
float gMCurl[10];                     // per side: index, middle, ring, pinky, thumb (0 straight .. 1 curled)
float gMSpread[2];
vec3 gMBC[2]; float gMBR[2];          // hand bounding spheres
float gBreath;
// ears: query rotation about the ear's base (mirrored head space, 0 = screen-left) and how far it stands out.
// fix: two named globals, never an array indexed at run time — the K620's GLSL compiler rendered the q.x > 0 half
// of the head black with gMEar[es] (CPU renders were fine; the GPU path showed a half-black head).
mat3 gMEar0, gMEar1;
float gMEarL0, gMEarL1;
const vec3 M_EAR_C = vec3(0.40, 0.50, -0.06);        // ear centre (mirrored head space, x = |x|)
const vec3 M_EAR_PIV = vec3(0.336, 0.410, -0.06);    // where it grows from: centre - 0.11 * its long axis (0.581, 0.814)

// ---- eval outputs
float gMSkin, gMBare, gMFurW, gMHeadD, gMSoleW, gMPalmW, gMFace;
float gMPanel, gMHood, gMCrease;     // style: face panel weight, the hood's rim round it, smooth-join crease
vec3  gMRest, gMPole, gQ;
int   gMPart;          // 0 head, 1 body, 2 arm, 3 hand, 4 leg, 5 foot

vec3 mPerp(vec3 d){ vec3 a = abs(d.y) < 0.9 ? vec3(0,1,0) : vec3(1,0,0); return normalize(cross(d, a)); }
mat3 mAxis(vec3 ax, float a){
  ax = normalize(ax); float c = cos(a), s = sin(a), t = 1.0 - c;
  return mat3(t*ax.x*ax.x + c, t*ax.x*ax.y + s*ax.z, t*ax.x*ax.z - s*ax.y,
              t*ax.x*ax.y - s*ax.z, t*ax.y*ax.y + c, t*ax.y*ax.z + s*ax.x,
              t*ax.x*ax.z + s*ax.y, t*ax.y*ax.z - s*ax.x, t*ax.z*ax.z + c);
}
// two-bone IK: root a, target t (clamped to reach), bone lengths, pole direction -> the middle joint
vec3 mIK(vec3 a, inout vec3 t, float l1, float l2, vec3 pole){
  vec3 d = t - a;
  float L = length(d);
  float lmax = (l1 + l2)*0.995, lmin = abs(l1 - l2) + 0.02;
  vec3 dir = d/max(L, 1e-4);
  L = clamp(L, lmin, lmax);
  t = a + dir*L;
  float ca = clamp((l1*l1 + L*L - l2*l2)/(2.0*l1*L), -1.0, 1.0);
  vec3 pp = pole - dir*dot(pole, dir);
  pp = dot(pp, pp) > 1e-6 ? normalize(pp) : mPerp(dir);
  return a + l1*(ca*dir + sqrt(1.0 - ca*ca)*pp);
}
vec3 upperToLocal(vec3 u){ return gMW*(u - M_WAIST) + M_WAIST + (gMHip - M_HIP); }
vec3 localToUpper(vec3 p){ return transpose(gMW)*(p - (gMHip - M_HIP) - M_WAIST) + M_WAIST; }

void charInit(){
  faceInit(); lipsInit();
  // a resting upturned smile: the mouth line bends up a little even at smile 0 (lips.glsl's own bend is small)
  float op = smoothstep(0.0, 0.6, max(uP[P_LIP_OPEN], uP[P_MOUTH_OPEN]));
  // r2-joy: a knit brow over a smile-less mouth turns the corners DOWN (sad, worried, cross) — the resting bend kept
  // it smiling through grief. Only where the smile is low; smile >= 0.5 or no knit = exactly the old bend.
  float frown = clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  // r2-joy R2: the resting bend is the idle smile (idle smile ~0.08-0.11 → exactly the old bend); a feeling that takes
  // the smile BELOW idle (sympathy, worry, alarm, a plain serious face) flattens the line, so it stops reading cheerful
  float rest = 0.13*mix(0.25, 1.0, smoothstep(0.0, 0.05, uP[P_SMILE]));
  // the frown holds while it TALKS (a4-a7: concern showed only in the pauses; the open mouth halved it and the spread
  // visemes bent the corners back up)
  gLBend += LIP_W*((rest + 0.06*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - 0.6*op) - MUSE_FROWN*frown*(1.0 - 0.15*op))*(1.0 - gLRnd);
  gBreath = uP[P_BREATH];
  // pelvis: body_bounce = vertical offset of everything above the feet (the knees take it)
  gMHip = M_HIP + vec3(0.0, 0.07*clamp(uP[P_BODY_BOUNCE], -1.0, 1.0), 0.0);
  // waist: the upper body over the hips
  gMW = rotY(0.40*uP[P_BODY_YAW])*rotX(0.28*uP[P_BODY_PITCH])*rotZ(-0.20*uP[P_BODY_LEAN]);
  // head: turns about the neck, carried by the waist
  mat3 F = rotY(uP[P_HEAD_YAW])*rotX(uP[P_HEAD_PITCH])*rotZ(-uP[P_HEAD_ROLL]);
  vec3 hc = M_NECK + F*(M_HEADC - M_NECK) + 0.30*vec3(uP[P_HEAD_X], uP[P_HEAD_Y], uP[P_HEAD_Z]);
  hc.y += 0.006*gBreath;
  gHO = upperToLocal(hc);
  gHR = transpose(gMW*F);
  // ---- ears: small round ears that move.  In the mirrored head space (+x = outward for both): ear_l/ear_r
  // (-1 droop .. +1 perk, contract) swing the ear in the face plane about its base (+ = up and in, - = down and out)
  // with a little forward / back tilt; ear_flick_* twitches it out and turns it; ear_*_turn rolls it about its own axis
  {
    float ea = clamp(uP[P_EAR_L], -1.0, 1.0), ef = clamp(uP[P_EAR_FLICK_L], 0.0, 1.0), et = clamp(uP[P_EAR_L_TURN], -1.0, 1.0);
    gMEar0 = transpose(mAxis(vec3(0.0, 0.0, 1.0), 0.85*ea - 0.55*ef)*mAxis(vec3(1.0, 0.0, 0.0), 0.30*ea)
             *mAxis(vec3(0.581, 0.814, 0.0), 0.70*et + 0.35*ef));
    gMEarL0 = 0.045*ea + 0.015*ef;                     // a perked ear stands a little further out of the head
  }
  {
    float ea = clamp(uP[P_EAR_R], -1.0, 1.0), ef = clamp(uP[P_EAR_FLICK_R], 0.0, 1.0), et = clamp(uP[P_EAR_R_TURN], -1.0, 1.0);
    gMEar1 = transpose(mAxis(vec3(0.0, 0.0, 1.0), 0.85*ea - 0.55*ef)*mAxis(vec3(1.0, 0.0, 0.0), 0.30*ea)
             *mAxis(vec3(0.581, 0.814, 0.0), 0.70*et + 0.35*ef));
    gMEarL1 = 0.045*ea + 0.015*ef;
  }
  // ---- arms (side 0 = screen-left, x < 0)
  for(int i=0;i<2;i++){
    float s = i == 0 ? -1.0 : 1.0;
    float hx = i == 0 ? uP[P_HAND_L_X] : uP[P_HAND_R_X];
    float hy = i == 0 ? uP[P_HAND_L_Y] : uP[P_HAND_R_Y];
    float hz = i == 0 ? uP[P_HAND_L_Z] : uP[P_HAND_R_Z];
    float sh = clamp(i == 0 ? uP[P_SHOULDER_L] : uP[P_SHOULDER_R], 0.0, 1.0);
    vec3 S = vec3(s*0.405, 1.035 + 0.10*sh, 0.0);
    float L = M_UPPER + M_FORE;
    vec3 rest = vec3(s*0.540, 0.555, 0.130);
    // contract: x outward (negative = across the chest, inward reach scaled so -0.45 meets at the midline), y up, z out
    vec3 T = rest + vec3(s*(hx > 0.0 ? 0.95*L*hx : 1.05*L*hx), 1.40*L*hy, 0.80*L*hz);     // short plush arms: y gains more
    T.x = s*max(s*T.x, 0.07);                                           // its own side; two hands meet at the midline
    // stay out of the belly / chest: push out of an inflated torso ellipsoid
    vec3 e = (T - vec3(0.0, 0.88, 0.0))/vec3(0.56, 0.56, 0.52);
    float el = length(e);
    if(el < 1.0) T = vec3(0.0, 0.88, 0.0) + e/el*vec3(0.56, 0.56, 0.52);
    // a paw that comes to the face rests ON it, in front — never inside the big head (the head is wider than
    // the arm is long: cover-eyes / mouth / chin targets used to bury the paw in it).  The palm (~0.09 above a raised
    // wrist) is pushed out of the head inflated by the paw's thickness, in head space so it follows the head: in front
    // of the face straight forward (it keeps its place over the eyes / mouth / cheek), toward the sides out from the
    // centre.  The short arm then stretches (cartoon, up to 1.7x, mostly along the view) to reach that spot.  An empty
    // paw only: a paw holding a prop keeps the calibrated reach (muse.props.json reach map) exactly as before.
    float kArm = 1.0;
    {
      float fr = 1.0;
#ifdef PROPS_HOLD
      fr = 1.0 - clamp(gHoldW[i], 0.0, 1.0);
#endif
      vec3 qh = transpose(F)*(T + vec3(0.0, 0.09, 0.0) - hc);
      vec3 Rh = M_HEADR + 0.085;
      vec3 eh = qh/Rh; float elh = length(eh);
      if(elh < 1.0 && fr > 0.0){
        vec3 radial = eh/max(elh, 1e-4)*Rh;
        vec3 front = vec3(qh.xy, Rh.z*sqrt(max(1.0 - dot(eh.xy, eh.xy), 0.0)));
        vec3 qn = mix(radial, front, smoothstep(0.0, 0.35, qh.z/Rh.z));
        vec3 Tn = hc + F*qn - vec3(0.0, 0.09, 0.0);
        kArm = mix(1.0, clamp(length(Tn - S)/(0.995*L), 1.0, 1.7), fr);
        T = mix(T, Tn, fr);
      }
    }
    float up = smoothstep(0.55, 1.35, T.y);
    vec3 pole = normalize(vec3(s*0.9, mix(-0.7, 0.25, up), mix(-0.55, -0.35, up)));
    vec3 E = mIK(S, T, M_UPPER*kArm, M_FORE*kArm, pole);
    gMSh[i] = upperToLocal(S); gMEl[i] = upperToLocal(E); gMWr[i] = upperToLocal(T);
    // hand frame: y along the forearm, z = palm normal (toward the midline at turn 0, rolled by hand_turn), x across
    vec3 f = normalize(T - E);
    // a hand raised to the shoulder and above stands its fingers up (the wrist bends); one held forward low points
    // its fingers forward and a little down (typing / holding)
    f = normalize(mix(f, vec3(0.0, 1.0, 0.0), 0.75*smoothstep(0.85, 1.35, T.y)*smoothstep(-0.2, 0.4, f.y + 0.3)));
    vec3 n0 = vec3(-s, 0.0, 0.0); n0 = n0 - f*dot(n0, f);
    n0 = dot(n0, n0) > 1e-4 ? normalize(n0) : normalize(cross(f, vec3(0,0,1)));
    float turn = clamp(i == 0 ? uP[P_HAND_L_TURN] : uP[P_HAND_R_TURN], -1.0, 1.0);
    // a hand raised by the head stands its fingers up toward the sky a little more
    vec3 nz = mAxis(f, s*turn*1.45)*n0;
    vec3 nx = normalize(cross(f, nz));
    gMHand[i] = gMW*mat3(nx, f, nz);
    gMPalm[i] = gMWr[i] + gMHand[i][1]*0.095;
    // digits: index, middle, ring, pinky, thumb
    float op = clamp(i == 0 ? uP[P_HAND_L_OPEN] : uP[P_HAND_R_OPEN], 0.0, 1.0);
    float pt = clamp(i == 0 ? uP[P_POINT_L] : uP[P_POINT_R], 0.0, 1.0);
    float th = clamp(i == 0 ? uP[P_THUMB_L] : uP[P_THUMB_R], 0.0, 1.0);
    float cnt = clamp(i == 0 ? uP[P_FINGERS_L] : uP[P_FINGERS_R], 0.0, 5.0);
    float relax = mix(0.55, 0.0, op);
    for(int k=0;k<5;k++){
      float c = k == 4 ? mix(0.35, 0.0, op) : relax;
      if(k > 0 && k < 4) c = mix(c, 1.0, pt);
      if(k == 0) c = mix(c, 0.0, pt);
      if(k == 4) c = mix(c, 0.0, th);
      else c = mix(c, 1.0, th*(1.0 - pt));             // thumbs-up: a fist with the thumb out
      // counting: that many digits straight, the rest curled; the next one part-way
      float upk = clamp(cnt - float(k), 0.0, 1.0);
      c = mix(c, mix(1.0, 0.0, upk), clamp(cnt, 0.0, 1.0));
#ifdef PROPS_HOLD
      // props: the digits close round what the paw holds (engine/props_hold.glsl: weight, radius, grip)
      float hw = gHoldW[i];
      if(hw > 0.0){
        float g = clamp(0.040/max(gHoldR[i], 0.02), 0.22, 0.95);             // a smaller object, a tighter wrap
        if(gHoldG[i] == 2) g = 0.16;                                          // two-paw hold: flat palms on its sides
        if(gHoldG[i] == 1) g = (k == 0 || k == 4) ? 0.55 : 0.90;             // pinch: index to thumb, the rest folded
        if(k == 4) g = gHoldG[i] == 2 ? 0.08 : min(g*0.85, 0.75);             // the thumb opposes
        c = mix(c, g, hw);
      }
#endif
      gMCurl[i*5 + k] = c;
    }
    gMSpread[i] = 0.25*op + 0.20*clamp(cnt - 0.5, 0.0, 1.0);
#ifdef PROPS_HOLD
    // props (engine/props_hold.glsl, included before the animal): the digits close round what this paw holds.
    // Power grip: the fingers wrap a handle of radius gHoldR (finger 0.10 long, 0.028 thick, under-curled a little so
    // they rest on its surface); pinch: index + thumb; two paws on it: palms flat on its sides, digits nearly straight.
    float hw = clamp(gHoldW[i], 0.0, 1.0);
    if(hw > 0.0){
      int g = gHoldG[i];
      float wrapc = clamp(0.70*0.10/((gHoldR[i] + 0.028)*1.45), 0.25, 0.95);
      for(int k=0;k<5;k++){
        float tc = g == 2 ? (k == 4 ? 0.10 : 0.18) : (g == 1 ? (k == 0 || k == 4 ? 0.55 : 0.70) : (k == 4 ? 0.55 : wrapc));
        gMCurl[i*5 + k] = mix(gMCurl[i*5 + k], tc, hw);
      }
      gMSpread[i] = mix(gMSpread[i], 0.08, hw);
    }
#endif
    gMBC[i] = gMPalm[i] + gMHand[i][1]*0.04; gMBR[i] = 0.29;
  }
  // ---- legs
  float H = M_H;
  for(int i=0;i<2;i++){
    float s = i == 0 ? -1.0 : 1.0;
    vec3 J = gMHip + vec3(s*0.205, -0.02, 0.0);
    vec3 off = H*vec3(i == 0 ? uP[P_FOOT_L_X] : uP[P_FOOT_R_X], i == 0 ? uP[P_FOOT_L_Y] : uP[P_FOOT_R_Y],
                      i == 0 ? uP[P_FOOT_L_Z] : uP[P_FOOT_R_Z]);
#ifdef ROOT_XFORM
    if(uRootOn > 0.5) off = uRootAi*off;                     // WORLD-axis offsets -> character-local
#endif
    vec3 T = vec3(s*0.225, M_ANK_Y, 0.035) + off;
    vec3 pole = normalize(vec3(s*0.18, 0.0, 1.0));             // knees bend forward
    vec3 K = mIK(J, T, M_THIGH, M_SHIN, pole);
    gMHipJ[i] = J; gMKn[i] = K; gMAnk[i] = T;
    // the foot: flat (local), toes out a little, toes down as it lifts off the floor
    float lift = smoothstep(0.0, 0.25, T.y - M_ANK_Y);
    gMFoot[i] = rotY(-s*0.16)*rotX(0.30*lift);
  }
}

// ---- the head (head space q): ellipsoid + ears
float museHeadHost(vec3 q){
  float d = sdEllipsoid(q, M_HEADR);
  // a soft chubby jowl: the lower half a little fuller
  d -= 0.025*smoothstep(0.1, -0.5, q.y);
  vec3 qa = vec3(abs(q.x), q.yz);
  bool el = q.x < 0.0; vec3 e = (el ? gMEar0 : gMEar1)*(qa - M_EAR_PIV) + M_EAR_PIV - M_EAR_C - (el ? gMEarL0 : gMEarL1)*vec3(0.581, 0.814, 0.0);
  e.xy = rot(0.62)*e.xy;
  float ear = sdEllipsoid(e, vec3(0.135, 0.175, 0.080));
  return smin(d, ear, 0.07);
}
float museTorso(vec3 p){
  // lower belly in pelvis space, chest in upper-body space: the waist bend deforms the soft body between them
  vec3 b = p - (gMHip - M_HIP);
  float lo = sdEllipsoid(b - vec3(0.0, 0.69, -0.01), vec3(0.47, 0.31, 0.41));
  vec3 u = localToUpper(p);
  float br = 1.0 + 0.012*gBreath;
  float hi = sdEllipsoid((u - vec3(0.0, 0.93, 0.0))/vec3(br, 1.0, br), vec3(0.48, 0.40, 0.41));
  float d = smin(lo, hi, 0.16);
  d = smin(d, sdEllipsoid(u - vec3(0.0, 1.12, -0.01), vec3(0.36, 0.16, 0.30)), 0.12);     // shoulders
  return d;
}
float museArm(vec3 p, int i, out float tt){
  vec3 a = gMSh[i], b = gMEl[i], c = gMWr[i];
  vec3 pa = p - a, ba = b - a;
  float h1 = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  float d1 = length(pa - ba*h1) - mix(0.115, 0.098, h1);
  vec3 pb = p - b, cb = c - b;
  float h2 = clamp(dot(pb, cb)/dot(cb, cb), 0.0, 1.0);
  float d2 = length(pb - cb*h2) - mix(0.098, 0.088, h2);
  tt = d1 < d2 ? 0.5*h1 : 0.5 + 0.5*h2;
  return smin(d1, d2, 0.05);
}
// the mitten: palm pad + four short fingers + a thumb, all soft
float museHand(vec3 p, int i){
  mat3 F = gMHand[i];
  vec3 q = transpose(F)*(p - gMPalm[i]);              // x across, y to the fingertips, z out of the palm
  float s = i == 0 ? -1.0 : 1.0;
  float ts = -s;                                      // thumb side along x (mirror-symmetric hands)
  float d = sdEllipsoid(q - vec3(0.0, -0.005, -0.012), vec3(0.112, 0.095, 0.064));
  float sp = gMSpread[i];
  for(int k=0;k<4;k++){
    float c = gMCurl[i*5 + k];
    float fk = float(k);
    vec3 base = vec3(ts*(0.070 - 0.047*fk), 0.062 - 0.010*abs(fk - 1.5), -0.008);
    float len = 0.105 - 0.014*abs(fk - 1.2);
    vec3 dir = normalize(vec3(ts*sp*(0.6 - 0.4*fk)*0.35, cos(c*1.45), sin(c*1.45)));     // curls toward the palm (+z)
    vec3 mid = base + dir*len*0.55;
    vec3 dir2 = normalize(vec3(dir.x, cos(c*2.7), sin(c*2.7)));
    float f = sdCapsule(q, base, mid, 0.029);
    f = smin(f, sdCapsule(q, mid, mid + dir2*len*0.5, 0.027), 0.010);
    d = smin(d, f, 0.010);
  }
  float c = gMCurl[i*5 + 4];
  vec3 tb = vec3(ts*0.085, -0.030, -0.004);
  vec3 tdir = normalize(mix(vec3(ts*0.75, 0.60, 0.28), vec3(-ts*0.45, 0.35, 0.82), c));   // folds across the palm
  d = smin(d, sdCapsule(q, tb, tb + tdir*0.095, 0.037), 0.016);
  gMPalmW = smoothstep(0.012, -0.02, q.z + 0.045)*smoothstep(0.10, 0.05, length(q.xy));
  return d;
}
float museLeg(vec3 p, int i){
  float d1 = sdTaper(p, gMHipJ[i], gMKn[i], 0.140, 0.122);
  float d2 = sdTaper(p, gMKn[i], gMAnk[i], 0.122, 0.112);
  return smin(d1, d2, 0.06);
}
float museFoot(vec3 p, int i, out float sole){
  vec3 q = transpose(gMFoot[i])*(p - gMAnk[i]);
  float d = sdEllipsoid(q - vec3(0.0, -0.050, 0.070), vec3(0.135, 0.100, 0.200));
  d = smax(d, -(q.y + M_ANK_Y - 0.002), 0.05);         // a flat soft sole
  sole = smoothstep(-0.10, -0.13, q.y);
  return d;
}

// style: the bare face panel (head space): a rounded squircle on the front of the head, eyes, brows, blush
// and mouth inside it.  Returns the panel weight; hood = the band of longer pile framing it.
float musePanel(vec3 q, out float hood){
  vec2 u = vec2(q.x/0.50, (q.y + 0.015)/0.405);
  vec2 a = abs(u);
  float sq = a.x*a.x*a.x + a.y*a.y*a.y;
  float front = smoothstep(0.12, 0.30, q.z);
  hood = smoothstep(0.92, 1.06, sq)*smoothstep(1.55, 1.10, sq)*front;
  return smoothstep(1.04, 0.90, sq)*front;
}
// ---- the whole character (local space)
void museEval(vec3 p){
  gMPart = 1; gMSoleW = 0.0; gMPalmW = 0.0; gMFace = 0.0;
  gMPanel = 0.0; gMHood = 0.0; gMCrease = 0.0;
  // the head (bounding sphere: radius ~0.9 about its centre)
  vec3 q = gHR*(p - gHO);
  gQ = q;
  float hb = length(q) - 0.92;
  float head, bare = 1e3;
  if(hb < 0.25){
    float host = museHeadHost(q);
    float skin = host;
    if(length(q - vec3(0.0, -0.05, 0.5)) < 0.55){       // only the face carries the face's work
      gMFace = 1.0;
      skin = lipsEval(q, host);
      skin = faceEval(q, skin, host);
      bare = min(gLipBare, gFEye);
    } else { gFEye = 1e3; gFBrow = 1e3; gFLid = 1e3; gFOffLips = 1.0; }
    head = skin;
  } else {
    head = hb + 0.05; gFEye = 1e3; gFBrow = 1e3; gFLid = 1e3; gFOffLips = 1.0;
  }
  gMHeadD = head;
  float body = museTorso(p);
  float d = smin(head, body, 0.16);
  gMCrease = clamp((min(head, body) - d)*25.0, 0.0, 1.0);      // the neck crease (smin k/4 = 0.04 deep)
  vec3 rest = q;
  if(body < head){ gMPart = 1; rest = localToUpper(p) + vec3(3.0, 0.0, 0.0); }
  else gMPart = 0;
  float best = d;
  // arms + hands
  for(int i=0;i<2;i++){
    float tt;
    float a = museArm(p, i, tt);
    float hd = length(p - gMBC[i]) - gMBR[i];
    float hn = hd < best + 0.05 ? museHand(p, i) : hd;
    float limb = smin(a, hn, 0.05);
    if(limb < best){
      vec3 ax = normalize(gMWr[i] - gMSh[i]);
      gMPart = hn < a ? 3 : 2;
      rest = gMPart == 3 ? transpose(gMHand[i])*(p - gMPalm[i])*1.6 + vec3(7.0 + float(i), 0.0, 0.0)
                         : vec3(tt*0.6, dot(p - gMSh[i], mPerp(ax)), dot(p - gMSh[i], cross(ax, mPerp(ax)))) + vec3(5.0 + float(i), 2.0, 0.0);
    }
    float dl = smin(d, limb, 0.07);
    gMCrease = max(gMCrease, clamp((min(d, limb) - dl)*45.0, 0.0, 1.0));      // armpit / paw on the belly
    d = dl;
    best = min(best, limb);
  }
  // legs + feet
  for(int i=0;i<2;i++){
    float leg = museLeg(p, i);
    float sole;
    float ft = museFoot(p, i, sole);
    float limb = smin(leg, ft, 0.06);
    if(limb < best){
      gMPart = ft < leg ? 5 : 4;
      gMSoleW = gMPart == 5 ? sole : 0.0;
      vec3 ax = normalize(gMAnk[i] - gMHipJ[i]);
      rest = transpose(gMFoot[i])*(p - gMAnk[i]) + vec3(9.0 + 2.0*float(i), 0.0, 0.0);
      if(gMPart == 4) rest = vec3(dot(p - gMHipJ[i], ax), dot(p - gMHipJ[i], mPerp(ax)), dot(p - gMHipJ[i], cross(ax, mPerp(ax)))) + vec3(13.0 + float(i), 0.0, 0.0);
    }
    float dg = smin(d, limb, 0.08);
    gMCrease = max(gMCrease, clamp((min(d, limb) - dg)*40.0, 0.0, 1.0));      // hip / groin
    d = dg;
    best = min(best, limb);
  }
#ifdef MUSE_DEBUG_SEAT
  // test only (tools): a plain box seat of height MUSE_DEBUG_SEAT in the WORLD (not part of the character)
  {
    vec3 pw = rootFwd(p) - vec3(0.0, 0.5*MUSE_DEBUG_SEAT, -0.05);
    vec3 bq = abs(pw) - vec3(0.40, 0.5*MUSE_DEBUG_SEAT, 0.32) + 0.03;
    float box = length(max(bq, 0.0)) + min(max(bq.x, max(bq.y, bq.z)), 0.0) - 0.03;
#ifdef ROOT_XFORM
    if(uRootOn > 0.5) box /= max(uRootLip, 1e-3);      // charEval scales local distances by uRootLip
#endif
    if(box < d){ gMPart = 6; rest = pw; }
    d = min(d, box);
  }
#endif
  gMSkin = d; gMBare = bare; gMRest = rest;
  if(gMPart == 0) gMPanel = musePanel(q, gMHood);
  // fur: a short even pile everywhere, none on the eyes, the lips, the brows' tips, the pads
  float onFace = (1.0 - smoothstep(0.004, 0.030, gFEye))*1.0;
  float onLips = 1.0 - gFOffLips;
  float onBrow = smoothstep(0.012, 0.002, gFBrow);
  // the tongue out has no pile: none within the pile's length of it either, or the shell grows over it
  onLips = max(onLips, gMFace > 0.5 && gLTgOut > 0.0 ? smoothstep(0.040, 0.028, gLTongue) : 0.0);
#ifdef LOOK_V2
  // look: pile grows on the skin, never in the air in front of the eyeball (the pile's weight is keyed to the
  // distance from the eye, so ~a pile's length off the eye the shell test passed on the tufts' peaks: cream specks
  // floating in the black of the eye)
  onFace = max(onFace, smoothstep(0.008, -0.004, gFEye - gMHeadD));
#endif
  gMFurW = float(gMPart != 6)*(1.0 - max(max(max(onFace, onLips), onBrow), gMPanel))*(1.0 - 0.85*gMSoleW)*(1.0 - 0.85*gMPalmW)*(gMPart == 3 ? 0.40 : 1.0);
}
void charEval(vec3 p){
  museEval(p);
  gcBase = min(gMSkin, gMBare - 0.0007); gcBare = gMBare; gcRest = gMRest;
  gcLen = 0.030*(1.0 + 0.30*gMHood)*gMFurW - 0.0006*(1.0 - gMFurW);      // the hood's rim stands a little fuller
}
float charTuft(){
  // plush pile: dense short tufts, no long flow
  vec3 c = gcRest*vec3(95.0);
  float a = vnoise3(c), b = vnoise3(c*2.3 + 4.1);
  return clamp(0.30 + 0.55*a + 0.35*b - 0.10, 0.0, 1.0);
}

//@shadow-begin
float charShadow(vec3 p){
  vec3 q = gHR*(p - gHO);
  float d = sdEllipsoid(q, M_HEADR*0.97);
  vec3 b = p - (gMHip - M_HIP);
  d = smin(d, sdEllipsoid(b - vec3(0.0, 0.69, -0.01), vec3(0.45, 0.29, 0.39)), 0.1);
  vec3 u = localToUpper(p);
  d = smin(d, sdEllipsoid(u - vec3(0.0, 0.93, 0.0), vec3(0.46, 0.38, 0.39)), 0.12);
  for(int i=0;i<2;i++){
    d = min(d, sdCapsule(p, gMSh[i], gMEl[i], 0.095));
    d = min(d, sdCapsule(p, gMEl[i], gMWr[i], 0.082));
    d = min(d, length(p - gMPalm[i]) - 0.095);
    d = min(d, sdCapsule(p, gMHipJ[i], gMKn[i], 0.12));
    d = min(d, sdCapsule(p, gMKn[i], gMAnk[i], 0.10));
    d = min(d, sdEllipsoid(transpose(gMFoot[i])*(p - gMAnk[i]) - vec3(0.0, -0.05, 0.07), vec3(0.12, 0.085, 0.18)));
  }
  return d;
}
//@shadow-end

// ---- for the props agent (character-local)
void museGrasp(int side, out vec3 pos, out mat3 frame){
  int i = side == 0 ? 0 : 1;
  frame = gMHand[i];                                   // x across the palm, y to the fingertips, z out of the palm
  pos = gMPalm[i] + frame[2]*0.070 + frame[1]*0.010;   // where a held thing's grip point sits, just off the palm
}
void museHeadFrame(out vec3 pos, out mat3 frame){
  frame = transpose(gHR);                              // head axes in local space: x right, y up, z out of the face
  pos = gHO + frame*vec3(0.0, 0.02, -0.03);            // the crown / ear line centre (head ellipsoid radii 0.70 0.64 0.62)
}

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  museEval(pw);
  vec3 n = s.n, q = gQ;
  vec3 r = gcRest;
  // style: a shaggy oatmeal pile — dark roots between the clumps, pale tips on them, broad soft clump shading
  float tuftC = vnoise3(r*vec3(95.0));
  float st = vnoise3(r*vec3(210.0));
  float big = vnoise3(r*7.0);
  float clump = vnoise3(r*vec3(28.0));
  vec3 alb = mix(C_PLUSH, C_PLUSH2, clamp(0.33 + 0.38*smoothstep(0.25, 0.80, tuftC) + 0.08*st + 0.34*(clump - 0.5), 0.0, 1.0));
  alb *= 0.93 + 0.12*big;
  // limbs a touch deeper than the body (the paws and feet read as their own masses against the belly)
  alb *= gMPart == 4 || gMPart == 5 ? 0.92 : (gMPart == 2 || gMPart == 3 ? 0.96 : 1.0);
  float panel = gMPanel*float(gMPart == 0);
  // cheeks: rosy blush, stronger with smile / cheek
  if(gMPart == 0){
    // the face panel: smooth peach cream with a faint warm falloff toward its edge (it sits recessed in the hood)
    vec2 pu = vec2(q.x/0.50, (q.y + 0.015)/0.405);
    vec3 face = C_FACE*(1.0 - 0.17*smoothstep(0.50, 1.05, dot(pu, pu)));       // recessed: it shades toward the hood
    alb = mix(alb, face, panel);
    vec2 ck = vec2(abs(q.x) - 0.355, q.y + 0.100);
    float blush = exp(-dot(ck*vec2(1.0, 1.25), ck*vec2(1.0, 1.25))/0.0115)*smoothstep(0.1, 0.4, q.z);
    alb = mix(alb, C_BLUSH, min(0.68 + 0.16*gLCheek + 0.12*gLSmile, 0.92)*blush*panel);
    // inner ears: a taupe-rose felt
    vec3 qa = vec3(abs(q.x), q.yz);
    bool el = q.x < 0.0; vec3 e = (el ? gMEar0 : gMEar1)*(qa - M_EAR_PIV) + M_EAR_PIV - M_EAR_C - (el ? gMEarL0 : gMEarL1)*vec3(0.581, 0.814, 0.0); e.xy = rot(0.62)*e.xy;
    float inEar = smoothstep(0.03, -0.01, sdEllipsoid(e - vec3(0.0, 0.0, 0.05), vec3(0.08, 0.11, 0.05)))*step(0.30, q.y);
    alb = mix(alb, mix(C_SOLE, C_BLUSH, 0.35), 0.75*inEar*(1.0 - panel));
  }
  if(gMSoleW > 0.0 || gMPalmW > 0.0) alb = mix(alb, C_SOLE, max(gMSoleW, gMPalmW)*0.85);
  if(gMPart == 6) alb = vec3(0.42, 0.30, 0.20)*(0.9 + 0.2*vnoise3(r*vec3(3.0, 40.0, 3.0)));
  s.alb = alb;
  vec3 pole = gMPart == 0 ? vec3(0.0, 0.1, 0.9) : vec3(0.0, 0.6, 0.0);
  s.T = furFlow(r - pole - vec3(0.0, 0.0, 0.0), n, vec3(0.0, -0.20, 0.0));
  float nb = (st - 0.5)*(1.0 - panel);
  vec3 bt = normalize(cross(s.T, n));
  s.n = normalize(n + bt*nb*0.12 + s.T*nb*0.04);
  s.fur = 0.75*gMFurW;
  s.wrap = mix(0.18, 0.30, panel);
  s.sheen = mix(0.40, 0.10, panel);
  // contours: the coat darkens between its clumps, in the smooth joins (neck, armpits, hips), and on its underside
  s.occ = (0.86 + 0.14*smoothstep(0.1, 0.7, tuftC))*mix(1.0, 0.92, smoothstep(0.55, 0.25, clump));
  s.occ = mix(s.occ, 1.0, panel);
  s.occ *= 1.0 - 0.50*gMCrease;
  s.occ *= mix(0.72, 1.0, smoothstep(-0.75, 0.25, n.y));
  // the panel's edge, where the hood's pile rolls over it: a soft shaded seam
  s.occ *= 1.0 - 0.20*gMHood*smoothstep(0.0, 0.5, panel + 0.3);
  s.spec = mix(0.03, 0.10, panel); s.shin = mix(12.0, 28.0, panel);
  s.trans = 0.25*smoothstep(0.30, 0.55, q.y)*float(gMPart == 0)*(1.0 - panel);      // the ears glow a little against the light
  s.transCol = vec3(1.0, 0.62, 0.50);
  s.emit = s.alb*mix(vec3(0.012, 0.009, 0.007), vec3(0.030, 0.022, 0.018), panel);   // plush scatters; the panel glows a touch
  if(gMPart == 0 && gMFace > 0.5){
    lipsPaint(n, s);
    facePaint(q, rd, n, gMSkin, s);
    float lipN = gPLip;
    s.fur *= 1.0 - max(lipN, smoothstep(0.012, 0.0, gFEye));
  }
}

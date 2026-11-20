// ---------------------------------------------------------------------------
// engine/mascot_rig.glsl — the MUSE SKELETON for other standing mascots (the gekkie, the owl, ...).
//
//  A mascot character file (animals/<name>.glsl) is:  its face / lips defines  ->  `#pragma include face`, `#pragma
//  include lips`  ->  its MR_* overrides  ->  `#pragma include mascot_rig`  ->  its own shapes and materials (the hooks
//  listed in engine/mascot_body.glsl)  ->  `#pragma include mascot_body`.
//
//  The joints, the hand map (rig hand_* -> wrist), the hand frame, the grasp point and the head frame are the muse's,
//  number for number (animals/muse.glsl charInit): a mascot shares animals/muse.body.json and muse.props.json (symlinks),
//  so the props' reach calibration, the chalkboard writing, sitting and walking carry over unchanged.  What a mascot
//  changes: the SHAPES hung on the joints (head, torso, limb radii, hands, feet, a tail / hat / crest), the bone
//  lengths of the arm (MR_LMAP keeps the map), shoulder / hip width, and every material.
//  Globals keep the muse's names (gMSh, gMEl, gMWr, gMHand, gMPalm, gMHipJ, gMKn, gMAnk, gHO, gHR ...): engine/props.glsl's
//  PROPS_BOUND_MUSE box and museGrasp / museHeadFrame (props JSON "renderer") work as they are.
// ---------------------------------------------------------------------------
#ifndef M_H
#define M_H 2.30
#endif
const vec3 M_HIP   = vec3(0.0, 0.54, 0.0);       // pelvis origin (rest)
const vec3 M_WAIST = vec3(0.0, 0.70, 0.0);       // waist pivot
const vec3 M_NECK  = vec3(0.0, 1.12, 0.0);       // neck pivot, upper-body space
#ifndef M_HEADC
#define M_HEADC vec3(0.0, 1.64, 0.03)            // head centre, upper-body space at rest (= SHAPE_WORLD)
#endif
#ifndef M_HEADR
#define M_HEADR vec3(0.70, 0.64, 0.62)           // head ellipsoid radii (head space): what an empty hand at the face rests on
#endif
#ifndef M_UPPER
#define M_UPPER 0.30
#endif
#ifndef M_FORE
#define M_FORE 0.27
#endif
const float M_THIGH = 0.215, M_SHIN = 0.215;     // leg bones
// MR_LEG_LIFT (optional, scene units) = a longer-legged character on the same skeleton: the pelvis and everything
// above it ride that much higher, each leg bone grows by half of it; the hand MAP stays where the muse's is in the world
// (props reach calibration, chalk writing).  The character then needs its OWN <name>.body.json (pelvis, hips, leg, sitting).
#ifndef MR_THIGH                                 // gekkie: a character may set its own leg bones (MR_THIGH / MR_SHIN)
#ifdef MR_LEG_LIFT
#define MR_THIGH (M_THIGH + 0.5*MR_LEG_LIFT)
#define MR_SHIN (M_SHIN + 0.5*MR_LEG_LIFT)
#else
#define MR_THIGH M_THIGH
#define MR_SHIN M_SHIN
#endif
#endif
const float M_ANK_Y = 0.145;                     // ankle height at rest (sole on the floor)
#ifndef MR_LMAP
#define MR_LMAP 0.57                             // the hand map's arm length (the muse's bones): never change
#endif
#ifndef MR_SH_X
#define MR_SH_X 0.405                            // shoulder half width / height / depth, upper-body space
#endif
#ifndef MR_SH_Y
#define MR_SH_Y 1.035
#endif
#ifndef MR_SH_Z
#define MR_SH_Z 0.0
#endif
#ifndef MR_HIP_X
#define MR_HIP_X 0.205
#endif
#ifndef MR_ANK_X
#define MR_ANK_X 0.225                           // resting ankle half spread (the gekkie stands narrower)
#endif
#ifndef MR_TOE_OUT
#define MR_TOE_OUT 0.16
#endif
#ifndef MR_PUSH_C
#define MR_PUSH_C vec3(0.0, 0.88, 0.0)           // a wrist target is pushed out of this torso ellipsoid
#endif
#ifndef MR_PUSH_R
#define MR_PUSH_R vec3(0.56, 0.56, 0.52)
#endif
#ifndef MR_REST_CURL
#define MR_REST_CURL 0.55                        // how far the fingers curl in a relaxed hand
#endif
#ifndef MR_REST_THUMB
#define MR_REST_THUMB 0.35                       // and the thumb
#endif
#ifndef MR_ANK_X
#define MR_ANK_X 0.225                           // resting ankle half spacing (a wider stance needs its own body.json ankle_rest)
#endif
#ifndef MR_HAND_THICK
#define MR_HAND_THICK 0.085                      // an empty hand at the face rests this far off the M_HEADR ellipsoid
#endif
#ifdef MR_INIT_HOOK
void mascotInit();                               // the character's own pose (tail, hat ...), at the end of charInit
#endif
#ifdef MR_CUSTOM_BEND
void mascotMouthBend();                          // the character's own resting mouth line (after lipsInit)
#endif

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
// of the head black with gMEar[es] (CPU renders were fine).
mat3 gMEar0, gMEar1;
float gMEarL0, gMEarL1;
#ifndef M_EAR_C
#define M_EAR_C vec3(0.40, 0.50, -0.06)        // ear centre (mirrored head space, x = |x|)
#endif
#ifndef M_EAR_PIV
#define M_EAR_PIV vec3(0.336, 0.410, -0.06)    // where it grows from
#endif

// ---- eval outputs
float gMSkin, gMBare, gMFurW, gMHeadD, gMSoleW, gMPalmW, gMFace;
float gMPanel, gMHood, gMCrease;     // face-skin weight (the emotion paints use it), unused, smooth-join crease
vec3  gMRest, gMPole, gQ;
int   gMPart;          // 0 head, 1 body, 2 arm, 3 hand, 4 leg, 5 foot, 7 extra (tail, hat ...)
int   gMSide;          // limb side of the nearest arm / hand / leg / foot (0 = screen-left)
float gMLimbT;         // 0..1 along that limb (shoulder -> wrist, hip -> ankle)
#ifndef T4408_URQ
#define T4408_URQ 1
uniform float uRQ;
#endif
float gMFp;          // pixel footprint at the character (scene units per pixel): detail finer than ~2 px must fade out

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
// a stack of rings along x: 0 in the crease between two rings .. 1 on a ring's crown
float mrRing(float x){ return abs(sin(3.14159265*x)); }
vec3 upperToLocal(vec3 u){ return gMW*(u - M_WAIST) + M_WAIST + (gMHip - M_HIP); }
vec3 localToUpper(vec3 p){ return transpose(gMW)*(p - (gMHip - M_HIP) - M_WAIST) + M_WAIST; }

void charInit(){
  faceInit(); lipsInit();
#ifdef MR_CUSTOM_BEND
  mascotMouthBend();
#else
  // a resting upturned smile: the mouth line bends up a little even at smile 0 (lips.glsl's own bend is small)
  float op = smoothstep(0.0, 0.6, max(uP[P_LIP_OPEN], uP[P_MOUTH_OPEN]));
  // a knit brow over a smile-less mouth turns the corners DOWN (sad, worried, cross) — the resting bend kept
  // it smiling through grief. Only where the smile is low; smile >= 0.5 or no knit = exactly the old bend.
  float frown = clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  // R2: the resting bend is the idle smile (idle smile ~0.08-0.11 → exactly the old bend); a feeling that takes
  // the smile BELOW idle (sympathy, worry, alarm, a plain serious face) flattens the line, so it stops reading cheerful
  float rest = 0.13*mix(0.25, 1.0, smoothstep(0.0, 0.05, uP[P_SMILE]));
  // the frown holds while it TALKS (a4-a7: concern showed only in the pauses; the open mouth halved it and the spread
  // visemes bent the corners back up)
  gLBend += LIP_W*((rest + 0.06*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - 0.6*op) - MUSE_FROWN*frown*(1.0 - 0.15*op))*(1.0 - gLRnd);
#endif
  gBreath = uP[P_BREATH];
  gMFp = length(uCamPos - rootFwd(M_HEADC))*2.0/max(uRes.y*uFl, 1.0);      // camRay: uv = (2 pix - uRes)/uRes.y
  // pelvis: body_bounce = vertical offset of everything above the feet (the knees take it)
  gMHip = M_HIP + vec3(0.0, 0.07*clamp(uP[P_BODY_BOUNCE], -1.0, 1.0), 0.0);
#ifdef MR_LEG_LIFT
  gMHip.y += MR_LEG_LIFT;
#endif
  // waist: the upper body over the hips
  gMW = rotY(0.40*uP[P_BODY_YAW])*rotX(0.28*uP[P_BODY_PITCH])*rotZ(-0.20*uP[P_BODY_LEAN]);
  // head: turns about the neck, carried by the waist
  mat3 F = rotY(uP[P_HEAD_YAW])*rotX(uP[P_HEAD_PITCH])*rotZ(-uP[P_HEAD_ROLL]);
  vec3 hc = M_NECK + F*(M_HEADC - M_NECK) + 0.30*vec3(uP[P_HEAD_X], uP[P_HEAD_Y], uP[P_HEAD_Z]);
  hc.y += 0.006*gBreath;
  gHO = upperToLocal(hc);
  gHR = transpose(gMW*F);
  // ---- ears : small round ears that move.  In the mirrored head space (+x = outward for both): ear_l/ear_r
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
    vec3 S = vec3(s*MR_SH_X, MR_SH_Y + 0.10*sh, MR_SH_Z);
    float L = MR_LMAP;          // the hand MAP is the muse's (props reach calibration); the bones may be longer
    vec3 rest = vec3(s*0.540, 0.555, 0.130);
#ifdef MR_LEG_LIFT
    rest.y -= MR_LEG_LIFT;                     // the hand map keeps its place in the world
#endif
#ifdef MR_REST_OFF
    {
      // a character's own resting hand place (hands on the hips), on top of the muse's map: all of it at rest, none
      // of it once the hand is sent somewhere or holds a prop — the map (props reach calibration) is untouched there
      float fr = 1.0 - smoothstep(0.04, 0.30, length(vec3(hx, hy, hz)));
#ifdef PROPS_HOLD
      fr *= 1.0 - clamp(gHoldW[i], 0.0, 1.0);
#endif
      rest += vec3(s*MR_REST_OFF.x, MR_REST_OFF.yz)*fr;
#ifdef MR_REST_ASYM
      if(i == 1) rest += MR_REST_ASYM*fr;             // gekkie: the screen-right hand rests a little differently (not a toy)
#endif
    }
#endif
    // contract: x outward (negative = across the chest, inward reach scaled so -0.45 meets at the midline), y up, z out
    vec3 T = rest + vec3(s*(hx > 0.0 ? 0.95*L*hx : 1.05*L*hx), 1.40*L*hy, 0.80*L*hz);     // short plush arms: y gains more
#ifdef MR_MAP_OFF
    {
      // a taller body (the gekkie): the whole hand map moves with its shoulders, except for a hand that holds a prop
      // (props reach calibration: world places)
      float fm = 1.0;
#ifdef PROPS_HOLD
      fm -= clamp(gHoldW[i], 0.0, 1.0);
#endif
      T += vec3(s*MR_MAP_OFF.x, MR_MAP_OFF.yz)*fm;
    }
#endif
    T.x = s*max(s*T.x, 0.07);                                           // its own side; two hands meet at the midline
    // stay out of the belly / chest: push out of an inflated torso ellipsoid
    vec3 e = (T - MR_PUSH_C)/MR_PUSH_R;
    float el = length(e);
    if(el < 1.0) T = MR_PUSH_C + e/el*MR_PUSH_R;
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
      vec3 Rh = M_HEADR + MR_HAND_THICK;
      vec3 eh = qh/Rh; float elh = length(eh);
      if(elh < 1.0 && fr > 0.0){
        vec3 radial = eh/max(elh, 1e-4)*Rh;
        vec3 front = vec3(qh.xy, Rh.z*sqrt(max(1.0 - dot(eh.xy, eh.xy), 0.0)));
        vec3 qn = mix(radial, front, smoothstep(0.0, 0.35, qh.z/Rh.z));
        vec3 Tn = hc + F*qn - vec3(0.0, 0.09, 0.0);
        kArm = mix(1.0, clamp(length(Tn - S)/(0.995*(M_UPPER + M_FORE)), 1.0, 1.7), fr);
        T = mix(T, Tn, fr);
      }
    }
    float up = smoothstep(0.55, 1.35, T.y);
    vec3 pole = normalize(vec3(s*0.9, mix(-0.7, 0.25, up), mix(-0.55, -0.35, up)));
#ifdef MR_REST_POLE
    {
      // a character's own resting elbow direction (the gekkie keeps its elbows in by its sides, bent back), faded out
      // like MR_REST_OFF as soon as the hand is sent somewhere or holds a prop
      float fp = 1.0 - smoothstep(0.04, 0.30, length(vec3(hx, hy, hz)));
#ifdef PROPS_HOLD
      fp *= 1.0 - clamp(gHoldW[i], 0.0, 1.0);
#endif
      pole = normalize(mix(pole, normalize(vec3(s*MR_REST_POLE.x, MR_REST_POLE.yz)), fp));
    }
#endif
    vec3 E = mIK(S, T, M_UPPER*kArm, M_FORE*kArm, pole);
    gMSh[i] = upperToLocal(S); gMEl[i] = upperToLocal(E); gMWr[i] = upperToLocal(T);
    // hand frame: y along the forearm, z = palm normal (toward the midline at turn 0, rolled by hand_turn), x across
    vec3 f = normalize(T - E);
    // a hand raised to the shoulder and above stands its fingers up (the wrist bends); one held forward low points
    // its fingers forward and a little down (typing / holding)
#ifdef MR_WRIST_UP_OFF
    // gekkie: its hand map rides MR_WRIST_UP_OFF higher (MR_MAP_OFF), so the fingers-up wrist starts that much higher
    f = normalize(mix(f, vec3(0.0, 1.0, 0.0), 0.75*smoothstep(0.85, 1.35, T.y - MR_WRIST_UP_OFF)*smoothstep(-0.2, 0.4, f.y + 0.3)));
#else
    f = normalize(mix(f, vec3(0.0, 1.0, 0.0), 0.75*smoothstep(0.85, 1.35, T.y)*smoothstep(-0.2, 0.4, f.y + 0.3)));
#endif
    vec3 n0 = vec3(-s, 0.0, 0.0); n0 = n0 - f*dot(n0, f);
    n0 = dot(n0, n0) > 1e-4 ? normalize(n0) : normalize(cross(f, vec3(0,0,1)));
    float turn = clamp(i == 0 ? uP[P_HAND_L_TURN] : uP[P_HAND_R_TURN], -1.0, 1.0);
    // a hand raised by the head stands its fingers up toward the sky a little more
#ifdef MR_PALM_FWD
    {
      // a character whose open hands show their palms (the gekkie): the turn-0 palm leans toward the camera (forward,
      // a little up) instead of facing the body; hand_turn still turns it from there.  Not for a hand that holds a prop.
      float fk = MR_PALM_FWD;
#ifdef PROPS_HOLD
      fk *= 1.0 - clamp(gHoldW[i], 0.0, 1.0);
#endif
      vec3 w = vec3(-s*0.30, 0.30, 1.0); w -= f*dot(w, f);
      if(dot(w, w) > 1e-3) n0 = normalize(mix(n0, normalize(w), fk));
    }
#endif
    vec3 nz = mAxis(f, s*turn*1.45)*n0;
    vec3 nx = normalize(cross(f, nz));
    gMHand[i] = gMW*mat3(nx, f, nz);
    gMPalm[i] = gMWr[i] + gMHand[i][1]*0.095;
    // digits: index, middle, ring, pinky, thumb
    float op = clamp(i == 0 ? uP[P_HAND_L_OPEN] : uP[P_HAND_R_OPEN], 0.0, 1.0);
    float pt = clamp(i == 0 ? uP[P_POINT_L] : uP[P_POINT_R], 0.0, 1.0);
    float th = clamp(i == 0 ? uP[P_THUMB_L] : uP[P_THUMB_R], 0.0, 1.0);
    float cnt = clamp(i == 0 ? uP[P_FINGERS_L] : uP[P_FINGERS_R], 0.0, 5.0);
    float relax = mix(MR_REST_CURL, 0.0, op);
    for(int k=0;k<5;k++){
      float c = k == 4 ? mix(MR_REST_THUMB, 0.0, op) : relax;
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
    vec3 J = gMHip + vec3(s*MR_HIP_X, -0.02, 0.0);
    vec3 off = H*vec3(i == 0 ? uP[P_FOOT_L_X] : uP[P_FOOT_R_X], i == 0 ? uP[P_FOOT_L_Y] : uP[P_FOOT_R_Y],
                      i == 0 ? uP[P_FOOT_L_Z] : uP[P_FOOT_R_Z]);
#ifdef ROOT_XFORM
    if(uRootOn > 0.5) off = uRootAi*off;                     // WORLD-axis offsets -> character-local
#endif
    vec3 T = vec3(s*MR_ANK_X, M_ANK_Y, 0.035) + off;
#ifdef MR_STANCE_ASYM
    if(i == 0) T += MR_STANCE_ASYM;                   // gekkie: the screen-left foot a little forward / out at rest
#endif
    vec3 pole = normalize(vec3(s*0.18, 0.0, 1.0));             // knees bend forward
    vec3 K = mIK(J, T, MR_THIGH, MR_SHIN, pole);
    gMHipJ[i] = J; gMKn[i] = K; gMAnk[i] = T;
    // the foot: flat (local), toes out a little, toes down as it lifts off the floor
    float lift = smoothstep(0.0, 0.25, T.y - M_ANK_Y);
    gMFoot[i] = rotY(-s*MR_TOE_OUT)*rotX(0.30*lift);
  }
#ifdef MR_INIT_HOOK
  mascotInit();
#endif
}

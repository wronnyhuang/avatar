// ---------------------------------------------------------------------------
// animals/bonbon.glsl — / : the red candy-coated chocolate candy in the muse's classroom.
// close to the reference art, plus ONE change — a small navy trilby.
//  Measured off the reference art: a round satin-red shell on long pale legs (MR_LEG_LIFT: its own bonbon.body.json), the eyes
//  high on the shell under thick red lid domes with tall black pupils and a slit catch-light, thick dark brows that FLOAT
//  above the head, a wide open smirk (black inside, no teeth) across 60% of the face, a big white slab-serif letter in the
//  lower third, thin arms ending in cuffed white gloves (three fingers and a thumb) resting on the hips, white sneakers.
// ---------------------------------------------------------------------------
#define M_H 2.30
#define MR_LEG_LIFT 0.26
#define M_HEADC vec3(0.0, 1.185, 0.0)
#define M_HEADR vec3(0.700, 0.665, 0.465)
#define M_UPPER 0.40
#define M_FORE 0.36
#define MR_SH_X 0.650
#define MR_SH_Y 1.165
#define MR_REST_OFF vec3(0.080, 0.150 + MR_LEG_LIFT, 0.050)     // hands on the hips at rest (fades as a hand is sent anywhere)
#define MR_HIP_X 0.225
#define MR_ANK_X 0.350
#define MR_TOE_OUT 0.34
#define MR_REST_THUMB 0.55
#define MR_HAND_THICK 0.07
#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.185, 0.0)
#define MUSE_EMO 1
#define EYE_R0 0.174
#define EYE_C0 vec3(0.196,0.288,0.348)
#define EYE_R gEyeR
#define EYE_C gEyeC
#define EYE_AU 0.21
#define EYE_TILT (-0.04)
#define EYE_AL (-1.00)
#define EYE_AC (-0.05)
#define EYE_LID_TH 0.015
#define EYE_SQUINT 0.55
#define EYE_IRIS 0.020
#define EYE_PUPIL 0.010
#define EYE_COL_A vec3(0.930,0.925,0.920)
#define EYE_COL_B vec3(0.930,0.925,0.920)
#define EYE_COL_W vec3(0.985,0.985,0.985)
#define LINER_COL vec3(0.470,0.010,0.014)
#define EYE_CATCH 2.0
#define FACE_EYE_HOOK 1
#define BROW_C vec3(4.0,4.0,0.0)
#define BROW_L 0.010
#define BROW_W 0.005
#define BROW_COL vec3(0.012,0.008,0.008)
#define BROW_H 0.004
#define BROW_RISE 0.0
#define MUSE_FROWN 0.42
#define BROW_SAD_TILT 1.4
#define LIP_O vec3(0.020,-0.020,0.461)
#define LIP_TILT 0.02
#define LIP_W 0.430
#define LIP_ZIP 0.16
#define LIP_PEEL (0.16*LIPU)
#define LIP_REST_OPEN (0.115 + 0.14*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE])))
#define LIP_HU (0.010*LIPU)
#define LIP_HL (0.034*LIPU)
#define LIP_TU (0.000*LIPU)
#define LIP_TL (0.006*LIPU)
#define LIP_BOW 0.0
#define LIP_SQUARE 0.10
#define LIP_OPEN_Y 1.00
#define LIP_PUSH (0.040*LIPU)
#define LIP_SKIN 0.18
#define LIP_BLEND (0.020*LIPU)
#define LIP_TOOTH_Z (-0.20*LIPU)
#define LIP_TEETH_H ((0.004 + 0.050*gEmTeeth)*LIPU)
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(0.550,0.009,0.022)
#define LIP_COL_L vec3(0.550,0.009,0.022)
#define LIP_COL_LINE vec3(0.030,0.001,0.002)
#define LIP_LINE 1.2
#define LIP_LINE_SOFT 0.08
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.085,0.006,0.008)
#define LIP_COL_DEEP vec3(0.006,0.001,0.001)
#define LIP_COL_GUM  vec3(0.030,0.003,0.004)
#define LIP_COL_TONGUE vec3(0.300,0.040,0.050)
#define LIP_COL_TOOTH mix(vec3(0.020,0.002,0.003), vec3(0.90,0.88,0.84), gEmTeeth)
#define LIP_SPEC 0.05
#define LIP_SHIN 60.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.05
#define LIP_WRAP 0.10
#define LIP_EMIT vec3(0.02,0.0,0.0)
#define LIP_TONGUE_FREE 0.11
#pragma include face
#pragma include lips

#define MR_ARM_R vec3(0.054, 0.062, 0.070)
#define MR_LEG_R vec3(0.078, 0.070, 0.064)
#define MR_PALM_R vec3(0.134, 0.124, 0.094)
#define MR_NFING 3
#define MR_FING_R 0.056
#define MR_FING_L 0.92
#define MR_THUMB_R 0.054
#define MR_FOOT_C vec3(0.0, -0.040, 0.075)
#define MR_FOOT_R vec3(0.200, 0.160, 0.340)
#define MR_FOOT_CUSTOM 1
#define MR_K_NECK 0.02
#define MR_K_ARM 0.020
#define MR_K_LEG 0.030
#define MR_K_HAND 0.030
#define MR_K_FOOT 0.030
#define MR_K_EXTRA 0.016
#define MR_HEAD_BOUND 1.02
#define MR_FACE_C vec3(0.0, 0.20, 0.30)
#define MR_FACE_RAD 0.80
#define MR_PUSH_C vec3(0.0, 1.185, 0.0)
#define MR_PUSH_R vec3(0.780, 0.740, 0.545)
#define MR_CHEEK vec2(0.40, 0.12)
#define MR_CHEEK_R 0.11
#define MR_CUSTOM_BEND 1
#define MR_REST_CURL 0.32
#define MR_INIT_HOOK 1
#define MR_HEAD_EXTRA 1
#define MR_EXTRA 1
#define MR_CREASE_OCC 0.0
//@shadow-begin
#define MR_SHADOW_HEAD vec3(0.680, 0.645, 0.445)
//@shadow-end
#pragma include mascot_rig

void mascotMouthBend(){
  // the smirk: the mouth is a wide grin that rides higher at the screen-right corner; a frown evens it out and shuts it
  float op = smoothstep(0.0, 0.6, max(uP[P_LIP_OPEN], uP[P_MOUTH_OPEN]));
  float frown = clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  float rest = 0.40*mix(0.45, 1.0, smoothstep(0.0, 0.05, uP[P_SMILE]));
  gLBend += LIP_W*((rest + 0.07*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - 0.5*op) - MUSE_FROWN*frown*(1.0 - 0.15*op))*(1.0 - gLRnd);
  gEmSkew = clamp(gEmSkew + 0.14*(1.0 - 0.6*op)*(1.0 - frown), -1.0, 1.0);
  gLSm = clamp(gLSm + 0.30*(1.0 - 0.5*op)*(1.0 - frown), -1.0, 1.0); gLSmX = gLSm*LIP_SMIRK*LIP_W;   // the whole mouth sits off-centre, its screen-right corner high
  gLSmile = max(gLSmile, 0.55*(1.0 - frown)); gLCheek = max(gLCheek, 0.35*(1.0 - frown));   // the cheek fold and the dimples of a standing grin
}
const vec3 C_SHELL = vec3(0.560,0.009,0.022);       // candy red
const vec3 C_LIMB  = vec3(0.700,0.600,0.595);       // arms and legs: a very pale pink
const vec3 C_GLOVE = vec3(0.860,0.858,0.862);
const vec3 C_HAT   = vec3(0.034,0.066,0.170);       // the hat: navy felt
const vec3 C_BAND  = vec3(0.880,0.840,0.740);       // its ribbon: cream
const vec3 C_BROW  = vec3(0.040,0.020,0.014);       // the brows: very dark brown

// ---- the hat : a small trilby on the back of the crown, tipped back and to one side; the brows float in front of it
mat3 gHat; vec3 gHatO;
float gBrU0, gBrU1, gBrT;          // brow lift per side (0 = screen-left), brow tilt (+ inner ends up)
float gMMGrin;                     // 1 = the standing smirk, 0 = a frown (the smile's folds go with it)
void mascotInit(){
  gBrU0 = clamp(gBrowL, -1.0, 1.0); gBrU1 = clamp(gBrowR, -1.0, 1.0);
  gBrT = clamp(gBTilt + 0.8*gKnit*smoothstep(0.0, 0.3, 0.5*(gBrU0 + gBrU1)) - 0.9*gKnit*(1.0 - smoothstep(0.0, 0.3, 0.5*(gBrU0 + gBrU1))), -1.0, 1.0);
  gMMGrin = 1.0 - clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  float bu = 0.5*(max(gBrU0, 0.0) + max(gBrU1, 0.0));
  gHat = rotZ(0.46 + 0.03*bu)*rotX(-0.14 - 0.05*bu);
  gHatO = vec3(-0.305, 0.578 + 0.020*bu, -0.125);
}
int gPHat;         // 1 felt, 2 ribbon, 3 brow
float mmHat(vec3 q){
  vec3 h = transpose(gHat)*(q - gHatO);
  float bd = length(h - vec3(0.0, 0.08, 0.0)) - 0.42;
  if(bd > 0.10){ gPHat = 1; return bd; }
  vec2 e = vec2(h.x, h.z/0.84);                      // a little oval, like the candy under it
  float rr = length(e);
  // brim: thin, turned up a little at the edge
  float by = 0.030*smoothstep(0.24, 0.365, rr);
  float brim = max(rr - 0.365, abs(h.y - by) - 0.007) - 0.009;
  // crown: tapers up, a lengthwise crease in the top, a pinch each side of the front; it runs down into the shell
  float pinch = 0.040*smoothstep(0.08, 0.24, h.y)*smoothstep(0.02, 0.18, h.z)*smoothstep(0.0, 0.12, abs(h.x));
  float cr = rr - (0.248 - 0.100*max(h.y, 0.0) - pinch);
  float top = 0.300 - 0.046*exp(-h.x*h.x/0.0045)*smoothstep(0.27, 0.0, rr) - 0.045*smoothstep(-0.05, 0.20, h.z);
  float crown = smax(cr, max(h.y - top, -0.16 - h.y), 0.032);
  float band = max(abs(h.y - 0.062) - 0.042, abs(cr - 0.004) - 0.006);
  float hat = min(min(brim, crown), band);
  gPHat = (band < min(brim, crown) + 0.002) ? 2 : 1;
  return hat*0.85;
}
// the brows float above the lid domes: thick rounded strokes, fat at the inner end.  Screen-left: a low wavy hook;
// screen-right: a tall arch, higher and further out (the cocked one)
float mmBrow(vec3 q, float sx, float up){
  float ck = sx > 0.0 ? 1.0 : 0.0;
  vec3 b = q - vec3(sx*mix(0.150, 0.235, ck), mix(0.655, 0.725, ck) + 0.060*up, 0.300);
  float bd = length(b) - 0.26;
  if(bd > 0.10) return bd;
  b.x *= sx;                                         // + = outward
  b.xy = rot(mix(0.10, 0.16, ck) + 0.55*gBrT)*b.xy;  // tilt lifts the inner ends (worried) or drops them (cross)
  float L = mix(0.112, 0.108, ck);
  float t = clamp((b.x + L)/(2.0*L), 0.0, 1.0);
  float y = b.y + mix(0.8, 5.0, ck)*b.x*b.x - mix(0.020, 0.0, ck)*sin(t*6.28318);
  float r = mix(0.054, 0.022, smoothstep(0.10, 1.0, t))*(1.0 + 0.30*gBThick);
  float dx = max(abs(b.x) - L, 0.0);
  return (length(vec3(dx, y, b.z*0.85)) - r)*0.80;
}
float mascotHeadBare(vec3 q){
  float hat = mmHat(q);
  float br = min(mmBrow(q, -1.0, gBrU0), mmBrow(q, 1.0, gBrU1));
  if(br < hat){ gPHat = 3; return br; }
  return hat;
}
float mmLetter(vec2 p);
// distance to the smile creases: arcs hugging the mouth corners (head space xy), the screen-left one long, the right one short
float mmCrease(vec2 p){
  vec2 a = p - vec2(-0.365, 0.105); float aa = atan(a.y, -a.x);
  float d1 = abs(length(a) - 0.118) + 0.20*max(abs(aa - 0.25) - 1.05, 0.0);
  vec2 b = p - vec2(0.405, 0.150); float ab = atan(b.y, b.x);
  float d2 = abs(length(b) - 0.085) + 0.20*max(abs(ab - 0.35) - 0.70, 0.0);
  return min(d1, d2*1.5);
}
float mascotHead(vec3 q){
  // the candy: a lentil standing on its edge, a little fuller at the bottom
  float d = sdEllipsoid(q, vec3(0.700, 0.665, 0.455));
  d -= 0.010*smoothstep(0.2, -0.6, q.y);
  if(q.z > 0.0){
    // a rounded dome of shell over each eye (the lids ride on them), and the letter stands a hair proud of the shell
    vec3 qa = vec3(abs(q.x), q.yz);
    d = smin(d, sdEllipsoid(qa - vec3(0.206, 0.400, 0.322), vec3(0.142, 0.130, 0.140)), 0.030);
    if(q.y > -0.10 && q.y < 0.32 && abs(q.x) > 0.20){
      float cr = mmCrease(q.xy);
      d += gMMGrin*(0.0085*smoothstep(0.022, 0.0, cr) - 0.0050*smoothstep(0.070, 0.020, cr)*smoothstep(0.0, 0.022, cr));     // a groove with a soft cheek roll beside it
    }
    if(q.y > -0.12 && q.y < 0.30){
      // the upper lip: a soft fold of shell rolling over the smile line
      float u = (q.x - 0.045)/0.430, uu = u*u;
      float yl = -0.020 + 0.190*uu/sqrt(1.0 + 0.39*uu) + 0.045*u;
      float t = (q.y - yl - 0.058)/0.046;
      d -= gMMGrin*0.0240*exp(-t*t)*smoothstep(1.05, 0.70, abs(u));
    }
    if(q.y < -0.10) d -= 0.0050*smoothstep(0.006, -0.006, mmLetter(rot(0.06)*(q.xy - vec2(0.030, -0.335))/1.05)*1.05)*smoothstep(0.02, 0.09, q.z);
  }
  return d;
}
float mascotTorso(vec3 p){
  // nothing to see: a small core inside the shell that the legs hang from
  vec3 b = p - (gMHip - M_HIP);
  return sdEllipsoid(b - vec3(0.0, 0.700, 0.0), vec3(0.250, 0.110, 0.180));
}
// the glove cuffs: a soft rolled ring round each wrist
float mascotExtra(vec3 p, out vec3 rest){
  float d = 1e3; rest = vec3(0.0);
  for(int i=0;i<2;i++){
    vec3 pw = p - gMWr[i];
    float bd = length(pw) - 0.17;
    if(bd > min(d, 0.10)){ d = min(d, bd); continue; }
    vec3 w = transpose(gMHand[i])*pw;                 // y along the forearm
    float t = length(vec2(length(w.xz) - 0.074, (w.y + 0.036)*0.55)) - 0.030;
    if(t < d){ d = t; rest = w + vec3(float(i), 0.0, 0.0); }
  }
  return d;
}
// the sneaker (foot frame: x across, y up, z to the toes; the ankle is the origin, the floor at y = -M_ANK_Y): a low toe
// box running back into a taller heel, a padded collar, a tongue, and a thick sole slab a little wider than the upper
#define MM_SHOE 1.26
float museFoot(vec3 p, int i, out float sole){
  vec3 q = transpose(gMFoot[i])*(p - gMAnk[i]);
  float bd = length(q - vec3(0.0, -0.030, 0.100)) - 0.52;
  if(bd > 0.12){ sole = 0.0; return bd; }
  q = vec3(q.x, q.y + M_ANK_Y, q.z)/MM_SHOE; q.y -= M_ANK_Y;
  // upper: one clean form whose widest section sits ON the sole — a long toe box rising along the instep into the heel, a padded collar
  float up = sdEllipsoid(q - vec3(0.0, -0.104, 0.122), vec3(0.146, 0.122, 0.250));
  up = smin(up, sdEllipsoid(q - vec3(0.0, -0.070, -0.020), vec3(0.128, 0.186, 0.172)), 0.085);
  up = smin(up, length(vec2(length(q.xz - vec2(0.0, -0.006)) - 0.070, q.y - 0.100)) - 0.032, 0.020);
  if(up < 0.06 && q.z > 0.04 && q.y > -0.02){
    // three laces: rounded bars across the instep, standing proud of it
    float lt = clamp(floor((q.z - 0.0575)/0.035), 0.0, 2.0);
    float zc = 0.075 + 0.035*lt;
    float yc = 0.094 - 0.68*(zc - 0.075);
    up = min(up, sdCapsule(q, vec3(-0.050, yc - 0.004, zc), vec3(0.050, yc - 0.004, zc), 0.0140));
  }
  up = max(up, -(q.y + 0.112));
  // sole: a thin band of rubber, flush with the upper
  vec2 e = vec2(q.x/0.150, (q.z - 0.112)/0.256);
  float sl = max((length(e) - 1.0)*0.150, abs(q.y + 0.127) - 0.013) - 0.004;
  float d = min(up, sl);
  d = max(d, -(q.y + M_ANK_Y - 0.002));
  sole = step(sl, up + 0.0005);
  return d*MM_SHOE;
}
// the letter: a heavy slab-serif lower-case m (three stems with feet, two round shoulders, a spur top-left); < 0 inside
float mmBox(vec2 p, vec2 b){ vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }
float mmLetter(vec2 p){
  const float st = 0.054, w = 0.200, cy = 0.012, y0 = -0.160, y1 = 0.160;
  float d = mmBox(p - vec2(-w, 0.0), vec2(st, y1));
  d = min(d, mmBox(p - vec2(0.0, 0.5*(y0 + cy)), vec2(st, 0.5*(cy - y0))));
  d = min(d, mmBox(p - vec2(w, 0.5*(y0 + cy)), vec2(st, 0.5*(cy - y0))));
  for(int k=0;k<2;k++){
    vec2 c = vec2((float(k) - 0.5)*w, cy);
    float ring = abs(length(p - c) - 0.5*w) - st;
    d = min(d, max(ring, c.y - p.y));
  }
  for(int k=0;k<3;k++){                                 // the feet: slabs
    float x = (float(k) - 1.0)*w;
    d = min(d, mmBox(p - vec2(x, y0 + 0.018), vec2(st + 0.017, 0.018)));
  }
  d = min(d, mmBox(p - vec2(-w - st - 0.012, y1 - 0.022), vec2(0.030, 0.022)));     // the spur
  return d - 0.006;
}
void faceEyeHook(vec3 e, vec2 uv, float lc, float lidL, inout Surf s){
  // a cartoon eye: clean white, a big tall pupil (black with a breath of plum), one soft upright catch-light and a faint second
  vec2 pu = uv/vec2(0.500, 0.740);
  float lp = length(pu);
  float pup = smoothstep(1.03, 0.97, lp)*step(0.0, e.z);
  vec3 c = EYE_COL_W*mix(0.84, 1.0, smoothstep(1.0, 0.50, length(e.xy)));
  vec3 pc = mix(vec3(0.020, 0.008, 0.018), vec3(0.070, 0.022, 0.050), smoothstep(0.30, 1.0, lp)*smoothstep(0.1, -0.9, pu.y));
  c = mix(c, pc, pup);
  s.alb = c*mix(0.82, 1.0, lc);
  vec2 hu = (uv - vec2(-0.110, 0.130))/vec2(0.085, 0.290);
  float hl = smoothstep(1.06, 0.70, length(hu))*step(0.0, e.z);
#ifdef LOOK_V3
  s.coat = 0.30*lc*(1.0 - hl); s.rough = 0.18;
#endif
  s.emit = vec3(1.0, 0.99, 0.97)*2.4*hl*lidL*lc + s.alb*0.16;
}
void mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s){
  if(gMPart == 0){
    float hd = mascotHeadBare(q);
    if(hd < 0.004 && hd <= gMHeadD + 0.002){
      if(gPHat == 3){
        // the brows: dark, glossy, like moulded liquorice
        s.alb = C_BROW; s.wrap = 0.10; s.sheen = 0.10; s.spec = 0.55; s.shin = 80.0; s.emit = C_BROW*0.05; s.mirror = 0.0;
#ifdef LOOK_V3
        s.rough = 0.30; s.coat = 0.45; s.sss = 0.0; s.sheenK = 0.0;
#endif
        return;
      }
      // wool felt, a satin ribbon
      vec3 h = transpose(gHat)*(q - gHatO);
      float rib = float(gPHat == 2);
      vec3 alb = mix(C_HAT*(0.95 + 0.10*vnoise3(h*30.0))*(0.97 + 0.06*vnoise3(h*120.0)), C_BAND*(0.95 + 0.05*sin(h.y*520.0)), rib);
      s.alb = alb; s.wrap = mix(0.45, 0.20, rib); s.sheen = mix(0.35, 0.25, rib); s.spec = mix(0.05, 0.30, rib); s.shin = mix(12.0, 60.0, rib);
      s.emit = alb*0.030; s.mirror = 0.0;
      s.T = normalize(cross(transpose(gHR)*gHat[1], n) + vec3(1e-4));
#ifdef LOOK_V3
      s.rough = mix(0.88, 0.40, rib); s.sheenK = mix(0.55, 0.20, rib); s.sheenCol = mix(vec3(0.45, 0.55, 0.80), vec3(1.0, 0.95, 0.85), rib);
      s.coat = 0.0; s.sss = 0.0;
#endif
      return;
    }
  }
  if(gMPart == 0 || gMPart == 1){
    // the candy shell: an even satin red, deeper toward the rim and the underside; the letter printed in white low on the belly
    vec3 alb = C_SHELL*(0.99 + 0.02*vnoise3(q*3.0));
    float rimw = smoothstep(0.45, 1.0, length(q.xy/vec2(0.700, 0.665)));
    alb *= 1.0 - 0.22*rimw*rimw;
    alb *= 0.90 + 0.17*clamp(dot(normalize(q/vec3(0.700, 0.665, 0.455)), vec3(-0.48, 0.62, 0.62)), 0.0, 1.0);     // a breath of the key, in the candy itself
    alb *= 1.0 - 0.12*smoothstep(0.10, -0.60, q.y);
    float aa = max(0.004, 0.9*gMFp);
    float ld = mmLetter(rot(0.06)*(q.xy - vec2(0.030, -0.335))/1.05)*1.05;
    float lt = smoothstep(aa, -aa, ld)*smoothstep(0.02, 0.09, q.z);
    alb = mix(alb, vec3(0.940, 0.938, 0.930), lt);
    s.alb = alb;
    s.wrap = 0.26; s.sheen = 0.05; s.spec = 0.34; s.shin = 40.0; s.mirror = 0.0;
    if(gMPart == 0 && q.z > 0.0) s.alb *= 1.0 - 0.30*gMMGrin*smoothstep(0.014, 0.0, mmCrease(q.xy));
    s.occ *= 1.0 - 0.22*smoothstep(0.016, 0.0, ld)*smoothstep(-0.004, 0.004, ld)*smoothstep(0.02, 0.09, q.z);     // a hair of shade round the print: it sits ON the shell
    s.emit = alb*(0.022 + 0.24*smoothstep(-0.05, -0.85, n.y) + 0.05*lt + 0.03*gMFace);      // the floor's bounce under the shell: dark red, never black
#ifdef LOOK_V3
    s.rough = 0.40; s.coat = 0.03*(1.0 - lt); s.sss = 0.50; s.sssCol = vec3(1.0, 0.12, 0.08); s.sheenK = 0.0;
#endif
    return;
  }
  bool glove = gMPart == 3 || gMPart == 5 || gMPart == 7 || (gMPart == 2 && gMLimbT > 0.93);
  if(glove){
    // glove / cuff / shoe: white, soft-touch; the shoe a little glossier, with a grey sole
    vec3 alb = C_GLOVE;
    float shoe = float(gMPart == 5);
    if(gMPart == 5){
      // the sole: off-white rubber, grey underneath; the upper: a stitched toe cap, grey laces across the tongue, eyelets
      vec3 fq = r - vec3(9.0 + 2.0*float(gMSide), 0.0, 0.0);
      fq = vec3(fq.x, fq.y + M_ANK_Y, fq.z)/MM_SHOE; fq.y -= M_ANK_Y;
      alb = mix(C_GLOVE, mix(vec3(0.800, 0.800, 0.810), vec3(0.45, 0.45, 0.48), smoothstep(-0.137, -0.142, fq.y)), gMSoleW);
      float upw = 1.0 - gMSoleW;
      float lt2 = clamp(floor((fq.z - 0.0575)/0.035), 0.0, 2.0);
      float zc2 = 0.075 + 0.035*lt2, yc2 = 0.094 - 0.68*(zc2 - 0.075);
      float lace = smoothstep(0.004, -0.002, sdCapsule(fq, vec3(-0.050, yc2 - 0.004, zc2), vec3(0.050, yc2 - 0.004, zc2), 0.0140))*step(0.04, fq.z);
      alb = mix(alb, vec3(0.34, 0.35, 0.40), 0.95*lace*upw);
      alb *= 1.0 - 0.30*smoothstep(0.005, 0.002, abs(fq.y + 0.112))*upw;                        // the seam over the sole
    }
    s.alb = alb;
    s.wrap = 0.40; s.sheen = 0.20; s.spec = mix(0.08, 0.30, shoe); s.shin = mix(18.0, 50.0, shoe);
    s.emit = alb*0.020;
#ifdef LOOK_V3
    s.rough = mix(0.70, 0.42, shoe); s.sheenK = 0.30*(1.0 - shoe); s.sheenCol = vec3(1.0);
    s.sss = 0.30; s.sssCol = vec3(1.0, 0.80, 0.78); s.coat = 0.18*shoe;
#endif
    return;
  }
  // arms and legs: smooth, very pale pink; a blush of the shell's red where they leave it
  float nearShell = gMPart == 2 ? smoothstep(0.30, 0.0, gMLimbT) : smoothstep(0.22, 0.0, gMLimbT);
  s.alb = mix(C_LIMB, C_LIMB*vec3(1.0, 0.80, 0.78), 0.55*nearShell)*(0.97 + 0.04*vnoise3(r*8.0));
  s.wrap = 0.40; s.sheen = 0.15; s.spec = 0.14; s.shin = 30.0;
  s.emit = s.alb*0.020 + vec3(0.30, 0.050, 0.040)*0.22*nearShell;      // the shell's red bounce: its shade on the limbs is warm, never black
#ifdef LOOK_V3
  s.rough = 0.50; s.sss = 0.80; s.sssCol = vec3(1.0, 0.55, 0.45); s.sheenK = 0.12; s.sheenCol = vec3(1.0, 0.9, 0.86);
#endif
}
#pragma include mascot_body

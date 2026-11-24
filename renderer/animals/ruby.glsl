// ---------------------------------------------------------------------------
// animals/ruby.glsl — RUBY, an original jellybean girl in the muse's classroom (our own design, not a brand mascot).
//  The whole body is one tall glossy cherry-red jellybean (the rig's "head": it carries the face), its lower half kicked a
//  little to one side the way a bean curves.  The jelly is translucent: light glows through its edges under a hard glaze.
//  Big brown eyes with lashes, thin arched brows, rosy cheeks, a small smile, a pink polka-dot bow on top; slim pink arms
//  and legs, bare four-finger hands, white ribbed ankle socks and red patent Mary-Jane shoes with a strap.
//  Skeleton / hand map / grasp = the muse's (engine/mascot_rig.glsl).
// ---------------------------------------------------------------------------
#define M_H 2.30
#define M_HEADC vec3(0.0, 1.34, 0.0)
#define M_HEADR vec3(0.60, 0.84, 0.52)
#define M_UPPER 0.36
#define M_FORE 0.33
#define MR_SH_X 0.450
#define MR_SH_Y 1.150
#define MR_HIP_X 0.170
#define MR_HAND_THICK 0.06
#define SHAPE_PIVOT vec3(0.0)
#define SHAPE_WORLD vec3(0.0, 1.34, 0.0)
#define MUSE_EMO 1
#define EYE_R0 0.165
#define EYE_C0 vec3(0.202,0.220,0.386)
#define EYE_R gEyeR
#define EYE_C gEyeC
#define EYE_AU 0.72
#define EYE_AL (-1.00)
#define EYE_AC (-0.08)
#define EYE_LID_TH 0.015
#define EYE_SQUINT 0.55
#define EYE_IRIS 0.640
#define EYE_PUPIL 0.330
#define EYE_COL_A vec3(0.560,0.250,0.070)          // warm chestnut, glowing low in the iris
#define EYE_COL_B vec3(0.075,0.028,0.014)
#define EYE_COL_W vec3(0.930,0.920,0.900)
#define LINER_COL vec3(0.040,0.008,0.012)
#define EYE_GLOW vec3(0.090,0.035,0.008)
#define EYE_CATCH 0.925
#define BROW_C vec3(0.202,0.440,0.0)
#define BROW_L 0.112
#define BROW_W 0.016
#define BROW_COL vec3(0.110,0.008,0.020)
#define BROW_H 0.016
#define BROW_RISE 0.065
#define MUSE_FROWN 0.40
#define BROW_SAD_TILT 1.4
#define LIP_O vec3(0.0,-0.075,0.500)
#define LIP_TILT 0.04
#define LIP_W 0.190
#define LIP_NARROW 0.85                             // the mouth against this face (engine/lips.glsl)
#define LIP_REACH 1.12                              // how far a spread or a grin may carry the corners
#define LIP_HU (0.030*LIPU)
#define LIP_HL (0.040*LIPU)
#define LIP_TU (0.006*LIPU)
#define LIP_TL (0.008*LIPU)
#define LIP_BOW 0.15
#define LIP_SQUARE 0.20
#define LIP_OPEN_Y 1.30
#define LIP_PUSH (0.035*LIPU)
#define LIP_SKIN 0.32
#define LIP_BLEND (0.010*LIPU)
#define LIP_TOOTH_Z (-0.14*LIPU)
#define LIP_TEETH_H ((0.030 + 0.050*gEmTeeth)*LIPU)
#define LIP_SCALLOP 0.0
#define LIP_GROOVE 0.0
#define LIP_COL_U vec3(0.520,0.040,0.090)
#define LIP_COL_L vec3(0.600,0.060,0.110)
#define LIP_COL_LINE vec3(0.070,0.002,0.008)
#define LIP_LINE 2.4
#define LIP_LINE_SOFT 0.40
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.18,0.010,0.022)
#define LIP_COL_DEEP vec3(0.035,0.003,0.006)
#define LIP_COL_GUM  vec3(0.32,0.030,0.050)
#define LIP_COL_TONGUE vec3(0.90,0.32,0.40)
#define LIP_COL_TOOTH vec3(0.93,0.91,0.88)
#define LIP_SPEC 0.50
#define LIP_SHIN 90.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.05
#define LIP_WRAP 0.10
#define LIP_EMIT vec3(0.02,0.0,0.0)
#define LIP_TONGUE_FREE 0.11
#pragma include face
#pragma include lips

#define MR_ARM_R vec3(0.074, 0.067, 0.061)
#define MR_LEG_R vec3(0.088, 0.080, 0.073)
#define MR_PALM_R vec3(0.100, 0.094, 0.054)
#define MR_FING_R 0.029
#define MR_FING_L 1.00
#define MR_THUMB_R 0.031
#define MR_REST_CURL 0.35
#define MR_FOOT_C vec3(0.0, -0.050, 0.065)
#define MR_FOOT_R vec3(0.136, 0.096, 0.205)
#define MR_K_NECK 0.02
#define MR_K_ARM 0.015
#define MR_K_LEG 0.015
#define MR_K_HAND 0.035
#define MR_K_FOOT 0.030
#define MR_HEAD_BOUND 1.00
#define MR_FACE_C vec3(0.0, 0.14, 0.36)
#define MR_FACE_RAD 0.58
#define MR_PUSH_C vec3(0.0, 1.34, 0.0)
#define MR_PUSH_R vec3(0.68, 0.92, 0.60)
#define MR_CHEEK vec2(0.300, 0.020)
#define MR_CHEEK_R 0.085
#define MR_CUSTOM_BEND 1
#define MR_HEAD_EXTRA 1
//@shadow-begin
#define MR_SHADOW_HEAD vec3(0.58, 0.82, 0.50)
//@shadow-end
#pragma include mascot_rig

void mascotMouthBend(){
  // a small sweet smile at rest; a knit brow with no smile turns it down
  float op = smoothstep(0.0, 0.6, max(uP[P_LIP_OPEN], uP[P_MOUTH_OPEN]));
  float frown = clamp(uP[P_BROW_KNIT], 0.0, 1.0)*(1.0 - smoothstep(0.0, 0.5, uP[P_SMILE]));
  float rest = 0.17*mix(0.40, 1.0, smoothstep(0.0, 0.05, uP[P_SMILE]));
  gLBend += LIP_W*((rest + 0.09*clamp(uP[P_SMILE], 0.0, 1.0))*(1.0 - 0.6*op) - MUSE_FROWN*frown*(1.0 - 0.15*op))*(1.0 - gLRnd);
}
const vec3 C_JELLY = vec3(0.640,0.010,0.050);       // cherry red
const vec3 C_LIMB  = vec3(0.930,0.380,0.430);       // arms, legs, hands: soft pink
const vec3 C_SOCK  = vec3(0.900,0.895,0.880);
const vec3 C_SHOE  = vec3(0.400,0.006,0.022);       // red patent
const vec3 C_BOW   = vec3(0.960,0.300,0.520);       // the bow: pink
const vec3 C_DOT   = vec3(0.950,0.940,0.930);

float mascotHead(vec3 q){
  // the jellybean: a tall rounded bean, both ends swung a little toward screen-left (a kidney curve), fuller at the bottom,
  // with a soft dent on the inner (screen-right) side where a bean curves in
  float yn = q.y/0.84;
  vec3 b = q; b.x += 0.125*yn*yn + 0.030*max(-yn, 0.0);
  float w = 1.0 + 0.060*smoothstep(0.10, -0.60, q.y);
  float d = sdEllipsoid(b, vec3(0.580*w, 0.840, 0.500*w));
  d += 0.040*exp(-(q.y + 0.10)*(q.y + 0.10)/0.12)*smoothstep(0.10, 0.60, q.x);
  return d;
}
float mascotTorso(vec3 p){
  // nothing to see: a small core inside the bean that the legs hang from
  vec3 b = p - (gMHip - M_HIP);
  return sdEllipsoid(b - vec3(0.0, 0.660, 0.0), vec3(0.260, 0.120, 0.200));
}

// ---- head extras (never carved by the lips): lashes on each eye, the bow on top
int gRx;            // 1 lash, 2 bow loop, 3 bow knot
vec3 gBowL;         // bow-local position of the hit (for the dots)
float rubyLashes(vec3 q){
  vec3 qa = vec3(abs(q.x), q.yz);
  float blink = q.x < 0.0 ? gBlinkL : gBlinkR;
  vec3 c = gEyeC;
  float R = gEyeR + EYE_LID_TH + 0.004;
  float d = 1e3;
  for(int k=0;k<5;k++){
    float th = 0.78 - 0.17*float(k);                                   // the upper lid's outer half
    vec3 dir = normalize(vec3(cos(th), sin(th)*(1.0 - 0.85*blink), 0.62));
    vec3 a = c + dir*R;
    float L = (0.040 + 0.016*smoothstep(0.0, 3.0, float(k)))*(1.0 - 0.3*step(3.5, float(k)));
    vec3 m = a + normalize(vec3(cos(th) + 0.5, sin(th) + 0.25, 0.25))*L*0.55;
    vec3 tip = m + normalize(vec3(cos(th) + 0.35, sin(th) + 1.3, -0.30))*L*0.60;
    d = min(d, min(sdTaper(qa, a, m, 0.0090, 0.0060), sdTaper(qa, m, tip, 0.0060, 0.0016)));
  }
  return d;
}
float rubyBow(vec3 q){
  vec3 l = q - vec3(-0.060, 0.855, 0.020);
  l.xy = rot(0.30)*l.xy;                                               // tipped toward screen-right
  l /= 1.22;                                                           // bigger
  if(length(l) > 0.40) return (length(l) - 0.34)*1.22;
  gBowL = l;
  vec3 la = vec3(abs(l.x), l.yz);
  vec3 lp = la - vec3(0.165, 0.030, 0.0);
  lp.xy = rot(-0.30)*lp.xy;
  // a loop: a puffy pillow pinched toward the knot, with soft folds running into it
  float pinch = smoothstep(0.16, -0.06, lp.x);
  float loop = sdEllipsoid(lp, vec3(0.160, 0.112*(1.0 - 0.50*pinch), 0.072*(1.0 - 0.35*pinch)));
  loop += 0.005*sin(atan(lp.y, lp.x + 0.20)*9.0)*smoothstep(0.02, 0.12, lp.x);
  vec3 lt = la - vec3(0.080, -0.090, 0.034);
  lt.xy = rot(0.50)*lt.xy;
  float tails = sdEllipsoid(lt, vec3(0.040, 0.095, 0.018));
  vec3 lk = l - vec3(0.0, 0.0, 0.012);
  float knot = sdEllipsoid(lk, vec3(0.055, 0.074, 0.064)) + 0.004*sin(lk.y*70.0);   // a wrapped knot with two creases
  float d = min(smin(loop, tails, 0.02), knot);
  gRx = knot <= d + 1e-4 ? 3 : 2;
  return d*1.22;
}
float mascotHeadBare(vec3 q){
  float la = rubyLashes(q);
  float bw = rubyBow(q);
  if(la < bw) gRx = 1;
  return min(la, bw);
}

void mascotMaterial(vec3 pw, vec3 q, vec3 r, vec3 n, inout Surf s){
  float rx = gMPart == 0 ? mascotHeadBare(q) : 1e3;
  if(rx < gMHeadD + 0.002 && rx <= gMBare + 0.001){
    if(gRx == 1){
      s.alb = vec3(0.060, 0.012, 0.016); s.wrap = 0.20; s.sheen = 0.20; s.spec = 0.25; s.shin = 50.0; s.emit = vec3(0.004, 0.0, 0.0);
#ifdef LOOK_V3
      s.rough = 0.35; s.sheenK = 0.0; s.sss = 0.0; s.coat = 0.0;
#endif
      return;
    }
    // the bow: satin pink with white polka dots; the knot a deeper pink, slightly shiny
    vec3 bl = gBowL/0.062;
    vec3 ip = floor(bl), fp = fract(bl);
    float dot_ = smoothstep(0.30, 0.22, length(fp.xy - 0.5));
    vec3 alb = gRx == 3 ? C_BOW*0.82 : mix(C_BOW, C_DOT, dot_*step(0.5, fract(ip.z*0.5 + 0.25) + 0.6));
    s.alb = alb; s.wrap = 0.35; s.sheen = 0.40; s.spec = 0.45; s.shin = 50.0; s.emit = alb*0.04;
    s.occ *= 1.0 - 0.40*smoothstep(0.05, 0.0, mascotHead(q));
    s.T = normalize(cross(n, vec3(0.0, 0.0, 1.0)) + vec3(1e-4));
#ifdef LOOK_V3
    s.rough = 0.38; s.sheenK = 0.45; s.sheenCol = vec3(1.0, 0.85, 0.92); s.sss = 0.25; s.sssCol = vec3(1.0, 0.5, 0.6); s.coat = 0.0;
#endif
    return;
  }
  if(gMPart == 0 || gMPart == 1){
    // the jelly: deep cherry red, lighter where light passes through its thin edges, a hard glaze with a faint orange peel
    float edge = pow(1.0 - clamp(dot(n, normalize(uCamPos - pw)), 0.0, 1.0), 2.0);
    vec3 alb = C_JELLY*(0.98 + 0.04*vnoise3(q*3.0));
    float speck = smoothstep(0.62, 0.80, vnoise3(q*38.0))*0.5 + smoothstep(0.70, 0.85, vnoise3(q*85.0 + 3.1))*0.5;
    alb *= 1.0 - 0.10*speck;                                            // the faint mottling of a real candy coat
    // rosy cheeks: a soft pinker flush under each eye
    vec2 ck = vec2(abs(q.x), q.y) - vec2(0.305, 0.015);
    float blush = exp(-dot(ck, ck)/(0.080*0.080))*smoothstep(0.10, 0.30, q.z);
    alb = mix(alb, vec3(1.00, 0.40, 0.52), 0.62*blush);
    s.alb = alb;
    vec3 pn = vec3(vnoise3(q*22.0), vnoise3(q*22.0 + 7.3), vnoise3(q*22.0 + 13.1)) - 0.5;
    s.n = normalize(n + 0.006*pn);
    // the bow's contact shade on the top of the bean
    float bowSh = smoothstep(0.08, 0.0, rubyBow(q));
    s.occ *= 1.0 - 0.45*bowSh;
    s.wrap = 0.14; s.sheen = 0.55; s.spec = 1.60; s.shin = 900.0; s.mirror = 0.035;
    s.emit = alb*0.030 + vec3(0.110, 0.006, 0.020)*edge;
    s.alb *= 1.0 - 0.10*(1.0 - edge)*smoothstep(-0.2, -0.7, q.y);
    s.trans = 0.35; s.transCol = vec3(1.0, 0.18, 0.22);
    // the glaze's hot spot: a crisp soft-edged window highlight high on the bean's lit shoulder (it rides the bean), and a
    // thin second glint low on the opposite side — the way a lacquered candy catches a studio softbox
    vec3 nh = gHR*n;
    vec3 hd = normalize(vec3(-0.42, 0.52, 0.74));
    float hs = smoothstep(0.935, 0.990, dot(nh, hd))*smoothstep(0.0, 0.5, 1.0 - abs(dot(nh - hd*dot(nh, hd), normalize(vec3(0.6, 0.5, 0.0)))*6.0));
    s.emit += vec3(1.0, 0.90, 0.90)*0.0*hs;
#ifdef LOOK_V3
    s.rough = 0.14; s.coat = 1.0; s.sss = 0.70; s.sssCol = vec3(1.0, 0.16, 0.22); s.sheenK = 0.0;
#endif
    return;
  }
  bool shoe = gMPart == 5;
  bool sock = gMPart == 4 && gMLimbT > 0.74;
  if(shoe){
    // red patent Mary-Jane: a glossy rounded toe, a strap across the instep with a little white button, a dark sole
    vec3 fq = r - vec3(9.0 + 2.0*float(gMSide), 0.0, 0.0);
    vec3 alb = C_SHOE;
    float strap = smoothstep(0.012, 0.004, abs(fq.z - 0.000) - 0.020)*step(-0.035, fq.y);
    float btn = smoothstep(0.016, 0.010, length(vec2(abs(fq.x) - 0.085, fq.z - 0.0)))*step(0.0, fq.y + 0.06);
    alb = mix(alb, C_SHOE*0.70, strap);
    alb = mix(alb, vec3(0.92, 0.90, 0.86), btn);
    float welt = smoothstep(-0.062, -0.070, fq.y)*(1.0 - gMSoleW);
    alb = mix(alb, vec3(0.86, 0.80, 0.74), 0.85*welt);                 // a pale welt round the sole
    alb = mix(alb, vec3(0.06, 0.03, 0.03), gMSoleW*0.95);
    s.alb = alb; s.wrap = 0.08; s.sheen = 0.20; s.spec = 1.20; s.shin = 300.0; s.mirror = 0.05; s.emit = alb*0.03;
#ifdef LOOK_V3
    s.rough = 0.14; s.coat = 0.9; s.sss = 0.05; s.sssCol = vec3(1.0, 0.2, 0.2); s.sheenK = 0.0;
#endif
    return;
  }
  if(sock){
    // white ribbed ankle sock with a little frilled cuff
    float rib = 0.5 + 0.5*sin(atan(r.y, r.z)*18.0);
    float cuff = smoothstep(0.74, 0.78, gMLimbT)*smoothstep(0.84, 0.80, gMLimbT);
    vec3 alb = C_SOCK*(0.94 + 0.05*rib)*(1.0 - 0.06*cuff);
    s.alb = alb; s.wrap = 0.40; s.sheen = 0.25; s.spec = 0.06; s.shin = 16.0; s.emit = alb*0.045;
#ifdef LOOK_V3
    s.rough = 0.80; s.sheenK = 0.40; s.sheenCol = vec3(1.0); s.sss = 0.20; s.sssCol = vec3(1.0, 0.9, 0.85);
#endif
    return;
  }
  // arms, legs, hands: soft pink, smooth and a little satiny
  s.alb = C_LIMB*(0.96 + 0.06*vnoise3(r*8.0));
  s.wrap = 0.50; s.sheen = 0.30; s.spec = 0.45; s.shin = 120.0;
  s.emit = s.alb*0.060;
#ifdef LOOK_V3
  s.rough = 0.30; s.sss = 0.50; s.sssCol = vec3(1.0, 0.45, 0.45); s.sheenK = 0.08; s.sheenCol = vec3(1.0, 0.9, 0.9); s.coat = 0.55;
#endif
}
#pragma include mascot_body

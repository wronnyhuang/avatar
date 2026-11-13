// ---------------------------------------------------------------------------
//  engine/shape_arms.glsl — real 3-D arms with mitten hands for the six shape characters (docs/RIG.md).
//  Part of the character SDF (same march, TAA, DoF, sun, shadow and AO as the face), on the contract's positional
//  hand columns: hand_{l,r}_{x,y,z} = the wrist's offset from its rest in units of one full arm length (x outward
//  from the midline on that side, y up, z toward the camera), hand_*_open, hand_*_turn, thumb_*, point_*, shoulder_*.
//  Side 0 = screen left (_l), 1 = screen right (_r).
//
//  A shape defines, before `#pragma include shape_arms`:
//    SA_SHO   vec3   the screen-right shoulder, shape (head) space; mirrored for the left.  Just inside the host.
//    SA_REST  vec3   the rest wrist, offset from the shoulder in WORLD axes, x outward (mirrored)
//    SA_LEN   float  shoulder to wrist, fully stretched
//    SA_R     float  arm radius at the shoulder (tapers to 0.72 of it at the wrist)
//    SA_HAND  float  mitten size (palm length)
//    SA_SKIN  float  how far the shoulder melts into the host (smooth-min radius)
//  and calls saInit() at the end of charInit, saDist(p) in its Eval, saSurface() in charSurface for arm hits.
// ---------------------------------------------------------------------------
#ifndef SA_SKIN
#define SA_SKIN 0.06
#endif
#ifndef SA_ALB
#define SA_ALB RIG_ARM
#endif
#ifndef SA_SPEC
#define SA_SPEC 0.22
#endif
#ifndef SA_MIRROR
#define SA_MIRROR 0.0
#endif
#ifndef SA_HALB
#define SA_HALB RIG_PAW
#endif

float charShadowHost(vec3 pw);       // the shape's own proxy (defined after this include)
vec3  gSAS[2], gSAE[2], gSAW[2];     // shoulder, elbow, wrist (world)
mat3  gSAH[2];                       // hand frame: x toward the thumb, y to the fingertips, z out of the palm
float gSAOpen[2], gSAPoint[2], gSAThumb[2];
vec4  gSABound[2];                   // bounding sphere of each arm (world)
float gSAHandW;                      // 1 on the mitten .. 0 on the arm (paint)

vec3 saToWorld(vec3 q){ return transpose(gHR)*((q - SHAPE_PIVOT)*vec3(gSqX, gSqY, gSqX)) + gHO; }

void saSolve(int i){
  float s = i == 0 ? -1.0 : 1.0;
  float hx = uP[i == 0 ? P_HAND_L_X : P_HAND_R_X], hy = uP[i == 0 ? P_HAND_L_Y : P_HAND_R_Y], hz = uP[i == 0 ? P_HAND_L_Z : P_HAND_R_Z];
  float open = clamp(uP[i == 0 ? P_HAND_L_OPEN : P_HAND_R_OPEN], 0.0, 1.0);
  float turn = clamp(uP[i == 0 ? P_HAND_L_TURN : P_HAND_R_TURN], -1.0, 1.0);
  float shr = clamp(uP[i == 0 ? P_SHOULDER_L : P_SHOULDER_R], 0.0, 1.0);
  gSAPoint[i] = clamp(uP[i == 0 ? P_POINT_L : P_POINT_R], 0.0, 1.0);
  gSAThumb[i] = clamp(uP[i == 0 ? P_THUMB_L : P_THUMB_R], 0.0, 1.0);
  gSAOpen[i] = open*(1.0 - gSAPoint[i]);
  vec3 S = saToWorld(vec3(s*SA_SHO.x, SA_SHO.y, SA_SHO.z)) + vec3(0.0, 0.10*SA_LEN*shr, 0.0);
  vec3 h = clamp(vec3(hx, hy, hz), vec3(-1.2), vec3(1.4));
  // x inward is scaled to the shape's half width, so x = -0.45 (the contract's clap) meets at the midline
  float xin = (SA_SHO.x + SA_REST.x)/0.45;
  vec3 T = S + vec3(s*SA_REST.x, SA_REST.y, SA_REST.z) + vec3(s*(h.x >= 0.0 ? SA_LEN*h.x : xin*h.x), SA_LEN*h.y, SA_LEN*h.z);
  // the hand stays outside the host and on its own side of the midline (it may reach it: claps meet there)
  for(int k=0;k<3;k++){
    float d = charShadowHost(T);
    float need = SA_R*1.3 + 0.55*SA_HAND;
    if(d < need){
      const vec2 e = vec2(0.02, 0.0);
      vec3 g = normalize(vec3(charShadowHost(T + e.xyy) - charShadowHost(T - e.xyy), charShadowHost(T + e.yxy) - charShadowHost(T - e.yxy),
                              charShadowHost(T + e.yyx) - charShadowHost(T - e.yyx)) + 1e-5);
      T += g*(need - d);
    }
  }
  T.x = s*max(s*T.x, 0.55*SA_HAND*0.5 + 0.01);
  // two bones, fixed lengths, a soft reach cap; the elbow hangs down, back and out, and swings out as the hand rises
  float U = 0.5*SA_LEN, F = 0.5*SA_LEN;
  vec3 v = T - S;
  float d0 = length(v);
  vec3 dir = v/max(d0, 1e-4);
  float d = clamp(smin(d0, 0.97*SA_LEN, 0.10*SA_LEN), 0.30*SA_LEN, 0.97*SA_LEN);
  vec3 W = S + dir*d;
  float up = clamp((W.y - S.y)/SA_LEN + 0.3, 0.0, 1.3);
  vec3 pole = vec3(s*(0.45 + 0.45*up), -1.0 + 0.55*up, -0.35 + 0.15*up);
  vec3 n = normalize(pole - dir*dot(pole, dir) + 1e-4);
  float a = (U*U - F*F + d*d)/(2.0*d);
  float hh = sqrt(max(U*U - a*a, 0.0));
  vec3 E = S + dir*a + n*hh;
  gSAS[i] = S; gSAE[i] = E; gSAW[i] = W;
  // the hand: along the forearm, a slack one droops; the palm faces the midline at turn 0, the camera at +1
  vec3 df = normalize(W - E);
  float slack = 1.0 - smoothstep(0.0, 0.8, open);
  vec3 hyv = normalize(df + vec3(0.0, -0.25*slack*(1.0 - gSAPoint[i]), 0.0) + vec3(0.0, 0.45, 0.0)*gSAPoint[i]*smoothstep(0.3, 0.9, df.z));
  // (a finger aimed at the lens tilts up a little, so it reads as a finger and not a dot)
  vec3 p0 = vec3(-s, 0.0, 0.0);
  p0 = normalize(p0 - hyv*dot(p0, hyv) + vec3(0.0, 0.0, 0.05));
  vec3 b = cross(hyv, p0);
  b *= b.z < 0.0 ? -1.0 : 1.0;
  float ang = 1.45*turn;
  vec3 pn = normalize(cos(ang)*p0 + sin(ang)*b);
  // a raised palm-out hand faces the lens more than its forearm allows; a hand at the face or chest faces it
  vec3 hx_ = normalize(s*cross(pn, hyv));
  gSAH[i] = mat3(hx_, hyv, pn);
  float r = 0.5*SA_LEN + SA_HAND*1.6 + SA_R;
  gSABound[i] = vec4(0.5*(S + W + hyv*SA_HAND), r);
}

void saInit(){ saSolve(0); saSolve(1); }

// the mitten, hand frame (y along the fingers, z out of the palm, x toward the thumb)
float saMitten(int i, vec3 l){
  float H = SA_HAND, o = gSAOpen[i], pt = gSAPoint[i], th = gSAThumb[i];
  float fist = 1.0 - o;
  // palm + finger pad: open it is a flat paddle, closed a ball
  vec3 c = vec3(0.0, mix(0.42, 0.55, o)*H, mix(0.06, 0.0, o)*H);
  vec3 rr = vec3(mix(0.40, 0.44, o), mix(0.42, 0.60, o), mix(0.34, 0.20, o))*H;
  float d = sdEllipsoid(l - c, rr);
  d = smin(d, sdTaper(l, vec3(0.0, -0.10*H, 0.0), vec3(0.0, 0.25*H, 0.0), 0.30*H, 0.36*H), 0.08*H);    // the wrist
  // the finger end curls toward the palm when closed
  vec3 tip = vec3(0.0, mix(0.62, 0.98, o)*H, mix(0.30, 0.02, o)*H);
  d = smin(d, sdEllipsoid(l - tip, vec3(0.36, 0.22, 0.16)*H), 0.10*H);
  // the index finger, pointing
  if(pt > 0.01){
    float pf = sdTaper(l, vec3(0.16*H, 0.70*H, 0.04*H), vec3(0.18*H, (0.70 + 0.70*pt)*H, 0.05*H), 0.13*H, 0.10*H);
    d = smin(d, pf, 0.05*H);
  }
  // the thumb: tucked over the fist .. out at the side .. straight up the hand's x for a thumbs-up
  vec3 t0 = vec3(0.26*H, 0.30*H, 0.06*H);
  vec3 tOut = vec3(0.58*H, 0.62*H, 0.08*H), tIn = vec3(0.10*H, 0.58*H, 0.36*H), tUp = vec3(0.80*H, 0.36*H, 0.04*H);
  vec3 t1 = mix(mix(tIn, tOut, o), tUp, th);
  d = smin(d, sdTaper(l, t0, t1, 0.15*H, 0.12*H), 0.06*H);
  return d;
}

float saArm(int i, vec3 p, out float hand){
  vec3 S = gSAS[i], E = gSAE[i], W = gSAW[i];
  float up = sdTaper(p, S, E, SA_R, 0.86*SA_R);
  float lo = sdTaper(p, E, W, 0.86*SA_R, 0.72*SA_R);
  float arm = smin(up, lo, 0.35*SA_R);
  vec3 l = (p - W)*gSAH[i];
  float m = saMitten(i, l);
  hand = smoothstep(0.02, -0.02, m - arm);
  return smin(arm, m, 0.20*SA_R);
}

// distance to both arms; guarded so the rest of the shape does not pay for them
float saDist(vec3 p){
  float d = 1e3;
  gSAHandW = 0.0;
  for(int i=0;i<2;i++){
    float b = length(p - gSABound[i].xyz) - gSABound[i].w;
    float di = b;
    float hw = 0.0;
    for(int k=0;k<1 && b < 0.05;k++) di = saArm(i, p, hw);
    if(di < d){ d = di; gSAHandW = hw; }
  }
  return d;
}

// a cheap stand-in for the shadow pass: two capsules and a ball per arm
float saShadow(vec3 p){
  float d = 1e3;
  for(int i=0;i<2;i++){
    d = min(d, sdTaper(p, gSAS[i], gSAE[i], 0.8*SA_R, 0.7*SA_R));
    d = min(d, sdTaper(p, gSAE[i], gSAW[i], 0.7*SA_R, 0.6*SA_R));
    d = min(d, length(p - gSAW[i] - gSAH[i][1]*0.5*SA_HAND) - 0.40*SA_HAND);
  }
  return d;
}

void saSurface(vec3 p, vec3 rd, inout Surf s){
  saDist(p);
  vec3 alb = mix(SA_ALB, SA_HALB, gSAHandW);
  float g = vnoise3(p*14.0);
  s.alb = alb*(0.92 + 0.14*g);
  s.spec = SA_SPEC; s.shin = 36.0; s.wrap = 0.22; s.sheen = 0.30; s.fur = 0.0; s.occ = 1.0; s.mirror = SA_MIRROR;
  s.trans = 0.0; s.emit = vec3(0.0);
}

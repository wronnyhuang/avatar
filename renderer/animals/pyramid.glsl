// ---------------------------------------------------------------------------
//  animals/pyramid.glsl — a squat, friendly sandstone pyramid sitting in the dunes.  Head space: origin on the
//  axis, +x screen right, +y up, +z toward the camera, foot of the profile at y = PY_Y0, tip at PY_Y0 + PY_H.
//
//  The host is a rounded square whose half width follows a hyperbola (a round tip, straight below it) plus a
//  belly: a chubby triangle, and a front face that leans back less low down, where the mouth and the eyes sit.
//  The rig mirrors the eyes about x = 0, so the 3/4 turn has to be SHAPE_YAW0, not a turned host.
//
//  The sand.  The character is always composited over the background, so it removes itself below the set's own
//  ground (sandNear, world space).  It rocks IN the sand, and the sand line needs no matching.
// ---------------------------------------------------------------------------
#ifndef SET_SAND
#define sandNear(xz) SP_GROUND_Y
#endif

const float PY_Y0 = -0.72;                          // foot of the profile, a little under the sand
const float PY_H  = 1.60;
const float PY_TIPR = 0.28;                         // the hyperbola's blunt end: the tip's softness
const float PY_K  = 2.06;
const float PY_BELLY = 0.10;
const float PY_RE = 0.20;                           // ridge radius

#define SHAPE_PIVOT vec3(0.0,-0.72,0.0)
#define SHAPE_WORLD vec3(0.0,-0.90,0.0)
#define SHAPE_YAW0  (-0.33)                         // front face toward the sun, the second face shows on the right
#define SHAPE_SQUASH 0.020                          // stone, but a chubby one
#define SHAPE_JIGGLE 0.012
// the sloped face narrows upward: eyes wide apart at mid height, mouth low and wide
#define EYE_R 0.182
#define EYE_C vec3(0.345,0.165,0.560)
#define EYE_AU 0.68
#define EYE_AL (-0.74)
#define EYE_AC (-0.30)
#define EYE_TILT 0.08
#define EYE_IRIS 0.86
#define EYE_PUPIL 0.52
#define EYE_COL_A vec3(0.62,0.30,0.085)
#define EYE_COL_B vec3(0.13,0.058,0.024)
#define BROW_W 0.050
#define BROW_L 0.190
#define BROW_H 0.030
#define BROW_RISE 0.135
#define BROW_COL vec3(0.35,0.190,0.100)
#define BROW_C vec3(0.305,0.405,0.0)
#define LINER_COL vec3(0.14,0.066,0.052)
#define LIP_TILT (-0.03)                            // aims at the camera: the host normal here leans 0.42 the other way
#define LIP_O vec3(0.0,-0.155,0.8080)               // on the surface, y = -0.155
#define LIP_W 0.352                                 // a broad carved mouth, like a pharaoh's
#define LIP_HU (0.060*LIPU)                         // shallow relief: cut from the block, not laid on it
#define LIP_HL (0.072*LIPU)
#define LIP_TU (0.016*LIPU)
#define LIP_TL (0.020*LIPU)
#define LIP_BOW 0.75
#define LIP_SQUARE 0.14                             // a crisp cut corner
#define LIP_OPEN_Y 0.92
#define LIP_BLEND (0.0055*LIPU)
#define LIP_SKIN 0.30
#define LIP_PUSH (0.060*LIPU)
#define LIP_TOOTH_Z (-0.155*LIPU)                   // the face recedes above the mouth: the teeth stay behind it
#define LIP_AIM 0.37                                // radians about y: see mouthPoint
#define LIP_SCALLOP 0.5
#define LIP_GROOVE 0.24
#define LIP_TEETH_H (0.064*LIPU)
#define LIP_TOOTH_GAIN 1.12
#define LIP_COL_U vec3(0.60,0.34,0.235)
#define LIP_COL_L vec3(0.68,0.40,0.270)
#define LIP_COL_LINE vec3(0.115,0.044,0.034)
#define LIP_LINE 1.8                                // sand on sand: the closed line has to carry the mouth
#define LIP_LINE_SOFT 0.40
#define LIP_FLOW 0.45
#define LIP_COL_CAVE vec3(0.36,0.150,0.090)         // shadow in a rock-cut doorway, warm
#define LIP_COL_DEEP vec3(0.095,0.040,0.034)
#define LIP_COL_GUM  vec3(0.42,0.20,0.13)
#define LIP_COL_TONGUE vec3(0.78,0.36,0.24)
#define LIP_COL_TOOTH vec3(0.95,0.87,0.72)           // pale limestone
#define LIP_SPEC 0.03
#define LIP_SHIN 14.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.40
#define LIP_WRAP 0.42                                // the sun rakes this face: a lip must not throw a hard terminator
#define LIP_EMIT vec3(0.10,0.055,0.040)
#pragma include face
#pragma include lips
// real 3-D arms with mitten hands on the contract's hand columns (engine/shape_arms.glsl, docs/RIG.md)
#define SA_SHO vec3(0.780,-0.250,0.050)
#define SA_REST vec3(0.140,-0.300,0.250)
#define SA_LEN 0.850
#define SA_R 0.090
#define SA_HAND 0.180
#define SA_ALB vec3(0.78,0.56,0.34)
#define SA_HALB vec3(0.82,0.62,0.40)
#define SA_SPEC 0.06
#define SA_MIRROR 0.0
#pragma include shape_arms

float gSkin, gBare, gChip, gEb;
vec3  gQ;

void charInit(){ faceInit(); lipsInit();  saInit(); }

//@shadow-begin
float gRa;                 // near 0 along a ridge, negative on a face
float pyrHost(vec3 q){
  float tip = PY_Y0 + PY_H;
  float Y = PY_TIPR + max(tip - q.y, 0.0)/PY_K;
  float sq = sqrt(max(Y*Y - PY_TIPR*PY_TIPR, 1e-6));
  float t = clamp((q.y - PY_Y0)/PY_H, 0.0, 1.0);
  float w = sq + PY_BELLY*4.0*t*(1.0 - t);
  float dw = -Y/(sq*PY_K) + PY_BELLY*4.0*(1.0 - 2.0*t)/PY_H;
  float r = w*PY_RE/sqrt(w*w + PY_RE*PY_RE);
  vec2 a = abs(q.xz) - (w - r);
  gRa = min(a.x, a.y);
  float d2 = length(max(a, vec2(0.0))) + min(max(a.x, a.y), 0.0) - r;
  return max(d2/sqrt(1.0 + min(dw*dw, 9.0)), q.y - tip);
}
float charShadowHost(vec3 pw){
  return pyrHost(gHR*(pw - gHO) + SHAPE_PIVOT) + 0.040;
}
float charShadow(vec3 pw){ return min(charShadowHost(pw), saShadow(pw)); }
//@shadow-end

// a few small chips, bites out of the ridges.  Irregular, or they read as drill holes or paper cut-outs
#define NCH 5
const vec4 CHIP[NCH] = vec4[NCH](
  vec4(-0.500, 0.37, 0.500,0.034), vec4(-0.765,-0.12, 0.765,0.044), vec4( 0.522, 0.31, 0.522,0.030),
  vec4( 0.712,-0.03, 0.712,0.040), vec4(-0.890,-0.40, 0.890,0.040));

// LIP_TILT aims the mouth in pitch only, and this face is turned 0.33 away: the opening's carve, teeth and tongue
// would all be seen from the side, sliding out of the opening.  So the rig is handed the point turned about LIP_O: its
// frame looks at the camera, and depth still comes from the true host.  0.33 of turn, plus 0.05 because the mouth sits
// left of the lens.  Any less and the view rays drift across the cavity and its wall hides one end of the teeth
vec3 mouthPoint(vec3 q){
  vec3 v = q - LIP_O;
  float c = cos(LIP_AIM), sn = sin(LIP_AIM);
  return LIP_O + vec3(c*v.x - sn*v.z, v.y, sn*v.x + c*v.z);
}

float layerTone(float y, float f, float seed){
  float v = y*f + 0.6*sin(y*f*0.37 + seed);
  float i = floor(v);
  return mix(hash11(i*1.37 + seed), hash11((i + 1.0)*1.37 + seed), smoothstep(0.42, 0.58, v - i));
}
float rockH(vec3 p){ return vnoise3(p*14.0)*0.56 + vnoise3(p*36.0 + 3.0)*0.36 + vnoise3(p*97.0 + 7.0)*0.08; }

void pyrEval(vec3 pw){
  vec3 q = shapeSpace(pw);
  gQ = q;
  float host = pyrHost(q);
  // the ridges weather along the beds: the outline steps a little and never runs ruler straight
  float rg = smoothstep(-0.13, 0.03, gRa);
  host += rg*0.0030*(sin(q.y*29.0 + 3.0*sin(q.y*7.0 + q.x*5.0)) + 0.8*sin(q.y*71.0 + 2.0*sin(q.y*17.0)));
  // the eyes sit in a soft dish that turns the slope toward the camera: below the eye the surface falls back, above
  // it comes forward.  Without this the slope cuts the lower half of the eyeball, and the eye reads half shut
  vec2 ed = vec2(abs(q.x) - EYE_C.x, q.y - EYE_C.y);
  float eb = lipsBell(length(ed/vec2(0.33,0.31)));
  gEb = eb;
  host += eb*(0.030 - 0.34*ed.y);
  float cn = sin(q.x*61.0 + 1.0)*sin(q.y*53.0)*sin(q.z*47.0 + 2.0) + 0.5*sin(q.y*131.0 + q.x*97.0);
  float chip = 1e3;
  for(int i=0;i<NCH;i++){
    vec3 v = q - CHIP[i].xyz;
    chip = min(chip, length(v*vec3(1.0,1.5,1.0)) - CHIP[i].w*(1.0 + 0.42*cn));
  }
  gChip = chip;
  host = smax(host, -chip, 0.012);
  float skin = lipsEval(mouthPoint(q), host);
  skin = faceEval(q, skin, host);
  gSkin = max(skin, sandNear(pw.xz) - pw.y);
  gBare = min(gLipBare, gFEye);
}
void charEval(vec3 p){
  pyrEval(p);
  // bare skin, so: fur length a hair below zero and skin just under the bare parts (see monkey.glsl)
  gcBase = smin(min(gSkin, gBare - 0.0007), saDist(p), SA_SKIN); gcBare = gBare; gcRest = gQ; gcLen = -0.0006;
}
float charTuft(){ return 0.0; }

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  pyrEval(pw);
  if(saDist(pw) < gSkin){ saSurface(pw, rd, s); return; }
  vec3 q = gQ, n = s.n;
  // optics follow the HOST's normal: features must not throw their own terminators
  const vec2 e = vec2(0.006, 0.0);
  float h0 = pyrHost(q);
  vec3 gh = normalize(vec3(pyrHost(q + e.xyy) - h0, pyrHost(q + e.yxy) - h0, pyrHost(q + e.yyx) - h0));
  vec3 nh = normalize(gh*gHR);
  float above = pw.y - sandNear(pw.xz);
  n = normalize(mix(n, nh, smoothstep(0.024, 0.004, above)));       // the clip's edge has no normal worth lighting: a pen line

  // ---- sandstone: a real rock height field first, so raking light has something to bite on
  vec2 hE = vec2(0.0035, 0.0);
  vec3 qf = q; qf.xy -= lipsFlow();                       // the rock rides with the jaw and the corners, or the lips slide over it
  float rk = rockH(qf);
  vec3 rkG = vec3(rockH(qf + hE.xyy) - rk, rockH(qf + hE.yxy) - rk, rockH(qf + hE.yyx) - rk)/hE.x;
  rkG -= gh*dot(rkG, gh);
  float cavity = smoothstep(0.37, 0.27, rk);             // eroded pockets, small and few
  float crown = smoothstep(0.58, 0.76, rk);              // the proud grains
  // bedding: discrete layers with soft edges, not a smooth noise, or it toasts like bread.  They wander a little
  float wob = 0.040*(vnoise3(qf*vec3(1.6,0.8,1.6) + 3.0) - 0.5) + 0.016*(vnoise3(qf*5.7) - 0.5);
  float yy = qf.y + wob;
  float bed1 = layerTone(yy, 8.5, 1.3), bed2 = layerTone(yy, 31.0, 4.1), bed3 = layerTone(yy, 87.0, 2.2);
  float calm = smoothstep(0.30, 0.74, length(vec2(q.x*0.74, q.y - 0.02)));        // the face stays quieter
  float patch_ = vnoise3(q*2.1 + 5.0)*0.60 + vnoise3(q*5.3)*0.40;
  float bed = bed1*0.52 + bed2*0.33 + bed3*0.15;
  vec3 a = mix(vec3(0.76,0.480,0.300), vec3(0.91,0.660,0.450), smoothstep(0.22, 0.78, bed));
  a = mix(a, vec3(0.76,0.40,0.265), 0.42*smoothstep(0.62, 0.86, layerTone(yy, 5.3, 7.7)));      // a rosier bed here and there
  a *= 1.0 + mix(0.05, 0.10, calm)*(bed2 - 0.5) - 0.07*(0.5 - patch_);
  a *= mix(1.0, mix(0.92, 0.84, calm), cavity*(1.0 - 0.7*smoothstep(0.22, -0.20, dot(nh, uSunDir))))*mix(1.0, 1.05, crown);
  float streak = vnoise3(q*vec3(15.0,1.3,15.0) + 2.0)*0.5 + 0.25;
  a *= 0.95 + 0.10*vnoise3(q*233.0);                                                           // grit, for the close crops
  a *= 1.0 - 0.08*smoothstep(0.54, 0.74, streak)*smoothstep(0.70, 0.10, q.y);                  // old run-off, down the slope
  // hairline cracks along the beds, broken up, and never across the eyes or the mouth
  float crk = 0.0;
  for(int i=0;i<5;i++){
    float fi = float(i);
    float lv = -0.52 + fi*0.27 + 0.05*sin(fi*7.3);
    float c = smoothstep(0.0058, 0.0012, abs(q.y - lv + 0.55*wob + 0.012*(vnoise2(vec2(q.x*6.0 + fi*3.0, q.z*6.0)) - 0.5)));
    crk = max(crk, c*smoothstep(0.40, 0.56, vnoise3(vec3(q.x*2.6, fi*2.7, q.z*2.6))));
  }
  crk *= 0.25 + 0.75*calm;
  a *= 1.0 - 0.46*crk;
  float chipM = smoothstep(0.010, -0.010, gChip + 0.008);
  a = mix(a, a*vec3(1.08,1.02,0.95), 0.6*chipM);
  float lipSoft = 0.45*smoothstep(0.012*LIPU, 0.0015*LIPU, gLRing);      // the lips are the same rock: most of the grain stays
  // the bump fades at the terminator: on rough rock at a grazing sun it flickers pixel to pixel, and the edge reads as fur
  float bumpK = mix(0.30, 1.0, smoothstep(0.04, 0.44, dot(nh, uSunDir)));
  s.n = normalize(n - rkG*gHR*mix(0.0085, 0.0165, calm)*bumpK*(1.0 - lipSoft));
  s.alb = a;
  // the raking sun would turn the lid's far shoulder dark: soot beside the eye.  Around the eye, light the host
  // and the same on the far flank of a pucker.  Lips keep their own normal
  float mz = lipsBell(length(vec2(gLQ.x/(1.75*LIP_W), (gLQ.y + 0.02)/(1.55*LIP_W))));
  float soft = max(smoothstep(0.05, 0.60, gEb), 0.70*mz*(1.0 - lipSoft));
  soft = max(soft, 0.85*(1.0 - gLCornT)*smoothstep(0.84, 1.02, gLU)*lipsBell(gLV/(0.075*LIPU)));          // the pad at each lip corner
  s.n = normalize(mix(s.n, nh, 0.62*soft));

  // cheeks: a warm flush, low and wide of the eye, that rises with `cheek`
  vec2 ck = vec2(abs(qf.x) - 0.47, qf.y + 0.055);
  float blush = exp(-dot(ck*vec2(1.0,1.25), ck*vec2(1.0,1.25))/0.021);
  s.alb = mix(s.alb, vec3(0.86,0.40,0.28), (0.20 + 0.34*gLCheek + 0.10*gLSmile)*blush);

  // ---- light: the sun's side glows, the far side falls into a deep cool shadow, warmed from below by the sand
  float ndl = dot(nh, uSunDir);
  float shade = smoothstep(0.22, -0.20, ndl);
  float graze = smoothstep(-0.10, 0.10, ndl)*smoothstep(0.66, 0.26, ndl);
  // a flat face lights flat, so the falloff is painted: the sun-side ridge is hottest, the far ridge has cooled
  float rel = q.x/(0.30 + 0.72*max(0.88 - q.y, 0.0));
  float across = smoothstep(-1.0, 1.0, rel);
  s.alb *= mix(vec3(1.14,1.08,1.00), vec3(0.80,0.71,0.70), across);
  s.alb *= mix(vec3(1.0), vec3(0.45,0.43,0.72), shade);
  s.occ = mix(1.0, 0.70, shade);
  s.spec = 0.03; s.shin = 14.0; s.mirror = 0.0; s.sheen = 0.55; s.wrap = mix(0.05, 0.30, soft); s.fur = 0.0;
  float low = smoothstep(0.80, 0.05, above);
  float sunRidge = smoothstep(-0.10, 0.03, gRa)*smoothstep(0.0, -0.30, q.x);
  s.emit = vec3(1.0,0.46,0.12)*(0.64*graze + 0.24*sunRidge)*(0.55 + 0.45*a.r)
         + SP_BOUNCE*a*0.34*shade*low*low
         + vec3(0.036,0.040,0.098)*shade*(0.80 + 0.20*(1.0 - cavity));

  // the foot: sand drifted against the stone, in the shade of the bank
  float dust = smoothstep(0.17, 0.0, above + 0.13*(vnoise3(q*vec3(5.0,2.2,5.0)) - 0.5) + 0.018*(rk - 0.5));
  s.alb = mix(s.alb, mix(vec3(0.78,0.47,0.23), s.alb, 0.25*shade), 0.80*dust);
  s.alb *= mix(0.70, 1.0, smoothstep(0.0, 0.11, above));
  s.occ *= mix(0.62, 1.0, smoothstep(0.0, 0.24, above));

  // voice shakes a few grains loose: they run down the slope, wide of the face, in columns that follow the taper
  float vc = smoothstep(0.08, 0.55, uP[P_VOICE]);
  float uu = q.x/(0.20 + 0.68*max(0.88 - q.y, 0.0))*9.0;
  float ci = floor(uu), cf = uu - ci;
  float ch = hash11(ci*2.31 + 4.0);
  float fall = (q.y + uTime*(0.32 + 0.42*ch))*2.6 + ch*19.0;
  float fi2 = floor(fall), ff = fall - fi2;
  float keep = step(0.58, hash11(fi2*3.7 + ci*1.9))*step(0.52, abs(uu)/9.0);
  vec2 gd = vec2((cf - 0.5 - 0.3*(hash11(fi2 + ci*7.0) - 0.5))*0.11, (ff - 0.5)*0.385);
  float grain = smoothstep(0.0065, 0.0042, length(gd))*keep*vc*smoothstep(0.02, 0.10, above);
  float gShade = smoothstep(0.0100, 0.0060, length(gd - vec2(0.0035,-0.0035)))*keep*vc*smoothstep(0.02, 0.10, above);
  s.alb = mix(s.alb, s.alb*0.62, 0.7*gShade);
  s.alb = mix(s.alb, vec3(0.96,0.72,0.44), grain);

  vec3 stone = s.alb;
  lipsPaint(n, s);
  // the lips are the same block: its beds run on across them, a rosier bed.  They sit IN the stone, inside an incised
  // line, the way a statue's lips do: a soft shade outside it, or they float like a sticker
  float lipM = gPLip, lipN = lipM*(1.0 - gPLine);
  s.alb = mix(s.alb, mix(stone, s.alb, 0.22)*vec3(0.93,0.79,0.72), lipN);
  float lipAO = smoothstep(0.046*LIPU, 0.004*LIPU, gLRing)*(1.0 - lipM);
  s.alb *= mix(vec3(1.0), vec3(0.76,0.68,0.66), 0.80*lipAO);
  float cut = smoothstep(0.12, 0.04, abs(gLE/max(gLH, 1e-3) - 0.88))*(1.0 - gPCave)*smoothstep(1.10, 0.96, gLU + gLCornT);   // the outer edge of the lip
  s.alb *= 1.0 - 0.50*cut;
  s.spec *= 1.0 - 0.25*lipM;
  // the brow is cut from the same stone: it keeps the grain, and throws a little shade onto the lid under it
  float bShade = smoothstep(0.046, 0.004, gFBrow)*(1.0 - smoothstep(0.012, 0.0, gFBrow));
  s.alb *= 1.0 - 0.34*bShade;
  facePaint(q, rd, n, gSkin, s);
  float bm = smoothstep(0.0060, 0.0010, gFBrow);
  s.alb *= mix(1.0, mix(0.86, 1.10, smoothstep(0.34, 0.66, rk)), bm);
}

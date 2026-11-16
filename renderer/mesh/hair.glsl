// ---------------------------------------------------------------------------
// renderer/mesh/hair.glsl — render-hair: strand ribbons (scalp, fuzz, brows, lashes). Assembled by hair.py as
//  header + #define <SECTION> + #define VS_HAIR (mesh.glsl's camera helpers) [+ scene lib for the fragment] + mesh.glsl
//  + this file.  Sections: HAIR_VERT/HAIR_FRAG (forward, MSAA, alpha-to-coverage), HAIR_SVERT/HAIR_SFRAG (key shadow).
// ---------------------------------------------------------------------------

#if defined(HAIR_VERT) || defined(HAIR_SVERT) || defined(HAIR_DVERT)
uniform sampler2D tPts;        // RGBA32F: (S*P) rest point (bone groups) / root-frame local point (surface), w = u
uniform sampler2D tStr;        // RGBA32F: 3 texels / strand: (wr, wt, seed, clump), (v0, v1, v2, surface), (bary, noise)
uniform sampler2D tPosed;      // R32F: a copy of vb_posed (10 floats / vertex: pos, nrm, tan)
uniform mat4  uBoneM;          // skin matrix of the follow bone (bone @ inverse bind)
uniform float uScale;          // asset metres -> scene units
uniform int   uP, uSeg, uSurface;
uniform float uWidthK;
uniform float uLenK;           // strand length about the root (knob)
uniform vec3  uCtr;            // centre of the hair volume (world) -> mass normal
#ifdef HAIR_V2
uniform mat4 uFollowM;        // follow: the follow bone's skin matrix (bone groups)
uniform mat4  uLagM;           // the lagged (spring-damped) head skin matrix
uniform vec4  uFollow;         // x hang_k, y u0, z lag_k, w on (1)
uniform vec2  uFollowY;
#ifdef HAIR_LAYER
uniform vec3 uLenVar;         // HAIR2 layer: x per-strand length cut (fraction), y per-lock, z tip taper start (u)
uniform int   uFollowN;        // follower strands per drawn strand (instances = strands x N; 1 = none)
uniform float uTaperU;         // u where the strand's width taper starts (layer; >= 1: the groom's own taper)
uniform vec4  uFollowR;        // follower spread (scene m): x root, y tip, z normal-direction share, w along-strand curl
#endif
uniform float uFarEdge;        // far field: extra width of lock-rim strands (x lock_edge)        // rest-space height band: rigid above .x, full follow below .y (the fringe stays put)
#endif
const int TW = 2048;
vec4 fetchT(sampler2D t, int i){ return texelFetch(t, ivec2(i % TW, i / TW), 0); }
float fetchT1(int i){ return texelFetch(tPosed, ivec2(i % TW, i / TW), 0).r; }
vec3 posedV(int v){ int b = v*10; return vec3(fetchT1(b), fetchT1(b + 1), fetchT1(b + 2)); }
#endif
// strand point -> world (shared by both vertex sections)
#if defined(HAIR_VERT) || defined(HAIR_SVERT) || defined(HAIR_DVERT)
struct Strand { vec3 root; mat3 F; vec4 a; vec4 b; vec4 c; };
Strand gS;
void loadStrand(int s){
  gS.a = fetchT(tStr, 3*s); gS.b = fetchT(tStr, 3*s + 1); gS.c = fetchT(tStr, 3*s + 2);
  if(uSurface == 1){
    vec3 v0 = posedV(int(gS.b.x + 0.5)), v1 = posedV(int(gS.b.y + 0.5)), v2 = posedV(int(gS.b.z + 0.5));
    gS.root = gS.c.x*v0 + gS.c.y*v1 + gS.c.z*v2;
    vec3 e0 = normalize(v1 - v0);
    vec3 n = normalize(cross(v1 - v0, v2 - v0));
    vec3 b = cross(n, e0);
    gS.F = mat3(e0, n, b);            // columns: local (x,y,z) -> world
  }
}
vec4 strandPoint(int s, int k){
  vec4 q = fetchT(tPts, s*uP + clamp(k, 0, uP - 1));
  if(uLenK != 1.0){
    if(uSurface == 1) q.xyz *= uLenK;
    else { vec3 r0 = fetchT(tPts, s*uP).xyz; q.xyz = r0 + (q.xyz - r0)*uLenK; }
  }
  vec3 w = uSurface == 1 ? gS.root + gS.F*(q.xyz*uScale) : (uBoneM*vec4(q.xyz, 1.0)).xyz;
#ifdef HAIR_V2
  // free-length follow + lag: past the scalp the strand keeps only part of the head's rotation (hangs toward the
  // follow bone) and trails a spring-damped copy of the head; deterministic per frame
  if(uSurface == 0 && uFollow.w > 0.5){
    float t = smoothstep(uFollow.y, 1.0, q.w)*smoothstep(uFollowY.x, uFollowY.y, q.y);
    vec4 p4 = vec4(q.xyz, 1.0);
    vec3 wf = (uFollowM*p4).xyz, wl = (uLagM*p4).xyz;
    w = mix(w, wf, uFollow.x*t) + (wl - w)*(uFollow.z*t);
  }
#endif
  return vec4(w, q.w);
}
// the ribbon centre point, tangent, u and the strand's true width (scene units) for this vertex
#ifdef HAIR_LAYER
float gFk;                     // HAIR2: follower index of this instance (0 = the strand itself)
float lh(float x){ return fract(sin(x*78.233)*43758.5453); }
#endif
void ribbonPoint(out vec3 P, out vec3 T, out float u, out float w, out float side){
  int s = gl_InstanceID;
#ifdef HAIR_LAYER
  int fn = max(uFollowN, 1);
  s = gl_InstanceID/fn;
  gFk = float(gl_InstanceID - s*fn);
#endif
  loadStrand(s);
  int i = gl_VertexID/2;
  side = (gl_VertexID & 1) == 0 ? -1.0 : 1.0;
  int k = min(i*uSeg, uP - 1);
  vec4 c = strandPoint(s, k);
  vec3 pa = strandPoint(s, k - uSeg).xyz, pb = strandPoint(s, k + uSeg).xyz;
  P = c.xyz; u = c.w;
  T = pb - pa;
  T = dot(T, T) > 1e-14 ? normalize(T) : vec3(0.0, -1.0, 0.0);
  // width: root -> tip, a soft point at the very tip
  w = mix(gS.a.x, gS.a.y, u)*(1.0 - 0.6*smoothstep(0.85, 1.0, u))*uScale*uWidthK;
#ifdef HAIR_LAYER
  // HAIR2: followers = extra strands around each drawn strand (offset in the plane across the strand, growing
  // root -> tip, mostly sideways in the lock layer), and every strand ends at its own length: the width tapers to 0
  // before a per-strand / per-lock cut, so ends are wispy and ragged instead of one blunt line
  float sd = gS.a.z + gFk*0.618;
  // real hair keeps its width to the last ~cm: the groom's linear root->tip taper made every free end (and so the
  // whole lower mass) go translucent; in the layer the width tapers only over u >= uTaperU, ends thin by COUNT (cut)
  if(uSurface == 0 && uTaperU < 1.0)
    w = mix(gS.a.x, gS.a.y, smoothstep(uTaperU, 1.0, u))*(1.0 - 0.6*smoothstep(0.85, 1.0, u))*uScale*uWidthK;
  if(uSurface == 0){
    float lc = 1.0 - uLenVar.x*lh(sd*17.31 + 1.3) - uLenVar.y*lh(gS.c.w*3.1 + 0.7);
    w *= smoothstep(lc, lc - max(uLenVar.z, 0.02), u);
    if(gFk > 0.5){
      vec3 nv = normalize(P - uCtr);
      vec3 b1 = cross(T, nv);
      b1 = dot(b1, b1) > 1e-10 ? normalize(b1) : vec3(1.0, 0.0, 0.0);
      vec3 b2 = cross(b1, T);
      float an = 6.2831853*lh(sd*91.7 + 0.37) + uFollowR.w*u;
      float rr = sqrt(lh(sd*53.9 + 2.1));
      float sp = mix(uFollowR.x, uFollowR.y, smoothstep(0.0, 1.0, u))*uScale;
      P += (b1*cos(an) + b2*sin(an)*uFollowR.z*smoothstep(0.0, 0.25, u))*rr*sp;
    }
  }
#endif
#ifdef HAIR_V2
  if(uSurface == 0) w *= 1.0 + uFarEdge*gS.a.w*smoothstep(0.2, 0.6, u);   // far field: lock-rim strands fill the gaps
#endif
}
#endif

// ============================================================ forward
#ifdef HAIR_VERT
#ifdef HAIR_V2
invariant gl_Position;          // hair-shade: the pre-pass and the early-Z shading pass must agree bit for bit
#endif
uniform float uMinPx;
uniform float uPull;            // surface groups: nudge toward the camera (asset metres) so brows never z-fight the skin
out vec3 vP;
out vec3 vT;
out vec3 vNv;
out vec3 vSideV;
out float vU, vAlpha, vSide, vSeed, vNoise;
out float vMass;          // render5: lock-on-lock depth (hair.py _mass_depth)
#ifdef HAIR_V2
out float vEdge;          // 0 lock core .. 1 lock rim (hair.py _lock_edge)
#endif
#ifdef HAIR_LAYER
out float vWpx, vHalfPx;  // HAIR2: true strand width (px), drawn half width (px)
uniform vec2 uCovTent;    // x tent radius (px), y mass-depth cut of the coverage pass (> 1 = off)
#endif
void main(){
  vec3 P, T; float u, w, side;
  ribbonPoint(P, T, u, w, side);
  vec3 V = normalize(uCamPos - P);
  float z = max(dot(P - uCamPos, uCamW), 1e-3);
  float pw = 2.0*z/(uRes.y*uFl);                 // world size of one pixel at this depth
  float dw = max(w, uMinPx*pw);
#ifdef HAIR_LAYER
  vWpx = w/pw;
#ifdef HAIR_COV
  dw = w > 1e-9 ? w + 2.0*uCovTent.x*pw : 0.0;   // analytic coverage: a +-tent px tent around the true width
  if(uSurface == 0 && gS.b.w > uCovTent.y) dw = 0.0;   // deep-interior strands add nothing visible (opaque already)
#ifdef HAIR_COVW0
  dw = 0.0;
#endif
#else
  dw = w > 1e-9 ? max(w, uMinPx*pw) : 0.0;       // colour: the front strand of every covered sample
#endif
  vHalfPx = 0.5*dw/pw;
  vSeed = gS.a.z + gFk*0.618;
#endif
  vec3 sv = cross(T, V);
  sv = dot(sv, sv) > 1e-12 ? normalize(sv) : vec3(1.0, 0.0, 0.0);
  P += V*uPull*uScale;
  vP = P + sv*(0.5*dw*side);
  vT = T; vSideV = sv; vU = u; vSide = side;
  vAlpha = clamp(w/dw, 0.0, 1.0);
#ifndef HAIR_LAYER
  vSeed = gS.a.z;
#endif
  vNoise = gS.c.w; vMass = uSurface == 1 ? 0.0 : gS.b.w;
#ifdef HAIR_V2
  vEdge = uSurface == 1 ? 0.0 : gS.a.w;
#endif
  vNv = uSurface == 1 ? gS.F[1] : normalize(P - uCtr);
  gl_Position = camClip(vP);
}
#endif

#if defined(HAIR_FRAG) || defined(HAIR_CAPF)
layout(location=0) out vec4 fragC;
layout(location=1) out vec4 fragD;
layout(location=2) out vec4 fragS;     // render: the skin-SSS attachment (mesh.glsl fragS): hair is not skin -> (0, coverage)
#ifdef MESH_TAA_R
layout(location=5) out vec4 fragR;     // shade2-taa (gated): this group's TAA class (react, sharpen, gamma/4) * a, a
uniform vec4 uTaaR;
#endif
uniform vec3  uKeyDir, uKeyCol, uRimDir2, uRimCol2, uFillDir, uFillCol;
uniform float uKeyGain, uRimGain, uAmbGain, uCharK;
uniform sampler2D uShadow;
uniform mat4  uLightVP;
uniform vec2  uShadowZ;          // world depth range of the ortho key map, 1/res
uniform float uDeep;             // self-shadow: depth into the hair (world) where the key falls to 1/e (<0: off)
uniform vec3  uColRoot, uColTip;
uniform int   uKind;             // 0 scalp, 1 fuzz, 2 brows, 3 lashes
uniform float uRough;
uniform float uDifK;
uniform int   uA2C;
uniform int   uSamples;
uniform int   uDebug;
uniform float uArcN;           // hash cells along a strand
uniform vec4 uHairK;          // render3 (mesh.hair.shade_k): x R gain, y R width, z broad sheen, w clump shadow
uniform vec4 uHairK2;         // render3 (mesh.hair.shade_k2): x per-strand band jitter, y R hue share
uniform vec4 uHairK3;         // render7 (mesh.hair.shade_k3): x tip -> root colour, y tip coverage, z root shade floor, w fuzz alpha

// key light transmittance through the hair volume (deep-opacity from one depth map: exp(-depth behind the first
// occluder / uDeep)), 4-tap rotated PCF so it is soft and stable
float keyTrans(vec3 p, vec3 nv){
  vec4 c = uLightVP*vec4(p + nv*0.0005, 1.0);
  vec3 s = c.xyz/c.w*0.5 + 0.5;
  if(any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0)))) return 1.0;
  float acc = 0.0;
  for(int i=0;i<4;i++){
    vec2 o = vec2(i == 0 ? 1.0 : (i == 1 ? -1.0 : 0.3), i == 2 ? 1.0 : (i == 3 ? -1.0 : -0.3))*1.5*uShadowZ.y;
    float zm = texture(uShadow, s.xy + o).r;
    float d = max((s.z - zm)*uShadowZ.x - 0.0006, 0.0);
    acc += uDeep > 0.0 ? exp(-d/uDeep) : 1.0;
  }
  return acc*0.25;
}
float h21(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
#ifdef HAIR_V2
// hair-shade: deep opacity map (strands + the v2 cap)
uniform sampler2D uDom;   // deep opacity map: cumulative coverage at uDomZ slab boundaries behind the first occluder
uniform vec4  uDomZ;      // slab boundaries (world)
uniform float uDomK;      // optical depth per unit coverage (<0: DOM off -> single-map keyTrans)
uniform float uDomRes;
// opacity in front of world depth d behind the first occluder: linear in the cumulative slabs
float domOpacity(vec4 L, float d){
  if(d <= uDomZ.x) return L.x*d/max(uDomZ.x, 1e-6);
  if(d <= uDomZ.y) return mix(L.x, L.y, (d - uDomZ.x)/(uDomZ.y - uDomZ.x));
  if(d <= uDomZ.z) return mix(L.y, L.z, (d - uDomZ.y)/(uDomZ.z - uDomZ.y));
  return mix(L.z, L.w, clamp((d - uDomZ.z)/(uDomZ.w - uDomZ.z), 0.0, 1.0));
}
// key transmittance: deep opacity (4 rotated bilinear taps, each with its own first-occluder depth) or the old map
float keyTransV2(vec3 p, vec3 nv, float jit){
  if(uDomK < 0.0) return keyTrans(p, nv);
  vec4 c = uLightVP*vec4(p, 1.0);
  vec3 s = c.xyz/c.w*0.5 + 0.5;
  if(any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0)))) return 1.0;
  float acc = 0.0;
  float an = jit*6.2831853;
  mat2 R = mat2(cos(an), sin(an), -sin(an), cos(an));
  for(int i=0;i<4;i++){
    vec2 o = R*vec2(i == 0 ? 1.0 : (i == 1 ? -1.0 : 0.35), i == 2 ? 1.0 : (i == 3 ? -1.0 : -0.35))*(1.2/uDomRes);
    float zm = texture(uShadow, s.xy + o).r;
    float d = max((s.z - zm)*uShadowZ.x - 0.0004, 0.0);
    // deeper than the last slab = behind something the hair alone cannot be (the skull / body in the same map)
    acc += exp(-uDomK*domOpacity(texture(uDom, s.xy + o), d) - max(d - uDomZ.w, 0.0)/0.004);
  }
  return acc*0.25;
}

#endif
#endif

#ifdef HAIR_FRAG
in vec3 vP;
in vec3 vT;
in vec3 vNv;
in vec3 vSideV;
in float vU, vAlpha, vSide, vSeed, vNoise;
in float vMass;
#ifdef HAIR_V2
in float vEdge;
#endif
#ifdef HAIR_LAYER
in float vWpx, vHalfPx;
uniform float uCovK;       // coverage multiplier of this group
uniform vec2  uCovTent;
#endif
uniform float uHairMass;

float az0(vec3 nv, vec3 H){ return smoothstep(0.0, 0.9, dot(nv, H)); }
// shifted-tangent (Marschner-flavoured Kajiya-Kay) lobe
float lobe(vec3 T, vec3 nv, float shift, vec3 H, float e){
  vec3 t = normalize(T + nv*shift);
  float th = dot(t, H);
  float s = sqrt(max(1.0 - th*th, 0.0));
  return pow(s, e)*(e + 2.0)/(2.0*PI);
}

#ifdef HAIR_V2
// ---------------------------------------------------------------- hair-shade: physically based strand shading
// Marschner / d'Eon lobes in Karis 2016 ("Physically Based Hair Shading in Unreal") form: longitudinal Gaussians in
// sin(theta) with cuticle-tilt shifts, cheap azimuthal terms, Fresnel (eta 1.55), absorption from melanin (Chiang 2016)
// so TT / TRT carry the hair's colour; Karis's fake multiple scattering for the soft saturated body.
uniform vec4  uV2Mel;     // x eumelanin, y pheomelanin, z tip melanin x
uniform vec2  uV2MelJ;    // melanin jitter: per strand, per clump
uniform float uV2MelK;    // per-kind melanin x (brows / lashes darker)
uniform vec4  uV2Lobe;    // x cuticle alpha (rad), y beta R, z roughness jitter, w fill area widening
uniform vec4  uV2Jit;     // shift jitter (rad): x per strand, y per clump, z along-strand wave
uniform vec4  uV2Gain;    // R, TT, TRT, MS
uniform vec4  uV2Misc;    // x R tint, y ms_pow, z ms volume-normal share, w ms key-transmittance exponent
uniform vec4  uV2Occ;     // x root occlusion floor, y its length (u), z mass occlusion, w ambient gain
uniform vec4  uV2Tip;     // x tip fade start (u), y tip alpha, z ambient R sheen, w specular AA
uniform float uV2RVar;
uniform float uV2FillR;
uniform float uV2Far;     // far-field smoothing 0..1: per-strand / per-lock variance fades (MSAA cannot resolve it)   // R gain of the fill / rim lights relative to the key's (broad lights: no grey wash)
uniform vec3  uV2Lock;    // per lock: MS brightness +-, R gain +-, rim-strand occlusion
uniform vec2  uV2Sat;     // x TT/TRT colour saturation (1 = melanin's own), y MS shadow-saturation exponent    // per-strand R brightness variation (+- fraction): the band reads as many fine streaks
uniform int   uPrepass;   // 1 = the depth pre-pass (coverage + depth only); the shading pass then tests EQUAL
#ifdef HAIR_EARLYZ
layout(early_fragment_tests) in;
#endif

float hg(float b, float x){ return exp(-0.5*x*x/(b*b))/(2.50662827*b); }
float hF(float c){ const float F0 = 0.04652; float m = clamp(1.0 - c, 0.0, 1.0); float m2 = m*m; return F0 + (1.0 - F0)*m2*m2*m; }
float h11(float x){ return fract(sin(x*91.3458)*47453.5453); }

vec3 sigmaA(float eu, float ph){ return eu*vec3(0.419, 0.697, 1.37) + ph*vec3(0.187, 0.4, 1.05); }

// R + TT + TRT for one light (radiance per unit irradiance, Karis's normalisation); area widens the lobes
// lobes: x R, y TT, z TRT on/off (fill / rim skip what they cannot show)
vec3 hairLobes(vec3 T, vec3 V, vec3 L, vec3 sig, float beta, float shift, float area, vec3 hue, vec3 lobes){
  float sL = clamp(dot(T, L), -1.0, 1.0), sV = clamp(dot(T, V), -1.0, 1.0);
  // cos((thV - thL)/2) without trig: cos(thV - thL) = cV cL + sV sL
  float cD = sqrt(max(1.0 - sV*sV, 0.0)*max(1.0 - sL*sL, 0.0)) + sV*sL;
  float cosThD = sqrt(clamp(0.5 + 0.5*cD, 0.0, 1.0));
  vec3 Lp = L - sL*T, Vp = V - sV*T;
  float cosPhi = dot(Lp, Vp)*inversesqrt(dot(Lp, Lp)*dot(Vp, Vp) + 1e-4);
  float cosHPhi = sqrt(clamp(0.5 + 0.5*cosPhi, 0.0, 1.0));
  float VoL = dot(V, L);
  float b2 = beta*beta;
  float bR = area + b2, bTT = area + 0.5*b2, bTRT = area + 2.0*b2;
  float al = uV2Lobe.x;
  float x = sL + sV;
  const float n = 1.55;
  vec3 S = vec3(0.0);
  // R: white-ish surface reflection, a touch of the hair's hue (r_tint)
  if(lobes.x > 0.0){
    float Mp = hg(bR*sqrt(2.0)*max(cosHPhi, 0.2), x - (-al + shift));
    float Np = 0.25*cosHPhi;
    float Fp = hF(sqrt(clamp(0.5 + 0.5*VoL, 0.0, 1.0)));
    S += lobes.x*uV2Gain.x*Mp*Np*Fp*mix(vec3(1.0), hue, uV2Misc.x);
  }
  // TT: through the fibre, strongest backlit
  if(lobes.y > 0.0){
    float Mp = hg(bTT, x - (0.5*al + shift));
    float a = 1.0/n;
    float h = cosHPhi*(1.0 + a*(0.6 - 0.8*cosPhi));
    float f = hF(cosThD*sqrt(clamp(1.0 - h*h, 0.0, 1.0)));
    float Fp = (1.0 - f)*(1.0 - f);
    vec3 Tp = exp(-sig*2.0*sqrt(max(1.0 - h*h*a*a, 0.0))/max(cosThD, 0.2));
    Tp = mix(vec3(dot(Tp, vec3(0.2126, 0.7152, 0.0722))), Tp, uV2Sat.x);
    float Np = exp(-3.65*cosPhi - 3.98);
    S += uV2Gain.y*Mp*Np*Fp*Tp;
  }
  // TRT: the coloured secondary highlight
  if(lobes.z > 0.0){
    float Mp = hg(bTRT, x - (1.5*al + shift));
    float f = hF(cosThD*0.5);
    float Fp = (1.0 - f)*(1.0 - f)*f;
    vec3 Tp = exp(-sig*3.2/max(cosThD, 0.2));
    Tp = mix(vec3(dot(Tp, vec3(0.2126, 0.7152, 0.0722))), Tp, uV2Sat.x);
    float Np = exp(17.0*cosPhi - 16.78);
    S += uV2Gain.z*Mp*Np*Fp*Tp;
  }
  return S;
}
// Karis's fake multiple scattering: wrap diffuse on a normal facing the viewer (blended with the volume normal),
// saturating as the light is shadowed (ScatterTint)
vec3 hairMS(vec3 T, vec3 V, vec3 L, vec3 nv, vec3 msAlb, float shadow){
  vec3 fn = V - T*dot(V, T);
  fn = dot(fn, fn) > 1e-8 ? normalize(fn) : nv;
  vec3 N = normalize(mix(fn, nv, uV2Misc.z));
  float NoL = clamp((dot(N, L) + 1.0)*0.25, 0.0, 1.0);
  float tl = dot(T, L);
  float kd = sqrt(max(1.0 - tl*tl, 0.0));
  float ds = (1.0/PI)*mix(NoL, kd*0.5, 0.33);
  float luma = max(dot(msAlb, vec3(0.2126, 0.7152, 0.0722)), 1e-5);
  vec3 tint = pow(max(msAlb/luma, vec3(1e-4)), vec3(uV2Sat.y*(1.0 - clamp(shadow, 0.0, 1.0))));
  return uV2Gain.w*msAlb*ds*tint;
}

void main(){
#if HAIR_DIAG == 2
  if(uPrepass == 1){ fragC = vec4(0.0); fragD = vec4(0.0); fragS = vec4(0.0); return; }
#endif
  float a = vAlpha;
  // roots emerge from the skin, a soft brow root; soft TIPS (alpha fades to uV2Tip.y) instead of hard cut ends
  if(uKind == 0) a *= mix(0.5, 1.0, smoothstep(0.0, 0.06, vU));
  if(uKind <= 1) a *= mix(1.0, uV2Tip.y, smoothstep(uV2Tip.x, 1.0, vU));
  if(uKind == 2) a *= mix(0.55, 1.0, smoothstep(0.0, 0.3, vU));
  a *= smoothstep(1.0, 0.55, abs(vSide));
#ifdef HAIR_COV
  // HAIR2 analytic coverage: the strand's true width spread over a +-1 px tent (sums to the true width in px
  // per pixel row whatever the sub-pixel position) -> thin strands are faint and smooth, never dotted / crawling
  {
    float xp = abs(vSide)*vHalfPx;
    float tr = max(uCovTent.x, 0.25);
    float cv = min(vWpx, 1.0)*clamp(1.0 - max(xp - 0.5*max(vWpx - 1.0, 0.0), 0.0)/tr, 0.0, 1.0)/tr;
    if(uKind == 0) cv *= mix(0.5, 1.0, smoothstep(0.0, 0.06, vU));
    if(uKind <= 1) cv *= mix(1.0, uV2Tip.y, smoothstep(uV2Tip.x, 1.0, vU));
    if(uKind == 2) cv *= mix(0.55, 1.0, smoothstep(0.0, 0.3, vU));
    fragC = vec4(cv*uCovK, 0.0, 0.0, 0.0); fragD = vec4(0.0); fragS = vec4(0.0);
    return;
  }
#endif
#if defined(HAIR_EARLYZ) || defined(HAIR_LAYER)
  // shading pass after the pre-pass: depth EQUAL selects exactly the samples this strand won there; no discard / no
  // sample mask so the hardware keeps early-Z
  a = 1.0;
#else
  float hd = h21(vec2(vSeed*1931.0 + vNoise*17.0, floor(vU*uArcN)));
  float ns = float(max(uSamples, 1));
  float k = floor(a*ns + hd);
#ifndef HAIR_NODISCARD
  if(k < 0.5) discard;
#endif
#ifdef HAIR_SMASK
  {
    int N = max(uSamples, 1), kk = min(int(k), N);
    int rot = int(h21(vec2(vSeed*733.0 + floor(vU*uArcN)*0.618, vNoise*91.0 + 3.7))*float(N)) % N;
    int m = (1 << kk) - 1;
    m = ((m << rot) | (m >> (N - rot))) & ((1 << N) - 1);
#if HAIR_DIAG != 1
    gl_SampleMask[0] = m;
#endif
  }
  a = 1.0;
#else
  a = uA2C == 1 ? k/ns : 1.0;
#endif
#endif
  if(uPrepass == 1){ fragC = vec4(0.0); fragD = vec4(0.0); fragS = vec4(0.0); return; }   // depth-only pre-pass
  float hs = h11(vSeed*13.7 + 0.5), hs2 = h11(vSeed*29.1 + 3.3), hs3 = h11(vSeed*7.7 + 9.1);
  vec3 V = normalize(uCamPos - vP);
  vec3 T = normalize(vT);
  vec3 nv = normalize(vNv);
  // colour (multiple-scattering albedo): root -> tip, clump-coherent variation
  vec3 alb = mix(uColRoot, uColTip, smoothstep(0.05, 1.0, vU));
  float lk = h11(vNoise*71.3 + 4.1);                   // a second per-lock hash
  alb *= mix((0.94 + 0.12*hs)*(1.0 + uV2Lock.x*(vNoise - 0.5)*2.0), 1.0, uV2Far);
  vec3 msAlb = pow(max(alb, vec3(1e-5)), vec3(uV2Misc.y));
  // absorption: melanin per strand / clump, tips lighter
  float mj = 1.0 + (uV2MelJ.x*(hs2 - 0.5)*2.0 + uV2MelJ.y*(vNoise - 0.5)*2.0)*(1.0 - uV2Far);
  float mt = mix(1.0, uV2Mel.z, smoothstep(0.2, 1.0, vU));
  vec3 sig = sigmaA(uV2Mel.x, uV2Mel.y)*max(mj*mt*uV2MelK, 0.05);
  vec3 hue = exp(-sig*1.5); hue /= max(max(hue.r, hue.g), 1e-4);
  // per-strand / per-clump / along-strand cuticle shift + roughness: the band breaks into strands and locks
  float shift = uV2Jit.x*(hs - 0.5)*2.0 + uV2Jit.y*(vNoise - 0.5)*2.0 + uV2Jit.z*sin(vU*17.0 + vSeed*40.0);
  shift *= 1.0 - uV2Far;
  float beta = uV2Lobe.y*(1.0 + uV2Lobe.z*(hs3 - 0.5)*2.0);
  // specular anti-aliasing: a sub-pixel strand (vAlpha = width / drawn width) widens its lobes and loses its private
  // shift (the pixel averages many strands) -> less sparkle / crawl in motion
  float sub = clamp(1.0 - vAlpha, 0.0, 1.0)*uV2Tip.w;
  beta = sqrt(beta*beta + 0.05*sub);
  shift *= 1.0 - 0.5*sub;
  // occlusion: roots, lock-under-lock (load time), a little per clump
  float occ = 1.0;
  if(uKind <= 1){
    occ = mix(uV2Occ.x, 1.0, smoothstep(0.0, max(uV2Occ.y, 1e-3), vU));
    occ *= 1.0 - uV2Occ.z*vMass;
    occ *= mix(0.85, 1.0, vNoise);
    occ *= 1.0 - uV2Lock.z*(1.0 - uV2Far)*smoothstep(0.35, 1.0, vEdge)*smoothstep(0.15, 0.5, vU);   // rim strands of a lock: the gaps
  }
  float kspec = uKind >= 2 ? 0.3 : 1.0;
  float rv = (1.0 + uV2RVar*(h11(vSeed*53.1 + 1.7) - 0.5)*2.0)*max(1.0 + uV2Lock.y*(lk - 0.5)*2.0, 0.0);
  rv = mix(rv, 1.0, uV2Far);
  msAlb *= uKind == 2 ? 0.7 : (uKind == 3 ? 0.5 : 1.0);     // brows / lashes: denser, darker short hair
  vec3 col = vec3(0.0);
  // key (deep-opacity self shadow)
  {
    vec3 L = uKeyDir;
    float tr = keyTransV2(vP, nv, hs);
    vec3 S = hairLobes(T, V, L, sig, beta, shift, 0.0, hue, vec3(1.0))*kspec*rv;
    vec3 M = hairMS(T, V, L, nv, msAlb, tr);
    col += uKeyCol*uKeyGain*(S*tr*mix(0.6, 1.0, occ) + M*pow(max(tr, 0.0), uV2Misc.w)*occ);
  }
  // fill (broad sky side, no shadow map): wider lobes
  {
    vec3 L = uFillDir;
    vec3 S = hairLobes(T, V, L, sig, beta, shift, uV2Lobe.w, hue, vec3(1.0, 0.0, 1.0))*kspec*mix(1.0, rv, 0.5)*uV2FillR;
    vec3 M = hairMS(T, V, L, nv, msAlb, 1.0);
    col += uFillCol*(S + M)*occ;
  }
  // rim (warm lamp behind): TT shines through thin strands at the silhouette
  {
    vec3 L = uRimDir2;
    vec3 S = hairLobes(T, V, L, sig, beta, shift, 0.05, hue, vec3(uV2FillR, 1.0, 0.0))*kspec*rv;
    vec3 M = hairMS(T, V, L, nv, msAlb, 1.0);
    col += uRimCol2*uRimGain*(S + M)*occ;
  }
  // room ambient: soft diffuse on the volume normal + a faint R sheen
  vec3 amb = setAmbient(vP, nv)*uAmbGain;
  col += amb*(msAlb*0.9*uV2Occ.w + 0.04652*uV2Tip.z*mix(vec3(1.0), hue, uV2Misc.x))*occ;
  if(uDebug == 1) col = vec3(0.0, 1.0, 0.0)/max(uCharK, 1e-3);
  if(uDebug == 2) col = vec3(keyTransV2(vP, nv, hs))/max(uCharK, 1e-3);
  col = jFin3(col, 0.0, 6.0e4);
  fragC = vec4(col*uCharK, a);
  float t = length(vP - uCamPos);
  fragD = vec4(jFin1(t, 0.0, 6.0e4), 0.0, 0.0, a);
  fragS = vec4(0.0, 0.0, 0.0, a);
#ifdef MESH_TAA_R
  fragR = vec4(uTaaR.rgb*a, a);
#endif
}
#else
void main(){
  float a = vAlpha;
  // root fade: strands emerge from the skin (soft hairline, no helmet edge); a faint tip fade
  if(uKind == 0) a *= mix(0.5, 1.0, smoothstep(0.0, 0.06, vU));
  if(uKind == 0) a = mix(a, max(a, smoothstep(0.35, 0.9, vU)*0.85), uHairK3.y);   // render7: tips keep coverage
  if(uKind == 1) a *= mix(1.0, uHairK3.w, smoothstep(0.2, 0.8, vU));               // render7: flyaways fade out along their length
  if(uKind == 2) a *= mix(0.55, 1.0, smoothstep(0.0, 0.3, vU));
  // across the ribbon: the cylinder edge fades so a wide (close-up) strand is round, not a flat card
  float xs = vSide;                                   // -1..1 across
  a *= smoothstep(1.0, 0.55, abs(xs));
  // stochastic coverage that sticks to the strand (hash of strand + arc position, not of the pixel -> no crawl when the
  // head moves): A2C gets an exact integer sample count k = floor(a*N + h), so thin strands keep their expected coverage
  float hd = h21(vec2(vSeed*1931.0 + vNoise*17.0, floor(vU*uArcN)));
  float ns = float(max(uSamples, 1));
  float k = floor(a*ns + hd);
  if(k < 0.5) discard;
#ifdef HAIR_SMASK
  // own sample mask: k of N samples, rotated by a per-strand hash, so overlapping strands land on DIFFERENT samples
  // (plain alpha-to-coverage uses one fixed pattern per alpha value -> stacked strands never get past k/N)
  {
    int N = max(uSamples, 1), kk = min(int(k), N);
    int rot = int(h21(vec2(vSeed*733.0 + floor(vU*uArcN)*0.618, vNoise*91.0 + 3.7))*float(N)) % N;
    int m = (1 << kk) - 1;
    m = ((m << rot) | (m >> (N - rot))) & ((1 << N) - 1);
    gl_SampleMask[0] = m;
  }
  a = 1.0;
#else
  a = uA2C == 1 ? k/ns : 1.0;
#endif
  vec3 V = normalize(uCamPos - vP);
  vec3 T = normalize(vT);
  vec3 nv = normalize(vNv);
  // per-strand colour: root -> tip, melanin jitter per strand + per clump (reads as strands / clumps, not a flat cap)
  vec3 alb = mix(uColRoot, mix(uColTip, uColRoot, uHairK3.x*(uKind == 1 ? 1.0 : 0.6)), smoothstep(0.05, 1.0, vU));   // render7 x
  alb *= (0.92 + 0.16*vSeed)*(0.80 + 0.40*vNoise);     // clump-coherent, little per-strand noise
  // occlusion inside the mass: roots and the lower layers sit in the hair's own shadow
  float occ = uKind == 0 ? mix(uHairK3.z, 1.0, smoothstep(0.0, 0.55, vU)) : (uKind == 1 ? 0.9 : 1.0);   // render7 z (was 0.30)
  occ *= mix(0.75, 1.0, vNoise);
  // render3: clump shadow -- strands toward a clump's edge (low clump noise) sit deeper in the mass; reads as
  // separated locks instead of an even sheet
  occ *= mix(1.0, smoothstep(0.0, 0.7, vNoise)*0.7 + 0.3, uHairK.w*(uKind == 0 ? 1.0 : 0.0));
  occ *= 1.0 - uHairMass*vMass*0.7;           // render5: locks under other locks sit in their shade
  vec3 col = vec3(0.0);
  // key
  {
    vec3 L = uKeyDir;
    float tr = keyTrans(vP, nv);
    float tl = dot(T, L);
    float kk = sqrt(max(1.0 - tl*tl, 0.0));
    float wrap = pow(clamp(dot(nv, L)*0.5 + 0.5, 0.0, 1.0), 1.6);
    float dif = mix(kk*0.8, wrap, 0.7)*uDifK;
    vec3 H = normalize(L + V);
    float sh = ((vSeed - 0.5)*0.012 + (vNoise - 0.5)*0.05)*uHairK2.x;   // clump-coherent shift, a little per strand: crisp bands with fine lines (look-dev)
    float rgh = uRough*max(uHairK.y, 0.05)*(0.85 + 0.3*vSeed);   // render3: per-strand roughness jitter
    float e1 = 10.0/max(rgh*rgh, 0.01);                      // primary (R): a defined band, not a wash (look-dev)
    float e2 = e1*3.0;                                       // thinner, shifted secondary (TRT)
    float face = smoothstep(-0.2, 0.3, dot(nv, L));
    // azimuthal falloff on the mass normal (the band sits where the volume faces the half vector), weaker on the
    // short surface hairs (brows / lashes catch only a glint)
    float az = smoothstep(0.25, 0.95, dot(nv, H));
    float kspec = uKind >= 2 ? 0.25 : 1.0;
    float r1 = lobe(T, nv, -0.08 + sh, H, e1)*0.011*az*kspec*uHairK.x;
    // render3: a broad soft sheen around the band (the reference's satin glow over each lock), light-coloured
    r1 += lobe(T, nv, -0.08 + sh*0.5, H, 6.0)*0.010*uHairK.z*smoothstep(0.0, 0.9, dot(nv, H))*kspec;
    float r2 = lobe(T, nv, 0.10 + sh*1.6, H, e2)*0.004*az*kspec*uHairK2.z;   // render3: TRT gain
    vec3 hue = alb/max(max(alb.r, alb.g), 1e-3);
    vec3 spec = r1*mix(vec3(0.95, 0.97, 1.0), hue, uHairK2.y) + r2*mix(vec3(1.0), hue, 0.85);   // R keeps the light's colour (a cool cast on dark hair), TRT the hair's
    col += uKeyCol*uKeyGain*tr*(alb*dif*occ + spec*face*mix(0.5, 1.0, occ));
  }
  // fill (sky side): diffuse + a faint broad sheen
  {
    vec3 L = uFillDir;
    float wrap = clamp(dot(nv, L)*0.5 + 0.5, 0.0, 1.0);
    vec3 H = normalize(L + V);
    col += uFillCol*(alb*wrap*occ*uDifK + alb*lobe(T, nv, -0.08, H, 4.0/max(uRough*uRough, 0.01))*0.6*occ*az0(nv, H));   // (no grey sheen wash: tinted, narrower)
  }
  // rim (warm lamp behind): edge + transmission through thin strands (TT)
  {
    vec3 L = uRimDir2;
    float edge = pow(1.0 - abs(dot(nv, V)), 2.5)*smoothstep(-0.3, 0.5, dot(nv, L));
    float tt = pow(clamp(dot(-V, L), 0.0, 1.0), 4.0);
    vec3 H = normalize(L + V);
    float rs = lobe(T, nv, -0.08, H, 2.0/max(uRough*uRough, 0.01))*0.04;
    col += uRimCol2*uRimGain*((alb*1.6 + 0.004)*edge*occ + alb*tt*1.5*(1.0 - 0.5*a) + rs*edge*(alb/max(max(alb.r, alb.g), 1e-3)));
    // render3 (shade_k2.w): a light-coloured backlit sheen on the silhouette of the mass (the reference's rim on
    // the hair edge: dark hair still separates from the room)
    col += uRimCol2*uRimGain*uHairK2.w*0.05*pow(1.0 - abs(dot(nv, V)), 4.0)*smoothstep(-0.2, 0.6, dot(nv, L))*mix(0.5, 1.0, occ);
  }
  // room ambient on the hair mass normal
  col += setAmbient(vP, nv)*uAmbGain*alb*occ*0.9*uDifK;
  if(uDebug == 1) col = vec3(0.0, 1.0, 0.0)/max(uCharK, 1e-3);
  col = jFin3(col, 0.0, 6.0e4);                      // render6: finite guard (mesh.glsl jFin)
  fragC = vec4(col*uCharK, a);
  float t = length(vP - uCamPos);
  fragD = vec4(jFin1(t, 0.0, 6.0e4), 0.0, 0.0, a);
  fragS = vec4(0.0, 0.0, 0.0, a);
#ifdef MESH_TAA_R
  fragR = vec4(uTaaR.rgb*a, a);
#endif
}
#endif
#endif

// ============================================================ key shadow
#ifdef HAIR_SVERT
#ifdef HAIR_V2
out float vSA, vSSeed, vSU;
#endif
uniform mat4  uLightVP;
uniform vec3  uLightDir;
uniform float uTexelW;
void main(){
  vec3 P, T; float u, w, side;
  ribbonPoint(P, T, u, w, side);
  vec3 sv = cross(T, uLightDir);
  sv = dot(sv, sv) > 1e-12 ? normalize(sv) : vec3(1.0, 0.0, 0.0);
  float dw = max(w, uTexelW);
  gl_Position = uLightVP*vec4(P + sv*(0.5*dw*side), 1.0);
#ifdef HAIR_V2
  vSA = clamp(w/dw, 0.0, 1.0); vSSeed = gS.a.z; vSU = u;
#endif
}
#endif
#ifdef HAIR_SFRAG
#ifdef HAIR_V2
// fractional hair occlusion in the binary key map -- a strand survives in the map with probability ~ its true
// coverage of a map texel x uShadowKeep (a hash of the strand + an arc cell: stable under motion), so a sparse
// hairline casts a soft partial shadow after the skin's PCF while the dense mass stays opaque
in float vSA, vSSeed, vSU;
uniform float uShadowKeep;     // <= 0: every strand (binary, the old map)
float sh21(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
void main(){
  if(uShadowKeep > 0.0 && sh21(vec2(vSSeed*977.0, floor(vSU*6.0) + 0.5)) > vSA*uShadowKeep) discard;
}
#else
void main(){}
#endif
#endif

// ============================================================ hair-shade: deep opacity map (key light space)
#ifdef HAIR_DVERT
uniform mat4  uLightVP;
uniform vec3  uLightDir;
uniform float uTexelW;
out float vA;
void main(){
  vec3 P, T; float u, w, side;
  ribbonPoint(P, T, u, w, side);
  vec3 sv = cross(T, uLightDir);
  sv = dot(sv, sv) > 1e-12 ? normalize(sv) : vec3(1.0, 0.0, 0.0);
  float dw = max(w, uTexelW);
  vA = clamp(w/dw, 0.0, 1.0);
  gl_Position = uLightVP*vec4(P + sv*(0.5*dw*side), 1.0);
}
#endif
#ifdef HAIR_DFRAG
in float vA;
uniform sampler2D uShadow;
uniform vec2  uShadowZ;
uniform vec4  uDomZ;
uniform float uDomA;
uniform float uDomRes;
layout(location=0) out vec4 fragO;
void main(){
  vec2 uv = gl_FragCoord.xy/uDomRes;
  float h = 0.5/uDomRes;                              // the key map is finer: front = min over this texel's footprint
  float z0 = min(min(texture(uShadow, uv + vec2(-h, -h)*0.5).r, texture(uShadow, uv + vec2(h, -h)*0.5).r),
                 min(texture(uShadow, uv + vec2(-h, h)*0.5).r, texture(uShadow, uv + vec2(h, h)*0.5).r));
  float d = max((gl_FragCoord.z - z0)*uShadowZ.x, 0.0);
  // cumulative slabs, a soft step (half the first slab) so a strand never pops between layers
  float sw = 0.5*uDomZ.x;
  vec4 m = clamp((uDomZ - vec4(d))/sw + 0.5, 0.0, 1.0);
  fragO = m*vA*uDomA;
}
#endif

// ============================================================ hair cap: the groom's root density painted onto the skin
// (head submesh again, premultiplied over the skin, same depth): a dark scalp under the strands with a soft 3-5 px
// hairline, a soft base under the brows, a lash line.  Mask baked once from the strand roots (root_uv) by hair.py.
#ifdef HAIR_CAPV
in vec3 aP;
in vec3 aN;
in vec2 aUV;
out vec3 vP;
out vec3 vN;
out vec2 vUV;
void main(){ vP = aP; vN = aN; vUV = aUV; gl_Position = camClip(aP); }
#endif
#ifdef HAIR_CAPF
in vec3 vP;
in vec3 vN;
in vec2 vUV;
uniform sampler2D tCap;          // rgb = premultiplied root colour (linear), a = coverage
uniform float uCapK;
void main(){
  vec4 c = texture(tCap, vUV);
  float a = clamp(c.a*uCapK, 0.0, 1.0);
  if(a < 0.003) discard;
  vec3 alb = c.rgb/max(c.a, 1e-4);
  vec3 n = normalize(vN);
  vec3 V = normalize(uCamPos - vP);
  if(dot(n, V) < 0.0) n = -n;
#ifdef HAIR_V2
  float tr = keyTransV2(vP + n*0.002, n, 0.5);     // the scalp under the hair sits in the hair's deep shadow
#else
  float tr = keyTrans(vP + n*0.002, n);
#endif
  float wrap = clamp(dot(n, uKeyDir)*0.5 + 0.5, 0.0, 1.0);
  vec3 col = alb*(uKeyCol*uKeyGain*tr*wrap*0.6 + setAmbient(vP, n)*uAmbGain*0.5 + uFillCol*wrap*0.3);
  col = jFin3(col, 0.0, 6.0e4);                      // render6: finite guard
  fragC = vec4(col*uCharK*a, a);
  fragD = vec4(jFin1(length(vP - uCamPos), 0.0, 6.0e4)*a, 0.0, 0.0, a);
  fragS = vec4(0.0, 0.0, 0.0, a);
#ifdef MESH_TAA_R
  fragR = vec4(uTaaR.rgb*a, a);
#endif
}
#endif

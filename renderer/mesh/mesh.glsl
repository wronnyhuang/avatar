// ---------------------------------------------------------------------------
// renderer/mesh/mesh.glsl — render: Juno's rasterized passes (docs/CONTRACT--render.md section 1).
//  renderer/mesh/character.py assembles one program per section:  header (#version + engine defines) + #define <SECTION>
//  [+ MAT_<kind>] + (forward only: engine common.glsl + the set, so the mesh takes the room's light from the same
//  setAmbient / setCookie / setSky functions the SDF characters use) + this file.
//  Camera = the engine's pinhole: pixel -> ray = normalize(uv.x U + uv.y V + fl W), uv = (2 px - res)/res.y.
// ---------------------------------------------------------------------------

// ============================================================ M1: skinning + blendshapes (transform feedback)
#ifdef VS_SKIN
in vec3  aPos;
in vec3  aNrm;
in vec4  aTan;
in vec4  aJnt;
in vec4  aWt;
in float aReg;                 // index into the morph region (-1 = no shape touches this vertex)
uniform sampler2D      uBones;     // RGBA32F, 3 x nb: rows 0..2 of each skin matrix (bones @ inverse_bind)
uniform sampler2DArray uMorph;     // RGBA16F, MW x MH x (2 nshapes): layer 2s = dpos, 2s+1 = dnrm
uniform int   uMorphN;
uniform int   uMorphW;             // texels per row
uniform int   uMorphId[32];
uniform float uMorphWt[32];
out vec3 oPos;
out vec3 oNrm;
out vec4 oTan;
mat4 boneM(int b){
  vec4 r0 = texelFetch(uBones, ivec2(0, b), 0), r1 = texelFetch(uBones, ivec2(1, b), 0), r2 = texelFetch(uBones, ivec2(2, b), 0);
  return transpose(mat4(r0, r1, r2, vec4(0.0, 0.0, 0.0, 1.0)));
}
void main(){
  vec3 p = aPos, n = aNrm;
  if(aReg >= 0.0){
    int r = int(aReg + 0.5);
    ivec2 tc = ivec2(r % uMorphW, r / uMorphW);
    for(int k=0;k<32;k++){
      if(k >= uMorphN) break;
      float w = uMorphWt[k];
      int s = uMorphId[k];
      p += w*texelFetch(uMorph, ivec3(tc, 2*s), 0).xyz;
      n += w*texelFetch(uMorph, ivec3(tc, 2*s + 1), 0).xyz;
    }
  }
  mat4 M = aWt.x*boneM(int(aJnt.x + 0.5)) + aWt.y*boneM(int(aJnt.y + 0.5))
         + aWt.z*boneM(int(aJnt.z + 0.5)) + aWt.w*boneM(int(aJnt.w + 0.5));
  oPos = (M*vec4(p, 1.0)).xyz;
  mat3 R = mat3(M);
  vec3 nw = normalize(R*n);
  vec3 tw = R*aTan.xyz;
  tw = normalize(tw - nw*dot(nw, tw) + vec3(1e-6));
  oNrm = nw;
  oTan = vec4(tw, aTan.w);
}
#endif

// ============================================================ camera helpers (raster passes)
#if defined(VS_SHADOW) || defined(VS_FWD) || defined(FS_PREFILL) || defined(FS_FWD) || defined(VS_HAIR) || defined(FS_COMP)
uniform vec3  uCamPos, uCamU, uCamV, uCamW;
uniform float uFl;
uniform vec2  uRes;
uniform vec2  uNF;                 // near, far (camera depth along W)
#ifdef MESH_TAA_JIT
uniform vec2 uJit;                // shade-taa: projection jitter, NDC (2 px / res)
#endif
// render6: finite guard -- NaN / Inf -> 0, then clamp to +-hi (fp16 max 65504).  One bad texel (an uninitialised
// read, a 0/0 on one GPU's precision path) must never spread: the screen-space SSS blur turned single NaN pixels into
// sharp black rectangles over her face on one cloud A40 (PERF2 05:46).  isnan/isinf + a boolean mix (a select, no
// arithmetic on the NaN), never max(NaN, 0) (implementation-defined).
#ifdef JUNO_NOGUARD
vec4 jFin(vec4 x, float lo, float hi){ return x; }
#else
vec4 jFin(vec4 x, float lo, float hi){ return clamp(mix(x, vec4(0.0), bvec4(vec4(isnan(x)) + vec4(isinf(x)))), lo, hi); }
#endif
#ifdef JUNO_NOGUARD
vec3 jFin3(vec3 x, float lo, float hi){ return x; }
float jFin1(float x, float lo, float hi){ return x; }
#else
vec3 jFin3(vec3 x, float lo, float hi){ return clamp(mix(x, vec3(0.0), bvec3(vec3(isnan(x)) + vec3(isinf(x)))), lo, hi); }
float jFin1(float x, float lo, float hi){ return clamp((isnan(x) || isinf(x)) ? 0.0 : x, lo, hi); }
#endif
vec4 camClip(vec3 p){
  vec3 d = p - uCamPos;
  float x = dot(d, uCamU), y = dot(d, uCamV), z = dot(d, uCamW);
  float n = uNF.x, f = uNF.y;
  float A = (f + n)/(f - n), B = -2.0*f*n/(f - n);
#ifdef MESH_TAA_JIT
  // shade-taa (gated, mesh.pbr.taa): this frame's subpixel projection jitter (NDC units) -> every raster pass
  // that projects with camClip (forward, hair strands, hair cap) samples a different subpixel each frame
  return vec4(uFl*x*uRes.y/uRes.x + uJit.x*z, uFl*y + uJit.y*z, A*z + B, z);
#else
  return vec4(uFl*x*uRes.y/uRes.x, uFl*y, A*z + B, z);
#endif
}
vec3 camRayM(vec2 pix){
  vec2 uv = (2.0*pix - uRes)/uRes.y;
  return normalize(uv.x*uCamU + uv.y*uCamV + uFl*uCamW);
}
float depthFromT(float t, vec3 rd){      // ray distance -> window depth of this projection
  float z = max(t*dot(rd, uCamW), uNF.x);
  float n = uNF.x, f = uNF.y;
  float A = (f + n)/(f - n), B = -2.0*f*n/(f - n);
  return clamp(0.5*((A*z + B)/z) + 0.5, 0.0, 1.0);
}
#endif
#if defined(FS_PREFILL) || defined(FS_COMP)
// render2: the SDF character layer (props) at THIS pixel, gated exactly like the engine's own composite
// (rt_passes comp: tile = floor(px * uScale / 8) at the SDF INTERNAL res, mask .g > 0.5).  Neither layer is valid outside
// the tiles the mask marks: the raw per-tile layer (no TAA) is only written there, and the TAA history buffers keep STALE
// pixels outside this frame's tiles (prewarm walks every tier = other internal scales -> a ghost chair from another
// scale sat at her thigh in one of the two ping-pong buffers: the board-coloured rectangle + the dark stick, AVATAR_ADAPTIVE=0).
uniform sampler2D uChM;            // the SDF pass's tile mask for this frame (8 px tiles at uChTileRes), g = character tile
uniform float uChMOn;              // 1 = gate by uChM
uniform vec2  uChTileRes;          // the SDF char pass's internal size (iw, ih)
uniform float uChInv;              // 1/spp of the raw layer (1 for the history)
void chLayer(sampler2D sc, sampler2D sd, vec2 fc, out vec4 c, out vec4 d){
  if(uChMOn > 0.5 && texelFetch(uChM, ivec2(floor(fc*uChTileRes/uRes/8.0)), 0).g < 0.5){ c = vec4(0.0); d = vec4(0.0); return; }
  ivec2 ip = ivec2(fc*vec2(textureSize(sc, 0))/uRes);
  c = jFin(texelFetch(sc, ip, 0)*uChInv, 0.0, 6.0e4);     // render6: finite guard (TAA history / tile buffers)
  d = jFin(texelFetch(sd, ip, 0)*uChInv, 0.0, 6.0e4);
}
#endif

// ============================================================ M2: shadow map (key light, orthographic)
#ifdef VS_SHADOW
in vec3 aP;
uniform mat4 uLightVP;
void main(){ gl_Position = uLightVP*vec4(aP, 1.0); }
#endif
#ifdef FS_SHADOW
void main(){}
#endif

// ============================================================ M3: scene depth -> the mesh's depth buffer
#ifdef VS_FULLM
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
#endif
#ifdef FS_PREFILL
uniform sampler2D uBgD;          // the living background: a = ray distance
uniform sampler2D uChC, uChD;    // the SDF character layer (props): premultiplied, (t*a, ., ., a)
layout(location=0) out vec4 fragC;
layout(location=1) out vec4 fragD;
void main(){
  // render-perf: the forward pass may run below output size (quality.py mesh_scale): fetch each layer at its own
  // resolution (exactly ivec2(gl_FragCoord.xy) at scale 1)
  float t = texelFetch(uBgD, ivec2(gl_FragCoord.xy*vec2(textureSize(uBgD, 0))/uRes), 0).a;
  vec4 cc, cd; chLayer(uChC, uChD, gl_FragCoord.xy, cc, cd);     // render2: tile-mask gated (no-TAA path)
  // coverage from the COLOUR layer's alpha: the TAA history depth (rt_passes outD) is (t*a, 0, 0, 1) with
  // a = 1 everywhere it resolved, incl. empty pixels (t 0) -> 'cd.a > .995' made empty char-layer pixels depth 0
  float ca = cc.a;
  if(ca > 0.995) t = min(t, cd.r/ca);    // render-light: partly covered prop edges are re-laid over the mesh in FS_COMP
  gl_FragDepth = depthFromT(t, camRayM(gl_FragCoord.xy));
  fragC = vec4(0.0); fragD = vec4(0.0);
}
#endif

// ============================================================ M4: forward
#ifdef VS_FWD
in vec3 aP;
in vec3 aN;
in vec4 aT;
in vec2 aUV;
out vec3 vP;
out vec3 vN;
out vec4 vT;
out vec2 vUV;
#ifdef MESH_TAA
// shade-taa (gated): motion vectors -- this frame's and the previous frame's (posed vertex, camera) projections,
// both without the jitter
in vec3 aPp;
uniform vec3  uPCamPos, uPCamU, uPCamV, uPCamW;
uniform float uPFl;
out vec4 vCc;
out vec4 vCp;
#endif
void main(){
  vP = aP; vN = aN; vT = aT; vUV = aUV;
  gl_Position = camClip(aP);
#ifdef MESH_TAA
  vec3 d = aP - uCamPos;
  vCc = vec4(uFl*dot(d, uCamU)*uRes.y/uRes.x, uFl*dot(d, uCamV), 0.0, dot(d, uCamW));
  vec3 dp = aPp - uPCamPos;
  vCp = vec4(uPFl*dot(dp, uPCamU)*uRes.y/uRes.x, uPFl*dot(dp, uPCamV), 0.0, dot(dp, uPCamW));
#endif
}
#endif

#ifdef FS_FWD
in vec3 vP;
in vec3 vN;
in vec4 vT;
in vec2 vUV;
layout(location=0) out vec4 fragC;
layout(location=1) out vec4 fragD;
layout(location=2) out vec4 fragS;   // render-skin: skin diffuse for the SSS blur (sss.py); a = 1 - skin coverage
#ifdef MESH_TAA
in vec4 vCc;
in vec4 vCp;
layout(location=MESH_TAA_LOC) out vec4 fragV;   // shade-taa: (velocity px (cur - prev), ray distance, 1)
#endif
#ifdef MESH_TAA_R
layout(location=5) out vec4 fragR;   // shade2-taa: the material's TAA class (react, sharpen, gamma/4) * a, a
uniform vec4 uTaaR;
#endif
vec4 gSss = vec4(0.0);               // render-skin: set by mat_skin (rgb diffuse premultiplied, a = skin coverage)
#ifdef PBR_ALB_SPLIT
layout(location=3) out vec4 fragA;   // shade: skin albedo * skin coverage, skin coverage (sss.py SSS_ALB)
vec3 gAlb = vec3(0.0);               // set by mat_skin_pbr (gSss then carries irradiance, not radiance)
#endif
uniform vec3  uKeyDir, uKeyCol, uRimDir2, uRimCol2, uFillDir, uFillCol;
#ifdef PBR_SPOT
// shade2 (gated: mesh.pbr.light.spot): a portrait key that falls off away from the face (a snooted / feathered
// softbox): x,y,z centre (between the eyes, world), w the falloff start below it (m); uPbrSpot2: x falloff end (m),
// y key floor (x of the key at/after the end), z lateral radius scale, w ambient floor (the room light on the body)
uniform vec4 uPbrSpot, uPbrSpot2;
float pbrSpotW(vec3 p){
  vec3 d = p - uPbrSpot.xyz;
  float dn = max(-d.y, 0.0) + 0.35*max(d.y - 0.10, 0.0);       // below the eyes mainly (above: the crown, mildly)
  float r = length(vec2(dn, length(d.xz)*uPbrSpot2.z));
  return smoothstep(uPbrSpot2.x, uPbrSpot.w, r);
}
#endif
uniform float uKeyGain, uRimGain, uAmbGain, uCharK;
uniform vec3 uAmbTint;           // render look-dev: white balance of the room light on her (mesh.amb_tint)
uniform sampler2D uShadow;
uniform mat4  uLightVP;
uniform vec2  uShadowTexel;        // 1/res, world size of one texel
uniform float uShadowTaps;         // PCF taps (quality)
uniform sampler2D tAlb, tNrm, tPack, tDet;
uniform vec4  uTexOn;              // albedo, normal, packed, detail present
uniform vec3  uAlbC;               // albedo colour when there is no texture (linear)
uniform vec4  uMatP;               // per-material params (x roughness, y spec level, z sss/sheen strength, w detail tile)
uniform vec3  uMatC;               // per-material colour param (sss colour / sheen colour)
uniform vec4  uEyeP;               // eye: iris uv radius, pupil uv radius, iris depth, limbus width
uniform float uAlphaMul;
#pragma include light
// ( render-light: light.glsl = PCSS key shadow, capsule AO, the radiance probe)

vec3 srgb2lin(vec3 c){ return pow(max(c, 0.0), vec3(2.2)); }

const vec2 POIS[16] = vec2[16](vec2(-0.94201624,-0.39906216), vec2(0.94558609,-0.76890725), vec2(-0.094184101,-0.92938870),
  vec2(0.34495938,0.29387760), vec2(-0.91588581,0.45771432), vec2(-0.81544232,-0.87912464), vec2(-0.38277543,0.27676845),
  vec2(0.97484398,0.75648379), vec2(0.44323325,-0.97511554), vec2(0.53742981,-0.47373420), vec2(-0.26496911,-0.41893023),
  vec2(0.79197514,0.19090188), vec2(-0.24188840,0.99706507), vec2(-0.81409955,0.91437590), vec2(0.19984126,0.78641367),
  vec2(0.14383161,-0.14100790));

// soft shadow from the key: PCF over a rotated Poisson disc, radius in shadow texels; returns (sharp, wide)
vec2 keyShadow(vec3 p, vec3 ng, float rad){
  vec3 L = uKeyDir;
  float ndl = dot(ng, L);
  vec3 po = p + ng*uShadowTexel.y*(1.2 + 2.0*(1.0 - abs(ndl))) + L*uShadowTexel.y*0.5;
  vec4 c = uLightVP*vec4(po, 1.0);
  vec3 s = c.xyz/c.w*0.5 + 0.5;
  if(any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0)))) return vec2(1.0);
  float ang = hash12(gl_FragCoord.xy)*6.2831;
  mat2 R = mat2(cos(ang), sin(ang), -sin(ang), cos(ang));
  float a = 0.0, b = 0.0;
  int N = int(uShadowTaps);
  for(int i=0;i<16;i++){
    if(i >= N) break;
    vec2 o = R*POIS[i]*uShadowTexel.x;
    float d1 = texture(uShadow, s.xy + o*rad).r;
    float d2 = texture(uShadow, s.xy + o*rad*2.6).r;
    a += step(s.z - 0.0008, d1);
    b += step(s.z - 0.0008, d2);
  }
  return vec2(a, b)/float(max(N, 1));
}

float D_GGX(float nh, float a){ float a2 = a*a; float d = nh*nh*(a2 - 1.0) + 1.0; return a2/(PI*d*d); }
float V_SmithJ(float nv, float nl, float a){ float a2 = a*a;
  float gv = nl*sqrt(nv*nv*(1.0 - a2) + a2), gl = nv*sqrt(nl*nl*(1.0 - a2) + a2); return 0.5/max(gv + gl, 1e-5); }
vec3 F_Schlick(vec3 f0, float vh){ return f0 + (1.0 - f0)*pow(1.0 - vh, 5.0); }
float specGGX(vec3 n, vec3 v, vec3 l, float rough){
  vec3 h = normalize(v + l);
  float nl = clamp(dot(n, l), 0.0, 1.0), nv = clamp(dot(n, v), 1e-4, 1.0), nh = clamp(dot(n, h), 0.0, 1.0);
  float a = max(rough*rough, 0.002);
  return D_GGX(nh, a)*V_SmithJ(nv, nl, a)*nl;
}
// Charlie sheen (cloth)
float D_Charlie(float nh, float r){ float inv = 1.0/max(r, 0.05); float s2 = 1.0 - nh*nh; return (2.0 + inv)*pow(max(s2, 0.0), inv*0.5)/(2.0*PI); }

vec3 envSpec(vec3 p, vec3 r){
  if(uProbeP.x > 0.5) return envProbe(r, gEnvRough);      // render-light: the baked room at the material's roughness
  // the room in a reflected direction: bright window glass above the horizon toward -x, the warm room elsewhere
  vec3 a = setAmbient(p, r);
  float win = smoothstep(0.2, 0.9, -r.x)*smoothstep(-0.05, 0.35, r.y)*smoothstep(0.95, 0.4, r.y);
  return a*0.9 + SP_AMB_WIN*win*1.4;
}

// what every material gets (filled by main before matShade): world point, geometric + mapped normal, view, tangent
// frame, uv (after matUV), albedo (linear) + alpha, the packed maps, the key's soft shadow (x sharp, y wide), the key
// colour (with the room's window cookie), the room's ambient at the mapped normal
struct SIn {
  vec3 p, ng, n, v, T, B;
  vec2 uv;
  vec3 alb;
  float alpha, ao, rough, specL, thick, nv;
  vec2 shw;
  vec3 keyC, amb;
};
// rim from the lamp behind (outer silhouettes), shared
vec3 rimLight(SIn S){
  float edge = pow(1.0 - S.nv, 3.0);
  float rimB = smoothstep(-0.2, 0.6, dot(S.ng, uRimDir2));
  return uRimCol2*uRimGain*edge*rimB*S.alb*(0.6 + 0.4*S.ao)*1.6;
}
// ---- the material (one file per material kind; character.py splices the right one in: mat_<kind>.glsl)
//      defines  void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng)
//               vec4 matShade(SIn S, inout vec4 fragDOut)   -> premultiplied colour (rgb*a, a), radiance before uCharK
#pragma include mat

void main(){
  vec3 ng = normalize(vN);
  vec3 v = normalize(uCamPos - vP);
  if(!gl_FrontFacing) ng = -ng;
  vec3 T = normalize(vT.xyz - ng*dot(ng, vT.xyz) + vec3(1e-6));
  vec3 B = cross(ng, T)*(vT.w < 0.0 ? -1.0 : 1.0);
  vec2 uv = vUV;
  matUV(uv, v, T, B, ng);
  SIn S;
  S.p = vP; S.ng = ng; S.v = v; S.T = T; S.B = B; S.uv = uv;
  S.alb = uAlbC; S.alpha = 1.0; S.ao = 1.0; S.rough = uMatP.x; S.specL = uMatP.y; S.thick = 0.0;
  if(uTexOn.x > 0.5){ vec4 a4 = texture(tAlb, uv); S.alb = srgb2lin(a4.rgb); S.alpha = a4.a; }
  vec3 n = ng;
  if(uTexOn.y > 0.5){
    vec3 tn = texture(tNrm, uv).xyz*2.0 - 1.0;
    if(uTexOn.w > 0.5){ vec3 dn = texture(tDet, uv*uMatP.w).xyz*2.0 - 1.0; tn = vec3(tn.xy + dn.xy*0.5, tn.z); }
    n = normalize(T*tn.x + B*tn.y + ng*max(tn.z, 0.05));
  } else if(uTexOn.w > 0.5){
    vec3 dn = texture(tDet, uv*uMatP.w).xyz*2.0 - 1.0;
    n = normalize(T*dn.x*0.6 + B*dn.y*0.6 + ng*max(dn.z, 0.05));
  }
  S.n = n;
  if(uTexOn.z > 0.5){
    vec4 pk = texture(tPack, uv);
    S.ao = pk.r; S.rough = pk.g; S.specL = pk.b*2.0*uMatP.y; S.thick = pk.a;
  }
  S.alpha *= uAlphaMul;
  S.nv = clamp(dot(n, v), 1e-4, 1.0);
  float ck = mix(0.80, 1.0, setCookie(vP));
  S.shw = uLightP.w > 0.5 ? lightShadow(vP, ng) : keyShadow(vP, ng, 1.6);   // render-light: PCSS
  S.ao *= capsuleAO(vP, ng);                                                // render-light: capsule AO
  gEnvRough = S.rough;
  S.keyC = uKeyCol*uKeyGain*ck;
#ifdef PBR_SPOT
  float spw = pbrSpotW(vP);
  S.keyC *= mix(uPbrSpot2.y, 1.0, spw);
#endif
  S.amb = setAmbient(vP, n)*uAmbGain;
  if(uProbeP.x > 0.5 && uProbeP.z > 0.0)        // render-light: the room's own bounce colour/direction, same level
    S.amb = mix(S.amb, envProbe(n, 1.0)*uAmbGain, uProbeP.z);
  S.amb *= uAmbTint;
#ifdef PBR_SPOT
  S.amb *= mix(uPbrSpot2.w, 1.0, spw);
#endif                                                        // render look-dev: white balance
  // render6: the inputs every material shades with, finite and in range (a bad texel in a map / the probe / the
  // shadow map stays one dim pixel instead of a NaN the SSS spreads): light >= 0, ao / shadow / coverage 0..1
  S.amb = jFin3(S.amb, 0.0, 6.0e4); S.keyC = jFin3(S.keyC, 0.0, 6.0e4); S.alb = jFin3(S.alb, 0.0, 4.0);
  S.shw = vec2(jFin1(S.shw.x, 0.0, 1.0), jFin1(S.shw.y, 0.0, 1.0)); S.ao = jFin1(S.ao, 0.0, 1.0);
  S.alpha = jFin1(S.alpha, 0.0, 1.0); S.thick = jFin1(S.thick, 0.0, 1.0);
  S.rough = jFin1(S.rough, 0.0, 1.0); S.specL = jFin1(S.specL, 0.0, 4.0);
  if(any(isnan(S.n)) || dot(S.n, S.n) < 1e-8) S.n = ng;

  vec4 dOut = vec4(-1.0);
  vec4 c = matShade(S, dOut);
  if(uProbeP.w > 0.5) c = vec4(vec3(S.shw.x, S.shw.y, S.ao)*c.a, c.a)/max(uCharK, 1e-3);   // render-light: debug view
  // render6: every attachment finite and in range (premultiplied colour >= 0, coverage 0..1) -- see jFin
  c = vec4(jFin3(c.rgb, 0.0, 6.0e4), jFin1(c.a, 0.0, 1.0));
  vec4 gs = vec4(jFin3(gSss.rgb, 0.0, 6.0e4), jFin1(gSss.a, 0.0, 1.0));
  fragC = vec4(c.rgb*uCharK, c.a);
  fragS = vec4(gs.rgb*uCharK, c.a*(1.0 - gs.a));   // render-skin
#ifdef PBR_ALB_SPLIT
  float sc4 = c.a*gs.a;                                   // shade: this fragment's skin coverage
  fragA = vec4(clamp(gAlb, 0.0, 1.0)*sc4, sc4);
#endif
  float t = length(vP - uCamPos);
  fragD = dOut.x >= 0.0 ? jFin(dOut, 0.0, 6.0e4) : vec4(t*c.a, 0.0, 0.0, c.a);
#ifdef MESH_TAA
  vec2 vel = (vCc.xy/max(vCc.w, 1e-4) - vCp.xy/max(vCp.w, 1e-4))*0.5*uRes;
  float tv = t;
#ifdef MAT_EYE
  tv = -t;                         // the eye (refracted iris, catchlight): REACTIVE in the resolve (sign of the depth)
#endif
  fragV = vec4(jFin3(vec3(vel, tv), -6.0e4, 6.0e4), 1.0);
#endif
#ifdef MESH_TAA_R
  fragR = vec4(uTaaR.rgb*c.a, c.a);
#endif
}
#endif

// ============================================================ render7: the room for the zoom-driven background blur
// The live close-up is a turn + ZOOM of the rest camera (focus stays ~7.7): the baked room keeps the rest lens's blur,
// magnified -- a real long lens blurs ~kz^2.  This pass copies the living background (no Juno in it) to a half-res
// mipmapped target, premultiplied by "behind the focus plane" (a nearer desk edge never bleeds into the bokeh).
#ifdef FS_BBDOWN
uniform sampler2D uBgD;
uniform vec2  uBBRes;
uniform float uBBFocus;
uniform vec2 uBBH;            // render8: highlight weight (x gain, y luminance threshold): bright patches dominate the gather -> bokeh discs
out vec4 fragC;
void main(){
  vec4 b = texture(uBgD, gl_FragCoord.xy/uBBRes);
  b = clamp(mix(b, vec4(0.0), bvec4(vec4(isnan(b)) + vec4(isinf(b)))), 0.0, 6.0e4);   // finite guard (as jFin)
  float w = max(smoothstep(uBBFocus + 0.15, uBBFocus + 0.9, b.a), 1.0e-3);
  if(uBBH.x > 0.0) w *= 1.0 + uBBH.x*smoothstep(uBBH.y, 3.0*uBBH.y, dot(b.rgb, vec3(0.2126, 0.7152, 0.0722)));
  fragC = vec4(b.rgb*w, w);
}
#endif

// ============================================================ M7: composite (mesh layer over the scene, own DoF)
#ifdef FS_COMP
uniform sampler2D uMC, uMD;        // resolved mesh layer: premultiplied colour, (t*a, ., ., a)
uniform float uFocus, uDofK, uDofMax;
uniform int   uFrame;
uniform int   uDofN;
out vec4 fragColor;
// render-light: her contact occlusion on the scene + held props over the mesh
#pragma include light
uniform sampler2D uBgD, uChC, uChD;   // the scene: ray distance (a), the SDF character layer (props): premultiplied, (t*a,..,a)
float gOcc = 1.0;                      // her occlusion of the scene at this pixel (1 = none)
// render5: the room behind her in a CLOSE view (card / home close-up) -- depth haze toward a light, desaturated
// tone (the reference's lifted creamy bokeh): she separates from the set.  uBgK: x strength (0 = off; the engine gates
// it by the camera's focus distance), y start behind the focus plane, z ramp length (scene units); uBgCol linear HDR.
uniform vec4 uBgK;
uniform vec3 uBgCol;
float gFog = 0.0;
// render7: zoom-driven room blur (live close-up).  uBBK: x radius (output px) for a point far behind the focus
// plane (0 = off), y max radius, z lod bias; uBB the half-res mipmapped room (premultiplied by its far weight).
uniform sampler2D uBB;
uniform vec4 uBBK;
float gBB = 0.0;
vec3  gBBc = vec3(0.0);
uniform sampler2DShadow uShadowC;      // the key's shadow map (compare): her key shadow laid softly on the set
uniform mat4  uLightVP;
uniform vec2  uShadowTexel;
uniform vec3  uSunDirC;                // the key direction
uniform float uKeySet;                 // strength of her key shadow on the set (0 = off: only the engine's live shadow)
float keyOnSet(vec3 p, vec3 n){
  vec4 c = uLightVP*vec4(p + n*uShadowTexel.y*3.0, 1.0);
  vec3 s = c.xyz/c.w*0.5 + 0.5;
  if(any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0))) || s.z >= 1.0) return 1.0;
  float rot = fract(52.9829189*fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))))*6.2831853;
  float a = 0.0;
  for(int i=0;i<10;i++){
    float rr = sqrt((float(i) + 0.5)/10.0), an = float(i)*2.39996 + rot;
    a += texture(uShadowC, vec3(s.xy + vec2(cos(an), sin(an))*rr*uShadowTexel.x*14.0, s.z - 0.002));
  }
  return a/10.0;
}
// render8: close-view background grade (mesh.bg_grade): the room behind her about a stop down, a soft gradient
// lighter toward the key side, a slight cool lift -- she pops off the wall.  uBgG: x level (1 = off, .5 = one stop),
// y gradient, z key side on screen (+1 = right, -1 = left), w weight (close-view / zoom eased, 0 = off); uBgCool lift.
uniform vec4 uBgG;
uniform vec3 uBgCool;
float gBgW = 0.0;
vec4 compOut(vec4 c){                  // mesh (premultiplied) over the scene, the scene darkened by her contact shade
  float o = jFin1(gOcc, 0.0, 1.0), f = jFin1(gFog, 0.0, 1.0);    // render6: finite guard (a NaN scene depth)
  float keep = (1.0 - c.a)*o*(1.0 - f*uBgK.w);   // w: share of the room replaced (0 = a pure lift)
  float gw = jFin1(gBgW, 0.0, 1.0);
  if(gw > 0.0){                                   // render8
    float sx = (gl_FragCoord.x/uRes.x*2.0 - 1.0)*uBgG.z;
    keep *= mix(1.0, uBgG.x*clamp(1.0 + uBgG.y*sx, 0.0, 2.0), gw);
    c.rgb += (1.0 - c.a)*o*gw*uBgCool;
  }
  float bb = jFin1(gBB, 0.0, 1.0);                // render7: room -> mix(room, blurred room, bb), then shade + haze
  vec4 r = vec4(c.rgb + (1.0 - c.a)*f*uBgCol + keep*bb*jFin3(gBBc, 0.0, 6.0e4), 1.0 - keep*(1.0 - bb));
  return vec4(jFin3(r.rgb, 0.0, 6.0e4), jFin1(r.a, 0.0, 1.0));
}
float h12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
#ifdef PBR_AO
uniform sampler2D uAO;                 // shade-post: the mesh layer's contact AO (ssao.glsl), r = 1 open
uniform float uAoAmt;
float aoAt(vec2 q){ return mix(1.0, clamp(texture(uAO, q).r, 0.0, 1.0), uAoAmt); }
#endif
float cocPx(float d){ return clamp(uDofK*abs(1.0/uFocus - 1.0/max(d, 0.05)), 0.0, uDofMax); }
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec4 c0 = jFin(texture(uMC, uv), 0.0, 6.0e4);   // render6: finite guard (jFin)
  vec4 d0 = jFin(texture(uMD, uv), 0.0, 6.0e4);
#ifdef PBR_AO
  c0.rgb *= aoAt(uv);
#endif
  // ---- render-light{  the scene point behind this pixel: her capsules occlude its ambient (contact shade)
  ivec2 ip = ivec2(gl_FragCoord.xy);
  float ts = texelFetch(uBgD, ip, 0).a;
  vec4 chc, chd; chLayer(uChC, uChD, gl_FragCoord.xy, chc, chd);   // render2: own res + tile mask (no-TAA path)
  chd.a = chc.a;                       // coverage = colour alpha (TAA history depth has a = 1 everywhere)
  if(chd.a > 0.5) ts = min(ts, chd.r/chd.a);
  vec3 rdS = camRayM(gl_FragCoord.xy);
  vec3 pS = uCamPos + rdS*ts;
  // render5: the scene depth is fp16 (t_live .a): at 1080p a 1-px dFdx of the reconstructed point is below its
  // quantum (~5e-3 at 5 units vs ~2e-3 per px) -> degenerate / flipped normals along iso-depth lines = the dotted RINGS
  // in her shadow on the board (INTEG 04:41).  Normal from a 4-px baseline, the one-sided side with the smaller step
  // (never across a silhouette).
  vec3 nS;
  {
    const int B = 4;
    ivec2 sz = textureSize(uBgD, 0) - 1;
    float tx1 = texelFetch(uBgD, clamp(ip + ivec2(B, 0), ivec2(0), sz), 0).a, tx0 = texelFetch(uBgD, clamp(ip - ivec2(B, 0), ivec2(0), sz), 0).a;
    float ty1 = texelFetch(uBgD, clamp(ip + ivec2(0, B), ivec2(0), sz), 0).a, ty0 = texelFetch(uBgD, clamp(ip - ivec2(0, B), ivec2(0), sz), 0).a;
    float tb = texelFetch(uBgD, ip, 0).a;
    vec3 pB = uCamPos + rdS*tb;
    vec3 px1 = uCamPos + camRayM(gl_FragCoord.xy + vec2(B, 0))*tx1 - pB, px0 = pB - (uCamPos + camRayM(gl_FragCoord.xy - vec2(B, 0))*tx0);
    vec3 py1 = uCamPos + camRayM(gl_FragCoord.xy + vec2(0, B))*ty1 - pB, py0 = pB - (uCamPos + camRayM(gl_FragCoord.xy - vec2(0, B))*ty0);
    vec3 dx = abs(tx1 - tb) < abs(tb - tx0) ? px1 : px0;
    vec3 dy = abs(ty1 - tb) < abs(tb - ty0) ? py1 : py0;
    nS = cross(dx, dy);
    if(chd.a > 0.5 && chd.r/chd.a < tb - 1e-3) nS = cross(dFdx(pS), dFdy(pS));   // a prop in front: its own surface
  }
  nS = dot(nS, nS) > 1e-14 ? normalize(nS) : vec3(0.0, 1.0, 0.0);
  if(dot(nS, rdS) > 0.0) nS = -nS;
  if(uCapP.w > 0.0 && ts < 200.0){
    float occ = capsuleOcc(pS + nS*0.004, nS) + 0.7*capsuleContact(pS + nS*0.01);
    gOcc = clamp(1.0 - uCapP.w*occ, 0.25, 1.0);
    if(uKeySet > 0.0) gOcc *= 1.0 - uKeySet*(1.0 - keyOnSet(pS, nS))*clamp(dot(nS, uSunDirC), 0.0, 1.0);
  }
  if(uBgK.x > 0.0) gFog = uBgK.x*smoothstep(uFocus + uBgK.y, uFocus + uBgK.y + uBgK.z, ts);
  if(uBgG.w > 0.0) gBgW = uBgG.w*smoothstep(uFocus + 0.3, uFocus + 1.0, ts);   // render8: the room behind her only
  // render7: the room behind the focus plane through a longer lens (zoom-driven, eases with the zoom)
  if(uBBK.x > 0.0){
    float tr = texelFetch(uBgD, ip, 0).a;                      // the room itself (props keep their own layer)
    float r = min(uBBK.x*max(0.0, 1.0 - uFocus/max(tr, 0.05)), uBBK.y);
    if(r > 0.5){
      float lod = max(0.0, log2(0.3*r) + uBBK.z);
      float rot = h12(gl_FragCoord.xy)*6.2831;                 // fixed per pixel: no temporal crawl (no TAA after us)
      int nt = uBBK.w > 0.5 ? int(uBBK.w) : 8;                 // render8: more taps = a disc kernel (bokeh), 8 = render7
      if(uBBK.w > 0.5) lod = max(0.0, log2(2.2*r/sqrt(float(nt))) + uBBK.z);
      vec4 acc = textureLod(uBB, uv, lod);
      for(int i=0;i<32;i++){
        if(i >= nt) break;
        float rr = sqrt((float(i) + 0.5)/float(nt)), an = float(i)*2.39996 + rot;
        acc += textureLod(uBB, uv + vec2(cos(an), sin(an))*rr*r/uRes, lod);
      }
      gBBc = acc.rgb/max(acc.a, 1.0e-4);
      gBB = smoothstep(0.5, 2.0, r)*(1.0 - clamp(chc.a, 0.0, 1.0))*smoothstep(0.02, 0.2, acc.a/float(nt + 1));
    }
  }
  // a held prop in front of her with a partly covered (antialiased / blurred) edge: lay it over the mesh again (the
  // scene below has it over the empty room: src = (1-a) M + m P, alpha m)
  if(chd.a > 0.002 && chd.a <= 0.995 && d0.a > 0.01 && chd.r/chd.a < d0.r/d0.a){
    c0 = vec4(c0.rgb*(1.0 - chd.a) + chc.rgb*c0.a, c0.a);
  }
  // ---- render-light}
  // the widest blur any mesh pixel near here could have: from this pixel, or the nearest mesh in a small ring
  float dd = d0.a > 0.01 ? d0.r/d0.a : -1.0;
  float cc = dd > 0.0 ? cocPx(dd) : 0.0;
  if(cc < 0.6){
    // in focus: a probe for a blurred neighbour (only matters for a defocused close-up)
    if(c0.a < 0.001 && uDofK < 1e-3){ if(gOcc > 0.999 && gFog < 1e-4 && gBB < 1e-4 && gBgW < 1e-4) discard; fragColor = compOut(vec4(0.0)); return; }
    if(cc < 0.6 && c0.a < 0.001){
      float m = 0.0;
      for(int i=0;i<4;i++){
        vec2 o = vec2(i == 0 ? 6.0 : (i == 1 ? -6.0 : 0.0), i == 2 ? 6.0 : (i == 3 ? -6.0 : 0.0));
        vec4 dn = texture(uMD, uv + o/uRes);
        if(dn.a > 0.01) m = max(m, cocPx(dn.r/dn.a));
      }
      if(m < 0.6){ if(gOcc > 0.999 && gFog < 1e-4 && gBB < 1e-4 && gBgW < 1e-4) discard; fragColor = compOut(vec4(0.0)); return; }
      cc = m;
    } else {
      fragColor = compOut(c0);
      return;
    }
  }
  float rot0 = h12(gl_FragCoord.xy + float(uFrame)*0.37)*6.2831;
  vec4 acc = c0*1.5; float wsum = 1.5;
  for(int i=0;i<48;i++){
    if(i >= uDofN) break;
    float r = sqrt((float(i) + 0.5)/float(uDofN));
    float a = float(i)*2.39996 + rot0;
    vec2 uv2 = uv + vec2(cos(a), sin(a))*r*cc/uRes;
    vec4 s = jFin(texture(uMC, uv2), 0.0, 6.0e4);
#ifdef PBR_AO
    s.rgb *= aoAt(uv2);
#endif
    vec4 sd = jFin(texture(uMD, uv2), 0.0, 6.0e4);
    // a sample spreads here only if its own blur reaches this far
    float sc = sd.a > 0.01 ? cocPx(sd.r/sd.a) : cc;
    float w = smoothstep(r*cc - 1.0, r*cc + 1.0, sc + 0.5);
    acc += s*w; wsum += w;
  }
  fragColor = compOut(acc/wsum);
}
#endif

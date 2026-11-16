// mat_skin.glsl — render-face: skin (skin_head, skin_body).  Spliced into mesh.glsl FS_FWD for MAT_SKIN.
//  Diffuse (key + fill + ambient + transmission) -> gSss (the SSS attachment, blurred by sss.py) when uSssSplit = 1,
//  else the pre-integrated wrap fallback straight into the colour.  Colour = dual-lobe GGX + env specular + rim.
//  Packed map: R AO, G roughness, B spec level (0.5 = F0 0.028), A thickness (1 = thin: ears, nostril wings, lids).
//  Face fallbacks (eyesmouth.py, uFaceProc): when the asset's packed map is a constant, crease AO / thin parts /
//  blush / under-eye violet / lip rose come from the landmarks (eye centres, lip seam) in the head frame, in units of
//  the eye-to-mouth distance D, so they follow any head shape and every expression.
uniform float uSssSplit;       // 1 = screen-space SSS active (diffuse to the S attachment)
uniform float uSssWrap;        // pre-integrated wrap amount (fallback 1.0; with the blur a residual share)
uniform float uTransKey, uTransRim;
uniform vec3  uSssRadSu;       // scatter radius per channel, scene units
uniform mat3  uHeadR;          // head frame (x right, y up, z out of the face)
uniform float uHeadS;          // scene units per asset metre
uniform vec3  uEyeC[2];
uniform float uEyeR;
uniform vec3  uMouthC, uMouthU, uMouthL;
uniform float uMouthW;
uniform float uFaceProc;       // procedural face maps weight (1 when the asset's maps are flat)
uniform float uFaceTint;       // blush / violet / lip tint weight
uniform float uSkinSpec;       // render look-dev: overall skin specular gain (juno.json mesh.skin.spec)
uniform float uFaceLiner;      // render2: upper lash-line darkening on the lid margin (mesh.face.liner)
uniform float uFaceAO;         // render2: the landmark crease AO over the asset's own maps (mesh.face.ao)
uniform vec4 uSkinK;         // render3 (mesh.face.skin_k): x terminator glow, y coloured-penumbra spread, z grazing sheen,
                              //   w shadow-side chroma clamp (0 = off): how far the unlit side may out-saturate the lit side

uniform vec4 uSkinK2;        // render4 (mesh.face.skin_k2): x micro-detail amplitude, y pore frequency (per uv unit),
                              //   z region spec (wet tight lobe on lower lip / nose tip / inner eye corner), w broad soft sheen
uniform vec4 uSkinK3;        // render5 (mesh.face.skin_k3): x de-blotch (blush/flush toward the base skin hue: warm
                               //   orange-brown, not pink), y warm bounce from below-front (cardigan / desk), z tan depth, w broad sheen at bust size
uniform vec4 uSkinK4;        // render5 (mesh.face.skin_k4): x under-jaw neck shade, y form falloff toward the silhouette
uniform vec4 uRimK;          // render7 (mesh.face.rim_k): the rim only where the head's own silhouette is (x strength, y..z gate on |radial. view|)
uniform vec4 uSssK;          // render4 (mesh.face.sss_k): x the skin's own albedo luminance (linear); texels well below it
                              //   are PAINTED hair (brows / hairline under the strands) and stay out of the scatter (0 = off)
uniform vec4 uDetailK;      // render8 (mesh.face.detail_k): x normal-map strength, y baked AO strength, z albedo
                              //   mottling flatten (edge-preserving: lips / brows / painted hair kept), w landmark crease AO scale
                              //   (default 1,1,0,1 = the picture before render8): the face as ONE smooth stylized form
uniform vec4 uWarmK;        // render8 (mesh.face.warm_k): stylized colour zones over the asset's maps -- x warm cheeks /
                              //   nose / ears (rosy-warm, never pink), y thin-part boost (ears / nose wings transmit), z cooler
                              //   brow / jaw, w lip tint (0 = off)
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}

// render4: procedural skin micro-detail (UV space, so it sticks to the skin; faded by the pixel footprint so it
// never sparkles): value noise + its analytic gradient
float skHash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x*p.y); }
vec3 skNoise(vec2 x){                         // value noise in [0,1] + d/dx, d/dy
  vec2 i = floor(x), f = fract(x);
  vec2 u = f*f*(3.0 - 2.0*f), du = 6.0*f*(1.0 - f);
  float a = skHash(i), b = skHash(i + vec2(1, 0)), c = skHash(i + vec2(0, 1)), d = skHash(i + vec2(1, 1));
  float k = a - b - c + d;
  return vec3(a + (b - a)*u.x + (c - a)*u.y + k*u.x*u.y, du*vec2(b - a + k*u.y, c - a + k*u.x));
}
// x height-ish (pore dimples dark), yz gradient in uv units, w the visibility fade
vec4 skinMicro(vec2 uv, float freq){
  float fp = max(length(dFdx(uv)), length(dFdy(uv)));     // uv per pixel
  vec4 acc = vec4(0.0);
  float f = freq, amp = 1.0, tot = 0.0, vis = 0.0;
  for(int o = 0; o < 3; o++){
    float w = smoothstep(0.7, 0.25, fp*f)*amp;            // feature (1/f) must be >= ~1.5-4 px to be shown
    vec3 nn = skNoise(uv*f + float(o)*17.3);
    // pores: sharp dimples where the noise is low
    float pore = smoothstep(0.32, 0.0, nn.x);
    acc.x += w*(nn.x - 0.5 - 0.8*pore);
    acc.yz += w*nn.yz*f*(1.0 + 2.5*smoothstep(0.32, 0.0, nn.x));
    tot += amp; vis += w;
    f *= 2.17; amp *= 0.6;
  }
  return vec4(acc.x/tot, acc.yz/tot, vis/tot);
}

// key-light thickness through the mesh from the shadow map (orthographic: depth is linear; one window-depth unit =
// 2 * the map's half depth range = 2 * texel world size / texel uv size)
float skinKeyThickness(vec3 p, vec3 ng){
  vec3 po = p - ng*uShadowTexel.y*0.75;
  vec4 c = uLightVP*vec4(po, 1.0);
  vec3 s = c.xyz/c.w*0.5 + 0.5;
  if(any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0)))) return 1e3;
  float d = 0.0;
  float r = uShadowTexel.x*2.0;
  d += texture(uShadow, s.xy).r;
  d += texture(uShadow, s.xy + vec2( r, 0.0)).r;
  d += texture(uShadow, s.xy + vec2(-r, 0.0)).r;
  d += texture(uShadow, s.xy + vec2(0.0,  r)).r;
  d += texture(uShadow, s.xy + vec2(0.0, -r)).r;
  d *= 0.2;
  return max(s.z - d, 0.0)*2.0*uShadowTexel.y/uShadowTexel.x;
}

// Karis' analytic split-sum environment BRDF
vec2 skinEnvBRDF(float r, float nv){
  const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
  const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
  vec4 r4 = r*c0 + c1;
  float a004 = min(r4.x*r4.x, exp2(-9.28*nv))*r4.x + r4.y;
  return vec2(-1.04, 1.04)*a004 + r4.zw;
}

float gauss2(vec2 d, vec2 r){ vec2 q = d/r; return exp(-dot(q, q)); }
float segDist(vec2 p, vec2 a, vec2 b){ vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0); return length(pa - ba*h); }

// face regions: x = crease AO (1 open, <1 occluded), y = thin (0..1), z = blush, w = violet; lip = lip mask; mouthIn =
// how deep inside the mouth (0 outside .. 1 deep)
float r_up(vec3 p){
  if(uHeadS <= 0.0) return 0.0;
  vec3 em = 0.5*(uEyeC[0] + uEyeC[1]);
  mat3 Ri = transpose(uHeadR);
  float D = max(-(Ri*(uMouthC - em)).y/uHeadS, 1e-3);
  vec3 q = Ri*(p - em)/uHeadS/D;
  return q.y*step(length(q), 3.0);
}
struct Face { float ao, thin, blush, violet, lip, mouthIn; };
Face faceRegions(vec3 p, vec3 S_ng_head){
  Face F; F.ao = 1.0; F.thin = 0.0; F.blush = 0.0; F.violet = 0.0; F.lip = 0.0; F.mouthIn = 0.0;
  if(uHeadS <= 0.0) return F;
  vec3 em = 0.5*(uEyeC[0] + uEyeC[1]);
  mat3 Ri = transpose(uHeadR);
  vec3 q = Ri*(p - em)/uHeadS;                      // metres in the head frame, origin between the eyes
  vec3 mq = Ri*(uMouthC - em)/uHeadS;
  float D = max(-mq.y, 1e-3);                       // eye line -> lip seam
  vec3 r = q/D;
  if(dot(r, r) > 9.0) return F;                     // far from the face (body, hands)
  vec3 e0 = Ri*(uEyeC[0] - em)/uHeadS/D;
  float ex = abs(e0.x);                             // half eye spacing (units of D)
  float eR = uEyeR/uHeadS/D;
  vec2 a = vec2(abs(r.x), r.y);
  vec3 uq = Ri*(uMouthU - em)/uHeadS/D, lq = Ri*(uMouthL - em)/uHeadS/D;
  float mw = 0.5*uMouthW/uHeadS/D;                  // mouth half width
  float front = smoothstep(-0.9, -0.2, r.z - e0.z); // the face's front half (not the back of the head)
  // ---- crease AO: inner canthus, eye socket rim, nostrils / under the nose, nasolabial fold, mouth corners, under lip
  float ao = 1.0;
  ao *= 1.0 - 0.45*gauss2(a - vec2(ex - 1.05*eR, -0.04), vec2(0.07, 0.06));            // inner canthus
  float de = length(vec2(a.x - ex, a.y + 0.02))/eR;
  ao *= 1.0 - 0.12*smoothstep(2.0, 1.1, de)*smoothstep(0.9, 1.1, de);                 // socket rim around the lids
  ao *= 1.0 - 0.40*gauss2(a - vec2(0.14, -0.66), vec2(0.07, 0.045));                     // nostril / alar crease
  ao *= 1.0 - 0.25*gauss2(a - vec2(0.0, -0.74), vec2(0.10, 0.04));                       // under the nose (philtrum top)
  float nl = segDist(a, vec2(0.24, -0.62), vec2(mw + 0.07, uq.y - 0.08));
  ao *= 1.0 - 0.20*smoothstep(0.06, 0.0, nl)*front;                                       // nasolabial fold
  ao *= 1.0 - 0.25*gauss2(a - vec2(mw + 0.01, 0.5*(uq.y + lq.y)), vec2(0.04, 0.04));     // mouth corners
  ao *= 1.0 - 0.20*gauss2(a - vec2(0.0, lq.y - 0.20), vec2(0.16, 0.04));                 // under the lower lip
  F.ao = mix(1.0, pow(ao, 1.6), front);
  // ---- thin parts: ears (lateral, behind the face plane), nose wings + tip, eyelids
  // render3: the ear window also in depth (between ~0.5 and ~1.7 D behind the eye plane) and no lower than the
  // nose: the old lateral band also took the jaw / neck under the ear -> a red "thin" glow there seen from behind
  float dz = r.z - e0.z;
  float ear = smoothstep(1.08, 1.25, abs(r.x))*smoothstep(-0.95, -0.75, r.y)*smoothstep(0.45, 0.2, r.y)
            *smoothstep(-0.35, -0.6, dz)*smoothstep(-1.9, -1.6, dz);
  float nose = gauss2(a - vec2(0.12, -0.60), vec2(0.09, 0.08))*smoothstep(-0.1, 0.25, r.z - e0.z);
  float lid = smoothstep(1.3, 1.0, de)*smoothstep(0.85, 0.95, de);
  F.thin = max(ear, max(0.75*nose, 0.3*lid));
  // ---- colour zones
  F.blush = gauss2(a - vec2(ex + 0.10, -0.50), vec2(0.30, 0.22))*front + 0.8*gauss2(a - vec2(0.0, -0.57), vec2(0.12, 0.09))*smoothstep(0.0, 0.3, r.z - e0.z) + 0.8*ear;
  F.violet = gauss2(a - vec2(ex, -0.33), vec2(0.26, 0.09)) + 0.6*gauss2(a - vec2(ex, 0.30), vec2(0.24, 0.07));
  float cy = 0.5*(uq.y + lq.y);
  float lx = clamp(a.x/max(mw*0.95, 1e-3), 0.0, 1.0);
  float shape = sqrt(max(1.0 - lx*lx, 0.0));
  float top = uq.y + 0.07*shape, bot = lq.y - 0.085*shape;
  F.lip = smoothstep(0.03, 0.0, max(r.y - top, bot - r.y))*smoothstep(1.0, 0.8, lx)*front;
  // ---- inside the mouth: anything behind the lip seam seen THROUGH the lip opening (projects inside it) is mouth
  //      interior; the light reaching it falls off with depth behind the seam, scaled by the lip gap (the same model
  //      as mat_generic's teeth / tongue so the cavity grades consistently)
  float behindM = (mq.z/D - r.z)*D;                      // metres behind the seam
  float gap = max(uq.y - lq.y, 0.0)*D;
  float ey = (r.y - 0.5*(uq.y + lq.y))/max(0.5*(uq.y - lq.y) + 0.04, 0.02);
  float ex2 = a.x/max(mw, 1e-3);
  float inOpen = smoothstep(1.15, 0.85, length(vec2(ex2, ey)));
  float reach = 0.004 + 0.9*gap;
  float lit = exp(-max(behindM - 0.001, 0.0)/reach);
  // render3: only the mouth's own depth (~5 cm behind the seam): the 2D lip-opening test alone also caught the
  // NAPE / back of the neck (same height, any depth) -> a dark ellipse + a red cavity glow on her neck from behind
  F.mouthIn = (1.0 - lit)*inOpen*smoothstep(0.0005, 0.003, behindM)*smoothstep(0.075, 0.05, behindM);
  return F;
}

vec4 matShade(SIn S, inout vec4 fragDOut){   // inout (an out param is undefined until written: NVIDIA gave 0 -> mesh depth layer 0 -> no SSS, no depth)
  // render: skin never shows INSIDE an eyeball (a fitted head can leave socket/lid-interior faces between the
  // iris and the cornea: the pink wedge on v2) -- the eyeball is the eyes' own, by construction
  if(uEyeR > 0.0 && (distance(S.p, uEyeC[0]) < 0.985*uEyeR || distance(S.p, uEyeC[1]) < 0.985*uEyeR)) discard;
  vec3 L = uKeyDir, v = S.v, ng = S.ng;
  // render8: detail_k -- realistic map detail on toon proportions read as creases / mottling at card size
  if(uDetailK.x < 1.0) S.n = normalize(mix(ng, S.n, max(uDetailK.x, 0.0)));
  S.ao = mix(1.0, S.ao, uDetailK.y);
  if(uDetailK.z > 0.0 && uTexOn.x > 0.5){
    vec3 lo3 = srgb2lin(textureLod(tAlb, S.uv, 3.5).rgb);
    float la0 = dot(S.alb, vec3(0.2126, 0.7152, 0.0722)), lo0 = dot(lo3, vec3(0.2126, 0.7152, 0.0722));
    float edge = abs(la0 - lo0)/max(lo0, 1e-3);
    S.alb = mix(S.alb, lo3, uDetailK.z*smoothstep(0.30, 0.10, edge)*smoothstep(0.35, 0.6, lo0/max(uSssK.x, 0.05)));
  }
  float split = uSssSplit;
  Face F = faceRegions(S.p, ng);
  float fp = uFaceProc;
  // ---- normals: the diffuse sees the macro normal (no micro detail; skin light diffuses under the surface), pulled
  //      a little toward the geometric normal; the specular sees everything (micro breakup lives only there)
  vec3 nd = S.n;
  if(uTexOn.y > 0.5){
    vec3 tn = texture(tNrm, S.uv).xyz*2.0 - 1.0;
    tn.xy *= max(uDetailK.x, 0.0);
    nd = normalize(S.T*tn.x + S.B*tn.y + ng*max(tn.z, 0.05));
  }
  nd = normalize(mix(nd, ng, 0.25));
  vec3 ns = normalize(mix(S.n, ng, 0.45));
  // render4: micro-detail -> the specular normals + roughness (pores break the sheen into a fine grain);
  //   density by region: full on nose / cheeks / forehead / chin, low on lips and lids
  float microH = 0.0, microV = 0.0;
  if(uSkinK2.x > 0.0){
    vec4 mi = skinMicro(S.uv, uSkinK2.y);
    float dens = (1.0 - 0.75*F.lip)*(1.0 - 0.6*smoothstep(0.0, 0.5, F.thin));
    float a = uSkinK2.x*dens/max(uSkinK2.y, 1.0);
    vec3 g = S.T*mi.y + S.B*mi.z;
    ns = normalize(ns - g*a*0.6);
    microH = mi.x*dens; microV = mi.w*dens;
  }
  float ao = S.ao*mix(1.0, F.ao, max(fp, uFaceAO)*uDetailK.w);      // render2: creases read even with baked maps
  float thick = max(S.thick, F.thin*max(fp, uWarmK.y));   // render8: warm_k.y
  // ---- albedo zones (cheek blush, faint violet eyelid / under-eye, rose lips); never more than a tint
  vec3 alb = S.alb;
  float hairShade = 1.0;
  // render look-dev: freckles / blush in the albedo read at card size -- local contrast against the map's own
  // low-pass (mip ~4), a texture-space detail lift (no screen-space crawl)
  if(uTexOn.x > 0.5){
    vec3 lo = srgb2lin(textureLod(tAlb, S.uv, 4.5).rgb);
    // render3: the lift is on LUMINANCE only (per-channel x1.6 turned every freckle / blotch into a saturated
    // orange-red dot: judge r1 "blotchy red patches")
    float la = dot(alb, vec3(0.2126, 0.7152, 0.0722)), ll = dot(lo, vec3(0.2126, 0.7152, 0.0722));
    alb = clamp(alb*pow(max(la, 1e-4)/max(ll, 1e-4), 0.35*(1.0 - clamp(uSkinK4.z, 0.0, 1.0))), 0.0, 1.0);   // render5: skin_k4.z eases the lift (judge: speckle noise)
    // render4 (sss_k.y): hair root shade -- skin next to the painted hairline / brows (the low-pass albedo dips
    // there) sits in the strands' soft contact shadow: the hairline is not a cut edge on a lit forehead
    if(uSssK.y > 0.0 && uSssK.x > 0.0){
      float lw = dot(srgb2lin(textureLod(tAlb, S.uv, 6.0).rgb), vec3(0.2126, 0.7152, 0.0722));
      float near = smoothstep(0.92, 0.55, min(ll, lw)/uSssK.x)*smoothstep(0.5, 0.8, la/uSssK.x);
      hairShade = 1.0 - uSssK.y*near;
    }
  }
  // render2: the upper lash line -- the lid margin hugging the eyeball's top reads as a soft dark liner (the
  // reference's strongest eye-appeal cue; lash cards alone are a few px at card size); the lower margin a fainter one
  if(uEyeR > 0.0 && uFaceLiner > 0.0){
    int ke = distance(S.p, uEyeC[0]) < distance(S.p, uEyeC[1]) ? 0 : 1;
    vec3 de = transpose(uHeadR)*(S.p - uEyeC[ke]);
    float dr = length(de)/uEyeR;
    float up = de.y/uEyeR;
    float fr = smoothstep(0.15, 0.55, de.z/uEyeR);                 // front of the eye only
    // render3: liner > 1 widens the upper band (a soft lash-mass shadow, the reference's thick dark lid line)
    float wl = 1.28 + 0.12*max(uFaceLiner - 1.0, 0.0);
    float rim = smoothstep(wl, 1.04, dr)*fr;
    float top = smoothstep(0.05, 0.40, up), bot = smoothstep(-0.10, -0.45, up);
    alb *= 1.0 - clamp(min(uFaceLiner, 1.0)*rim*(0.88*top + 0.22*bot), 0.0, 0.92);
  }
  // render5 (skin_k3.x, judge r4 "blotchy red-pink flush"): texels redder than the base skin (g/r, b/r below
  // the base ratios) are pulled toward the base hue, keeping most of their darkening (a flush = warmer and a touch
  // deeper, never pink); x .. 1 = all the way.  (z) tan depth: mid-tones a little deeper and warmer (less blue).
  if(uSkinK3.x > 0.0 && uMatP.z > 0.0){
    float l0 = dot(alb, vec3(0.2126, 0.7152, 0.0722));
    float rr0 = max(alb.r, 1e-4);
    vec2 q0 = vec2(alb.g, alb.b)/rr0, qb = vec2(0.47, 0.33);
    vec2 q1 = mix(q0, max(q0, qb*vec2(0.97, 0.94)), uSkinK3.x*(1.0 - smoothstep(0.02, 0.0, l0)));
    vec3 a1 = vec3(1.0, q1)*rr0;
    float l1 = dot(a1, vec3(0.2126, 0.7152, 0.0722));
    alb = a1*mix(1.0, l0/max(l1, 1e-4), 0.75);
  }
  if(uSkinK3.z > 0.0 && uMatP.z > 0.0) alb *= mix(vec3(1.0), vec3(0.98, 0.90, 0.80), uSkinK3.z);
  // render8 (warm_k): the feature-animation colour zones (warm cheeks / nose / ears, a cooler brow and jaw)
  if(uWarmK.x > 0.0) alb *= mix(vec3(1.0), vec3(1.12, 0.90, 0.84), clamp(F.blush, 0.0, 1.0)*uWarmK.x);
  if(uWarmK.z > 0.0){
    float ru = r_up(S.p);
    alb *= mix(vec3(1.0), vec3(0.96, 0.99, 1.03), (smoothstep(0.3, 0.9, ru) + smoothstep(-1.2, -1.6, ru))*uWarmK.z);
  }
  if(uWarmK.w > 0.0) alb *= mix(vec3(1.0), vec3(1.02, 0.86, 0.86), F.lip*uWarmK.w);
  float tw = uFaceTint*fp;
  alb *= mix(vec3(1.0), vec3(1.08, 0.84, 0.80), clamp(F.blush, 0.0, 1.0)*0.85*tw);
  alb *= mix(vec3(1.0), vec3(0.97, 0.99, 1.02), smoothstep(0.3, 0.9, r_up(S.p))*tw*0.6);   // forehead / temples cooler, yellower
  alb *= mix(vec3(1.0), vec3(0.93, 0.86, 0.97), clamp(F.violet, 0.0, 1.0)*0.55*tw);
  alb *= mix(vec3(1.0), vec3(0.88, 0.66, 0.70), F.lip*tw);
  // ---- inside the mouth: light does not get there (graded dark, warm)
  float mIn = F.mouthIn;                 // (the cavity grade is geometry, not a map fallback: always on)
  float cav = mix(1.0, 0.22, mIn);             // (look-dev: the cavity reads warm and lit near the lips)
  // ---- key diffuse: per-channel wrap (pre-integrated look) -- full in the fallback, a residual under the blur
  float ndl = dot(nd, L);
  vec3 w = vec3(0.34, 0.14, 0.08)*uMatP.z*uSssWrap*mix(1.0, 0.55, split);
  vec3 dif = clamp((vec3(ndl) + w)/(1.0 + w), 0.0, 1.0);
  dif = mix(dif, dif*dif*(3.0 - 2.0*dif), 0.5);
  // coloured penumbra: red sees the wide shadow, blue the sharp one
  vec3 sh = mix(vec3(S.shw.x), vec3(S.shw.y), mix(vec3(0.70, 0.28, 0.10), vec3(0.45, 0.15, 0.05), split)*uSkinK.y);
  vec3 Ekey = S.keyC*dif*sh*hairShade;
  // the terminator band itself picks up a red-orange glow (light scattered around the corner), never grey
  float term = smoothstep(-0.25, 0.05, ndl)*smoothstep(0.35, 0.05, ndl);
  Ekey += S.keyC*vec3(0.10, 0.025, 0.008)*term*mix(S.shw.y, 1.0, 0.3)*uMatP.z*uSkinK.x;
  // ---- fill (cool, large) + ambient with warm soft AO: creases go warm-dark, never black
  vec3 aoC = mix(vec3(0.66, 0.46, 0.40), vec3(1.0), ao);
  vec3 Efill = uFillCol*clamp((dot(nd, uFillDir) + 0.35)/1.35, 0.0, 1.0)*mix(aoC, vec3(1.0), 0.3);
  vec3 Eamb = S.amb*aoC*mix(1.0, hairShade, 0.6);
  // render5 (skin_k3.y): warm bounce from below-front (the ochre cardigan / desk / floor under the key): lifts the
  // underside of the chin, nose and jaw in a warm tone -- the reference's warm under-chin / under-nose fill
  if(uSkinK3.y > 0.0){
    vec3 bd = normalize(vec3(0.0, -1.0, 0.0) + 0.9*S.v);
    float bw = clamp((dot(nd, bd) + 0.2)/1.2, 0.0, 1.0);
    Eamb += S.keyC*vec3(1.0, 0.62, 0.36)*0.10*uSkinK3.y*bw*bw*mix(aoC, vec3(1.0), 0.4);
  }
  Efill *= mix(1.0, hairShade, 0.6);
  // ---- transmission: thin parts glow with the light behind them
  //  key: thickness from the shadow map (fingers, ears, nostril wings in silhouette against the key)
  vec3 tr = vec3(0.0);
  // render2: start past the terminator (at it the receiver is its own occluder in the raw-depth shadow map:
  // thickness ~0 -> a jagged bright band of fake transmission along the facets on the shadow side of the cheek)
  float back = smoothstep(0.25, 0.7, -dot(ng, L));
  if(back > 0.0 && uTransKey > 0.0){
    float th = skinKeyThickness(S.p, ng);
    vec3 Tk = exp(-th/(2.0*uSssRadSu + 1e-6));
    tr += S.keyC*Tk*back*uTransKey*mix(0.3, 1.0, thick);
  }
  //  rim + room (behind; no shadow map): the thickness map, a view-aligned lobe (Barre-Brisebois)
  // render3: only really thin parts (ears, nostril wings, lids) transmit the rim/room light; 0.35 let the
  // jaw/neck behind the ear glow red-orange from behind (ASSET2 rt_y140, ld3/y_strip3.png)
  float thin = smoothstep(0.6, 0.95, thick);
  if(thin > 0.0 && uTransRim > 0.0){
    vec3 Ld = normalize(uRimDir2 + ng*0.35);
    float vt = pow(clamp(dot(v, -Ld), 0.0, 1.0), 3.0)*1.6 + 0.35*clamp(-dot(ng, uRimDir2), 0.0, 1.0);
    vec3 Tr = exp(-(1.25 - thin)*1.2*uSssRadSu.r/(uSssRadSu + 1e-6));
    tr += uRimCol2*uRimGain*vt*thin*Tr*uTransRim;
    // thin parts also glow a little with the room light coming through (ears read warm red even front-lit)
    tr += S.amb*Tr*thin*0.35*uTransRim;
  }
  // render3 (skin_k.w, lead r3 tell #1): the shadow side darkens and DESATURATES in the lit side's hue instead of
  // going salmon-red -- the room/fill light reaching her shadow side is taken at its level (neutral), and the albedo
  // itself loses a little chroma where the key does not reach
  if(uSkinK.w > 0.0){
    float kv = clamp(dot(Ekey, vec3(0.3, 0.59, 0.11))/max(dot(Ekey + Efill + Eamb, vec3(0.3, 0.59, 0.11)), 1e-4), 0.0, 1.0);
    float sw = uSkinK.w*(1.0 - kv);
    Efill = mix(Efill, vec3(dot(Efill, vec3(0.3, 0.59, 0.11))), sw);
    Eamb = mix(Eamb, vec3(dot(Eamb, vec3(0.3, 0.59, 0.11))), sw);
    float la2 = dot(alb, vec3(0.3, 0.59, 0.11));
    alb = mix(alb, vec3(la2), 0.35*sw)*mix(1.0, 0.85, sw);
  }
  // render4 (sss_k.z): skin hue -- the shadow side went pink-red (hue ~8 deg vs the reference's ~12-17) and the
  // lit side under-saturated (sat .35 vs .5): the unlit share turns toward peach, the lit share gains a little chroma
  if(uSssK.z > 0.0){
    float kv2 = clamp(dot(Ekey, vec3(0.3, 0.59, 0.11))/max(dot(Ekey + Efill + Eamb, vec3(0.3, 0.59, 0.11)), 1e-4), 0.0, 1.0);
    vec3 E = Efill + Eamb;
    E *= mix(vec3(1.0), vec3(0.96, 1.06, 0.92), uSssK.z*(1.0 - kv2));
    Efill = E*(dot(Efill, vec3(0.333))/max(dot(Efill + Eamb, vec3(0.333)), 1e-5));
    Eamb = E - Efill;
    float la3 = dot(alb, vec3(0.3, 0.59, 0.11));
    alb = max(mix(vec3(la3), alb, 1.0 + 0.3*uSssK.z*kv2), 0.0);
  }
  // render5 (skin_k4): the reference's value structure at bust distance -- (x) the neck under the jaw sits in the
  // head's shadow (key and fill; a broad soft cast, not a line), (y) the face turns darker toward its silhouette (a
  // ball, not a lit mask)
  if(uSkinK4.x > 0.0 && uHeadS > 0.0){
    vec3 qn = transpose(uHeadR)*(S.p - uMouthC)/uHeadS;      // metres, head frame (y up, z out of the face)
    float below = smoothstep(-0.045, -0.085, qn.y)*smoothstep(-0.20, -0.12, qn.y);
    float behind = smoothstep(0.03, -0.01, qn.z);
    float nk = 1.0 - uSkinK4.x*below*behind;
    Ekey *= nk; Efill *= mix(1.0, nk, 0.6);
  }
  if(uSkinK4.y > 0.0){
    float fo = mix(1.0, mix(0.3, 1.0, smoothstep(0.0, 0.8, S.nv)), uSkinK4.y);   // floor .3: no dark outline at the jaw
    Ekey *= fo; Efill *= fo; Eamb *= mix(1.0, fo, 0.7);
  }
  vec3 sssC = uMatC;                            // the asset's sss colour tints what came through
  vec3 diffuse = (alb*(Ekey + Efill + Eamb) + alb*sssC*tr*0.9)*cav;
  // render3: the cavity glow from the room's LEVEL, not its colour (her room light is white-balanced cool: the
  // inner lips read navy-black in the open shots) -- warm red-brown soft tissue, a little lifted
  float ambL = dot(S.amb, vec3(0.30, 0.59, 0.11));
  diffuse += alb*vec3(1.0, 0.30, 0.22)*ambL*mIn*mix(1.0 - mIn, 0.6, 0.5)*3.0;
  // ---- specular: two GGX lobes, F0 0.028 * level; a broad soft sheen + a faint tighter lobe, no oily hot spots
  float rb = clamp(S.rough*1.25, 0.55, 0.9);    // (lips too: no plastic lip gloss)   // matte-soft base (lead 00:40: no wax)
  float rt = clamp(S.rough*0.6, 0.30, 0.42);                       // small crisp highlight
  rt = clamp(rt*(1.0 + 0.9*microH*uSkinK2.x), 0.22, 0.6);          // render4: pore grain in the crisp lobe
  float f0 = 0.028*min(S.specL, 1.0)*uSkinSpec;
  vec3 H = normalize(v + L);
  float vh = clamp(dot(v, H), 0.0, 1.0);
  float Fk = f0 + (1.0 - f0)*pow(1.0 - vh, 5.0);
  // render3: the tight lobe sees the FULL micro normal (pores / map detail break the small highlight up -- no
  // plastic hot spot), the broad lobe the smoothed one; mesh.face.skin_k.z moves energy toward the tight lobe
  float tightW = clamp(0.15 + 0.5*uSkinK.z, 0.0, 0.6);
  // render4: region spec -- wet tight lobes where the reference glints (lower lip centre, nose tip, inner eye corner)
  float wet = 0.0;
  if(uSkinK2.z > 0.0 && uHeadS > 0.0){
    vec3 em = 0.5*(uEyeC[0] + uEyeC[1]);
    mat3 Ri = transpose(uHeadR);
    float D = max(-(Ri*(uMouthC - em)).y/uHeadS, 1e-3);
    vec3 q = Ri*(S.p - em)/uHeadS/D;
    vec3 lq = Ri*(uMouthL - em)/uHeadS/D;
    float ex = abs((Ri*(uEyeC[0] - em)/uHeadS/D).x);
    vec2 a2 = vec2(abs(q.x), q.y);
    wet = max(wet, gauss2(a2 - vec2(0.0, lq.y - 0.05), vec2(0.20, 0.06))*F.lip);          // lower lip centre
    wet = max(wet, 0.8*gauss2(a2 - vec2(0.0, -0.52), vec2(0.07, 0.07)));                 // nose tip
    wet = max(wet, 0.6*gauss2(a2 - vec2(ex*0.55, -0.02), vec2(0.05, 0.05)));            // inner eye corner
    wet *= uSkinK2.z;
  }
  float sp = (1.0 - tightW)*specGGX(ns, v, L, rb) + tightW*specGGX(S.n, v, L, rt) + wet*0.6*specGGX(normalize(mix(S.n, ns, 0.5)), v, L, 0.2);
  // broad soft sheen (the reference's lit-side satin: forehead / cheek tops / nose bridge), broken by the pores
  sp += uSkinK2.w*specGGX(ns, v, L, 0.75)*(1.0 + 0.6*microH*uSkinK2.x);
  // render5 (skin_k3.w): a soft BROAD sheen that survives the card scale (forehead / cheek tops / nose bridge):
  // a very wide lobe on the smoothed normal, key-lit only, low F0 -- a satin, not oil
  sp += uSkinK3.w*0.5*specGGX(normalize(mix(nd, ns, 0.5)), v, L, 0.62)*S.shw.y;
  float specOcc = clamp(pow(S.nv + ao, exp2(-16.0*rb - 1.0)) - 1.0 + ao, 0.0, 1.0)*ao;
  vec3 spec = S.keyC*Fk*sp*S.shw.x*PI*specOcc;
  // fill: broad lobe only, weak
  spec += uFillCol*f0*specGGX(ns, v, uFillDir, rb)*PI*specOcc*0.5;
  // the room: diffuse-ish (rough skin sees the blurred room), split-sum weights, grazing Fresnel tamed
  vec3 r = reflect(-v, ns);
  vec3 env = setAmbient(S.p, r);
  vec2 ab = skinEnvBRDF(rb, S.nv);
  spec += env*uAmbGain*(f0*ab.x + ab.y*0.25)*specOcc*uSkinSpec;
  // render3: grazing sheen -- the fine vellus / oil film lifts the silhouette-facing skin a little toward the
  // room's colour (Fresnel at grazing, rough; never on the shadow side's core: x the key's visibility, floor 0.35)
  spec += env*uAmbGain*uSkinK.z*0.22*pow(1.0 - S.nv, 4.0)*specOcc*mix(0.35, 1.0, S.shw.y);
  spec *= cav*cav;
  // ---- rim (sharp edge line, not blurred)
  vec3 rim = rimLight(S)*(1.0 - mIn)*(1.0 - 0.8*F.lip*fp);   // no hot rim line on the inner lip edge
  // render7: inner creases that face away from the camera (nostril wing outline, lip corners) took the edge rim
  // as orange specks / a drawn line; a rim light behind the head reaches only its OUTER silhouette (radial from the
  // skull centre ~ perpendicular to the view): gate it there.
  if(uRimK.x > 0.0 && uHeadS > 0.0){
    vec3 hc = 0.5*(uEyeC[0] + uEyeC[1]) - uHeadR[2]*(0.085*uHeadS);
    vec3 hd = S.p - hc;
    float hl = length(hd);
    if(hl < 0.17*uHeadS) rim *= mix(1.0, smoothstep(uRimK.y, uRimK.z, abs(dot(hd/max(hl, 1e-6), S.v))), uRimK.x);
  }
  rim *= smoothstep(0.5, 0.12, S.nv);   // render3: a rim is an EDGE on skin (a wide pow-3 lobe glowed orange over the whole cheek seen from behind)
  vec3 col;
  // render4: the SSS blur scatters RADIANCE (albedo applied before the blur): a dark painted brow / hairline in the
  // albedo got its red channel filled in from the bright skin around it -> an orange-red fringe along every hair edge
  // (hairline, under the brows; gone with JUNO_SSS_TAPS=0: ld4/Hs2.png). Painted-hair texels are not skin for the
  // scatter: shaded directly, and they block the blur (no dark pull on the skin next to them either)
  float paint = uSssK.x > 0.0 ? smoothstep(0.55, 0.25, dot(S.alb, vec3(0.2126, 0.7152, 0.0722))/uSssK.x) : 0.0;
  if(split > 0.5){
    gSss = vec4(diffuse*S.alpha*(1.0 - paint), 1.0 - paint);   // a partial skin "coverage" for the blur
    col = spec + rim + diffuse*paint;
  } else {
    col = diffuse + spec + rim;
  }
  return vec4(col*S.alpha, S.alpha);
}

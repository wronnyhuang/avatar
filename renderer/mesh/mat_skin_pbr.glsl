// mat_skin_pbr.glsl — shade: physically based skin for a realistic character (pbr.py; gated by mesh.pbr.skin).
//  Spliced into mesh.glsl FS_FWD instead of mat_skin.glsl.  No stylised colour zones / liners / flush: the asset's maps
//  carry the look, the shader carries the physics.
//   diffuse   Lambert on the macro normal (asset normal map, softened), x (1 - specular albedo) = energy conserving;
//             multi-bounce coloured AO; key + fill + room (probe) + shadow-map thickness transmission (Burley
//             transmittance) + thin-part back light.  -> the S attachment as IRRADIANCE (PBR_ALB_SPLIT: the SSS blur
//             scatters light, the albedo is applied after the blur at full resolution) or as radiance.
//   specular  two GGX lobes (x0.75 / x1.3 roughness, mixed by lobe2) on the micro normal: the asset's normal map +
//             a tiling pore micro-normal (tMicro, UDN blend, faded by its own mip = no shimmer) with Toksvig roughness
//             from both maps' filtered length (specular anti-aliasing); key as an AREA light (roughness widened by its
//             angular size), fill, the room probe at the material roughness (split-sum, Karis analytic env BRDF),
//             specular occlusion (Lagarde) + micro cavity; peach-fuzz sheen (Charlie) at grazing; a silhouette rim.
//   regions   when the asset's packed map has no roughness (a constant): an oilier T-zone, smoother lips, rougher
//             cheeks from the face landmarks (eyesmouth.py), and the mouth interior darkening behind the lip seam.
//  Packed map: R AO, G roughness, B spec level (0.5 = F0 0.028), A thickness (1 = thin: ears, nostril wings, lids).
uniform float uSssSplit;
uniform float uSssWrap;
uniform float uTransKey, uTransRim;
uniform vec3  uSssRadSu;
uniform mat3  uHeadR;
uniform float uHeadS;
uniform vec3  uEyeC[2];
uniform float uEyeR;
uniform vec3  uMouthC, uMouthU, uMouthL;
uniform float uMouthW;
uniform vec4  uPbrSkin;    // x micro-normal strength, y micro tile (per uv unit), z spec gain (x F0), w roughness scale
uniform vec4  uPbrSkin2;   // x wide-lobe weight, y peach-fuzz strength, z micro cavity, w transmission gain
uniform vec4  uPbrSkin3;   // x diffuse normal-map share, y region roughness weight (0 when the asset has a map), z key size, w room spec gain
uniform vec4  uPbrSkin4;   // x AO strength, y rim gain, z fill spec gain, w wrinkle blend
uniform float uPbrMouthIn;
uniform float uPbrToks;    // Toksvig variance gain (pbr.skin.toksvig)  // the inner-mouth skin's light floor (pbr.mouth.inner)
uniform sampler2D tMicro;  // tiling micro relief: rg normal xy, b normal z, a cavity
uniform sampler2D tWrinkleN, tWrinkleM;   // the asset's wrinkle normal map + RGBA region mask (pbr.py)
uniform vec4  uWrinkleW;   // per mask channel: how far its driving shapes are on (0..1)
uniform float uWrinkleOn;
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}

float psGauss(vec2 d, vec2 r){ vec2 q = d/r; return exp(-dot(q, q)); }

// face regions for the fallbacks: x T-zone (forehead / nose), y lips, z ears+nose wings (thin), w mouth interior
vec4 psRegions(vec3 p){
  if(uHeadS <= 0.0) return vec4(0.0);
  vec3 em = 0.5*(uEyeC[0] + uEyeC[1]);
  mat3 Ri = transpose(uHeadR);
  vec3 q = Ri*(p - em)/uHeadS;
  vec3 mq = Ri*(uMouthC - em)/uHeadS;
  float D = max(-mq.y, 1e-3);
  vec3 r = q/D;
  if(dot(r, r) > 9.0) return vec4(0.0);
  vec3 e0 = Ri*(uEyeC[0] - em)/uHeadS/D;
  vec2 a = vec2(abs(r.x), r.y);
  float front = smoothstep(-0.9, -0.2, r.z - e0.z);
  vec3 uq0 = Ri*(uMouthU - em)/uHeadS/D, lq0 = Ri*(uMouthL - em)/uHeadS/D;
  vec3 uq = uq0.y >= lq0.y ? uq0 : lq0, lq = uq0.y >= lq0.y ? lq0 : uq0;   // seam points by height
  float mw = 0.5*uMouthW/uHeadS/D;
  // T-zone: forehead band above the brows + the nose bridge / tip
  float tz = smoothstep(0.25, 0.55, r.y)*smoothstep(1.0, 0.6, a.x)*front;
  tz = max(tz, psGauss(a - vec2(0.0, -0.35), vec2(0.13, 0.32))*smoothstep(-0.1, 0.2, r.z - e0.z));
  // lips
  float lx = clamp(a.x/max(mw*0.95, 1e-3), 0.0, 1.0);
  float shape = sqrt(max(1.0 - lx*lx, 0.0));
  float top = uq.y + 0.07*shape, bot = lq.y - 0.085*shape;
  float lip = smoothstep(0.03, 0.0, max(r.y - top, bot - r.y))*smoothstep(1.0, 0.8, lx)*front;
  // thin: ears (lateral, behind the face plane), nose wings
  float dz = r.z - e0.z;
  float ear = smoothstep(1.08, 1.25, abs(r.x))*smoothstep(-0.95, -0.75, r.y)*smoothstep(0.45, 0.2, r.y)
            *smoothstep(-0.35, -0.6, dz)*smoothstep(-1.9, -1.6, dz);
  float nose = psGauss(a - vec2(0.12, -0.60), vec2(0.09, 0.08))*smoothstep(-0.1, 0.25, dz);
  // the mouth interior seen through the lip opening (mat_skin.glsl faceRegions: only the mouth's own depth)
  float behindM = (mq.z/D - r.z)*D;
  float gap = length(uq - lq)*D;
  float ey = (r.y - 0.5*(uq.y + lq.y))/max(0.5*(uq.y - lq.y) + 0.04, 0.02);
  float inOpen = smoothstep(1.15, 0.85, length(vec2(a.x/max(mw, 1e-3), ey)));
  float lit = exp(-max(behindM - 0.001, 0.0)/(0.004 + 0.9*gap));
  float mIn = (1.0 - lit)*inOpen*smoothstep(0.0005, 0.003, behindM)*smoothstep(0.075, 0.05, behindM);
  return vec4(tz, lip, max(ear, 0.75*nose), mIn);
}

// key-light thickness through the mesh from the shadow map (orthographic, linear depth; mat_skin.glsl's estimator)
float psKeyThickness(vec3 p, vec3 ng){
  vec3 po = p - ng*uShadowTexel.y*0.75;
  vec4 c = uLightVP*vec4(po, 1.0);
  vec3 s = c.xyz/c.w*0.5 + 0.5;
  if(any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0)))) return 1e3;
  float r = uShadowTexel.x*2.0;
  float d = texture(uShadow, s.xy).r + texture(uShadow, s.xy + vec2(r, 0.0)).r + texture(uShadow, s.xy - vec2(r, 0.0)).r
          + texture(uShadow, s.xy + vec2(0.0, r)).r + texture(uShadow, s.xy - vec2(0.0, r)).r;
  return max(s.z - 0.2*d, 0.0)*2.0*uShadowTexel.y/uShadowTexel.x;
}
// Burley normalised diffusion: transmittance through thickness d for scatter distance s (per channel)
vec3 psBurleyT(float d, vec3 s){ vec3 x = d/max(s, vec3(1e-6)); return 0.25*(exp(-x) + 3.0*exp(-x/3.0)); }

vec2 psEnvBRDF(float r, float nv){
  const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
  const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
  vec4 r4 = r*c0 + c1;
  float a004 = min(r4.x*r4.x, exp2(-9.28*nv))*r4.x + r4.y;
  return vec2(-1.04, 1.04)*a004 + r4.zw;
}
// GGX with a separate alpha for D (area-widened) and visibility
float psGGX(vec3 n, vec3 v, vec3 l, float aD, float aV){
  vec3 h = normalize(v + l);
  float nl = clamp(dot(n, l), 0.0, 1.0), nv = clamp(dot(n, v), 1e-4, 1.0), nh = clamp(dot(n, h), 0.0, 1.0);
  return D_GGX(nh, max(aD, 0.002))*V_SmithJ(nv, nl, max(aV, 0.002))*nl;
}
// Charlie sheen + Neubelt visibility (peach fuzz)
float psSheen(vec3 n, vec3 v, vec3 l, float r){
  vec3 h = normalize(v + l);
  float nl = clamp(dot(n, l), 0.0, 1.0), nv = clamp(dot(n, v), 1e-4, 1.0), nh = clamp(dot(n, h), 0.0, 1.0);
  return D_Charlie(nh, r)*nl/(4.0*(nl + nv - nl*nv) + 1e-4);
}
// multi-bounce AO (Jimenez 2016): coloured, the albedo lifts what bounces inside a crease
vec3 psAOMulti(float ao, vec3 alb){
  vec3 a = 2.0404*alb - 0.3324, b = -4.7951*alb + 0.6417, c = 2.7552*alb + 0.6903;
  return max(vec3(ao), ((ao*a + b)*ao + c)*ao);
}

vec4 matShade(SIn S, inout vec4 fragDOut){
  if(uEyeR > 0.0 && (distance(S.p, uEyeC[0]) < 0.985*uEyeR || distance(S.p, uEyeC[1]) < 0.985*uEyeR)) discard;
  vec3 L = uKeyDir, v = S.v, ng = S.ng;
  vec4 R = psRegions(S.p);
  S.shw = pbrShadow(S.p, S.ng, S.shw);                                   // + the head cascade
#ifdef PBR_DBG_SHADOW
  { vec3 f = faceShadowW(S.p, S.ng); return vec4(vec3(f.x, S.shw.x, f.z)/max(uCharK, 1e-3), 1.0); }
#endif
  // ---------------- normals
  vec3 tn = vec3(0.0, 0.0, 1.0);
  float lenN = 1.0;
  if(uTexOn.y > 0.5){
    vec3 t0 = texture(tNrm, S.uv).xyz*2.0 - 1.0;
    if(uWrinkleOn > 0.5 && uPbrSkin4.w > 0.0){                          // wrinkles: the compressed-state normal by region
      float ww = clamp(dot(texture(tWrinkleM, S.uv), uWrinkleW)*uPbrSkin4.w, 0.0, 1.0);
      if(ww > 0.0){                                                     // UDN-add the crease relief over the base map
        vec3 wn = texture(tWrinkleN, S.uv).xyz*2.0 - 1.0;
        t0 = vec3(t0.xy + wn.xy/max(wn.z, 0.2)*ww, t0.z);
      }
    }
    lenN = clamp(length(t0), 0.05, 1.0);
    tn = t0/lenN;
  }
  // pore micro-normal: tiling relief in uv, density lower on lips (smooth) and the thin lids / ears
  float dens = (1.0 - 0.4*R.y)*(1.0 - 0.4*R.z);
  vec2 uvT = S.uv*uPbrSkin.y;
  float fpT = max(length(dFdx(uvT)), length(dFdy(uvT)));                   // tile units per pixel
  dens *= smoothstep(1.0/6.0, 1.0/20.0, fpT);                             // pore grain where it can still read (Toksvig keeps it calm)
  vec4 mt = texture(tMicro, uvT);
  vec3 mn = mt.xyz*2.0 - 1.0;
  float lenM = clamp(length(mn), 0.05, 1.0);
  float mk = uPbrSkin.x*dens;
  vec3 ts = normalize(vec3(tn.xy + mn.xy/max(mn.z, 0.2)*mk, tn.z));      // UDN blend
  vec3 nS = normalize(S.T*ts.x + S.B*ts.y + ng*max(ts.z, 0.05));
  if(dot(nS, v) < 0.0) nS = normalize(nS - v*dot(nS, v)*1.01);            // never a back-facing micro normal
  vec3 nM = normalize(S.T*tn.x + S.B*tn.y + ng*max(tn.z, 0.05));          // macro (map only)
  vec3 nD = normalize(mix(ng, nM, uPbrSkin3.x));                          // diffuse: softened macro (light diffuses)
  // ---------------- roughness: map or regions, Toksvig from both maps' filtered normal length
  float rough = S.rough;
  float rReg = mix(0.52, 0.40, R.x);            // cheeks / forehead base vs the oilier T-zone
  rReg = mix(rReg, 0.33, R.y);                   // lips: smoother, a little wet
  rough = mix(rough, rReg, uPbrSkin3.y);
  rough = mix(rough, min(rough, 0.30), 0.8*R.y);                            // lips: a moist, tighter sheen even over a map
  // the wet line: the inner (mucosa) part of the lips next to the seam -- a tight wet lobe
  float wetL = 0.0;
  if(uHeadS > 0.0){
    mat3 Rw = transpose(uHeadR);
    vec3 qw = Rw*(S.p - uMouthC)/uHeadS;
    vec3 uw = Rw*(uMouthU - uMouthC)/uHeadS, lw0 = Rw*(uMouthL - uMouthC)/uHeadS;
    float gapw = abs(uw.y - lw0.y);
    float dy = abs(qw.y) - 0.5*gapw;                                       // metres from the seam edge, outward
    wetL = R.y*smoothstep(0.0045, 0.0005, dy)*smoothstep(-0.002, 0.0, dy)*smoothstep(0.75*0.5*uMouthW/uHeadS, 0.2*0.5*uMouthW/uHeadS, abs(qw.x));
    rough = mix(rough, 0.16, wetL);
  }
  rough = clamp(rough*uPbrSkin.w, 0.08, 1.0);
  float var = (1.0 - lenN)/lenN + mk*mk*(1.0 - lenM)/lenM;
  float a0 = rough*rough;
  float aT = sqrt(min(a0*a0 + uPbrToks*min(var, 0.25), 1.0));             // Toksvig (alpha^2 += k variance; a scan map's mip length is noisy -> capped)
  // ---------------- occlusion
  float aoMap = mix(1.0, S.ao, uPbrSkin4.x);
  float cavM = mix(1.0, mt.a, clamp(uPbrSkin2.z*mk, 0.0, 1.0));           // pore cavity (faded with its mip)
  float ao = aoMap;
  vec3 alb = S.alb;
  vec3 aoD = psAOMulti(ao, alb);
  // ---------------- specular
  float f0 = 0.028*min(S.specL, 2.0)*uPbrSkin.z;
  float nv = clamp(dot(nS, v), 1e-4, 1.0);
  vec2 ab = psEnvBRDF(sqrt(aT), nv);
  float specAlb = f0*ab.x + ab.y;                                         // directional albedo of the spec layer
  float specOcc = clamp(pow(nv + ao, exp2(-16.0*aT - 1.0)) - 1.0 + ao, 0.0, 1.0)*cavM;
  float aKey = min(aT + 0.5*uLightP.x*uPbrSkin3.z, 1.0);                  // area key: lobe widened by its size
  float lw = uPbrSkin2.x;
  vec3 H = normalize(v + L);
  float Fk = f0 + (1.0 - f0)*pow(1.0 - clamp(dot(v, H), 0.0, 1.0), 5.0);
  float a1 = aT*0.75, a2 = min(aT*1.3, 1.0);
  float k1 = min(a1 + 0.5*uLightP.x*uPbrSkin3.z, 1.0), k2 = min(a2 + 0.5*uLightP.x*uPbrSkin3.z, 1.0);
  float sp = (1.0 - lw)*psGGX(nS, v, L, k1, a1) + lw*psGGX(nS, v, L, k2, a2);
  vec3 spec = S.keyC*Fk*sp*S.shw.x*PI*specOcc;
  // fill: a big soft source -> rough lobe
  float af = min(aT + 0.35, 1.0);
  spec += uFillCol*uPbrSkin4.z*f0*psGGX(nS, v, uFillDir, af, aT)*PI*specOcc;
  // the room (probe at the lobe's roughness, else the set's analytic ambient)
  vec3 rr = reflect(-v, nS);
  float hor = clamp(1.0 + 1.2*dot(rr, ng), 0.0, 1.0); hor *= hor;          // horizon occlusion
  vec3 env = uProbeP.x > 0.5 ? envProbe(rr, sqrt(aT)) : setAmbient(S.p, rr);
  spec += env*uAmbGain*uAmbTint*specAlb*specOcc*hor*uPbrSkin3.w;
  // ---------------- diffuse irradiance (energy: what the spec layer reflects does not enter the skin)
  float kd = 1.0 - specAlb;
  float ndl = dot(nD, L);
  vec3 shK = mix(vec3(S.shw.x), vec3(S.shw.y), vec3(0.30, 0.10, 0.03));  // the red light diffuses round the shadow edge
  vec3 E = S.keyC*clamp(ndl, 0.0, 1.0)*shK;
  if(uSssSplit < 0.5){                                                    // no blur (cheapest tiers): pre-integrated wrap
    vec3 w = vec3(0.34, 0.14, 0.08)*uSssWrap;
    E = S.keyC*clamp((vec3(ndl) + w)/(1.0 + w), 0.0, 1.0)*shK;
  }
  E += uFillCol*clamp(dot(nD, uFillDir), 0.0, 1.0)*aoD;
  E += S.amb*aoD;
  // transmission: key through the thin parts (shadow-map thickness), the rim / room through ears + nose wings
  float back = smoothstep(0.2, 0.7, -dot(ng, L));
  float thin = max(S.thick, 0.6*R.z);
  if(back > 0.0 && uTransKey > 0.0){
    float th = psKeyThickness(S.p, ng);
    E += S.keyC*psBurleyT(th, uSssRadSu)*back*uTransKey*uPbrSkin2.w*mix(0.25, 1.0, thin);
  }
  if(thin > 0.0 && uTransRim > 0.0){
    float vt = pow(clamp(dot(v, -normalize(uRimDir2 + ng*0.3)), 0.0, 1.0), 3.0)*1.5 + 0.3*clamp(-dot(ng, uRimDir2), 0.0, 1.0);
    vec3 Tr = psBurleyT((1.15 - thin)*1.5*uSssRadSu.r, uSssRadSu);
    E += (uRimCol2*uRimGain*vt + S.amb*0.15)*Tr*thin*uTransRim*uPbrSkin2.w;
  }
  E *= kd;
  // the mouth interior: light falls off behind the lip seam (warm, never black)
  // skin seen through the open lips (the head mesh's inner mouth): the key's shadow map says "occluded" for all of
  // it; the light that reaches it is what comes through the opening + bounces off teeth / tongue (as mat_mouth_pbr):
  // a dim warm level that fades with depth, never the black of an empty hole
  float cav = mix(1.0, 0.25, R.w);
  E *= cav; spec *= cav*cav;
  E = max(E, (S.amb*1.2 + S.keyC*0.10)*vec3(1.0, 0.75, 0.7)*R.w*uPbrMouthIn);
  // ---------------- peach fuzz (vellus) + a silhouette rim
  vec3 fuzzC = mix(alb, vec3(1.0), 0.5)*0.10*uPbrSkin2.y*(1.0 - 0.7*R.y);
  vec3 fuzz = fuzzC*(S.keyC*psSheen(nD, v, L, 0.45)*S.shw.y*PI + S.amb*aoD*0.6*pow(1.0 - nv, 3.0));
  // rim: the lamp behind lights only the head's OUTER silhouette (radial from the skull centre ~ perpendicular to the view)
  vec3 rim = vec3(0.0);
  if(uPbrSkin4.y > 0.0){
    float edge = pow(1.0 - clamp(dot(ng, v), 0.0, 1.0), 3.0);
    float rb = smoothstep(-0.1, 0.6, dot(ng, uRimDir2));
    float gate = 1.0;
    if(uHeadS > 0.0){
      vec3 hc = 0.5*(uEyeC[0] + uEyeC[1]) - uHeadR[2]*(0.085*uHeadS);
      vec3 hd = S.p - hc; float hl = length(hd);
      if(hl < 0.17*uHeadS) gate = smoothstep(0.75, 0.45, abs(dot(hd/max(hl, 1e-6), v)));
    }
    float rs = psGGX(nS, v, uRimDir2, min(aT + 0.15, 1.0), aT);
    rim = uRimCol2*uRimGain*uPbrSkin4.y*gate*(edge*rb*alb*0.5*aoD + (f0 + (1.0 - f0)*pow(1.0 - nv, 5.0))*rs*PI*specOcc)*(1.0 - R.w);
  }
#ifdef PBR_DBG_MTYPE
  return vec4(vec3(0.35) + vec3(0.5, 0.3, 0.0)*R.w + vec3(0.0, 0.0, 0.6)*R.y, 1.0)/max(uCharK, 1e-3);
#endif
#ifdef PBR_DBG_REGIONS
  return vec4(vec3(R.w, 0.5, R.y)/max(uCharK, 1e-3), 1.0);
#endif
#ifdef PBR_DBG_SPEC
  return vec4(spec, 1.0);
#endif
#ifdef PBR_DBG_FUZZ
  return vec4(fuzz + rim, 1.0);
#endif
#ifdef PBR_DBG_ENV
  return vec4(env*uAmbGain*uAmbTint*specAlb*specOcc*hor*uPbrSkin3.w, 1.0);
#endif
  vec3 col;
  if(uSssSplit > 0.5){
#ifdef PBR_ALB_SPLIT
    gSss = vec4(E*S.alpha, 1.0);
    gAlb = alb;
#else
    gSss = vec4(alb*E*S.alpha, 1.0);
#endif
    col = spec + fuzz + rim;
  } else {
    col = alb*E + spec + fuzz + rim;
  }
  return vec4(col*S.alpha, S.alpha);
}

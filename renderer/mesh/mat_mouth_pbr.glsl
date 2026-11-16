// mat_mouth_pbr.glsl — shade: teeth, gums, tongue, mouth interior for a realistic character (pbr.py "mouth";
//  PBR_T_TEETH / PBR_T_GUMS / PBR_T_TONGUE / PBR_T_MOUTH_INTERIOR pick the tissue).  Spliced into mesh.glsl FS_FWD.
//   light in the mouth  the key's shadow map says "occluded" for nearly everything behind the lip seam, so the light
//                       that reaches a mouth point is modelled from the geometry instead: a cone through the lip
//                       opening (falls off with depth behind the seam relative to the lip gap), darker toward the
//                       cheeks (the lip corners shade it), the upper lip's soft shadow on the upper teeth; the room's
//                       light comes in the same way.  Never pure black: a dim warm floor (light scattered by tissue).
//   teeth               enamel: off-white, a little translucent (wrapped diffuse + a warm dentin tint where light goes
//                       through, cooler grey-blue translucency at the biting edge), rough dielectric spec (IOR 1.6),
//                       per-tooth contact AO (surfaces turned away from the front = the gaps between teeth, darker).
//   gums / tongue       wet mucosa: red SSS-ish wrap, a tight wet specular lobe on top of a rough one (tongue
//                       papillae break it up); caruncle (type gums, far from the mouth) = the same wet tissue.
uniform mat3  uHeadR;
uniform float uHeadS;
uniform vec3  uMouthC, uMouthU, uMouthL;
uniform float uMouthW;
uniform vec4  uPbrMouth;    // x teeth translucency, y cavity darkening strength, z wet spec gain, w teeth roughness
uniform vec4  uPbrMouth2;   // x floor (share of the room light that always reaches the cavity), yzw min tissue albedo (linear)
// shade2-mouth (all 0 = the shade r4 model above, exactly): the APERTURE model -- light reaches a mouth point only
// along rays that pass through the lip opening (an ellipse in the seam plane), tested analytically per light
uniform vec4  uPbrMouth3;   // x aperture model on (1), y tongue saturation, z lower-teeth level, w interproximal (teeth AO on direct light)
uniform vec4  uPbrMouth4;   // x incisal translucency, y gum-line / upper-lip shade on the upper teeth, z throat falloff (m), w bounce
uniform vec4  uPbrMouth6;   // x room gain through the opening (aperture area scale), y key bounce onto the floor / tongue, z extra key softness at the aperture edge, w teeth albedo level
uniform vec4 uPbrMouth7;   // shade2: x gum albedo level, y gum saturation (1,1 = as before)
uniform vec4  uPbrMouth5;   // x caruncle gain (0 = as before), y caruncle wet glint, z buccal (side) darkening, w tongue sheen
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}

float pmHash(vec3 p){ p = fract(p*0.3183099 + 0.1); p *= 17.0; return fract(p.x*p.y*p.z*(p.x + p.y + p.z)); }
float pmNoise(vec3 x){
  vec3 i = floor(x), f = fract(x); f = f*f*(3.0 - 2.0*f);
  return mix(mix(mix(pmHash(i), pmHash(i + vec3(1,0,0)), f.x), mix(pmHash(i + vec3(0,1,0)), pmHash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(pmHash(i + vec3(0,0,1)), pmHash(i + vec3(1,0,1)), f.x), mix(pmHash(i + vec3(0,1,1)), pmHash(i + vec3(1,1,1)), f.x), f.y), f.z);
}

// x light reaching here through the lips (0..1), y mouth weight (0 = not near the mouth: caruncle), z depth behind the
// seam (m), w how far toward the corners (0 centre .. 1)
vec4 pmLight(vec3 p){
  if(uHeadS <= 0.0 || uMouthW <= 0.0) return vec4(1.0, 0.0, 0.0, 0.0);
  mat3 Ri = transpose(uHeadR);
  vec3 q = Ri*(p - uMouthC)/uHeadS;
  float hw = 0.5*uMouthW/uHeadS;
  if(length(q) > 3.0*hw) return vec4(1.0, 0.0, 0.0, 0.0);
  vec3 u = Ri*(uMouthU - uMouthC)/uHeadS, l = Ri*(uMouthL - uMouthC)/uHeadS;
  float open = length(u - l);            // the lip gap (the seam points' order is not trusted: a stand-in rig swaps them)
  float uy = max(u.y, l.y);
  float behind = max(-q.z, 0.0);
  // the share of the (front, brightest) hemisphere a point at depth d behind the seam sees through the lip opening:
  // the opening's area over (area + the hemisphere's at that depth); a closed mouth -> 0 just behind the seam
  float area = 3.14159*hw*0.5*open + 2.0e-6;
  float lit = area/(area + 3.14159*behind*behind*1.6);
  float side = smoothstep(0.30*hw, 1.0*hw, abs(q.x));
  lit *= mix(1.0, 0.45, side*smoothstep(0.0, 0.006, behind));
  float underU = smoothstep(0.0, 0.004, uy - q.y)*smoothstep(0.008, 0.0, uy - q.y);
  lit *= mix(1.0, 0.55, underU*smoothstep(-0.002, 0.003, behind));
  return vec4(clamp(lit, 0.0, 1.0), 1.0, behind, side);
}


// head-space aperture: share of a light of direction d (head space, toward the light) that a point q behind the seam
// sees through the opening (ellipse centre (0, yc), half axes hw, hh in the seam plane z = 0); soft edge = the light's
// size x the distance to the plane (area-light penumbra) + a base softness (lip thickness / curvature)
float pmThru(vec3 q, vec3 d, float hw, float yc, float hh, float soft, float tanR){
  if(q.z >= 0.0) return 1.0;
  if(d.z <= 0.02) return 0.0;
  float t = -q.z/d.z;
  vec2 h = q.xy + t*d.xy;
  vec2 e = vec2(h.x/hw, (h.y - yc)/hh);
  float r = length(e);
  float pen = min(soft + t*tanR/hh, 0.9);
  return smoothstep(1.0 + pen, 1.0 - pen, r);
}

float pmGGX(vec3 n, vec3 v, vec3 l, float a){
  vec3 h = normalize(v + l);
  float nl = clamp(dot(n, l), 0.0, 1.0), nv = clamp(dot(n, v), 1e-4, 1.0), nh = clamp(dot(n, h), 0.0, 1.0);
  return D_GGX(nh, max(a, 0.002))*V_SmithJ(nv, nl, max(a, 0.002))*nl;
}

vec4 matShade(SIn S, inout vec4 fragDOut){
  vec3 L = uKeyDir, n = S.n, v = S.v, ng = S.ng;
  S.shw = pbrShadow(S.p, S.ng, S.shw);                                   // + the head cascade
  vec4 ml = pmLight(S.p);
  float mouth = ml.y;
#if defined(PBR_DBG_MTYPE) || defined(PBR_DBG_DEPTH)
#if defined(PBR_T_TEETH)
  vec3 dc = vec3(1.0);
#elif defined(PBR_T_TONGUE)
  vec3 dc = vec3(1.0, 0.1, 0.1);
#elif defined(PBR_T_GUMS)
  vec3 dc = vec3(0.1, 1.0, 0.1);
#else
  vec3 dc = vec3(0.1, 0.2, 1.0);
#endif
#ifdef PBR_DBG_DEPTH
  return vec4(vec3(ml.x, ml.z/0.04, ml.w)/max(uCharK, 1e-3), 1.0);
#endif
  return vec4(dc*(0.25 + 0.75*clamp(ml.x, 0.0, 1.0))/max(uCharK, 1e-3), 1.0);
#endif
  // light bounced in through the lips and off the teeth/tongue: a mouth part never gets less than a share that fades
  // slowly with depth (dim pink at the back of an open mouth, never a black hole; uPbrMouth2.x = the share)


#if defined(PBR_T_GUMS)
  // shade2-mouth: the CARUNCLE (gums-type tissue far from the mouth, at the inner eye corner) as small pink wet
  // tissue: lit like the skin around it (shadow map, no cavity), a soft wet glint (pbr.mouth.caruncle / caruncle_wet)
  if(uPbrMouth5.x > 0.0 && mouth <= 0.0){
    vec3 ca = mix(S.alb, vec3(0.50, 0.22, 0.21), 0.5)*uPbrMouth5.x;
    float cw = 0.5;
    float cd = clamp((dot(n, L) + cw)/(1.0 + cw), 0.0, 1.0);
    vec3 cE = S.keyC*cd*S.shw.x + uFillCol*clamp((dot(n, uFillDir) + cw)/(1.0 + cw), 0.0, 1.0) + S.amb*S.ao;
    vec3 cc = ca*cE;
    vec3 cH = normalize(v + L);
    float cF = 0.03 + 0.97*pow(1.0 - clamp(dot(v, cH), 0.0, 1.0), 5.0);
    cc += S.keyC*cF*pmGGX(n, v, L, 0.10*0.10 + 0.5*uLightP.x)*PI*S.shw.x*uPbrMouth5.y;
    vec3 cr = reflect(-v, n);
    vec3 cenv = uProbeP.x > 0.5 ? envProbe(cr, 0.15) : setAmbient(S.p, cr);
    cc += cenv*uAmbGain*(0.03 + 0.97*pow(1.0 - clamp(dot(n, v), 1e-4, 1.0), 5.0))*uPbrMouth5.y*S.ao;
    return vec4(cc*S.alpha, S.alpha);
  }
#endif
  // ================================================================ shade2-mouth: the aperture model
  if(uPbrMouth3.x > 0.0 && mouth > 0.0){
    mat3 Ri = transpose(uHeadR);
    vec3 q = Ri*(S.p - uMouthC)/uHeadS;
    vec3 u0 = Ri*(uMouthU - uMouthC)/uHeadS, l0 = Ri*(uMouthL - uMouthC)/uHeadS;
    float yu = max(u0.y, l0.y), yl = min(u0.y, l0.y);
    float yc = 0.5*(yu + yl), hh = max(0.5*(yu - yl), 0.0012);
    float hw = 0.46*uMouthW/uHeadS;
#if defined(PBR_T_TONGUE)
    // the tongue as one rounded form: its mesh splits the normal at the midline (a hard dark groove) -> lean the
    // shading normal toward the head's up (the dorsum); the papillae glints stay in the specular below
    n = normalize(mix(n, uHeadR[1], 0.45));
#endif
    vec3 nq = Ri*n;
    float behind = max(-q.z, 0.0);
    float side = smoothstep(0.35*hw, 1.0*hw, abs(q.x));
    float kv = pmThru(q, Ri*L, hw, yc, hh, 0.12 + uPbrMouth6.z, uLightP.x);
    float fv = pmThru(q, Ri*uFillDir, hw, yc, hh, 0.45, 0.3);
    // the room through the opening: cosine-weighted share of a small aperture (area A at distance r), the surface
    // turned toward the opening sees more of it (sides / back of the tongue, cheeks: less)
    vec3 dq = vec3(0.0, yc, 0.0) - q;
    float r2 = dot(dq, dq) + 1e-8, ir = inversesqrt(r2);
    float A = 3.14159*hw*hh*uPbrMouth6.x;
    float cz = max(behind, 0.6*hh)*ir;
    float av = q.z >= 0.0 ? 1.0 : A*cz/(A*cz + 3.14159*r2*0.35);
    float cn = clamp(dot(nq, dq)*ir, -1.0, 1.0);
    av *= mix(0.5, 1.0, smoothstep(-0.4, 0.8, cn));
    av *= mix(1.0, 1.0 - 0.6*uPbrMouth5.z, side*smoothstep(0.0, 0.006, behind));      // buccal sides (cheeks in the way)
    float dk = exp(-behind/max(uPbrMouth4.z, 1e-3));                                   // the throat: light gone with depth
    vec3 alb = S.alb;
    float ao = S.ao;
    float rough = S.rough, f0 = 0.04, wrap = 0.3, wetW = 0.0, kao = 1.0;
    vec3 scat = vec3(1.0, 0.55, 0.45);
#if defined(PBR_T_TEETH)
    alb = mix(alb, vec3(dot(alb, vec3(0.333))), 0.3)*vec3(0.92, 0.86, 0.76);
    alb = min(alb, vec3(0.62));
    bool upper = q.y > yc;
    float facing = clamp(dot(n, uHeadR[2]), 0.0, 1.0);
    float ip = mix(0.40, 1.0, smoothstep(0.15, 0.75, facing));               // the interproximal faces turn away
    ao *= ip;
    kao = mix(1.0, ao, uPbrMouth3.w);                                         // gaps between the teeth: shadowed for direct light too
    // incisal edge: the biting edge (normal turned toward the bite) is thin enamel: greyer, bluish, lets the dark through
    float inc = smoothstep(0.15, 0.65, upper ? -nq.y : nq.y)*uPbrMouth4.x;
    alb = mix(alb, alb*vec3(0.62, 0.68, 0.80), inc);
    // gum line: the tooth neck under the lip / against the gum (upper row: above the upper lip's inner edge)
    // + the upper lip's soft shadow over the top of the visible upper teeth (a band ~4 mm under its inner edge)
    float gl = upper ? smoothstep(-0.0045, -0.0003, q.y - yu) : smoothstep(0.0035, -0.0015, q.y - yl);
    ao *= mix(1.0, 0.35, gl*uPbrMouth4.y);
    kao *= mix(1.0, 0.35, gl*uPbrMouth4.y);
    alb *= (upper ? 1.0 : uPbrMouth3.z)*uPbrMouth6.w;                          // the lower row sits in the upper's shade; level
    rough = uPbrMouth.w; f0 = 0.05; wrap = 0.5*uPbrMouth.x; scat = vec3(1.0, 0.78, 0.55); wetW = 0.35;
    kv = min(kv, mix(1.0, S.shw.x, 0.35));
#elif defined(PBR_T_TONGUE)
    alb = mix(vec3(dot(alb, vec3(0.30, 0.59, 0.11))), alb, uPbrMouth3.y);
    float pn = pmNoise(S.p*900.0/max(uHeadS, 1e-3));
    n = normalize(n + (vec3(pmNoise(S.p*1300.0/max(uHeadS, 1e-3)), pn, pmNoise(S.p*1700.0/max(uHeadS, 1e-3))) - 0.5)*0.25);
    // the tongue as one rounded form: its mesh splits the normal at the midline (a hard dark groove) -> lean the shading
    // normal toward the head's up (the dorsum), keep the papillae glints in the specular
    ao = mix(1.0, ao, 0.3);          // the map's AO carves the median groove as a black slit: keep a trace of it
    rough = 0.45; wrap = 0.35; wetW = 0.6*uPbrMouth5.w;
#elif defined(PBR_T_GUMS)
    alb = mix(vec3(dot(alb, vec3(0.30, 0.59, 0.11))), alb, uPbrMouth7.y)*uPbrMouth7.x;
    rough = 0.38; wrap = 0.45; wetW = 0.8;
#else
    alb = mix(vec3(dot(alb, vec3(0.30, 0.59, 0.11))), alb, uPbrMouth3.y);
    rough = 0.5; wrap = 0.4; wetW = 0.4;
#endif
    float ndl = dot(n, L);
    float dif = clamp((ndl + wrap)/(1.0 + wrap), 0.0, 1.0);
    vec3 sssTint = mix(vec3(1.0), scat, smoothstep(0.3, -0.2, ndl)*wrap);
    vec3 Ek = S.keyC*dif*sssTint*kv*kao;
    vec3 Ef = uFillCol*clamp((dot(n, uFillDir) + wrap)/(1.0 + wrap), 0.0, 1.0)*fv*kao;
    vec3 Ea = S.amb*ao*av;
    // bounce: light scattered off the lit teeth / tongue / lips into the cavity -- dim, warm, gone deep in
    float ambL = dot(S.amb, vec3(0.30, 0.59, 0.11)) + 0.08*dot(S.keyC, vec3(0.30, 0.59, 0.11));
    vec3 Eb = vec3(ambL)*vec3(1.0, 0.72, 0.66)*uPbrMouth4.w*dk*(1.0 - 0.7*side)*ao;
    // light the key put on the lower lip / lower teeth / tongue front, bounced up onto the tongue and the floor (warm)
    Eb += S.keyC*uPbrMouth6.y*dk*clamp(0.5 + 0.5*nq.y, 0.0, 1.0)*(1.0 - side)*vec3(1.0, 0.75, 0.68)*ao;
    vec3 col = alb*(Ek + Ef + Ea + Eb);
#if defined(PBR_T_TEETH)
    col += alb*S.keyC*vec3(1.0, 0.7, 0.45)*0.12*uPbrMouth.x*clamp(-ndl, 0.0, 1.0)*kv;
#endif
    vec3 H = normalize(v + L);
    float Fk = f0 + (1.0 - f0)*pow(1.0 - clamp(dot(v, H), 0.0, 1.0), 5.0);
    float sp = pmGGX(n, v, L, rough*rough) + wetW*uPbrMouth.z*0.6*pmGGX(n, v, L, 0.14*0.14 + 0.5*uLightP.x);
    col += S.keyC*Fk*sp*PI*kv*kao*ao;
    vec3 r = reflect(-v, n);
    float nv = clamp(dot(n, v), 1e-4, 1.0);
    vec3 env = uProbeP.x > 0.5 ? envProbe(r, mix(rough, 0.15, 0.5*wetW)) : setAmbient(S.p, r);
    col += env*uAmbGain*(f0 + (1.0 - f0)*pow(1.0 - nv, 5.0))*mix(0.6, 1.0, wetW)*ao*av;
#ifdef PBR_DBG_ALB
    return vec4(vec3(alb.r, ao, kv)/max(uCharK, 1e-3), 1.0);
#endif
#ifdef PBR_DBG_LUM
    return vec4(vec3(dot(Ea, vec3(0.333)), dot(Eb, vec3(.333)), dot(Ek, vec3(.333)))*3.0/max(uCharK, 1e-3), 1.0);
#endif
#ifdef PBR_DBG_AP
    return vec4(vec3(kv, av, dk)/max(uCharK, 1e-3), 1.0);
#endif
    return vec4(col*S.alpha, S.alpha);
  }
  float occ = mix(1.0, max(ml.x, 4.0*uPbrMouth2.x*exp(-ml.z/0.045)*(1.0 - 0.8*ml.w)), mouth);
  vec3 alb = S.alb;
  float ao = S.ao;
  float rough = S.rough;
  float f0 = 0.04;
  float wrap = 0.3;
  vec3 scat = vec3(1.0, 0.35, 0.25);     // colour of light scattered inside the tissue
  float wetW = 0.0;
  vec3 fwd = uHeadR[2];
#if defined(PBR_T_TEETH)
  // enamel: warm off-white (never chalk), a slight value spread between teeth from their own albedo
  alb = mix(alb, vec3(dot(alb, vec3(0.333))), 0.3)*vec3(0.92, 0.86, 0.76);
  alb = min(alb, vec3(0.62));
  // per-tooth contact AO: faces turned away from the front (the interproximal gaps) and toward the gums
  float facing = clamp(dot(n, fwd), 0.0, 1.0);
  ao *= mix(0.40, 1.0, smoothstep(0.15, 0.75, facing));      // interproximal: each tooth reads as its own form
  rough = uPbrMouth.w;
  f0 = 0.05;
  wrap = 0.5*uPbrMouth.x;
  scat = vec3(1.0, 0.78, 0.55);
  wetW = 0.35;
#elif defined(PBR_T_TONGUE)
  alb *= vec3(1.0, 0.92, 0.92);
  // papillae: a fine bumpy normal breaks the wet sheen into small glints
  float pn = pmNoise(S.p*900.0/max(uHeadS, 1e-3));
  n = normalize(n + (vec3(pmNoise(S.p*1300.0/max(uHeadS, 1e-3)), pn, pmNoise(S.p*1700.0/max(uHeadS, 1e-3))) - 0.5)*0.25);
  rough = 0.42; wrap = 0.6; wetW = 0.6;
#elif defined(PBR_T_GUMS)
  rough = 0.38; wrap = 0.6; wetW = 0.8;
#else   // mouth interior (cheeks, palate, throat)
  rough = 0.5; wrap = 0.7; wetW = 0.4;
#endif
#if !defined(PBR_T_TEETH)
  alb = max(alb, uPbrMouth2.yzw*mouth*(1.0 - ml.w));   // (only toward the centre: the buccal sides keep the asset's dark albedo)   // mucosa never darker than real soft tissue (a stylised map's near-black red)
#endif
  // ---- light through the lip opening (the shadow map is not trusted inside the mouth)
#if defined(PBR_T_TEETH)
  float shk = mix(S.shw.x, mix(S.shw.x, 1.0, 0.3), mouth);   // the front teeth are near the lips: the shadow map resolves the lip's shadow
#else
  float shk = mix(S.shw.x, mix(S.shw.x, 1.0, 0.8), mouth);   // deeper in, the map says "occluded" for everything: the opening model carries it
#endif
  float ndl = dot(n, L);
  float dif = clamp((ndl + wrap)/(1.0 + wrap), 0.0, 1.0);
  vec3 sssTint = mix(vec3(1.0), scat, smoothstep(0.3, -0.2, ndl)*wrap);       // the wrapped share is tissue-coloured
  vec3 Ek = S.keyC*dif*sssTint*shk*occ;
  vec3 Ef = uFillCol*clamp((dot(n, uFillDir) + wrap)/(1.0 + wrap), 0.0, 1.0)*occ;
  vec3 Ea = S.amb*ao*mix(occ, 1.0, uPbrMouth2.x*(1.0 - mouth));
  vec3 diffuse = alb*(Ek + Ef + Ea);
  // light scattered deep in the mouth is red, and never drops to black (a dim floor from the room's level)
  float ambL = dot(S.amb, vec3(0.30, 0.59, 0.11));
  diffuse *= 1.0 + mouth*(1.0 - occ)*vec3(0.35, 0.12, 0.09);        // inter-reflection inside the red cavity
  diffuse = max(diffuse, alb*ambL*uPbrMouth2.x*0.35*vec3(1.0, 0.8, 0.75)*mix(1.0, 0.5, ml.w));   // a low flat floor: the form comes from the light
#if defined(PBR_T_TEETH)
  // dentin warmth where the key passes through the enamel
  diffuse += alb*S.keyC*vec3(1.0, 0.7, 0.45)*0.12*uPbrMouth.x*clamp(-ndl, 0.0, 1.0)*occ;
#endif
  // ---- specular: rough base + a wet tight lobe, both occluded like the light
  vec3 H = normalize(v + L);
  float Fk = f0 + (1.0 - f0)*pow(1.0 - clamp(dot(v, H), 0.0, 1.0), 5.0);
  float sp = pmGGX(n, v, L, rough*rough) + wetW*uPbrMouth.z*0.6*pmGGX(n, v, L, 0.12*0.12 + 0.5*uLightP.x);
  float specOcc = ao*occ*occ;
  vec3 col = diffuse + S.keyC*Fk*sp*S.shw.x*PI*specOcc*mix(1.0, 0.7, mouth);
  vec3 r = reflect(-v, n);
  float nv = clamp(dot(n, v), 1e-4, 1.0);
  vec3 env = uProbeP.x > 0.5 ? envProbe(r, mix(rough, 0.15, 0.5*wetW)) : setAmbient(S.p, r);
  col += env*uAmbGain*(f0 + (1.0 - f0)*pow(1.0 - nv, 5.0))*mix(0.6, 1.0, wetW)*specOcc;
  // the cavity darkening by depth (strength knob): far behind the seam the light is gone
  col *= mix(1.0, mix(0.30, 1.0, exp(-ml.z/0.025)), mouth*uPbrMouth.y);   // depth: the back of the mouth falls toward dark
  return vec4(col*S.alpha, S.alpha);
}

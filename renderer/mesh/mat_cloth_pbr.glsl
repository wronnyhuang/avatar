// mat_cloth_pbr.glsl — shade-cloth: physically based cloth for a realistic character (pbr.py; gated by mesh.pbr.cloth).
//  Spliced into mesh.glsl FS_FWD instead of mat_cloth.glsl (pbr_common.glsl in front).  Same inputs as mat_cloth (cloth.py
//  setup: the procedural yarn / weave relief tCloth + uCloth*), plus uPbrCloth* (pbr.py CLOTH_DEFAULTS).
//   pattern   the relief in stitch space q, sampled with LATTICE-ALIGNED TILE SHUFFLING: every relief tile (12 x 16
//             stitches) takes its own random whole-stitch / whole-row offset (rib: 4-column, 2-row steps so the k2p2 and
//             purl parity hold) -> the knit's per-stitch jitter / colour no longer repeats every 8 cm; the hand-knit warp
//             (uClothK.y) bends rows on top.  Fine weaves (blouse plain, trouser twill) use a real thread pitch
//             (x weave_scale: 1 mm -> ~0.35 mm poplin) so they resolve to a smooth fabric with a micro-sheen at bust.
//   yarn      relief normal (Toksvig-faded with the footprint: no shimmer), parallax inside the loops, one relief
//             self-shadow tap; DIFFUSE on a softened relief normal (light diffuses through the fuzz), SHEEN on the full one;
//             multiple scattering inside the yarn: the cavities saturate (albedo^(1+k)/max^k: hue deepens, not grey) +
//             coloured multi-bounce AO (Jimenez) instead of black grooves; per-stitch variation (the relief's own stitch
//             ids, shuffled) + a 3-stitch heather.
//   fuzz      a fibre layer: Charlie sheen (Estevez-Kulla) with Neubelt visibility on the key + the room (directional
//             albedo approximation) + fill; energy taken from the base by the layer's albedo; a grazing halo where
//             fibres stick out of the silhouette (wrapped, so it survives the terminator)
//   cotton    plain-weave blouse: thin-fabric translucency (key + room through the cloth), a weak rough GGX fibre spec +
//             tight sheen = the poplin micro-sheen, deeper curvature fold occlusion
//   contact   the PBR key shadow (body map + head cascade), the neck as an occluder (collar / blouse around the neck:
//             a cylinder from the head frame, AO + contact), curvature fold AO, the asset AO / capsule AO
//  Energy: diffuse x (1 - sheen albedo); wrap normalised; every light the same as the skin (S.keyC, uFill*, S.amb/probe).
uniform sampler2D tCloth;
uniform vec4 uClothQ;   // x level, y parallax steps, z shadow taps, w fold AO (cloth.py)
uniform vec4 uClothX;   // mat2 uv -> q (stitch pitches)
uniform vec4 uClothT;   // x 1/tile.x, y 1/tile.y, z relief depth (height / pitch), w stitch (m)
uniform vec4 uClothS;   // x sheen roughness, y fuzz, z colour variation, w cavity strength
uniform vec4 uClothW;   // x diffuse wrap, y kind (0 none 1 knit 2 rib 3 plain 4 twill), z mean cavity AO, w wash
uniform vec4 uClothK;   // x knit relief contrast, y macro warp, z bright exposure, w bright form (face cfg cloth_k)
uniform vec4 uPbrCloth;   // x knit relief contrast, y tile shuffle, z yarn multiple scattering, w per-stitch variation
uniform vec4 uPbrCloth2;  // x fuzz layer, y fuzz (Charlie) roughness, z diffuse relief share, w knit stitch scale
uniform vec4 uPbrCloth3;  // x fine-weave pitch scale, y cotton translucency, z neck contact, w fold AO
uniform vec4 uPbrCloth4;  // xyz knit albedo tint (linear multiplier), w bright-cloth exposure
uniform vec4 uPbrCloth5;  // x neck radius scale, y cotton spec gain, z wash, w rim gain
uniform vec4 uPbrCloth6;  // x halo, y cotton sheen roughness, z heather, w relief self-shadow
uniform vec4 uPbrCloth7;  // x albedo flatten (the asset's tiled albedo -> its mean: its blotches repeat every tile_m),
                          // y fine-weave drape (soft rumple normal, ~2 cm), z cotton slub streaks, w twill sheen
uniform vec4 uPbrCloth8;  // x uv-stretch fade, y stretch ratio where it starts, z knit cavity gain
uniform mat3  uHeadR;
uniform float uHeadS;
uniform vec3  uEyeC[2];

vec2 cl_q;
float cl_h = 0.6, cl_var = 0.5, cl_fade = 0.0, cl_self = 1.0;
vec2 cl_dqx, cl_dqy;
bool cl_knit = false, cl_shuf = false;
float cl_m = 1e-3;      // metres per pattern pitch
float cl_str = 1.0;     // uv stretch ratio of the pattern on the surface

vec2 cpHash2(vec2 p){
  vec3 p3 = fract(vec3(p.xyx)*vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz)*p3.zy);
}
// relief tile uv of a pattern point: lattice-aligned per-tile offset (whole stitches / rows; rib keeps k2p2 + purl parity)
vec2 cpTuv(vec2 q){
  vec2 t = q*uClothT.xy;
  if(cl_shuf){
    vec2 h = cpHash2(floor(t) + 17.0);
    t += uClothW.y > 1.5 ? floor(h*vec2(3.0, 8.0))*vec2(4.0/12.0, 2.0/16.0)
                         : floor(h*vec2(12.0, 8.0))*vec2(1.0/12.0, 2.0/16.0);
  }
  return t;
}
vec4 clTex(vec2 q){ return textureGrad(tCloth, cpTuv(q), cl_dqx*uClothT.xy, cl_dqy*uClothT.xy); }

void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){
  mat2 M = mat2(uClothX.xy, uClothX.zw);
  float kind = uClothW.y;
  cl_knit = kind > 0.5 && kind < 2.5;
  cl_shuf = cl_knit && uPbrCloth.y > 0.5;
  float scl = cl_knit ? uPbrCloth2.w : (kind > 2.5 ? uPbrCloth3.x : 1.0);
  vec2 q = M*uv/max(scl, 0.05);
  cl_m = uClothT.w*max(scl, 0.05);
  if(uClothK.y > 0.0 && cl_knit){            // hand-knit irregularity: rows wander, stitches vary (as mat_cloth)
    vec2 qm = q*(1.0/18.0);
    q += uClothK.y*vec2(vnoise2(qm) - 0.5, vnoise2(qm + vec2(7.1, 3.3)) - 0.5)
       + 0.4*uClothK.y*vec2(vnoise2(qm*2.9 + 1.7) - 0.5, vnoise2(qm*2.9 + vec2(4.2, 9.1)) - 0.5);
  }
  cl_q = q;
  cl_dqx = dFdx(q); cl_dqy = dFdy(q);
  float fp = max(length(cl_dqx), length(cl_dqy));               // pixel footprint in stitch pitches
  cl_fade = (uClothQ.x < 0.5 || kind < 0.5) ? 0.0 : 1.0 - smoothstep(0.45, 1.0, fp);
  // UV stretch (the pattern's anisotropy on the SURFACE, not on screen): q per world unit along T, B from the screen
  // derivatives; where the asset's uv shears / stretches the stitches >~1.6:1 the relief turns into streaks -> fade it
  if(uPbrCloth8.x > 0.0 && cl_fade > 0.0){
    vec3 dpx = dFdx(vP), dpy = dFdy(vP);
    mat2 P = mat2(dot(dpx, T), dot(dpx, B), dot(dpy, T), dot(dpy, B));
    float dP = determinant(P);
    if(abs(dP) > 1e-14){
      mat2 J = mat2(cl_dqx, cl_dqy)*inverse(P);
      mat2 JJ = transpose(J)*J;
      float tr = JJ[0][0] + JJ[1][1], dt = max(determinant(JJ), 0.0);
      float dd = sqrt(max(tr*tr*0.25 - dt, 0.0));
      float st = sqrt(max(tr*0.5 + dd, 0.0)/max(tr*0.5 - dd, 1e-12));
      cl_str = st;
      cl_fade *= 1.0 - uPbrCloth8.x*smoothstep(uPbrCloth8.y, uPbrCloth8.y*1.6, st);
    }
  }
  if(cl_fade <= 0.0) return;
  vec2 c0 = normalize(M[0]), c1 = normalize(M[1]);
  int N = int(uClothQ.y);
  if(N > 0){
    vec3 vt = vec3(dot(v, T), dot(v, B), dot(v, ng));
    vec2 dir = -(c0*vt.x + c1*vt.y)/max(vt.z, 0.3)*uClothT.z*cl_fade;
    float dz = 1.0/float(N), z = 1.0;
    vec2 qq = q; float hp = clTex(qq).z, zp = 1.0;
    for(int i = 0; i < 8; i++){
      if(i >= N || hp >= z) break;
      zp = z; z -= dz; qq = q + dir*(1.0 - z);
      float hn = clTex(qq).z;
      if(hn >= z){ float a = (zp - hp)/max((zp - hp) - (z - hn), 1e-4); qq = q + dir*(1.0 - mix(zp, z, a)); break; }
      hp = hn;
    }
    cl_q = qq;
  }
}

vec3 cpAOMulti(float ao, vec3 alb){             // Jimenez 2016 multi-bounce AO: crevices keep the yarn's colour
  vec3 a = 2.0404*alb - 0.3324, b = -4.7951*alb + 0.6417, c = 2.7552*alb + 0.6903;
  return max(vec3(ao), ((ao*a + b)*ao + c)*ao);
}
float cpMax(vec3 c){ return max(max(c.r, c.g), c.b); }
// the neck as an occluder: a cylinder down from the head frame (collar stand, blouse round the neck); 1 = open
float cpNeck(vec3 p, vec3 n){
  if(uHeadS <= 0.0 || uPbrCloth3.z <= 0.0) return 1.0;
  vec3 up = uHeadR[1], fw = uHeadR[2];
  vec3 hc = 0.5*(uEyeC[0] + uEyeC[1]) - fw*(0.085*uHeadS);
  vec3 a = hc - up*(0.07*uHeadS) - fw*(0.01*uHeadS), b = hc - up*(0.24*uHeadS) - fw*(0.005*uHeadS);
  float r = 0.052*uHeadS*uPbrCloth5.x;
  vec3 ba = b - a;
  float t = clamp(dot(p - a, ba)/dot(ba, ba), 0.0, 1.0);
  vec3 d = a + ba*t - p;
  float l = length(d);
  float o = clamp(dot(n, d)/max(l, 1e-5), 0.0, 1.0)*(r*r)/max(l*l, 1e-8);
  o *= smoothstep(0.7*r, 1.0*r, l);
  float c = 1.0 - smoothstep(0.0, 0.03*uHeadS, l - r);             // within ~3 cm of the skin
  return clamp(1.0 - uPbrCloth3.z*(0.55*min(o, 1.0) + 0.30*c*c), 0.25, 1.0);
}

vec4 matShade(SIn S, inout vec4 fragDOut){
  vec3 L = uKeyDir, v = S.v, ng = S.ng;
  float kind = uClothW.y;
  bool plain = kind > 2.5 && kind < 3.5;
  S.shw = pbrShadow(S.p, S.ng, S.shw);
  vec3 nM = S.n;                                  // macro: the asset's fold normal
  vec3 nR = nM, nB = nM;
  vec3 alb = S.alb;
  if(uTexOn.x > 0.5 && uPbrCloth7.x > 0.0)       // the asset tile's mean colour (top mip); the shader's own variation replaces it
    alb = mix(alb, srgb2lin(textureLod(tAlb, S.uv, 12.0).rgb), uPbrCloth7.x);
  if(cl_knit) alb *= uPbrCloth4.xyz;              // documented hue knob (linear multiplier; 1 = the asset's albedo)
  float bright = smoothstep(0.45, 0.7, dot(S.alb, vec3(0.333)));
  alb *= mix(1.0, uPbrCloth4.w, bright);
  float cav = uClothW.y < 0.5 ? 1.0 : uClothW.z;
  float srough = uPbrCloth2.y;
  float valley = 0.0;
  mat2 M = mat2(uClothX.xy, uClothX.zw);
  vec2 c0 = normalize(M[0]), c1 = normalize(M[1]);
  // fine weaves: a soft rumple (two octaves over ~2 cm, fixed to the cloth's uv) -- a real blouse is never CG-flat
  if(kind > 2.5 && uPbrCloth7.y > 0.0){
    vec2 qd = cl_q*(cl_m/0.02)*vec2(1.0, 0.4);         // folds hang: elongated along the cloth's v (down)
    float e = 0.3;
    float n0 = vnoise2(qd), n1 = vnoise2(qd*2.3 + 7.7);
    vec2 g = vec2(vnoise2(qd + vec2(e, 0.0)) - n0, vnoise2(qd + vec2(0.0, e)) - n0)/e
           + 0.5*vec2(vnoise2(qd*2.3 + 7.7 + vec2(e, 0.0)) - n1, vnoise2(qd*2.3 + 7.7 + vec2(0.0, e)) - n1)/e;
    vec2 sd = vec2(dot(g, c0), dot(g, c1))*0.12*uPbrCloth7.y;
    nM = normalize(nM - (S.T*sd.x + S.B*sd.y));
  }
  // cotton slub: thread-thickness streaks along warp and weft (a few threads wide, ~1-2 cm long), faded by footprint
  if(plain && uPbrCloth7.z > 0.0){
    float fpq = max(length(cl_dqx), length(cl_dqy));
    float sw = vnoise2(vec2(cl_q.x*0.45, cl_q.y*0.025)) - 0.5, sf = vnoise2(vec2(cl_q.x*0.025 + 3.1, cl_q.y*0.45)) - 0.5;
    alb *= 1.0 + uPbrCloth7.z*0.10*(sw + sf)*(1.0 - smoothstep(1.2, 3.0, fpq));
  }
  if(cl_fade > 0.0){
    vec4 r = clTex(cl_q);
    cl_h = r.z; cl_var = r.w;
    float kc = cl_knit ? uPbrCloth.x : 1.0;
    vec2 sl = vec2(dot(r.xy, c0), dot(r.xy, c1))*uClothT.z*cl_fade*kc;
    nR = normalize(nM - (S.T*sl.x + S.B*sl.y));
    // the fuzz-blurred relief (a 2.5x wider footprint = ~1.3 mips up): the diffuse sees rounded, soft yarn
    vec4 rb = textureGrad(tCloth, cpTuv(cl_q), cl_dqx*uClothT.xy*2.5, cl_dqy*uClothT.xy*2.5);
    vec2 slb = vec2(dot(rb.xy, c0), dot(rb.xy, c1))*uClothT.z*cl_fade*kc;
    nB = normalize(nM - (S.T*slb.x + S.B*slb.y));
    float a0 = mix(1.0, mix(0.35, 1.0, smoothstep(0.0, 0.8, cl_h)), clamp(uClothS.w*(cl_knit ? uPbrCloth8.z : 1.0), 0.0, 1.0));
    cav = mix(uClothW.z, a0, cl_fade);
    valley = (1.0 - smoothstep(0.15, 0.8, cl_h))*cl_fade;
    // per-stitch variation from the relief's own stitch ids (shuffled -> no repeat)
    float cv = mix(0.5, cl_var, cl_fade);
    alb *= 1.0 + uPbrCloth.w*uClothS.z*(cv - 0.5)*0.9;
    // relief self-shadow toward the key (one tap at the yarn's scale)
    if(uClothQ.z > 0.5 && uPbrCloth6.w > 0.0){
      vec3 lt = vec3(dot(L, S.T), dot(L, S.B), dot(L, ng));
      if(lt.z > 0.02){
        vec2 ld = (c0*lt.x + c1*lt.y)/max(lt.z, 0.2)*uClothT.z;
        float hs = clTex(cl_q + ld*0.25).z;
        cl_self = 1.0 - 0.6*uPbrCloth6.w*cl_fade*clamp((hs - cl_h - 0.25)*3.0, 0.0, 1.0);
      }
    }
    srough = clamp(srough + 0.2*(1.0 - cl_fade), 0.05, 1.0);       // Toksvig-style: the filtered relief widens the lobe
  } else {
    srough = clamp(srough + 0.2, 0.05, 1.0);
  }
  if(plain) srough = clamp(uPbrCloth6.y + 0.15*(1.0 - cl_fade), 0.05, 1.0);
  // yarn multiple scattering: the deeper the light goes between fibres, the more often it is filtered -> saturated hue
  if(cl_knit && uPbrCloth.z > 0.0){
    float mx = max(cpMax(alb), 1e-3);
    float e = uPbrCloth.z*(0.15 + 0.85*valley)*0.4;
    alb = alb*pow(clamp(alb/mx, 1e-3, 1.0), vec3(e));
  }
  // heather (3 stitches) + wash (~25 stitches): smooth value noise, pure ALU, never aliases
  if(kind > 0.5){
    float fpq = max(length(cl_dqx), length(cl_dqy));
    if(cl_knit && uPbrCloth6.z > 0.0){
      float hn = vnoise2(cl_q*(1.0/3.1) + 11.0)*0.6 + vnoise2(cl_q*(1.0/1.3) + 3.0)*0.4;
      alb *= 1.0 + uPbrCloth6.z*0.14*(hn - 0.5)*(1.0 - smoothstep(0.6, 1.4, fpq));
    }
    if(uPbrCloth5.z > 0.0){
      vec2 qm = cl_q*(1.0/25.0)*(cl_knit ? 1.0 : 0.3);
      float wv = vnoise2(qm)*0.7 + vnoise2(qm*2.7 + 5.3)*0.3;
      alb *= 1.0 + uPbrCloth5.z*uClothW.w*(wv - 0.5)*(cl_knit ? 0.45 : 0.15);
    }
  }
  // ---------------- occlusion: asset/capsule AO x relief cavity x fold curvature x the neck
  float fold = 1.0;
  if(uClothQ.w > 0.0 && uPbrCloth3.w > 0.0){
    vec3 dpx = dFdx(S.p), dpy = dFdy(S.p);
    float k = dot(dFdx(S.n), dpx)/max(dot(dpx, dpx), 1e-10) + dot(dFdy(S.n), dpy)/max(dot(dpy, dpy), 1e-10);
    fold = clamp(1.0 + uPbrCloth3.w*(plain ? 0.018 : 0.012)*min(k, 0.0), 0.5, 1.0);
  }
  float neck = cpNeck(S.p, nM);
  float aoG = S.ao*fold*neck;                     // geometric occlusion (all light)
  float ao = aoG*cav;
  vec3 aoD = cpAOMulti(ao, alb);
  // ---------------- normals: diffuse on a softened relief (light diffuses through the fuzz), sheen on the full one
  vec3 nD = normalize(mix(nM, nB, uPbrCloth2.z));
  float nv = clamp(dot(nR, v), 1e-4, 1.0);
  float ndl = dot(nD, L);
  float w = uClothW.x*(plain ? 0.7 : 1.0);
  float shK = mix(S.shw.x, S.shw.y, 0.4)*cl_self;
  // ---------------- fuzz / sheen layer (energy: what it reflects does not reach the base)
  vec3 sheenC = (cl_knit ? mix(alb, uMatC, 0.45) : mix(alb, uMatC, 0.6))*uPbrCloth2.x*(plain ? 0.8 : (kind > 3.5 ? uPbrCloth7.w : 1.0));
  float Es = mix(0.10, 0.42, pow(1.0 - nv, 3.0))*(0.7 + 0.5*srough);   // Charlie directional albedo (approx.)
  float kd = 1.0 - cpMax(sheenC)*Es;
  // ---------------- diffuse
  float dif = clamp((ndl + w)/(1.0 + w), 0.0, 1.0)/(1.0 + w);
  vec3 E = S.keyC*dif*shK*mix(1.0, cav, 0.6)*mix(1.0, aoG, 0.3);
  float wf = 0.3;
  E += uFillCol*clamp((dot(nD, uFillDir) + wf)/(1.0 + wf), 0.0, 1.0)/(1.0 + wf)*aoD;
  E += S.amb*aoD;
  // thin cotton: the key and the room through the fabric (light from behind the surface)
  if(plain && uPbrCloth3.y > 0.0){
    float back = clamp(-dot(ng, L), 0.0, 1.0);
    E += (S.keyC*back*S.shw.y*0.35 + setAmbient(S.p, -ng)*uAmbGain*uAmbTint*0.25)*uPbrCloth3.y*aoG;
  }
  vec3 col = alb*E*kd;
  // key sheen (Charlie + Neubelt visibility) on the relief normal
  vec3 H = normalize(v + L);
  float nl = clamp(dot(nR, L), 0.0, 1.0);
  float Vis = 1.0/(4.0*(nl + nv - nl*nv) + 1e-3);
  float shn = D_Charlie(clamp(dot(nR, H), 0.0, 1.0), max(srough, 0.2))*Vis*nl;
  col += sheenC*S.keyC*shn*shK*PI*mix(1.0, cav, 0.5);
  float nlf = clamp(dot(nR, uFillDir), 0.0, 1.0);
  vec3 Hf = normalize(v + uFillDir);
  col += sheenC*uFillCol*D_Charlie(clamp(dot(nR, Hf), 0.0, 1.0), max(srough + 0.1, 0.2))/(4.0*(nlf + nv - nlf*nv) + 1e-3)*nlf*PI*aoG;
  col += sheenC*S.amb*aoD*Es;
  // fibre halo: fibres sticking out of the yarn catch the key past the terminator, strongest on the crowns
  float graze = pow(1.0 - clamp(dot(nM, v), 0.0, 1.0), 3.0);
  float crown = mix(1.0, smoothstep(0.2, 0.9, cl_h)*1.3, cl_fade);
  col += sheenC*S.keyC*clamp((dot(ng, L) + 0.4)/1.4, 0.0, 1.0)*S.shw.y*graze*crown*uPbrCloth6.x*0.35*aoG;
  // cotton fibre spec: weak, rough GGX = the poplin's micro-sheen (key as an area light)
  if(plain && uPbrCloth5.y > 0.0){
    float a = 0.5*0.5;
    float ak = min(a + 0.5*uLightP.x, 1.0);
    float f0 = 0.04*uPbrCloth5.y;
    vec3 Hh = normalize(v + L);
    float Fk = f0 + (1.0 - f0)*pow(1.0 - clamp(dot(v, Hh), 0.0, 1.0), 5.0);
    col += S.keyC*Fk*specGGX(nR, v, L, sqrt(ak))*S.shw.x*PI*aoG*cav;
  }
  // rim from the lamp behind (outer silhouettes)
  col += rimLight(S)*uPbrCloth5.w*mix(0.6, 1.0, aoG);
#ifdef PBR_DBG_CLOTH
  return vec4(vec3(clamp((cl_str - 1.0)/2.0, 0.0, 1.0), cl_fade, neck)/max(uCharK, 1e-3), 1.0);
#endif
  return vec4(col*S.alpha, S.alpha);
}

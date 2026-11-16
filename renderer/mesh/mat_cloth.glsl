// mat_cloth.glsl — render-cloth: knit cardigan, rib bands, blouse plain weave, twill trousers.  MAT_CLOTH.
// Procedural yarn / weave relief (renderer/mesh/cloth.py builds one tileable mip-mapped RGBA16F texture per weave kind:
// dh/dqx, dh/dqy, height 0..1, per-stitch colour variation; q = pattern space in stitch pitches).  On top of the asset's
// own albedo / fold normal / packed maps: relief normal, cavity AO, parallax inside the loops, a relief self-shadow tap,
// per-stitch heather, wrapped diffuse, Charlie sheen with Ashikhmin visibility, grazing fibre fuzz, soft fold occlusion.
// Anti-alias: the relief is band-limited by its mips; every detail term additionally fades to its tile mean when a pixel
// covers more than ~half a stitch, so nothing sparkles or moires under motion.
uniform sampler2D tCloth;
uniform vec4 uClothQ;   // x level (0 maps+sheen, 1 relief+cavity, 2 +parallax/shadow), y parallax steps, z shadow taps, w fold AO
uniform vec4 uClothX;   // mat2 uv -> q (stitch pitches)
uniform vec4 uClothT;   // x 1/tile.x, y 1/tile.y (pitches -> texture uv), z relief depth (height / pitch), w stitch (m)
uniform vec4 uClothS;   // x sheen roughness, y fuzz, z colour variation, w cavity strength
uniform vec4 uClothW;   // x diffuse wrap, y kind (0 none 1 knit 2 rib 3 plain 4 twill), z mean cavity AO of the tile, w wash
uniform vec4 uClothK;   // render4 (mesh.face.cloth_k): x knit relief contrast, y macro warp (stitch pitches; breaks
                        //   the tile), z bright-cloth exposure (blouse whites), w bright-cloth form (less wrap, deeper folds)

vec2 cl_q;              // pattern coordinate after parallax
float cl_h = 0.6, cl_var = 0.5, cl_fade = 0.0, cl_self = 1.0;
vec2 cl_dqx, cl_dqy;

vec4 clTex(vec2 q){ return textureGrad(tCloth, q*uClothT.xy, cl_dqx*uClothT.xy, cl_dqy*uClothT.xy); }

void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){
  mat2 M = mat2(uClothX.xy, uClothX.zw);
  vec2 q = M*uv;
  // render4: hand-knit irregularity -- a smooth low-frequency warp of the pattern (over ~18 stitches) so rows
  // wander and stitches vary in size instead of a printed tile (stitch-space; derivatives stay those of q)
  if(uClothK.y > 0.0 && uClothW.y > 0.5 && uClothW.y < 2.5){
    vec2 qm = q*(1.0/18.0);
    q += uClothK.y*vec2(vnoise2(qm) - 0.5, vnoise2(qm + vec2(7.1, 3.3)) - 0.5)
       + 0.4*uClothK.y*vec2(vnoise2(qm*2.9 + 1.7) - 0.5, vnoise2(qm*2.9 + vec2(4.2, 9.1)) - 0.5);
  }
  cl_q = q;
  cl_dqx = dFdx(q); cl_dqy = dFdy(q);
  float fp = max(length(cl_dqx), length(cl_dqy));               // pixel footprint in stitch pitches
  cl_fade = (uClothQ.x < 0.5 || uClothW.y < 0.5) ? 0.0 : 1.0 - smoothstep(0.45, 1.0, fp);
  if(cl_fade <= 0.0) return;
  vec2 c0 = normalize(M[0]), c1 = normalize(M[1]);
  // parallax: walk the view ray down through the relief (linear search + one secant refinement)
  int N = int(uClothQ.y);
  if(N > 0){
    vec3 vt = vec3(dot(v, T), dot(v, B), dot(v, ng));
    vec2 dir = -(c0*vt.x + c1*vt.y)/max(vt.z, 0.3)*uClothT.z*cl_fade;   // q shift for a full height unit
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

vec4 matShade(SIn S, inout vec4 fragDOut){   // inout (an out param is undefined until written: NVIDIA gave 0 -> mesh depth layer 0 -> no SSS, no depth)
  vec3 L = uKeyDir, v = S.v, ng = S.ng;
  vec3 n = S.n;
  vec3 alb = S.alb;
  float cav = 1.0, srough = uClothS.x;
  if(cl_fade > 0.0){
    vec4 r = clTex(cl_q);
    cl_h = r.z; cl_var = r.w;
    mat2 M = mat2(uClothX.xy, uClothX.zw);
    vec2 c0 = normalize(M[0]), c1 = normalize(M[1]);
    float kc = uClothW.y > 0.5 && uClothW.y < 2.5 && uClothK.x > 0.0 ? uClothK.x : 1.0;   // render4: knit contrast
    vec2 sl = vec2(dot(r.xy, c0), dot(r.xy, c1))*uClothT.z*cl_fade*kc;  // height slope along T, B
    vec3 np = normalize(n - (S.T*sl.x + S.B*sl.y));
    // Toksvig-style: mip-averaged slopes shorter than the true ones -> widen the sheen lobe instead of sparkling
    srough = clamp(srough + 0.25*(1.0 - cl_fade), 0.05, 1.0);
    n = np;
    float a0 = mix(1.0, mix(0.35, 1.0, smoothstep(0.0, 0.8, cl_h)), uClothS.w);
    cav = mix(uClothW.z, a0, cl_fade);
    cav = mix(uClothW.z, cav, kc < 1.0 ? kc : 1.0);
    float cv = mix(0.5, cl_var, cl_fade);
    alb *= 1.0 + uClothS.z*(cv - 0.5)*1.1;
    alb = mix(alb, alb*alb/max(dot(alb, vec3(0.333)), 1e-3)*0.95, uClothS.z*(0.5 - cv)*0.8); // deeper hue in the cavities
    // valleys: deeper, more saturated yarn (light scatters between fibres before it leaves) instead of grey darkening
    float vly = (1.0 - smoothstep(0.15, 0.75, cl_h))*cl_fade*uClothS.w;
    alb = mix(alb, pow(alb, vec3(1.35))*1.15, vly);
    // relief self-shadow toward the key
    int K = int(uClothQ.z);
    vec3 lt = vec3(dot(L, S.T), dot(L, S.B), dot(L, ng));
    if(K > 0 && lt.z > 0.02){
      vec2 ld = (c0*lt.x + c1*lt.y)/max(lt.z, 0.2)*uClothT.z;
      float occ = 0.0;
      for(int k = 1; k <= 2; k++){
        if(k > K) break;
        float dh = 0.22*float(k);
        float hs = clTex(cl_q + ld*dh).z;
        occ += clamp((hs - cl_h - dh)*3.0, 0.0, 1.0);
      }
      cl_self = 1.0 - 0.75*cl_fade*occ/float(K);
    }
  } else {
    cav = uClothW.y < 0.5 ? 1.0 : uClothW.z;          // q0 keeps the tile-mean cavity: no brightness jump on the ladder
  }
  // large-scale wash / heather: smooth 2-octave value noise over ~25 and ~9 stitches (pure ALU, never aliases)
  if(uClothW.w > 0.0 && uClothW.y > 0.5){
    vec2 qm = cl_q*(1.0/25.0);
    float wv = vnoise2(qm)*0.7 + vnoise2(qm*2.7 + 5.3)*0.3;
    alb *= 1.0 + uClothW.w*(wv - 0.5)*0.45;
  }
  float nv = clamp(dot(n, v), 1e-4, 1.0);
  // soft fold occlusion from screen-space curvature of the smooth normal (concave folds darken a little)
  float fold = 1.0;
  if(uClothQ.w > 0.0){
    vec3 dpx = dFdx(S.p), dpy = dFdy(S.p);
    float k = dot(dFdx(S.n), dpx)/max(dot(dpx, dpx), 1e-10) + dot(dFdy(S.n), dpy)/max(dot(dpy, dpy), 1e-10);
    fold = clamp(1.0 + uClothQ.w*0.012*min(k, 0.0), 0.6, 1.0);
  }
  float ao = S.ao*cav*fold;
  // key: wrapped diffuse (energy-normalised), soft shadow, relief self-shadow
  float ndl = dot(n, L);
  float w = uClothW.x;
  // render4: bright cloth (the cream blouse) reads blown/flat -- lower its exposure and give it form
  float bright = smoothstep(0.45, 0.7, dot(S.alb, vec3(0.333)));
  if(bright > 0.0 && uClothK.z > 0.0){
    alb *= mix(1.0, uClothK.z, bright);
    w = mix(w, w*(1.0 - uClothK.w), bright);
    fold = mix(fold, fold*fold, bright*uClothK.w);
    ao = S.ao*cav*fold;
  }
  float dif = clamp((ndl + w)/(1.0 + w), 0.0, 1.0)/(1.0 + w)*1.35;
  float sh = mix(S.shw.x, S.shw.y, 0.5)*cl_self;
  vec3 sheenC = mix(alb, uMatC, 0.6)*uMatP.z;
  float esheen = 0.25*max(max(sheenC.r, sheenC.g), sheenC.b);     // energy taken by the sheen layer
  vec3 fillT = uFillCol*clamp((dot(n, uFillDir) + 0.3)/1.3, 0.0, 1.0);
  vec3 diffuse = alb*(1.0 - esheen)*(S.keyC*dif*sh*mix(1.0, cav, 0.5) + fillT*ao + S.amb*ao);
  // Charlie sheen (Estevez-Kulla) with Ashikhmin visibility
  vec3 h = normalize(v + L);
  float nl = clamp(ndl, 0.0, 1.0);
  float Vis = 1.0/(4.0*(nl + nv - nl*nv) + 1e-3);
  float shn = D_Charlie(clamp(dot(n, h), 0.0, 1.0), max(srough, 0.25))*Vis*nl;
  vec3 col = diffuse + sheenC*S.keyC*shn*sh*PI;
  // grazing fibre fuzz: a fibre halo lit by the room and (wrapped) by the key, strongest on the yarn crowns
  float graze = pow(1.0 - nv, 2.5);
  float crown = mix(1.0, smoothstep(0.2, 0.9, cl_h)*1.3, cl_fade);
  vec3 fz = sheenC*(S.amb*0.9 + S.keyC*clamp(dot(ng, L) + 0.5, 0.0, 1.0)*sh*0.35)*graze*uClothS.y*crown;
  col += fz*mix(0.6, 1.0, ao);
  col += rimLight(S)*(0.6 + 0.6*uClothS.y);
  return vec4(col*S.alpha, S.alpha);
}

// mat_eye_pbr.glsl — shade-eyes: the physically based eyeball (eye_l / eye_r) for a realistic character (pbr.py;
// gated by mesh.pbr.eye).  Spliced into mesh.glsl FS_FWD instead of mat_eye.glsl.  The clear cornea's reflections are
// mat_cornea_pbr.glsl (blended over this); here: what is seen THROUGH it.
//   iris     matUV traces the view ray back to the cornea (a sphere cap over the iris: the limbus circle + the anterior
//            chamber depth uPbrEye.y), refracts it (ior uPbrEye.x) and intersects the iris (a shallow concave dish at
//            the limbus plane): every map is read at the refracted point -> real parallax + magnification, no swim at
//            grazing angles (refraction clamped, a smooth warp back to the mesh uv across the limbus)
//            light: the key refracted at the cornea above the iris point onto the dish normal (+ the iris normal
//            map), a caustic crescent on the side AWAY from the key, the limbus' own shadow on the near side, a soft
//            limbal ring, the pupil truly dark
//   sclera   a lit wet ball: wrap / translucent falloff (red travels further), faint procedural veins growing toward
//            the corners, warmer toward the caruncle, lid-contact AO + the upper lid's shade band (lid aperture,
//            eyesmouth.py) on top of the key shadow map (lid + lash cast shadows)
//  Frame uniforms (pbr.py EyeFrame): uEyeGX/GY/GZ[2] world frame per eye (GZ = gaze), uEyeUV[2] the iris uv layout,
//  uEyeIris (iris radius, limbus plane, dish depth, iris half-angle).  uEyeIris.x == 0 -> the mesh uv, no refraction.
uniform mat3  uHeadR;
uniform float uHeadS;
uniform vec3  uEyeC[2];
uniform float uEyeR;
uniform float uLidTop[26], uLidBot[26];
uniform vec3  uEyeGX[2], uEyeGY[2], uEyeGZ[2];
uniform vec4  uEyeUV[2];
uniform vec4  uEyeIris;
uniform vec4  uPbrEye;     // x ior, y anterior chamber depth (m, for a 6 mm iris; scaled by the iris), z limbus ring, w caustic
uniform vec4  uPbrEye2;    // x sclera veins, y lid AO, z iris gain, w sclera tone
uniform vec4  uPbrEye3;    // x upper-lid shade band, y refraction (0 = mesh uv), z iris dish (concavity) gain, w sclera scatter
uniform vec3  uPbrIrisC, uPbrIrisM, uPbrIrisR;   // iris grade palette: collarette, stroma, rim (linear)
uniform float uPbrIrisF;   // radial fibre contrast (1 = r3)
uniform vec4  uPbrEye5;    // x socket occlusion of the room + fill (the ball sits in the orbit: sees ~half the room), y sclera corner darkening
uniform vec4  uPbrEye4;    // x eye-occlusion strength, y tear-line strength (the other eye layers), z iris grade (0 = the map), w pupil size (1 = the map)

// set by matUV for matShade
int   gEk = 0;
float gEW = 0.0;           // refraction weight (0 = mesh uv)
vec3  gEHit = vec3(0.0);   // the refracted ray's iris point (eye frame)
float gEDome = 0.0;        // dome radius
float gEDomeZ = 0.0;       // dome centre (axial)

int peEye(vec3 p){ return distance(p, uEyeC[0]) < distance(p, uEyeC[1]) ? 0 : 1; }
vec3 peLoc(vec3 d, int k){ return vec3(dot(d, uEyeGX[k]), dot(d, uEyeGY[k]), dot(d, uEyeGZ[k])); }
vec3 peWld(vec3 l, int k){ return uEyeGX[k]*l.x + uEyeGY[k]*l.y + uEyeGZ[k]*l.z; }

// the lids around the fragment (mat_eye.glsl eyeLids): x deg below the upper lid, y deg above the lower lid, z inner corner
vec4 peLids(vec3 p){
  int k = peEye(p);
  mat3 Ri = transpose(uHeadR);
  vec3 d = normalize(Ri*(p - uEyeC[k]));
  float az = degrees(atan(d.x, d.z)), el = degrees(asin(clamp(d.y, -1.0, 1.0)));
  float f = clamp((az + 66.0)/11.0, 0.0, 11.999);
  if(isnan(f)) f = 6.0;
  int i = int(f); float t = fract(f);
  float top = mix(uLidTop[k*13 + i], uLidTop[k*13 + i + 1], t);
  float bot = mix(uLidBot[k*13 + i], uLidBot[k*13 + i + 1], t);
  vec3 o = Ri*(uEyeC[k] - 0.5*(uEyeC[0] + uEyeC[1]));
  float inner = (o.x > 0.0 ? -az : az);
  return vec4(top - el, el - bot, smoothstep(20.0, 48.0, inner), float(k));
}

void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){
  gEk = peEye(vP);
  if(uEyeIris.x <= 0.0 || uPbrEye3.y <= 0.0) return;
  int k = gEk;
  float rI = uEyeIris.x, zL = uEyeIris.y;
  float h = max(uPbrEye.y, 1e-4)/0.006*rI;                 // the cornea's height over the limbus plane
  float Rc = (rI*rI + h*h)/(2.0*h), zc = zL + h - Rc;
  gEDome = Rc; gEDomeZ = zc;
  vec3 lp = peLoc(vP - uEyeC[k], k), lv = peLoc(v, k);
  vec3 oc = lp - vec3(0.0, 0.0, zc);
  float b = dot(oc, lv), cc = dot(oc, oc) - Rc*Rc, disc = b*b - cc;
  if(disc <= 0.0) return;
  float t = -b + sqrt(disc);                               // the camera-side crossing of the dome
  if(t < 0.0) return;
  vec3 q = lp + lv*t;
  float cap = smoothstep(zL - 0.04*rI, zL + 0.06*rI, q.z);
  if(cap <= 0.0) return;
  vec3 nc = (q - vec3(0.0, 0.0, zc))/Rc;
  vec3 d = refract(-lv, nc, 1.0/max(uPbrEye.x, 1.0));
  if(dot(d, d) < 0.25) d = -lv;
  d.z = min(d.z, -0.2); d = normalize(d);                  // never skim along the iris (grazing: clamp)
  float dish = uEyeIris.z*uPbrEye3.z;
  float zI = zL - 0.5*dish;
  vec3 hit = q + d*((zI - q.z)/d.z);
  float s2 = min(dot(hit.xy, hit.xy)/(rI*rI), 1.0);
  hit = q + d*((zL - dish*(1.0 - s2) - q.z)/d.z);          // one step onto the concave dish
  float s = length(hit.xy)/rI;
  // pupil size (uPbrEye4.w, 1 = the map's): a radial remap of the iris, pupil edge pr -> pr*scale, rim fixed
  float pr = uEyeP.y/max(uEyeP.x, 1e-4), pk = clamp(pr*uPbrEye4.w, 0.05, 0.9);
  float sm = s < pk ? s*pr/pk : (s < 1.0 ? pr + (s - pk)*(1.0 - pr)/(1.0 - pk) : s);
  vec2 X = hit.xy*(sm/max(s, 1e-5))*(uEyeIris.w/rI);
  vec4 M = uEyeUV[k];
  vec2 uvR = 0.5 + vec2(dot(M.xy, X), dot(M.zw, X));
  float w = cap*smoothstep(1.25, 1.0, s)*clamp(uPbrEye3.y, 0.0, 1.0);
  uv = mix(uv, uvR, w);
  gEW = w; gEHit = hit;
}

float peHash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7)))*43758.5453); }
float peNoise(vec2 p){
  vec2 i = floor(p), f = fract(p); f = f*f*(3.0 - 2.0*f);
  return mix(mix(peHash(i), peHash(i + vec2(1.0, 0.0)), f.x), mix(peHash(i + vec2(0.0, 1.0)), peHash(i + vec2(1.0)), f.x), f.y);
}

vec4 matShade(SIn S, inout vec4 fragDOut){
  S.shw = pbrShadow(S.p, S.ng, S.shw);                                   // + the head cascade
  vec3 L = uKeyDir, v = S.v, ng = S.ng;
  int k = gEk;
  vec4 lid = peLids(S.p);
  bool frame = uEyeIris.x > 0.0;
  vec2 c = S.uv - 0.5;
  float rr = length(c)/max(uEyeP.x, 1e-4);                 // iris radii (refracted uv inside the cornea)
  float pr = uEyeP.y/max(uEyeP.x, 1e-4);
  float iris = smoothstep(1.04, 0.96, rr);
  float pupil = smoothstep(pr*1.08, pr*0.92, rr);
  vec3 lp = frame ? peLoc(S.p - uEyeC[k], k) : vec3(0.0, 0.0, 1.0);
  vec3 Ll = frame ? peLoc(L, k) : vec3(0.0, 0.0, 1.0);
  // ---------------- shared occlusion: lid contact AO + the upper lid's shade band (thickness + lashes over the ball)
  float aoT = smoothstep(0.0, 12.0, lid.x), aoB = smoothstep(0.0, 6.0, lid.y);
  float ao = mix(1.0, mix(0.30, 1.0, aoT)*mix(0.55, 1.0, aoB), uPbrEye2.y);
  float bandS = smoothstep(0.0, 18.0, lid.x);
  float band = mix(1.0, mix(0.22, 1.0, bandS*bandS), uPbrEye3.x);     // key + fill
  float bandA = mix(1.0, mix(0.55, 1.0, bandS), uPbrEye3.x);           // the room
  float shK = S.shw.x*band;
  // the orbit: the ball is recessed under the brow / beside the nose -> the room and the fill reach it only partly,
  // less toward the corners (deeper under the lids)
  float sideA = frame ? smoothstep(0.15, 0.85, abs(normalize(lp).x)) : 0.0;
  float sock = mix(1.0, 0.55 - 0.15*sideA, uPbrEye5.x);
  ao *= sock;
  // ---------------- iris (+ pupil)
  vec3 albI = S.alb;
  if(uPbrEye4.z > 0.0){
    // iris grade (a real amber-brown iris from a stylised map): the map's luminance STRUCTURE (fibres, crypts) with
    // its contrast compressed (no near-black outer band, no glowing collarette), on a LOW-saturation brown --
    // a subtly lighter amber collarette round the pupil, mid brown stroma, a gently darker rim -- plus fine radial
    // fibres (a few hash taps, no textures)
    float lum = dot(albI, vec3(0.30, 0.59, 0.11));
    float lref = 0.16;
    float m = clamp(pow(max(lum, 1e-4)/lref, 0.45), 0.55, 1.35);
    float ir = clamp((rr - pr)/max(1.0 - pr, 1e-3), 0.0, 1.0);          // 0 pupil edge .. 1 limbus
    vec3 coll = uPbrIrisC, mid = uPbrIrisM, rim = uPbrIrisR;            // the iris palette (pbr.eye.iris_*)
    vec3 br = mix(coll, mid, smoothstep(0.08, 0.38, ir));
    br = mix(br, rim, smoothstep(0.70, 1.0, ir));
    float an = atan(c.y, c.x);
    float fib = 0.55*peNoise(vec2(an*42.0, ir*3.0)) + 0.45*peNoise(vec2(an*97.0 + 7.0, ir*5.0));
    br *= m*mix(1.0 - 0.20*uPbrIrisF, 1.0 + 0.18*uPbrIrisF, fib);
    albI = mix(albI, br, clamp(uPbrEye4.z, 0.0, 1.0)*(1.0 - pupil));
  }
  albI *= uPbrEye2.z;
  albI = mix(albI, vec3(0.0025, 0.0022, 0.0020), pupil);
  // limbal ring: soft dark ring at the iris edge (width from the asset's limbus_width, as a fraction of the iris)
  float lw = clamp(uEyeP.w, 0.02, 0.25);
  float limb = smoothstep(1.0 - 2.2*lw, 1.0 - 0.3*lw, rr)*smoothstep(1.0 + 1.2*lw, 1.0 - 0.1*lw, rr);
  // iris surface normal: the concave dish (+ the map's own relief)
  vec3 nI = ng;
  vec3 LrI = L;
  float caus = 0.0, nearSh = 1.0, Tin = 1.0;
  if(frame){
    vec3 hp = gEW > 0.0 ? gEHit : lp;
    float rI = uEyeIris.x;
    vec2 u2 = hp.xy/rI;
    float slope = 2.0*uEyeIris.z*uPbrEye3.z/rI;
    vec3 nd = normalize(vec3(-u2*slope*1.0, 1.0));
    nI = normalize(peWld(nd, k) + (S.n - ng));
    // the key refracted at the cornea above this iris point
    float Rc = max(gEDome, 1e-6);
    vec2 xy = hp.xy;
    vec3 nc = normalize(vec3(xy, sqrt(max(Rc*Rc - dot(xy, xy), 1e-12))));
    float cl = dot(nc, Ll);
    vec3 lr = refract(-Ll, nc, 1.0/max(uPbrEye.x, 1.0));
    if(cl > 0.0 && dot(lr, lr) > 0.25) LrI = peWld(-lr, k);
    Tin = cl > 0.0 ? 1.0 - (0.025 + 0.975*pow(1.0 - cl, 5.0)) : 0.0;
    // caustic: the dome focuses the key onto the iris side AWAY from it; the limbus shades the near side
    float ob = length(Ll.xy);
    vec2 kd = Ll.xy/max(ob, 1e-4);
    float far = dot(-kd, u2);
    float obl = smoothstep(0.15, 0.75, ob)*smoothstep(-0.25, 0.25, Ll.z);
    caus = smoothstep(0.15, 0.85, far)*smoothstep(0.35, 0.9, length(u2))*obl;
    nearSh = 1.0 - 0.55*obl*smoothstep(0.45, 1.0, -far)*smoothstep(0.55, 1.0, length(u2));
  }
  float ndlI = dot(nI, LrI);
  vec3 EI = S.keyC*clamp((ndlI + 0.15)/1.15, 0.0, 1.0)*shK*Tin*nearSh;
  EI *= 1.0 + 0.9*uPbrEye.w*caus*vec3(1.05, 1.0, 0.94);
  // the iris sits mid-aperture: the lid-contact AO hardly reaches it; the orbit's share stays (floored) so an eye on
  // the shadow side still reads (dim, never a black hole)
  float aoI = max(mix(1.0, sock, 0.8)*mix(1.0, 0.75, 1.0 - aoT), 0.45);
  EI += uFillCol*clamp(dot(nI, uFillDir)*0.6 + 0.4, 0.0, 1.0)*mix(1.0, band, 0.5)*aoI + S.amb*aoI*mix(1.0, bandA, 0.5)*1.25;
  vec3 colI = albI*EI*(1.0 - uPbrEye.z*0.75*limb);
  // ---------------- sclera
  vec3 albS = S.alb*uPbrEye2.w;
  if(frame){
    // veins: faint, branching (ridged warped noise in the ball's angular coords), growing toward the corners
    vec3 dn = normalize(lp);
    float th = acos(clamp(dn.z, -1.0, 1.0));
    float ph = atan(dn.y, dn.x);
    vec2 q = vec2(ph*5.0, th*14.0);
    q += 1.3*vec2(peNoise(q*0.7 + 11.0), peNoise(q*0.7 + 37.0));
    float r1 = 1.0 - abs(2.0*peNoise(q) - 1.0);
    float r2 = 1.0 - abs(2.0*peNoise(q*2.3 + 5.0) - 1.0);
    float vein = pow(r1, 14.0) + 0.6*pow(r2, 18.0);
    float side = smoothstep(0.25, 0.9, abs(dn.x));
    float per = smoothstep(uEyeIris.w*1.15, uEyeIris.w*2.4, th);
    vein *= per*mix(0.35, 1.0, side)*uPbrEye2.x;
    albS = mix(albS, albS*vec3(0.82, 0.38, 0.34), clamp(vein, 0.0, 1.0));
    // the faint pinkish-grey toward the corners (conjunctival vessels, not a white sticker)
    albS *= mix(vec3(1.0), vec3(0.97, 0.88, 0.85), per*side*0.6*clamp(uPbrEye2.x*2.0, 0.0, 1.0));
  }
  albS = mix(albS, albS*vec3(1.0, 0.78, 0.74), lid.z*0.55);   // warmer / redder toward the caruncle
  // the sclera lit as the BALL (analytic sphere normal + the map's relief): the mesh's iris-dish wall, seen at the
  // limbus when the eye turns, would otherwise shade as a dark sideways-facing band
  vec3 nB = frame ? normalize(normalize(S.p - uEyeC[k]) + (S.n - ng)) : S.n;
  float nvB = clamp(dot(nB, v), 1e-4, 1.0);
  float ndl = dot(nB, L);
  vec3 wr = vec3(0.45, 0.28, 0.22)*uPbrEye3.w;                 // wrap: light scattered in the sclera (red further)
  vec3 dS = clamp((vec3(ndl) + wr)/(1.0 + wr), 0.0, 1.0);
  vec3 shC = mix(vec3(shK), vec3(S.shw.y*band), vec3(0.35, 0.15, 0.08)*uPbrEye3.w);
  vec3 ES = S.keyC*dS*shC;
  ES += uFillCol*clamp((dot(nB, uFillDir) + 0.3)/1.3, 0.0, 1.0)*band*ao + S.amb*ao*bandA;
  // translucent falloff toward the visible rim of the ball: less light comes back where the ball turns away
  ES *= mix(vec3(1.0), vec3(0.78, 0.70, 0.68), (1.0 - smoothstep(0.15, 0.75, nvB))*uPbrEye3.w);
  // the sclera turns away and under the lids toward both corners: darker, a touch warmer (key included)
  ES *= mix(vec3(1.0), vec3(0.62, 0.56, 0.54), sideA*uPbrEye5.y);
  vec3 colS = albS*ES;
  // the sclera right at the limbus: the ring's soft outer half
  colS *= 1.0 - uPbrEye.z*0.35*limb;
  vec3 col = mix(colS, colI, iris);
  // the tear meniscus: a thin concave water line on the ball along the lower lid margin (the cornea shell does not
  // reach the lids) -- its normal turns up and out, so it catches the room above and a sliver of the key
  float men = smoothstep(2.8, 0.8, lid.y)*smoothstep(-0.4, 0.4, lid.y)*uPbrEye4.y;
  if(men > 0.0){
    vec3 nm = normalize(ng + uHeadR[1]*0.9);
    vec3 rm = reflect(-v, nm);
    float Fm = 0.02 + 0.98*pow(1.0 - clamp(dot(nm, v), 0.0, 1.0), 5.0);
    vec3 em = uProbeP.x > 0.5 ? envProbe(rm, 0.08)*uAmbGain*uAmbTint : envSpec(S.p, rm);
    vec3 km = S.keyC*S.shw.x*specGGX(nm, v, L, 0.2)*PI;
    col += men*(em*max(Fm, 0.08)*1.5 + km*0.06);
  }
  // the wet film over the sclera where the cornea shell does not reach (near the lids): a faint room reflection
  {
    vec3 rw = reflect(-v, nB);
    float Fw = 0.025 + 0.975*pow(1.0 - nvB, 5.0);
    vec3 ew = uProbeP.x > 0.5 ? envProbe(rw, 0.1)*uAmbGain*uAmbTint : envSpec(S.p, rw);
    col += ew*Fw*(1.0 - iris)*smoothstep(10.0, 3.0, min(lid.x, lid.y + 2.0))*aoT*0.8;
  }
#ifndef EYE_HAS_CORNEA
  vec3 r = reflect(-v, ng);
  float Fr = 0.025 + 0.975*pow(1.0 - S.nv, 5.0);
  col += (envSpec(S.p, r)*Fr + S.keyC*shK*specGGX(ng, v, L, 0.08)*0.025*PI)*mix(0.3, 1.0, aoT);
#endif
  return vec4(col, 1.0);
}

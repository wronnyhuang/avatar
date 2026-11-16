// ---------------------------------------------------------------------------
// renderer/mesh/light.glsl — render-light: Juno's light terms in the forward pass (FS_FWD).  Spliced into
//  mesh.glsl at "#pragma include light" by character.py (renderer/mesh/light.py owns the uniforms).
//
//   lightShadow(p, ng)  key soft shadow, PCSS: a blocker search in the (texel-snapped, orthographic) key shadow map,
//                       then a Vogel-disc PCF whose radius = the penumbra of an area key of angular radius uLightP.x
//                       (contact-hardening: sharp where the shadow touches its caster, soft far from it).  Each tap is
//                       a hardware 2x2 compare (sampler2DShadow), the disc is rotated per pixel (interleaved
//                       gradient noise: fixed in screen space, no temporal crawl).  Returns (sharp, wide): `wide` = the
//                       same penumbra x2.2 (the skin's red channel scatters further).  Knobs: uLightP.w taps (0 =
//                       the legacy fixed PCF), uLightP.z max penumbra (texels).
//   capsuleAO(p, n)     analytic occlusion from the rig's capsule proxies (uCap: the same 18 capsules as Juno's SDF
//                       stand-in): cosine-weighted solid angle of each capsule's nearest sphere.  Chin on the neck,
//                       arms on the torso, hands on the thighs.  uCapP.x = strength (0 = off).
//   envProbe(r, rough)  the room's radiance from Juno's chest: a lat-long probe baked ONCE from the SDF set with the
//                       engine's own background pass (light.py), prefiltered into 4 roughness layers, calibrated to
//                       setAmbient's level so reflections sit in the same light as Muse and the props.
// ---------------------------------------------------------------------------
// capsule proxies: both the forward pass (AO on the mesh) and the composite (her contact shade on the scene)
uniform vec4  uCap[40];             // capsules (world): (a.xyz, r), (b.xyz, kind)
uniform vec4  uCapP;                // x AO strength on the mesh (0 = off), y floor of the AO, z capsule count,
                                    // w contact strength on the scene (composite)
#ifdef FS_FWD
uniform sampler2DShadow uShadowC;   // the key shadow map again, compare mode (hardware 2x2 PCF)
uniform float uLightB2;             // render2: minimum penumbra, texels (mesh.light.pen_min): the self-shadow at
                                    // the terminator is otherwise 1 texel hard -> a staircase along the facets
uniform float uLightB;              // render2: grazing normal offset (texels; mesh.light.term_bias)
uniform vec4  uLightP;              // x key angular radius (tan), y shadow depth range (world units per 1.0 depth),
                                    // z max penumbra (texels), w PCF taps (0 = legacy keyShadow)
uniform sampler2DArray uProbe;      // lat-long radiance, layers = roughness 0, 0.25, 0.5, 1.0
uniform vec4  uProbeP;              // x on, y gain (calibration x knob), z sharp-window mix (studio roomEnv), w -
float gEnvRough = 0.5;              // the material's roughness (main sets it before matShade): envSpec reads it

float lgIGN(vec2 p){ return fract(52.9829189*fract(dot(p, vec2(0.06711056, 0.00583715)))); }

vec2 lightShadow(vec3 p, vec3 ng){
  vec3 L = uKeyDir;
  float tx = uShadowTexel.y, range = max(uLightP.y, 1e-3);
  float ndl = dot(ng, L);
  float cs = clamp(abs(ndl), 0.08, 1.0);
  float tanT = min(sqrt(max(1.0 - cs*cs, 0.0))/cs, 6.0);
  // normal offset (one texel, more at grazing) + a slope-scaled depth bias: no acne, and small enough that the nose's
  // shadow still touches the nose (no peter-panning on the face)
  vec3 po = p + ng*tx*(0.8 + uLightB*(1.0 - cs));      // render2: more offset toward the terminator (facet acne)
  vec4 c = uLightVP*vec4(po, 1.0);
  vec3 s = c.xyz/c.w*0.5 + 0.5;
  if(any(lessThan(s.xy, vec2(0.0))) || any(greaterThan(s.xy, vec2(1.0))) || s.z >= 1.0) return vec2(1.0);
  float zr = s.z - tx*(0.7 + 0.6*tanT)/range;
  float rot = lgIGN(gl_FragCoord.xy)*6.2831853;
  float cr = cos(rot), sr = sin(rot);
  mat2 R = mat2(cr, sr, -sr, cr);
  // 1. blocker search (raw depths, 12 taps over the widest penumbra)
  float srch = uLightP.z*uShadowTexel.x;
  float bsum = 0.0, bn = 0.0;
  int NB = uLightP.w <= 8.5 ? 6 : 12;
  for(int i=0;i<12;i++){
    if(i >= NB) break;
    float rr = sqrt((float(i) + 0.5)/float(NB));
    float a = float(i)*2.39996;
    vec2 o = R*vec2(cos(a), sin(a))*rr*srch;
    float d = texture(uShadow, s.xy + o).r;
    if(d < zr){ bsum += d; bn += 1.0; }
  }
  if(bn < 0.5) return vec2(1.0);
  float db = bsum/bn;
  // 2. penumbra (texels) of an area key: receiver-to-blocker distance x the key's angular radius
  float pen = clamp((zr - db)*range*uLightP.x/tx, uLightB2, uLightP.z);   // render2: min penumbra (soft terminator)
  // 3. PCF: Vogel disc, hardware-filtered taps
  int N = int(uLightP.w);
  float a0 = 0.0, a1 = 0.0;
  float penW = min(pen*2.2 + 1.5, uLightP.z*1.6);
  for(int i=0;i<32;i++){
    if(i >= N) break;
    float rr = sqrt((float(i) + 0.5)/float(N));
    float a = float(i)*2.39996;
    vec2 d = R*vec2(cos(a), sin(a))*rr*uShadowTexel.x;
    a0 += texture(uShadowC, vec3(s.xy + d*pen, zr));
    a1 += texture(uShadowC, vec3(s.xy + d*penW, zr));
  }
  vec2 r = vec2(a0, a1)/float(max(N, 1));
  return r*r*(3.0 - 2.0*r);
}

#endif
// the summed occlusion of the capsules at p (normal n): 0 = open sky
float capsuleOcc(vec3 p, vec3 n){
  float occ = 0.0;
  int K = int(uCapP.z);
  for(int i=0;i<20;i++){
    if(i >= K) break;
    vec4 A = uCap[2*i], B = uCap[2*i + 1];
    float r = A.w;
    if(r <= 0.0) continue;
    vec3 ba = B.xyz - A.xyz;
    float h = clamp(dot(p - A.xyz, ba)/max(dot(ba, ba), 1e-8), 0.0, 1.0);
    vec3 d = A.xyz + ba*h - p;
    float l = length(d);
    float cosA = dot(n, d)/max(l, 1e-5);
    // a capsule's occlusion ~ its nearest sphere's, widened along the axis (x1.4); points inside a proxy (overlapping
    // joints, the capsule the point itself belongs to) do not count
    float o = clamp(cosA, 0.0, 1.0)*(r*r)/max(l*l, 1e-6)*1.4;
    o *= smoothstep(0.55*r, 1.0*r, l);
    occ += min(o, 1.0);
  }
  return occ;
}
// contact: how close p is to the nearest capsule surfaces (the cosine term above vanishes for a foot lying ON the floor:
// its axis is level with the floor point beside it).  0 = nothing within ~3-6 cm.
float capsuleContact(vec3 p){
  float c = 0.0;
  int K = int(uCapP.z);
  for(int i=0;i<20;i++){
    if(i >= K) break;
    vec4 A = uCap[2*i], B = uCap[2*i + 1];
    if(A.w <= 0.0) continue;
    vec3 ba = B.xyz - A.xyz;
    float h = clamp(dot(p - A.xyz, ba)/max(dot(ba, ba), 1e-8), 0.0, 1.0);
    float ds = length(A.xyz + ba*h - p) - A.w;
    float k = 0.035 + 0.9*A.w;
    float q = clamp(1.0 - ds/k, 0.0, 1.0);
    c += q*q;
  }
  return c;
}
float capsuleAO(vec3 p, vec3 n){
  if(uCapP.x <= 0.0) return 1.0;
  return max(1.0 - uCapP.x*capsuleOcc(p, n), uCapP.y);
}
#ifdef FS_FWD

vec3 envProbe(vec3 r, float rough){
  float u = atan(r.z, r.x)*(0.5/PI) + 0.5;
  float v = acos(clamp(r.y, -1.0, 1.0))/PI;
  float l = clamp(rough, 0.0, 1.0);
  // layers at roughness 0, 0.25, 0.5, 1.0
  float f = l < 0.25 ? l/0.25 : (l < 0.5 ? 1.0 + (l - 0.25)/0.25 : 2.0 + (l - 0.5)/0.5);
  float i0 = floor(min(f, 2.999));
  vec3 c0 = textureLod(uProbe, vec3(u, v, i0), 0.0).rgb, c1 = textureLod(uProbe, vec3(u, v, i0 + 1.0), 0.0).rgb;
  return mix(c0, c1, f - i0)*uProbeP.y;
}
#endif

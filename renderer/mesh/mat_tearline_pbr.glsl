// mat_tearline_pbr.glsl — shade-eyes: the wet meniscus along the lower lid (tearline_l/r), PBR path (pbr.py;
// gated by mesh.pbr.eye).  Blended, premultiplied: the tear film's reflection only (alpha = its Fresnel x coverage).
// A thin concave water strip: Fresnel (water F0 0.02), the key as an area light on a smooth lobe (GGX widened by the
// key's angular radius) through the shadow map, the room probe at a low roughness; uPbrEye4.y = strength.
uniform vec4  uPbrEye4;
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}
vec4 matShade(SIn S, inout vec4 fragDOut){
  S.shw = pbrShadow(S.p, S.ng, S.shw);                                   // + the head cascade
  vec3 L = uKeyDir, n = S.n, v = S.v;
  float nv = clamp(dot(n, v), 1e-3, 1.0);
  vec3 r = reflect(-v, n);
  float F = 0.02 + 0.98*pow(1.0 - nv, 5.0);
  float rough = 0.06;
  vec3 env = uProbeP.x > 0.5 ? envProbe(r, rough)*uAmbGain*uAmbTint : envSpec(S.p, r);
  float a = rough*rough, aK = min(a + 0.5*uLightP.x, 1.0);
  vec3 h = normalize(v + L);
  float nl = clamp(dot(n, L), 0.0, 1.0);
  float sp = D_GGX(clamp(dot(n, h), 0.0, 1.0), aK)*V_SmithJ(nv, nl, a)*nl;
  float Fk = 0.02 + 0.98*pow(1.0 - clamp(dot(v, h), 0.0, 1.0), 5.0);
  vec3 refl = env*F*1.3 + S.keyC*S.shw.x*sp*Fk*PI;
  float g = uPbrEye4.y;
  fragDOut = vec4(0.0);
  return vec4(refl*S.alpha*g, clamp(F*S.alpha*g, 0.0, 1.0));
}

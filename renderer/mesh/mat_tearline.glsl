// mat_tearline.glsl — render: the wet lower-lid line (tearline_l/r).  Blended, additive reflection.
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}
vec4 matShade(SIn S, inout vec4 fragDOut){   // inout (an out param is undefined until written: NVIDIA gave 0 -> mesh depth layer 0 -> no SSS, no depth)
  vec3 L = uKeyDir, n = S.n, v = S.v;
  vec3 r = reflect(-v, n);
  float Fr = 0.03 + 0.9*pow(1.0 - S.nv, 5.0);
  vec3 refl = envSpec(S.p, r)*Fr*1.2 + S.keyC*S.shw.x*specGGX(n, v, L, 0.05)*0.08*PI;
  fragDOut = vec4(0.0);
  return vec4(refl*S.alpha, 0.0);
}

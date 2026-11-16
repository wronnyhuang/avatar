// mat_lashes.glsl — render-face: lash CARDS (the asset's fallback; strand lashes go through the hair pass).
// MAT_LASHES.  The alpha is the card's strands (an opaque placeholder texture is replaced by eyesmouth.fix_lash_cards);
// blended premultiplied (order 2) so thin strands resolve soft, not as a solid band.
uniform vec4 uLashBox;
uniform vec4 uEyeK2;          // render5: w lash line warmth (dark brown, softer: not a black liner decal)        // the procedural card's uv box (eyesmouth.fix_lash_cards); zero = the asset's own texture
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){ if(uLashBox.z > 0.0) uv = (uv - uLashBox.xy)/uLashBox.zw; }
vec4 matShade(SIn S, inout vec4 fragDOut){   // inout (an out param is undefined until written: NVIDIA gave 0 -> mesh depth layer 0 -> no SSS, no depth)
  float a = clamp(S.alpha*2.2 - 0.05, 0.0, 1.0);   // mip-averaged strands: boost (a lash row reads dense)
  if(a < 0.02) discard;
  vec3 L = uKeyDir;
  vec3 col = S.alb*(S.keyC*clamp(dot(S.n, L)*0.5 + 0.5, 0.0, 1.0)*S.shw.x*0.5 + S.amb*0.5)*0.6 + rimLight(S)*0.3;
  col += S.keyC*specGGX(S.n, S.v, L, 0.35)*0.02*S.shw.x;
  a *= 1.0 - 0.3*uEyeK2.w;
  col = mix(col, max(col, vec3(0.045, 0.026, 0.017)*(S.keyC*0.5 + S.amb*2.0)), uEyeK2.w);
  return vec4(col*a, a);
}

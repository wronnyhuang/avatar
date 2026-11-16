// mat_trim.glsl — render-cloth: garment trims.  buttons (MAT_GENERIC define + TRIM_BUTTONS), shoes (TRIM_SHOES).
// Buttons: polished horn / resin: wrapped diffuse, GGX + a tight clear-coat lobe, room reflection, darker rim (cavity).
// Shoes: smooth leather: procedural fine pebble grain (object-scale value noise on the world point, faded by footprint),
// broad GGX with a soft sheen, grazing darkening.  uMatP: x roughness, y spec level, z clear-coat (buttons) / grain (shoes).
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}
vec4 matShade(SIn S, inout vec4 fragDOut){   // inout (an out param is undefined until written: NVIDIA gave 0 -> mesh depth layer 0 -> no SSS, no depth)
  vec3 L = uKeyDir, v = S.v, n = S.n;
  float rough = S.rough;
#if defined(TRIM_SHOES)
  // leather grain: 0.6 mm pebbles; uv-space (stable under motion), fades out when sub-pixel
  vec2 g = S.uv*vec2(900.0);
  float fp = max(length(dFdx(g)), length(dFdy(g)));
  float fd = (1.0 - smoothstep(0.4, 1.0, fp))*uMatP.z;
  float e = 0.5;
  float h0 = vnoise2(g), hx = vnoise2(g + vec2(e, 0.0)), hy = vnoise2(g + vec2(0.0, e));
  n = normalize(n - (S.T*(hx - h0) + S.B*(hy - h0))/e*0.25*fd);
  rough = clamp(rough + 0.12*(1.0 - fd), 0.05, 1.0);
#endif
  float nv = clamp(dot(n, v), 1e-4, 1.0);
  float ndl = dot(n, L);
  float dif = clamp((ndl + 0.2)/1.2, 0.0, 1.0);
  float sh = S.shw.x;
  vec3 diffuse = S.alb*(S.keyC*dif*sh + uFillCol*clamp(dot(n, uFillDir)*0.5 + 0.5, 0.0, 1.0)*0.5 + S.amb*S.ao);
  vec3 h = normalize(v + L);
  float vh = clamp(dot(v, h), 0.0, 1.0);
  vec3 F = F_Schlick(vec3(0.04*S.specL), vh);
  vec3 r = reflect(-v, n);
  vec3 col = diffuse*S.ao + S.keyC*F*specGGX(n, v, L, rough)*sh*PI;
  col += envSpec(S.p, r)*F_Schlick(vec3(0.04*S.specL), nv)*(1.0 - rough)*S.ao;
#if defined(TRIM_BUTTONS)
  float cc = uMatP.z;                                         // clear coat
  col += S.keyC*F_Schlick(vec3(0.04), vh)*specGGX(n, v, L, 0.08)*sh*PI*cc;
  col *= mix(1.0, 0.55, pow(1.0 - nv, 2.0));                  // the button's rolled edge reads darker
#else
  col *= mix(1.0, 0.75, pow(1.0 - nv, 3.0));
#endif
  col += rimLight(S)*0.7;
  return vec4(col*S.alpha, S.alpha);
}

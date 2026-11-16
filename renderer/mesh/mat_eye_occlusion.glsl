// mat_eye_occlusion.glsl — render: the soft shade where the lids meet the eye (eye_occl_l/r).  Blended:
// premultiplied black, alpha = shade.
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}
vec4 matShade(SIn S, inout vec4 fragDOut){   // inout (an out param is undefined until written: NVIDIA gave 0 -> mesh depth layer 0 -> no SSS, no depth)
  fragDOut = vec4(0.0);
  return vec4(vec3(0.0), S.alpha*0.75);
}

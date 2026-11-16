// mat_eye_occlusion_pbr.glsl — shade-eyes: the soft contact shade where the lids meet the eye (eye_occl_l/r), PBR
// path (pbr.py; gated by mesh.pbr.eye).  Blended premultiplied: the eye behind is darkened by alpha and gets back a
// little warm light bounced off the lid skin (the shade is the skin's, not a grey veil); uPbrEye4.x = strength.
uniform vec4  uPbrEye4;
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}
vec4 matShade(SIn S, inout vec4 fragDOut){
  fragDOut = vec4(0.0);
  float a = clamp(S.alpha*0.8*uPbrEye4.x, 0.0, 0.95);
  vec3 bounce = S.amb*vec3(0.30, 0.14, 0.10)*0.35;
  return vec4(bounce*a, a);
}

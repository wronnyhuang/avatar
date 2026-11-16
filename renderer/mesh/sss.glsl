// ---------------------------------------------------------------------------
// renderer/mesh/sss.glsl — render-skin: screen-space separable subsurface scattering (M5).
//  Input  uS   (resolved MSAA attachment 2 of the forward pass): rgb = skin DIFFUSE radiance (premultiplied by the skin
//              coverage), a = 1 - skin coverage (blend-safe: blended occluders over skin attenuate it like colour).
//         uMD  the resolved mesh depth layer (t*a, ., ., a).
//  H pass (uPass 0): rgb = per-channel normalised blur along x, a = the green weight sum (skin fraction under the kernel)
//  V pass (uPass 1): normalised blur of the H result along y, * the centre's skin coverage -> ADDED onto the char layer
//  Kernel (sss.py): Burley-shaped per channel + a delta (the unscattered share), offsets in [-1,1] of the half range;
//  the half range in pixels = uWorldR (scene units) projected at the pixel's depth; depth-aware (gaussian in t).
// ---------------------------------------------------------------------------
#ifdef VS_SSS
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
#endif

#ifdef FS_SSS
uniform sampler2D uS;        // H: the resolved skin buffer; V: the H result
uniform sampler2D uS0;       // the resolved skin buffer (V: centre coverage)
uniform sampler2D uMD;
uniform vec2  uRes;
uniform vec2  uDir;          // (1,0) or (0,1)
uniform int   uPass;
uniform int   uN;
uniform vec4  uK[17];        // xyz weight per channel (each channel sums to 1), w = offset in [-1, 1]
uniform float uFlPx;         // camera focal length in pixels per unit (fl*res.y/2)
uniform float uWorldR;       // kernel half range, scene units
uniform float uMaxPx;        // clamp of the half range in pixels
uniform float uDepthK;       // 1/(2 sigma_t^2), scene units
out vec4 fragColor;
#ifdef SSS_ALB
uniform sampler2D uAlb;      // shade: resolved skin albedo * skin coverage (rgb), skin coverage (a)
#endif
// render6: finite guard (mesh.glsl jFin): a NaN / Inf / negative texel in the S buffer or the depth layer must not
// spread through the separable blur (one bad pixel -> a sharp (2 rpx)^2 black rectangle on the face: PERF2's A40 blobs)
#ifdef JUNO_NOGUARD
vec4 jFin(vec4 x, float lo, float hi){ return x; }
#else
vec4 jFin(vec4 x, float lo, float hi){ return clamp(mix(x, vec4(0.0), bvec4(vec4(isnan(x)) + vec4(isinf(x)))), lo, hi); }
#endif
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  vec4 d0 = jFin(texelFetch(uMD, ip, 0), 0.0, 6.0e4);
  if(d0.a < 0.004) discard;
  float t0 = d0.r/d0.a;
  float mc = 1.0 - jFin(texelFetch(uS0, ip, 0), 0.0, 1.0).a;     // skin coverage of this pixel
  if(uPass == 1 && mc < 0.002) discard;
  float rpx = min(uWorldR*uFlPx/max(t0, 1e-3), uMaxPx);
  vec2 px = gl_FragCoord.xy;
  vec3 acc = vec3(0.0), wac = vec3(0.0);
  for(int i=0;i<17;i++){
    if(i >= uN) break;
    vec4 k = uK[i];
    vec2 uv = (px + uDir*(k.w*rpx))/uRes;
    vec4 s = jFin(texture(uS, uv), 0.0, 6.0e4);
    s.a = min(s.a, 1.0);                               // H: 1 - skin coverage; V: the skin fraction (both 0..1)
    vec4 d = jFin(texture(uMD, uv), 0.0, 6.0e4);
    float t = d.a > 0.004 ? d.r/d.a : 1e4;
    float dz = t - t0;
    float wz = exp(-dz*dz*uDepthK);
    if(uPass == 0){
      float m = 1.0 - s.a;
      acc += k.xyz*wz*s.rgb;                        // s.rgb is already diffuse * coverage
      wac += k.xyz*wz*m;
    } else {
      acc += k.xyz*wz*s.rgb*s.a;
      wac += k.xyz*wz*s.a;
    }
  }
  vec3 r = jFin(vec4(acc/max(wac, vec3(1e-5)), 0.0), 0.0, 6.0e4).rgb;
  if(uPass == 0){
    if(wac.g < 1e-4 && wac.r < 1e-4) discard;
    fragColor = vec4(r, clamp(wac.g, 0.0, 1.0));
  } else {
#ifdef SSS_ALB
    vec4 al = jFin(texelFetch(uAlb, ip, 0), 0.0, 1.0);   // shade: post-scatter albedo (S carries irradiance)
    r *= al.rgb/max(al.a, 1e-3);
#endif
    fragColor = vec4(r*mc, 0.0);                       // added (ONE, ONE) onto the premultiplied char layer
  }
}
#endif

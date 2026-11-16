// pbr_common.glsl — shade: shared light terms of the PBR materials (spliced before every mat_*_pbr.glsl).
//   faceShadow(p, ng)  the key's shadow from the HEAD cascade (pbr.py FaceShadow: ~0.3 mm texels), PCSS with the key's
//                      angular size (uLightP.x) -> (sharp, wide) like light.glsl lightShadow; vec2(1) outside it
//   pbrShadow(p, ng, body)  min(body map, face map), faded to the body map at the cascade's border
uniform sampler2D uFaceSh;
uniform sampler2DShadow uFaceShC;
uniform mat4  uFaceVP;
uniform vec4  uFaceShP;       // x on, y texel (world), z depth range (world per 1.0), w PCF taps
vec3 faceShadowW(vec3 p, vec3 ng){            // (sharp, wide, weight)
  if(uFaceShP.x < 0.5) return vec3(1.0, 1.0, 0.0);
  vec3 L = uKeyDir;
  float tx = uFaceShP.y, range = max(uFaceShP.z, 1e-4);
  float cs = clamp(abs(dot(ng, L)), 0.08, 1.0);
  float tanT = min(sqrt(max(1.0 - cs*cs, 0.0))/cs, 6.0);
  vec3 po = p + ng*tx*(1.0 + 1.5*(1.0 - cs));
  vec4 c = uFaceVP*vec4(po, 1.0);
  vec3 s = c.xyz/c.w*0.5 + 0.5;
  vec2 e2 = min(s.xy, 1.0 - s.xy);
  float wgt = smoothstep(0.0, 0.08, min(e2.x, e2.y));
  if(wgt <= 0.0 || s.z >= 1.0) return vec3(1.0, 1.0, 0.0);
  float zr = s.z - tx*(1.0 + 1.2*tanT)/range;
  float rot = fract(52.9829189*fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))))*6.2831853;
  mat2 R = mat2(cos(rot), sin(rot), -sin(rot), cos(rot));
  float texUV = 1.0/float(textureSize(uFaceSh, 0).x);
  // blocker search over the widest penumbra (key angular size x up to ~8 cm of blocker distance)
  float maxPen = clamp(0.08*uLightP.x/tx, 2.0, 48.0);
  float bsum = 0.0, bn = 0.0;
  for(int i=0;i<10;i++){
    float rr = sqrt((float(i) + 0.5)/10.0), a = float(i)*2.39996;
    vec2 o = R*vec2(cos(a), sin(a))*rr*maxPen*texUV;
    float d = texture(uFaceSh, s.xy + o).r;
    if(d < zr){ bsum += d; bn += 1.0; }
  }
  if(bn < 0.5) return vec3(1.0, 1.0, wgt);
  float db = bsum/bn;
  float pen = clamp((zr - db)*range*uLightP.x/tx, 1.0, maxPen);
  int N = int(uFaceShP.w);
  float a0 = 0.0, a1 = 0.0;
  for(int i=0;i<24;i++){
    if(i >= N) break;
    float rr = sqrt((float(i) + 0.5)/float(N)), a = float(i)*2.39996;
    vec2 d = R*vec2(cos(a), sin(a))*rr*texUV;
    a0 += texture(uFaceShC, vec3(s.xy + d*pen, zr));
    a1 += texture(uFaceShC, vec3(s.xy + d*(pen*2.0 + 2.0), zr));
  }
  vec2 r = vec2(a0, a1)/float(max(N, 1));
  return vec3(r*r*(3.0 - 2.0*r), wgt);
}
vec2 pbrShadow(vec3 p, vec3 ng, vec2 body){
  vec3 f = faceShadowW(p, ng);
  return mix(body, min(body, f.xy), f.z);
}

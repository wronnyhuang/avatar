// ---------------------------------------------------------------------------
// mesh/post_pbr.glsl -- shade-post: the final image for a physically based character (junor).  GATED: compiled
//  and used only when the mesh config has mesh.pbr.post (engine._post_init); every other character keeps its post.
//
//  Sections (one file, one #define each; engine.PbrPost assembles them):
//    PBR_BLOOM_DOWN  13-tap (Jimenez / CoD:AW) downsample of the HDR frame into a mip pyramid, Karis-averaged on the
//                    first level (a single specular firefly cannot flash the whole bloom)
//    PBR_BLOOM_UP    3x3 tent upsample + the level below it (the dual-filter / "physically based bloom" chain)
//    PBR_POST        the frame:  scene-linear  (exposure, optical vignette, energy-preserving bloom = a LERP toward the
//                    blurred image, not an add)  ->  AgX (Rec.2020 working space, log2 encoding, sigmoid, ASC-CDL look)
//                    with an Oklab hue restore (saturated skin / mustard keep their hue instead of skewing toward
//                    yellow) -> display linear grade (saturation, gain, lift) -> piecewise sRGB OETF -> fine
//                    luminance-weighted grain that doubles as the 8-bit dither.
//  Uniforms of PBR_POST (the shared grade keeps its meaning):
//    uExposure, uSaturation, uGain, uLift, uVignette, uGrain   juno.json "grade" (exposure = a scene-linear multiplier)
//    uPP   x exposure (stops, added), y bloom mix (0..1, energy preserving), z vignette (optical, x uVignette), w grain k
//    uPP2  x look slope, y look power (contrast about AgX mid), z look saturation, w hue restore (0..1)
//    uPP3  x black floor (display linear, the toe: punchy, never crushed), y highlight desat start, z ca px, w debug
//    uPP4  x white headroom (stops above mid grey, AgX 6.5; less = brighter highlights), y shadow range (stops, AgX 10)
// ---------------------------------------------------------------------------
uniform vec2 uRes;
out vec4 fragColor;
vec3 pFin(vec3 x, float hi){ return clamp(mix(x, vec3(0.0), bvec3(vec3(isnan(x)) + vec3(isinf(x)))), 0.0, hi); }

#ifdef PBR_BLOOM_DOWN
uniform sampler2D uSrc;
uniform vec2 uSrcTexel;        // 1 / source size
uniform float uKaris;          // 1 on the first level
vec3 S(vec2 uv){ return pFin(textureLod(uSrc, uv, 0.0).rgb, 6.0e4); }
float kw(vec3 c){ return 1.0/(1.0 + dot(c, vec3(0.2126, 0.7152, 0.0722))); }
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec2 t = uSrcTexel;
  vec3 a = S(uv + t*vec2(-2, 2)), b = S(uv + t*vec2(0, 2)), c = S(uv + t*vec2(2, 2));
  vec3 d = S(uv + t*vec2(-2, 0)), e = S(uv), f = S(uv + t*vec2(2, 0));
  vec3 g = S(uv + t*vec2(-2, -2)), h = S(uv + t*vec2(0, -2)), i = S(uv + t*vec2(2, -2));
  vec3 j = S(uv + t*vec2(-1, 1)), k = S(uv + t*vec2(1, 1)), l = S(uv + t*vec2(-1, -1)), m = S(uv + t*vec2(1, -1));
  vec3 o;
  if(uKaris > 0.5){
    // Karis average per 2x2 group (weights 1/(1+luma)): fireflies lose their weight
    vec3 g0 = (j + k + l + m)*0.25, g1 = (a + b + d + e)*0.25, g2 = (b + c + e + f)*0.25, g3 = (d + e + g + h)*0.25,
         g4 = (e + f + h + i)*0.25;
    float w0 = kw(g0)*0.5, w1 = kw(g1)*0.125, w2 = kw(g2)*0.125, w3 = kw(g3)*0.125, w4 = kw(g4)*0.125;
    o = (g0*w0 + g1*w1 + g2*w2 + g3*w3 + g4*w4)/(w0 + w1 + w2 + w3 + w4);
  } else {
    o = e*0.125 + (a + c + g + i)*0.03125 + (b + d + f + h)*0.0625 + (j + k + l + m)*0.125;
  }
  fragColor = vec4(o, 1.0);
}
#endif

#ifdef PBR_BLOOM_UP
uniform sampler2D uLow;        // the coarser level (already upsampled chain)
uniform sampler2D uCur;        // this level of the down chain
uniform vec2 uLowTexel;
uniform float uMixLow;         // weight of the wider levels (the bloom's falloff shape)
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec2 t = uLowTexel;
  vec3 s = textureLod(uLow, uv, 0.0).rgb*4.0
         + (textureLod(uLow, uv + vec2(t.x, 0), 0.0).rgb + textureLod(uLow, uv - vec2(t.x, 0), 0.0).rgb
          + textureLod(uLow, uv + vec2(0, t.y), 0.0).rgb + textureLod(uLow, uv - vec2(0, t.y), 0.0).rgb)*2.0
         + textureLod(uLow, uv + t, 0.0).rgb + textureLod(uLow, uv - t, 0.0).rgb
         + textureLod(uLow, uv + vec2(t.x, -t.y), 0.0).rgb + textureLod(uLow, uv + vec2(-t.x, t.y), 0.0).rgb;
  s /= 16.0;
  vec3 c = textureLod(uCur, uv, 0.0).rgb;
  fragColor = vec4(mix(c, s, uMixLow), 1.0);
}
#endif

#ifdef PBR_POST
uniform int   uFrame;
uniform sampler2D uHdr;
uniform sampler2D uBloomT;      // the upsampled bloom (half res), scene linear
uniform float uExposure, uBloom, uSaturation, uVignette, uGrain, uFade;
uniform vec3  uGain, uLift;
uniform vec4  uPP, uPP2, uPP3, uPP4;
float h12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }

// ---- AgX (Sobotka; Blender 4 / Filament constants), Rec.2020 working space
const mat3 SRGB_TO_2020 = mat3(vec3(0.6274, 0.0691, 0.0164), vec3(0.3293, 0.9195, 0.0880), vec3(0.0433, 0.0113, 0.8956));
const mat3 R2020_TO_SRGB = mat3(vec3(1.6605, -0.1246, -0.0182), vec3(-0.5876, 1.1329, -0.1006), vec3(-0.0728, -0.0083, 1.1187));
const mat3 AGX_IN = mat3(vec3(0.856627153315983, 0.137318972929847, 0.11189821299995),
                         vec3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
                         vec3(0.0482516061458583, 0.101439036467562, 0.811302368396859));
const mat3 AGX_OUT = mat3(vec3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
                          vec3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
                          vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
vec3 agxCurve(vec3 x){
  vec3 x2 = x*x, x4 = x2*x2;
  return 15.5*x4*x2 - 40.14*x4*x + 31.96*x4 - 6.868*x2*x + 0.4298*x2 + 0.1191*x - 0.00232;
}
vec3 agx(vec3 c){
  c = AGX_IN*(SRGB_TO_2020*max(c, 0.0));
  // log2 encoding about mid grey 0.18: uPP4.x stops of white headroom, uPP4.y stops of shadow (AgX: 6.5 / 10)
  float emin = log2(0.18) - uPP4.y, emax = log2(0.18) + uPP4.x;
  c = clamp((log2(max(c, 1e-10)) - emin)/(emax - emin), 0.0, 1.0);
  c = agxCurve(c);
  // ASC-CDL look in the AgX display encoding (Blender's "Punchy" = power 1.35, sat 1.4); here gentle and knob-driven
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = pow(max(c*uPP2.x, 0.0), vec3(uPP2.y));
  float l2 = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = l2 + uPP2.z*(c - l2);
  c = AGX_OUT*c;
  c = pow(max(c, 0.0), vec3(2.2));                    // the AgX encoding is display gamma 2.2 -> display linear
  return R2020_TO_SRGB*c;
}
// ---- Oklab (Ottosson) for the hue restore
vec3 toOklab(vec3 c){
  vec3 lms = mat3(vec3(0.4122214708, 0.2119034982, 0.0883024619), vec3(0.5363325363, 0.6806995451, 0.2817188376),
                  vec3(0.0514459929, 0.1073969566, 0.6299787005))*max(c, 0.0);
  lms = pow(lms + 1e-9, vec3(1.0/3.0));
  return mat3(vec3(0.2104542553, 1.9779984951, 0.0259040371), vec3(0.7936177850, -2.4285922050, 0.7827717662),
              vec3(-0.0040720468, 0.4505937099, -0.8086757660))*lms;
}
vec3 fromOklab(vec3 L){
  vec3 lms = mat3(vec3(1.0, 1.0, 1.0), vec3(0.3963377774, -0.1055613458, -0.0894841775),
                  vec3(0.2158037573, -0.0638541728, -1.2914855480))*L;
  lms = lms*lms*lms;
  return mat3(vec3(4.0767416621, -1.2684380046, -0.0041960863), vec3(-3.3077115913, 2.6097574011, -0.7034186147),
              vec3(0.2309699292, -0.3413193965, 1.7076147010))*lms;
}
vec3 srgbOETF(vec3 c){
  c = clamp(c, 0.0, 1.0);
  return mix(12.92*c, 1.055*pow(c, vec3(1.0/2.4)) - 0.055, step(vec3(0.0031308), c));
}
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec3 col;
  if(uPP3.z > 0.0){                       // lateral CA (px at 1080 rows, grows with r^2); 0 = off
    vec2 q = uv - 0.5;
    vec2 o = q*dot(q, q)*4.0*uPP3.z*(uRes.y/1080.0)/uRes;
    col = vec3(textureLod(uHdr, uv + o, 0.0).r, textureLod(uHdr, uv, 0.0).g, textureLod(uHdr, uv - o, 0.0).b);
  } else {
    col = textureLod(uHdr, uv, 0.0).rgb;
  }
  col = pFin(col, 6.0e4);
  vec3 bl = pFin(textureLod(uBloomT, uv, 0.0).rgb, 6.0e4);
  // energy preserving: a share of every pixel's light is spread by the lens (no threshold, no added light)
  col = mix(col, bl, clamp(uPP.y*uBloom, 0.0, 1.0));
  // optical vignette (cos^4-like falloff in scene light, before the tone curve: highlights roll, not grey)
  vec2 qa = uv - 0.5; qa.x *= uRes.x/uRes.y;
  float r2 = dot(qa, qa);
  col *= mix(1.0, 1.0/((1.0 + r2*1.6)*(1.0 + r2*1.6)), clamp(uVignette*uPP.z, 0.0, 2.0));
  col *= uExposure*exp2(uPP.x);
  vec3 tm = agx(col);
  // hue restore: the tone-mapped colour keeps its lightness + chroma, its hue turns back toward the scene's own hue
  // (fades out toward white, where a highlight should desaturate)
  if(uPP2.w > 0.0){
    vec3 Li = toOklab(col/(1.0 + dot(col, vec3(0.2126, 0.7152, 0.0722)))), Lo = toOklab(max(tm, 0.0));
    float ci = length(Li.yz), co = length(Lo.yz);
    if(ci > 1e-4 && co > 1e-4){
      float k = uPP2.w*(1.0 - smoothstep(uPP3.y, 1.0, Lo.x));
      vec2 hdir = normalize(mix(Lo.yz/co, Li.yz/ci, k));
      Lo.yz = hdir*co;
      tm = fromOklab(Lo);
    }
  }
  tm = max(tm, 0.0);
  // display linear grade: saturation (protected at black / white), gain, lift, the black floor (toe)
  float l = dot(tm, vec3(0.2126, 0.7152, 0.0722));
  float sp = smoothstep(0.0, 0.05, l)*smoothstep(1.0, 0.8, l);
  tm = max(mix(vec3(l), tm, mix(1.0, uSaturation, sp)), 0.0);
  tm = tm*uGain + uLift*(1.0 - tm);
  tm = uPP3.x + (1.0 - uPP3.x)*tm;
  vec3 o = srgbOETF(tm)*uFade;
  // grain: fine, luminance weighted (mid-tones), mostly monochrome; the pattern cross-fades over 3 frames (no 30 Hz
  // fizz in the encoder) and its amplitude also covers the 8-bit quantisation (dither)
  float lg = dot(o, vec3(0.299, 0.587, 0.114));
  float gw = 0.35 + 2.6*lg*(1.0 - lg);
  float ft = float(uFrame)/3.0, f0 = floor(ft), fa = fract(ft);
  vec2 p = gl_FragCoord.xy;
  float n0 = h12(p + f0*17.31) + h12(p.yx*1.37 + f0*5.13) - 1.0;
  float n1 = h12(p + (f0 + 1.0)*17.31) + h12(p.yx*1.37 + (f0 + 1.0)*5.13) - 1.0;
  float g = mix(n0, n1, fa)*inversesqrt(fa*fa + (1.0 - fa)*(1.0 - fa));
  float gc = h12(p*0.71 + f0*3.7) - 0.5;
  o += (g*gw*vec3(1.0) + gc*vec3(0.25, -0.1, -0.2)*gw)*(uGrain*uPP.w/255.0);
  if(uPP3.w > 0.5) o = vec3(textureLod(uBloomT, uv, 0.0).rgb);
  fragColor = vec4(clamp(o, 0.0, 1.0), 1.0);
}
#endif

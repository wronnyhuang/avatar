// ---------------------------------------------------------------------------
//  engine/post.glsl — screen-space passes (no scene knowledge).
//    PASS_CHARDOF  per frame: small DoF over the character, premultiplied
//    PASS_COMP     per frame: live background + god rays + character
//    PASS_POST     per frame: bloom, ACES, grade, vignette, grain (rgb, a ignored)
// ---------------------------------------------------------------------------
uniform vec2  uRes;
uniform float uFocus;
uniform float uDofK;
uniform int   uFrame;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }

#ifdef PASS_CHARDOF
// Premultiplied gather inside the character's rectangle.  rgb+a from uChar, depth from uCharD.
#define MAXC 7.0
layout(location=0) out vec4 fragC;
layout(location=1) out vec4 fragD;
uniform sampler2D uChar, uCharD;
uniform float uPxScale, uInvSpp;
float coc(float d){ return clamp(uDofK*abs(1.0/uFocus - 1.0/max(d,0.05)), 0.0, MAXC)*uPxScale; }
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec4 c0 = texture(uChar, uv)*uInvSpp;
  vec4 dd = texture(uCharD, uv)*uInvSpp;
  float d0 = c0.a > 0.001 ? dd.r/c0.a : uFocus;
  float cc = max(coc(d0), 0.45*uPxScale);      // 0.45 px: the lens, a soft edge-aware AA
  const int N = 18;
  float rot0 = hash12(gl_FragCoord.xy + float(uFrame)*0.37)*6.2831;
  vec4 acc = c0*1.5; float wsum = 1.5; vec2 dAcc = dd.rg*1.5;
  for(int i=0;i<N;i++){
    float r = sqrt((float(i)+0.5)/float(N));
    float a = float(i)*2.39996 + rot0;
    vec2 dir = vec2(cos(a), sin(a));
    vec2 uv2 = uv + dir*r*cc/uRes;
    vec4 s = texture(uChar, uv2)*uInvSpp;
    vec4 sd = texture(uCharD, uv2)*uInvSpp;
    acc += s; dAcc += sd.rg; wsum += 1.0;
  }
  acc /= wsum; dAcc /= wsum;
  fragC = acc;
  fragD = vec4(dAcc.r, 0.0, 0.0, 1.0);
}
#endif

#ifdef PASS_COMP
out vec4 fragColor;
uniform sampler2D uLive, uVol, uChar, uCharD;
uniform float uHasChar;
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec4 bg = texture(uLive, uv);
  vec4 ch = texture(uChar, uv)*uHasChar;
  float dd = texture(uCharD, uv).r;
  vec2 px = 3.0/uRes;      // the volume is 1/3 res: tent filter it
  vec3 vol = 0.25*(texture(uVol, uv + px*vec2(0.75,0.25)).rgb + texture(uVol, uv + px*vec2(-0.25,0.75)).rgb
                 + texture(uVol, uv + px*vec2(-0.75,-0.25)).rgb + texture(uVol, uv + px*vec2(0.25,-0.75)).rgb);
  float d = mix(bg.a, ch.a > 0.001 ? dd/ch.a : bg.a, ch.a);
  fragColor = vec4(bg.rgb*(1.0 - ch.a) + ch.rgb + vol*(1.0 - 0.85*ch.a), d);
}
#endif

#ifdef PASS_POST
out vec4 fragColor;
uniform sampler2D uHdr;
uniform float uExposure, uBloom, uSaturation, uVignette, uGrain, uFade;
uniform vec3 uGain, uLift;
//look-v2{
#ifdef LOOK_V2
uniform float uClarity, uContrast;
#endif
//look-v2}
vec3 aces(vec3 x){ const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14; return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.,1.); }
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec3 col = textureLod(uHdr, uv, 0.0).rgb;
  vec3 bl = vec3(0.0);
  bl += textureLod(uHdr, uv, 2.0).rgb*0.30;
  bl += textureLod(uHdr, uv, 3.0).rgb*0.25;
  bl += textureLod(uHdr, uv, 4.0).rgb*0.20;
  bl += textureLod(uHdr, uv, 5.0).rgb*0.15;
  bl += textureLod(uHdr, uv, 6.0).rgb*0.10;
//look-v2{  (stripped from the background cache key: the frame's grade only)
#ifdef LOOK_V2
  // clarity: a little local contrast against the first blur level, before the bloom mix (no ringing: clamped)
  vec3 soft = textureLod(uHdr, uv, 1.5).rgb;
  col += clamp(col - soft, -0.08, 0.08)*uClarity;
#endif
//look-v2}
  col = mix(col, bl, 0.07*uBloom) + max(bl - 1.6, 0.0)*0.12*uBloom;
  col *= uExposure;
  col = aces(col);
  float l = dot(col, vec3(0.2126,0.7152,0.0722));
  col = mix(vec3(l), col, uSaturation);
  col = col*uGain + uLift*(1.0 - col);
  vec2 q = uv - 0.5; q.x *= uRes.x/uRes.y;
  col *= 1.0 - uVignette*pow(length(q)*1.15, 2.6);
  col = pow(clamp(col,0.0,1.0), vec3(1.0/2.2));
//look-v2{
#ifdef LOOK_V2
  // a gentle S curve about mid grey: deeper shade, the lights kept
  col = mix(col, col*col*(3.0 - 2.0*col), uContrast);
#endif
//look-v2}
  col *= uFade;
  col += (hash12(gl_FragCoord.xy + float(uFrame)*17.31) - 0.5)*(uGrain/255.0);
  fragColor = vec4(col, 1.0);
}
#endif

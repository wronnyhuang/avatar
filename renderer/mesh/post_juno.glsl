// ---------------------------------------------------------------------------
// mesh/post_juno.glsl — render5: Juno's own final grade (replaces engine/post.glsl PASS_POST for the mesh
//  backend only when juno.json mesh.post is set; Muse and every SDF character keep the shared post untouched).
//  Same inputs/uniforms as the shared post (uHdr with its mip chain, exposure, saturation, gain/lift, vignette, grain)
//  plus uPostK / uPostK2:
//    uPostK.x  highlight bloom gain (soft-kneed threshold, no veil over the frame)
//    uPostK.y  chromatic softness at the frame edges (px at 1080 rows, grows with r^2)
//    uPostK.z  mid-contrast S after the tone curve (0 = none)
//    uPostK.w  shoulder: highlights roll off softer (0 = the plain ACES fit)
//    uPostK2.x hue-kept share of the tone curve (saturated colour keeps its depth in the light)
//    uPostK2.y grain mids weighting (grain strongest in the mid-tones, never in black / white)
//    uPostK2.z bloom threshold (HDR)
//    uPostK2.w halation veil (wide, tiny)
// ---------------------------------------------------------------------------
uniform vec2  uRes;
uniform int   uFrame;
uniform sampler2D uHdr;
uniform float uExposure, uBloom, uSaturation, uVignette, uGrain, uFade;
uniform float uClarity, uContrast;
uniform vec3  uGain, uLift;
uniform vec4  uPostK, uPostK2;
out vec4 fragColor;
vec3 jFin3(vec3 x, float lo, float hi){ return clamp(mix(x, vec3(0.0), bvec3(vec3(isnan(x)) + vec3(isinf(x)))), lo, hi); }   // render6
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
vec3 acesN(vec3 x){ const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14; return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.,1.); }
float acesN1(float x){ const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14; return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.,1.); }
vec3 tone(vec3 c){
  c = max(c, 0.0);
  // soft shoulder: compress the top before the curve (skin highlights / the window roll off instead of clipping)
  float sh = uPostK.w;
  if(sh > 0.0){
    float m = max(c.r, max(c.g, c.b));
    float k = 0.8;
    float mm = m > k ? k + (m - k)/(1.0 + sh*(m - k)) : m;
    c *= mm/max(m, 1e-5);
  }
  vec3 pc = acesN(c);
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  vec3 lc = c*(acesN1(l)/max(l, 1e-5));
  vec3 m = mix(pc, lc, uPostK2.x*smoothstep(1.0, 0.6, max(lc.r, max(lc.g, lc.b))));
  return clamp(m, 0.0, 1.0);
}
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec2 q = uv - 0.5;
  vec3 col = textureLod(uHdr, uv, 0.0).rgb;
  // chromatic softness: red/blue a fraction of a pixel apart toward the frame edge (lens, not a glitch)
  if(uPostK.y > 0.0){
    vec2 o = q*dot(q, q)*4.0*uPostK.y*(uRes.y/1080.0)/uRes;
    col.r = textureLod(uHdr, uv + o, 0.0).r;
    col.b = textureLod(uHdr, uv - o, 0.0).b;
  }
  vec3 soft = textureLod(uHdr, uv, 1.5).rgb;
  col += clamp(col - soft, -0.08, 0.08)*uClarity;
  vec3 b1 = textureLod(uHdr, uv, 2.0).rgb, b2 = textureLod(uHdr, uv, 3.0).rgb, b3 = textureLod(uHdr, uv, 4.0).rgb,
       b4 = textureLod(uHdr, uv, 5.0).rgb, b5 = textureLod(uHdr, uv, 6.0).rgb;
  float th = uPostK2.z;
  vec3 bl = max(b1 - th, 0.0)*0.30 + max(b2 - th*0.85, 0.0)*0.25 + max(b3 - th*0.70, 0.0)*0.20
          + max(b4 - th*0.60, 0.0)*0.15 + max(b5 - th*0.50, 0.0)*0.10;
  col += bl*uPostK.x*uBloom;
  col = mix(col, 0.30*b1 + 0.25*b2 + 0.20*b3 + 0.15*b4 + 0.10*b5, uPostK2.w*uBloom);
  col *= uExposure;
  col = tone(col);
  float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
  float sp = smoothstep(0.0, 0.10, l)*smoothstep(1.0, 0.75, l);
  col = max(mix(vec3(l), col, mix(1.0, uSaturation, sp)), 0.0);
  col = col*uGain + uLift*(1.0 - col);
  vec2 qa = q; qa.x *= uRes.x/uRes.y;
  col *= 1.0 - uVignette*pow(length(qa)*1.15, 2.6);
  col = pow(clamp(col, 0.0, 1.0), vec3(1.0/2.2));
  col = mix(col, col*col*(3.0 - 2.0*col), clamp(uContrast + uPostK.z, 0.0, 1.0));
  col *= uFade;
  // grain: two decorrelated hashes (a finer, less digital texture), weighted toward the mid-tones
  float lg = dot(col, vec3(0.299, 0.587, 0.114));
  float gw = mix(1.0, 4.0*lg*(1.0 - lg), uPostK2.y);
  float g = (hash12(gl_FragCoord.xy + float(uFrame)*17.31) + hash12(gl_FragCoord.yx*1.37 + float(uFrame)*5.13) - 1.0);
  col += g*gw*(uGrain/255.0);
  fragColor = vec4(jFin3(col, 0.0, 1.0), 1.0);        // render6: a NaN never reaches the encoder (black, not garbage)
}

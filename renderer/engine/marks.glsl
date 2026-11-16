// ---------------------------------------------------------------------------
// engine/marks.glsl — EMO-R2: the muse's floating cartoon emotion marks (T8/emo/r2/PARAMS2.md):
//  ? ! hearts sparkles steam scribble dots shake-lines.  Instanced camera-facing quads (one per mark piece), placed
//  from the head pose the POSE pass already writes (4x1 texels: world->head rotation + head origin), drawn into the
//  HDR frame after the composite: they move with the head (bob, lean, tilt), sit at the head's depth (a paw in front
//  hides them), stay sharp (no DoF / TAA history) and cost nothing when every mark param is 0 (no draw at all).
//  Compiled as both stages: rt_engine adds `#define VS` or `#define FS`.  fx_green is a face paint (muse.glsl).
// ---------------------------------------------------------------------------
uniform vec2  uRes;
uniform float uTime;
uniform float uP[NPARAMS];
uniform vec3  uCamPos, uCamU, uCamV, uCamW;
uniform float uFl;
uniform vec3  uSunCol;
uniform float uHeadR;          // the head's radius, world units (the offsets below are in head radii)

#define MK_N 28
// instance layout: 0-1 ?, 2 !, 3-5 hearts, 6-11 sparkles, 12-17 steam, 18 scribble, 19-21 dots, 22-27 lines
#define K_Q 0
#define K_EX 1
#define K_HEART 2
#define K_SPARK 3
#define K_STEAM 4
#define K_SCRIB 5
#define K_DOTS 6
#define K_LINES 7
#ifndef DOTS_O
#define DOTS_O vec2(0.05 + 0.42*fs, 1.00 + 0.02*sin(T*1.5 + fs))   // above the head, a little right of centre (clear of ? and !)
#define DOTS_S 0.19
#endif

float mkHash(float p){ p = fract(p*.1031); p *= p+33.33; p *= p+p; return fract(p); }
mat2 mkRot(float a){ float c=cos(a), s=sin(a); return mat2(c,s,-s,c); }

#ifdef VS
uniform sampler2D uPose;
flat out int vKind;
flat out float vA, vSeed;
out vec2 vUv;
out float vDepth;
void main(){
  int id = gl_InstanceID;
  int kind; int sub;
  if(id < 2){ kind = K_Q; sub = id; }
  else if(id < 3){ kind = K_EX; sub = 0; }
  else if(id < 6){ kind = K_HEART; sub = id - 3; }
  else if(id < 12){ kind = K_SPARK; sub = id - 6; }
  else if(id < 18){ kind = K_STEAM; sub = id - 12; }
  else if(id < 19){ kind = K_SCRIB; sub = 0; }
  else if(id < 22){ kind = K_DOTS; sub = id - 19; }
  else { kind = K_LINES; sub = id - 22; }
  float a = clamp(uP[kind == K_LINES ? P_FX_LINES : P_FX_QUESTION + kind], 0.0, 1.0);   // (fx_green, between, is a face paint)
  float T = uTime, fs = float(sub);
  vec2 o = vec2(0.0);          // offset from the head centre, head radii, screen axes (x right, y up)
  float sz = 0.0, ang = 0.0, al = 0.0;
  float grow = 0.45 + 0.55*smoothstep(0.0, 0.6, a);       // a mark grows in with its param, never pops
  float fade = smoothstep(0.0, 0.35, a);
  if(kind == K_Q){
    float second = sub == 1 ? smoothstep(0.55, 0.85, a) : 1.0;
    o = sub == 0 ? vec2(1.18, 0.55) : vec2(1.55, 0.92);
    o.y += 0.05*sin(T*2.1 + fs*1.7);
    ang = 0.16*sin(T*1.3 + fs*2.0) - 0.18 + 0.30*fs;
    sz = (sub == 0 ? 0.40 : 0.26)*grow*(0.6 + 0.4*second);
    al = fade*second;
  } else if(kind == K_EX){
    o = vec2(-1.12, 0.58);
    float sp = sin(T*5.5);
    ang = 0.15 + 0.10*sp;
    o.y += 0.03*sin(T*3.1);
    sz = 0.42*grow*(1.0 + 0.04*sin(T*7.0));
    al = fade;
  } else if(kind == K_HEART){
    float n = 1.0 + 2.0*smoothstep(0.25, 0.95, a);           // 1 -> 3 hearts
    float vis = clamp(n - fs, 0.0, 1.0);
    float per = 2.8, ph = fract(T/per + fs*0.37 + 0.11);
    o = vec2(0.95 + 0.22*fs*0.6, 0.05) + vec2(0.10*sin(ph*6.2832*1.3 + fs*2.0) + 0.06*fs, 1.05*ph);
    if(sub == 1) o.x = -o.x + 0.05;                          // the second heart rises on the other side
    ang = 0.25*sin(ph*6.2832 + fs) + (sub == 1 ? 0.25 : -0.20);
    sz = (sub == 0 ? 0.30 : 0.24)*grow*(0.75 + 0.25*smoothstep(0.0, 0.25, ph));
    al = fade*vis*smoothstep(0.0, 0.12, ph)*smoothstep(1.0, 0.70, ph);
  } else if(kind == K_SPARK){
    vec2 P6[6] = vec2[6](vec2(-1.05, 0.55), vec2(1.08, 0.48), vec2(-0.86, 0.98), vec2(0.92, 0.95), vec2(-1.22, -0.05), vec2(1.20, 0.02));
    float n = 2.0 + 4.0*smoothstep(0.15, 0.95, a);
    float vis = clamp(n - fs, 0.0, 1.0);
    o = P6[sub];
    float tw = 0.5 + 0.5*sin(T*3.2 + fs*2.39);
    ang = 0.25*sin(T*0.8 + fs);
    sz = (sub < 2 ? 0.30 : 0.22)*grow*(0.55 + 0.45*tw);
    al = fade*vis*(0.55 + 0.45*tw);
  } else if(kind == K_STEAM){
    float sd = sub < 3 ? 1.0 : -1.0; float k = mod(fs, 3.0);
    float per = 1.6, ph = fract(T/per + k/3.0);
    o = vec2(sd*(0.62 + 0.55*ph), 0.50 + 0.30*ph + 0.06*sin(ph*9.0 + k));
    ang = sd*0.4*ph;
    sz = (0.14 + 0.22*ph)*grow*(0.7 + 0.3*a);
    al = fade*smoothstep(0.0, 0.15, ph)*smoothstep(1.0, 0.55, ph);
  } else if(kind == K_SCRIB){
    o = vec2(-0.72, 1.06) + 0.03*vec2(sin(T*1.7), cos(T*1.3));   // above the head, left of centre
    ang = 0.0;
    sz = 0.46*grow;
    al = fade;
  } else if(kind == K_DOTS){
    // r2-sig (scene judge: 'dots pop in'): a slow cycle (4 s); each dot grows in over ~0.8 s, all fade over ~0.8 s, and
    // they never drop below a floor (no blink to empty)
    float per = 4.0, ph = fract(T/per);
    float on = smoothstep(0.04 + 0.15*fs, 0.24 + 0.15*fs, ph)*smoothstep(1.0, 0.80, ph);
    float lo = 0.6*smoothstep(0.25, 0.9, a);
    o = DOTS_O;   // trailing off beside the mouth: speechless, not 'thinking'
    ang = 0.0;
    sz = DOTS_S*grow*(0.6 + 0.4*max(on, lo));
    al = fade*max(on, lo);
  } else {
    float sd = sub < 3 ? 1.0 : -1.0; float k = mod(fs, 3.0);
    float th = (0.36 - 0.36*k);                               // angle up the side of the head (radians)
    float sh = 0.05*sin(T*11.0 + k*2.1 + sd);                 // the shake
    float r = 1.12 + sh + 0.04*k;
    o = vec2(sd*r*cos(th), r*sin(th) + 0.05);
    ang = sd > 0.0 ? th : -th;
    sz = 0.32*grow;
    al = fade*(0.70 + 0.30*sin(T*2.4 + k*1.9 + sd*1.3));
  }
  vKind = kind; vA = a; vSeed = fs;
  if(a <= 0.0 || al <= 0.002 || sz <= 0.0){ gl_Position = vec4(4.0, 4.0, 0.0, 1.0); vUv = vec2(0.0); vDepth = 0.0; return; }
  // the head pose (rt_engine POSE pass): R maps world -> head, texel 3 = the head origin in the world
  vec3 c0 = texelFetch(uPose, ivec2(0, 0), 0).xyz, c1 = texelFetch(uPose, ivec2(1, 0), 0).xyz, c2 = texelFetch(uPose, ivec2(2, 0), 0).xyz;
  vec3 ho = texelFetch(uPose, ivec2(3, 0), 0).xyz;
  vec3 hy = vec3(c0.y, c1.y, c2.y);                            // the head's up axis in the world (row 1 of R)
  float roll = atan(dot(hy, uCamU), max(dot(hy, uCamV), 0.2)); // its tilt on screen
  vec3 hc = ho + hy*0.04*uHeadR;
  vec2 ro = mkRot(roll)*o;                                      // offsets turn with the head's tilt
  vec3 C = hc + (uCamU*ro.x + uCamV*ro.y)*uHeadR - uCamW*0.35*uHeadR;
  vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID>>1) & 1))*2.0 - 1.0;
  vec2 q = mkRot(ang + roll)*corner*sz*uHeadR;
  // keep the whole mark inside the frame: shift its centre (in the camera plane) when the push-in crowds it
  vec3 v = C - uCamPos; float z = max(dot(v, uCamW), 0.05);
  vec2 ndc = vec2(dot(v, uCamU), dot(v, uCamV))*uFl/z;          // y in [-1,1], x in [-aspect, aspect]
  float asp = uRes.x/uRes.y, hs = 1.42*sz*uHeadR*uFl/z, m = 0.03;
  vec2 lim = vec2(asp, 1.0) - hs - m;
  vec2 cl = clamp(ndc, -lim, lim);
  C += (uCamU*(cl.x - ndc.x) + uCamV*(cl.y - ndc.y))*z/uFl;
  vec3 p = C + uCamU*q.x + uCamV*q.y;
  vec3 vv = p - uCamPos; float zz = max(dot(vv, uCamW), 0.05);
  gl_Position = vec4(dot(vv, uCamU)*uFl/zz*uRes.y/uRes.x, dot(vv, uCamV)*uFl/zz, 0.0, 1.0);
  vUv = corner; vDepth = length(C - uCamPos);
  vA = al;
}
#endif

#ifdef FS
uniform sampler2D uChar, uCharD;
uniform float uInvSpp;
flat in int vKind;
flat in float vA, vSeed;
in vec2 vUv;
in float vDepth;
out vec4 fragColor;

float sdSeg(vec2 p, vec2 a, vec2 b){ vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0); return length(pa - ba*h); }
float sdHeart(vec2 p){
  // iq's heart (point at the bottom), unit height ~1.1, centred
  p.y += 0.55; p.x = abs(p.x);
  if(p.y + p.x > 1.0) return sqrt(dot(p - vec2(0.25, 0.75), p - vec2(0.25, 0.75))) - sqrt(2.0)/4.0;
  return sqrt(min(dot(p - vec2(0.00, 1.00), p - vec2(0.00, 1.00)), dot(p - 0.5*max(p.x + p.y, 0.0), p - 0.5*max(p.x + p.y, 0.0))))*sign(p.x - p.y);
}
float sdArc(vec2 p, vec2 c, float r, float a0, float a1){
  // distance to the arc of radius r about c from angle a0 to a1 (radians, a0 < a1, ccw)
  vec2 d = p - c; float a = atan(d.y, d.x);
  float mid = 0.5*(a0 + a1), hw = 0.5*(a1 - a0);
  float da = mod(a - mid + 3.14159265, 6.2831853) - 3.14159265;
  if(abs(da) <= hw) return abs(length(d) - r);
  vec2 e0 = c + r*vec2(cos(a0), sin(a0)), e1 = c + r*vec2(cos(a1), sin(a1));
  return min(length(p - e0), length(p - e1));
}

void main(){
  vec2 p = vUv*1.2;            // the glyph space: about [-1.2, 1.2]
  float aa = max(fwidth(p.x), fwidth(p.y))*0.9;
  float d = 1e3;               // the glyph's signed distance (fill < 0)
  vec3 fill = vec3(1.0), ink = vec3(0.025, 0.018, 0.015);
  float outl = 0.11, glow = 0.0;
  float T = uTime;
  if(vKind == K_Q){
    float w = 0.15;
    float arc = sdArc(p, vec2(0.0, 0.40), 0.40, -1.5708, 3.0) - w;
    float stem = sdSeg(p, vec2(0.0, 0.0), vec2(0.0, -0.30)) - w;
    float dot_ = length(p - vec2(0.0, -0.70)) - 0.17;
    d = min(min(arc, stem), dot_);
    fill = vec3(0.95, 0.97, 1.0);
  } else if(vKind == K_EX){
    vec2 pa = p - vec2(0.0, -0.18); float h = clamp((pa.y)/0.95, 0.0, 1.0);
    float bar = length(vec2(pa.x, pa.y - clamp(pa.y, 0.0, 0.95))) - mix(0.11, 0.21, h);
    float dot_ = length(p - vec2(0.0, -0.66)) - 0.18;
    d = min(bar, dot_);
    fill = vec3(0.95, 0.03, 0.02);
  } else if(vKind == K_HEART){
    d = sdHeart(p*1.05);
    fill = mix(vec3(0.98, 0.06, 0.22), vec3(1.0, 0.36, 0.52), smoothstep(0.2, 0.7, p.y + 0.3*p.x));
    outl = 0.09;
  } else if(vKind == K_SPARK){
    vec2 a2 = abs(p)/0.95;
    float s = pow(a2.x, 0.5) + pow(a2.y, 0.5) - 1.0;       // the four-point star (astroid)
    d = s*0.45;
    fill = mix(vec3(1.0, 0.86, 0.35), vec3(1.0, 1.0, 0.92), smoothstep(0.35, 0.0, length(p)));
    outl = 0.0; glow = 1.0;
  } else if(vKind == K_STEAM){
    float c1 = length(p - vec2(-0.28, -0.10)) - 0.50;
    float c2 = length(p - vec2(0.30, -0.05)) - 0.46;
    float c3 = length(p - vec2(0.02, 0.32)) - 0.48;
    d = min(min(c1, c2), c3);
    fill = vec3(0.97, 0.97, 0.98)*(0.90 + 0.10*smoothstep(-0.3, 0.5, p.y));
    ink = vec3(0.55, 0.55, 0.60); outl = 0.07;
  } else if(vKind == K_SCRIB){
    // a tangled ball: a few overlapping wobbly loops, slowly churning
    float l = 1e3;
    for(int i = 0; i < 6; i++){
      float fi = float(i);
      vec2 c = 0.30*vec2(sin(fi*2.4 + T*0.9), cos(fi*1.7 + T*0.7));
      vec2 r = vec2(0.42 + 0.18*mkHash(fi*3.1), 0.26 + 0.14*mkHash(fi*5.7));
      vec2 q = mkRot(fi*1.1 + T*(0.6 + 0.2*fi))*(p - c);
      float e = length(q/r);
      l = min(l, abs(e - 1.0)*min(r.x, r.y));
    }
    d = l - 0.060;
    fill = vec3(0.03, 0.025, 0.03); outl = 0.0;
  } else if(vKind == K_DOTS){
    d = length(p) - 0.78;
    fill = vec3(0.03, 0.025, 0.025); ink = vec3(0.80, 0.78, 0.74); outl = 0.09;    // a light rim: reads on the dark TV too
  } else {
    // a short curved stroke bowing away from the head
    vec2 q = vec2(p.y, p.x);
    float s = sdArc(q, vec2(0.0, -1.6), 1.6, 1.5708 - 0.50, 1.5708 + 0.50) - 0.17*(1.0 - 0.5*abs(q.x));
    s = min(s, sdArc(q, vec2(0.0, -2.05), 1.6, 1.5708 - 0.40, 1.5708 + 0.40) - 0.13*(1.0 - 0.5*abs(q.x)));   // a second, shorter: a vibration
    d = s;
    fill = vec3(0.03, 0.025, 0.02); outl = 0.0;
  }
  float cov = smoothstep(aa, -aa, d - outl);                    // fill + outline
  float inner = smoothstep(aa, -aa, d);
  vec3 col = mix(ink, fill, outl > 0.0 ? inner : 1.0);
  float a = cov;
  if(glow > 0.0){                                               // sparkles: a soft halo around the star
    float h = exp(-length(p)*3.2)*0.55;
    a = max(cov, h);
    col = mix(vec3(1.0, 0.92, 0.6), col, cov/max(a, 1e-4));
  }
  a *= vA;
  if(a <= 0.001) discard;
  // a paw (or anything of the character) in front of the mark hides it
  vec2 uv = gl_FragCoord.xy/uRes;
  vec4 ch = texture(uChar, uv)*uInvSpp;
  if(ch.a > 0.02){
    float cd = texture(uCharD, uv).r*uInvSpp/ch.a;
    a *= 1.0 - ch.a*smoothstep(-0.02, 0.02, vDepth - cd);
  }
  // lit by the set a little: the key's colour, never dark
  col *= mix(vec3(1.0), uSunCol/max(max(uSunCol.r, uSunCol.g), uSunCol.b), 0.25)*1.05;
  fragColor = vec4(col*a, a);
}
#endif

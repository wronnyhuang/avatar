// ---------------------------------------------------------------------------
// renderer/mesh/taa.glsl -- shade-taa: temporal AA resolve of the MESH layer (GATED: mesh.pbr.taa; taa.py).
//  Runs after the MSAA resolve + SSS on the resolved, premultiplied mesh layer (t_mc) at the forward pass's size.
//  history (premultiplied rgba) reprojected by per-pixel motion: the forward pass's velocity attachment (skinned mesh:
//  previous posed vertex + previous camera), the head bone's rigid motion for hair (strands never write velocity),
//  3x3 closest-depth dilation; YCoCg+alpha variance clipping against this frame's 3x3; disocclusion by depth /
//  velocity change; luma-weighted blend.  Out: the new history (colour) + aux (velocity, depth, coverage).
// ---------------------------------------------------------------------------
#ifdef VS_TAA
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
#endif

#ifdef FS_TAA
uniform sampler2D uCur;      // resolved mesh layer, premultiplied colour (after SSS)
uniform sampler2D uCurD;     // (t*a, ., ., a)
uniform sampler2D uVel;      // resolved velocity attachment (v*cov, t*cov, cov)
uniform sampler2D uHist;     // previous output colour (premultiplied)
uniform sampler2D uHistA;    // previous aux: (velocity px, t, a)
uniform vec2  uRes;          // forward-pass size (all five textures)
uniform vec4  uK;            // x blend (new frame weight), y clip gamma, z velocity reject (px), w reset (1 = no history)
uniform vec4  uK2;           // x hair depth eps (scene units), y depth reject (scene units, 0 = off), z luma weight k, w catmull-rom (1)
uniform vec4  uK3;           // x motion blend (extra new weight at 1 px/frame motion), y max blend, z unjitter (1), w static blend
uniform vec4  uK4;           // x static clip gamma, y motion (px/frame) at which the moving values take over, z reactive blend, w reactive gamma
uniform vec2  uJitPx;        // this frame's jitter in pixels
uniform sampler2D uCls;      // shade2-taa: the resolved class attachment ((react, sharpen, gamma/4)*a, a)
uniform float uClsOn;        // 1 = classes configured (else uCls is not bound)
uniform mat4  uHeadRel;      // world now -> world previous (the head bone's rigid motion)
uniform vec3  uCamPos, uCamU, uCamV, uCamW;
uniform float uFl;
uniform vec3  uPCamPos, uPCamU, uPCamV, uPCamW;
uniform float uPFl;
layout(location=0) out vec4 oC;
layout(location=1) out vec4 oA;

vec4 fin(vec4 x){ return clamp(mix(x, vec4(0.0), bvec4(vec4(isnan(x)) + vec4(isinf(x)))), -6.0e4, 6.0e4); }
vec4 toY(vec4 c){ return vec4(dot(c.rgb, vec3(0.25, 0.5, 0.25)), dot(c.rgb, vec3(0.5, 0.0, -0.5)), dot(c.rgb, vec3(-0.25, 0.5, -0.25)), c.a); }
vec4 fromY(vec4 y){ return vec4(y.x + y.y - y.z, y.x + y.z, y.x - y.y - y.z, y.w); }
vec3 rayAt(vec2 pix){ vec2 uv = (2.0*pix - uRes)/uRes.y; return normalize(uv.x*uCamU + uv.y*uCamV + uFl*uCamW); }
vec2 prevPix(vec3 p){
  vec3 d = p - uPCamPos;
  float z = max(dot(d, uPCamW), 1e-4);
  vec2 ndc = vec2(uPFl*dot(d, uPCamU)*uRes.y/uRes.x, uPFl*dot(d, uPCamV))/z;
  return (ndc*0.5 + 0.5)*uRes;
}
// Catmull-Rom history fetch from 5 bilinear taps (Jimenez 2016): keeps the history sharp under motion
vec4 histCR(vec2 uv){
  vec2 sp = uv*uRes, tp = floor(sp - 0.5) + 0.5, f = sp - tp;
  vec2 w0 = f*(-0.5 + f*(1.0 - 0.5*f)), w1 = 1.0 + f*f*(-2.5 + 1.5*f), w2 = f*(0.5 + f*(2.0 - 1.5*f)), w3 = f*f*(-0.5 + 0.5*f);
  vec2 w12 = w1 + w2, o12 = w2/w12;
  vec2 t0 = (tp - 1.0)/uRes, t3 = (tp + 2.0)/uRes, t12 = (tp + o12)/uRes;
  vec4 r = texture(uHist, vec2(t12.x, t0.y))*(w12.x*w0.y) + texture(uHist, vec2(t0.x, t12.y))*(w0.x*w12.y)
         + texture(uHist, t12)*(w12.x*w12.y) + texture(uHist, vec2(t3.x, t12.y))*(w3.x*w12.y)
         + texture(uHist, vec2(t12.x, t3.y))*(w12.x*w3.y);
  float ws = w12.x*w0.y + w0.x*w12.y + w12.x*w12.y + w3.x*w12.y + w12.x*w3.y;
  return max(r/ws, vec4(0.0));
}
// clip q toward the box centre (not a per-channel clamp: keeps the colour's hue)
vec4 clipBox(vec4 q, vec4 lo, vec4 hi){
  vec4 c = 0.5*(hi + lo), e = 0.5*(hi - lo) + 1e-5;
  vec4 v = q - c;
  vec4 u = abs(v/e);
  float m = max(max(u.x, u.y), max(u.z, u.w));
  return m > 1.0 ? c + v/m : q;
}

void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  ivec2 sz = ivec2(uRes) - 1;
  // empty neighbourhood (the room around her): 4 bilinear taps cover the 3x3 -> nothing to resolve (the clip box
  // would be [0, 0] anyway); most of a framing's pixels leave here
  vec2 px = 1.0/uRes, cq = gl_FragCoord.xy*px;
  float a4 = texture(uCurD, cq + vec2(-0.5, -0.5)*px).a + texture(uCurD, cq + vec2(0.5, -0.5)*px).a
           + texture(uCurD, cq + vec2(-0.5, 0.5)*px).a + texture(uCurD, cq + vec2(0.5, 0.5)*px).a;
  if(!(a4 > 1e-5)){ oC = vec4(0.0); oA = vec4(0.0); return; }
  vec4 cur = fin(texelFetch(uCur, ip, 0));
  // ---- this frame's 3x3: moments (YCoCg + alpha), the box, the closest covered pixel (velocity dilation)
  vec4 m1 = vec4(0.0), m2 = vec4(0.0), mn = vec4(1e9), mx = vec4(-1e9);
  float best = 1e9; ivec2 bp = ip; float cov = 0.0;
  // unjittered current: a tent over the 3x3 centred on this pixel's true centre (the samples sit at +jitter)
  vec4 cu = vec4(0.0); float cw = 0.0;
  for(int j = -1; j <= 1; j++) for(int i = -1; i <= 1; i++){
    ivec2 q = clamp(ip + ivec2(i, j), ivec2(0), sz);
    vec4 s = fin(texelFetch(uCur, q, 0));
    vec4 y = toY(s);
    m1 += y; m2 += y*y; mn = min(mn, y); mx = max(mx, y);
    vec2 dq = vec2(i, j) - uJitPx;          // sample q shows the surface at q - jitter
    float wq = max(0.0, 1.0 - abs(dq.x))*max(0.0, 1.0 - abs(dq.y));
    cu += s*wq; cw += wq;
    vec4 d = fin(texelFetch(uCurD, q, 0));
    if(d.a > 0.05){
      float t = d.r/d.a;
      cov = max(cov, d.a);
      if(t < best){ best = t; bp = q; }
    }
  }
  if(uK3.z > 0.5 && cw > 1e-3) cur = cu/cw;
  // ---- motion of the surface at this pixel
  vec2 v = vec2(0.0);
  float tl = best;
  float react = 0.0;
  if(cov > 0.0){
    vec4 vv = fin(texelFetch(uVel, bp, 0));
    bool hair = true;
    if(vv.a > 0.05){
      float tm = vv.b/vv.a;
      if(tm < 0.0){ react = 1.0; tm = -tm; }
      if(tl > tm - uK2.x){ v = vv.rg/vv.a; hair = false; }      // the skinned mesh is the front surface here
    }
    if(hair){                                                    // a strand (or anything not in the velocity pass)
      vec2 pc = vec2(bp) + 0.5;
      vec3 p = uCamPos + rayAt(pc)*tl;
      vec3 pp = (uHeadRel*vec4(p, 1.0)).xyz;
      v = pc - prevPix(pp);
    }
  }
  vec2 uvp = (gl_FragCoord.xy - v)/uRes;
  // a still surface: accumulate longer + a looser box (the jitter's own sample spread never clips the history)
  float mf = smoothstep(0.0, uK4.y, length(v));
  float gam = mix(mix(uK4.x, uK.y, mf), uK4.w, react);
  float rr = 0.0;                                                // shade2-taa: the material class
  if(uClsOn > 0.5){
    vec4 cl = texelFetch(uCls, ip, 0);
    if(cl.a > 0.05){ rr = cl.r/cl.a; gam *= 4.0*cl.b/cl.a; }
  }
  vec4 mu = m1/9.0, sg = sqrt(max(m2/9.0 - mu*mu, 0.0));
  vec4 lo = max(mn, mu - gam*sg), hi = min(mx, mu + gam*sg);
  float w = mix(uK3.w, uK.x, mf);
  vec4 outc;
  bool off = any(lessThan(uvp, vec2(0.0))) || any(greaterThan(uvp, vec2(1.0)));
  if(uK.w > 0.5 || off){
    outc = cur; w = 1.0;
  } else {
    vec4 h = uK2.w > 0.5 ? histCR(uvp) : texture(uHist, uvp);
    vec4 ha = texture(uHistA, uvp);
    // disocclusion: the history's velocity / depth at the reprojected spot disagree with this surface
    float dv = length(v - ha.xy);
    float rej = smoothstep(0.5*uK.z, uK.z, dv);
    if(uK2.y > 0.0 && ha.w > 0.05 && cov > 0.0)
      rej = max(rej, smoothstep(uK2.y, 2.0*uK2.y, abs(ha.z - tl)));
    // motion: resampled history softens -> lean on the new frame a little more while it moves
    float mv = length(v);
    w = clamp(w + uK3.x*mv, w, max(uK3.y, w));
    w = max(w, react*uK4.z);
    w = max(w, rr);
    w = mix(w, 1.0, rej);
    vec4 hy = clipBox(toY(h), lo, hi);
    vec4 hc = fromY(hy);
    hc = vec4(max(hc.rgb, 0.0), clamp(hc.a, 0.0, 1.0));
    // luma weighting (Karis): bright fireflies never dominate the average
    float lc = toY(cur).x, lh = hy.x;
    float wc = w/(1.0 + uK2.z*max(lc, 0.0)), wh = (1.0 - w)/(1.0 + uK2.z*max(lh, 0.0));
    outc = (cur*wc + hc*wh)/max(wc + wh, 1e-6);
  }
  oC = fin(outc);
  oA = vec4(v, tl < 1e8 ? tl : 0.0, cov);
}
#endif

#ifdef FS_SHARP
// shade2-taa: CAS-like sharpen of the resolved mesh layer (premultiplied rgba), clamped to the cross's min/max (no
// halos, no overshoot past the neighbourhood); strength uSharp.x x the pixel's class sharpen.  Never fed back to history.
uniform sampler2D uIn;
uniform sampler2D uCls;
uniform vec2 uSharp;         // x strength, y classes on
uniform vec2 uRes;
layout(location=0) out vec4 oC;
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy), sz = ivec2(uRes) - 1;
  vec4 c = texelFetch(uIn, ip, 0);
  vec4 n = texelFetch(uIn, clamp(ip + ivec2(0, 1), ivec2(0), sz), 0), s = texelFetch(uIn, clamp(ip - ivec2(0, 1), ivec2(0), sz), 0);
  vec4 e = texelFetch(uIn, clamp(ip + ivec2(1, 0), ivec2(0), sz), 0), w = texelFetch(uIn, clamp(ip - ivec2(1, 0), ivec2(0), sz), 0);
  vec4 mx = max(max(max(n, s), max(e, w)), c);
  if(!(mx.a > 1e-5)){ oC = c; return; }
  vec4 mn = min(min(min(n, s), min(e, w)), c);
  float k = uSharp.x;
  if(uSharp.y > 0.5){
    vec4 cl = texelFetch(uCls, ip, 0);
    k *= cl.a > 0.05 ? cl.g/cl.a : 1.0;
  }
  vec4 o = c + k*(c - 0.25*(n + s + e + w));
  oC = clamp(o, mn, mx);
}
#endif

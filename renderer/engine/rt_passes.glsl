// ---------------------------------------------------------------------------
//  engine/rt_passes.glsl — the real-time engine's own passes (engine/rt_engine.py).  The scene, the character and
//  the lighting are the v2 shaders unchanged; these passes only replace the CPU round trip and the full-screen
//  bookkeeping around them:
//    PASS_MASK     1/8 res: 3x3 min of the prepass -> march start (r), R1 = tiles that march (g), R2 = R1 + 1 ring (b)
//    VS_CHARPTS    vertex shader of the character pass: one 8x8 point per R2 tile, culled on the GPU (no readback)
//    VS_TILEQUAD   instanced quads over the R2 tiles at output resolution (the temporal resolve)
//    PASS_RESOLVE  temporal accumulation / upsampling of the character layer (premultiplied rgba + depth)
//    PASS_COMPDOF  the character's small DoF gather fused into the composite (no full-res DoF targets)
// ---------------------------------------------------------------------------
uniform vec2  uRes;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }

// ===========================================================================
#ifdef PASS_MASK
uniform sampler2D uPre;
uniform ivec2 uTiles;
out vec4 fragColor;
void main(){
  ivec2 tc = ivec2(gl_FragCoord.xy);
  float k3 = 1e6, k5 = 1e6;
  for(int dy=-2; dy<=2; dy++) for(int dx=-2; dx<=2; dx++){
    ivec2 q = tc + ivec2(dx, dy);
    if(q.x < 0 || q.y < 0 || q.x >= uTiles.x || q.y >= uTiles.y) continue;
    float v = texelFetch(uPre, q, 0).r;
    k5 = min(k5, v);
    if(abs(dx) <= 1 && abs(dy) <= 1) k3 = min(k3, v);
  }
  fragColor = vec4(k3, k3 < 1e5 ? 1.0 : 0.0, k5 < 1e5 ? 1.0 : 0.0, 0.0);
}
#endif

// ===========================================================================
#ifdef VS_CHARPTS
uniform sampler2D uMask;
uniform ivec2 uTiles;
uniform float uTilePx;
flat out float vTStart;
void main(){
  ivec2 tc = ivec2(gl_VertexID % uTiles.x, gl_VertexID / uTiles.x);
  vec4 m = texelFetch(uMask, tc, 0);
  gl_PointSize = uTilePx;
  vTStart = (m.g > 0.5) ? max(m.r - 0.12, 0.2) : -1.0;     // ring-2 tiles: zeros, so later bilinear reads stay clean
  if(m.b < 0.5){ gl_Position = vec4(4.0, 4.0, 0.0, 1.0); return; }
  gl_Position = vec4((vec2(tc) + 0.5)*uTilePx/uRes*2.0 - 1.0, 0.0, 1.0);
}
#endif

// ===========================================================================
#ifdef VS_ROIPTS
uniform sampler2D uMask;
uniform ivec2 uRoiTiles;
uniform vec2  uRoiOrg;       // output px of the ROI layer's pixel (0,0)
uniform float uRoiScale, uScale, uTilePx;
uniform vec2  uRoiRes;
flat out float vTStart;
void main(){
  ivec2 tc = ivec2(gl_VertexID % uRoiTiles.x, gl_VertexID / uRoiTiles.x);
  vec2 cRoi = (vec2(tc) + 0.5)*uTilePx;                       // ROI px
  vec2 cOut = uRoiOrg + cRoi/uRoiScale;                       // output px
  vec4 m = texelFetch(uMask, ivec2(floor(cOut*uScale/8.0)), 0);
  gl_PointSize = uTilePx;
  vTStart = (m.g > 0.5) ? max(m.r - 0.12, 0.2) : -1.0;
  gl_Position = (m.b < 0.5) ? vec4(4.0, 4.0, 0.0, 1.0) : vec4(cRoi/uRoiRes*2.0 - 1.0, 0.0, 1.0);
}
#endif

// ===========================================================================
#ifdef VS_TILEQUAD
uniform sampler2D uMask;
uniform ivec2 uTiles;
uniform vec2  uTileOut;     // output pixels per tile (8 / scale)
void main(){
  int id = gl_InstanceID;
  ivec2 tc = ivec2(id % uTiles.x, id / uTiles.x);
  float m = texelFetch(uMask, tc, 0).b;
  vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
  vec2 px = min((vec2(tc) + corner)*uTileOut, uRes);
  gl_Position = (m < 0.5) ? vec4(4.0, 4.0, 0.0, 1.0) : vec4(px/uRes*2.0 - 1.0, 0.0, 1.0);
}
#endif

// ===========================================================================
#ifdef PASS_RESOLVE
// History lives at OUTPUT resolution; the character was marched at internal resolution uResI with this frame's
// sub-pixel jitter uJit (internal pixels).  An output pixel that holds one of this frame's samples takes it with
// weight uAlpha (box integration over the frames = supersampling); one that does not takes a bilinear estimate
// with the weight uAlphaMiss.  History is clipped to the variance box of the current neighbourhood, so a moving
// mouth does not ghost; it is fetched where the head's rigid motion says the point was (or in place, whichever is
// nearer this frame); where the tile was not marched last frame the history is dropped.
// Optional mouth ROI: inside uRoiBox (output px) the current frame comes from a second, finer march (uRoiC/D at
// uRoiScale), feathered into the base over uRoiFeather output px.
layout(location=0) out vec4 outC;
layout(location=1) out vec4 outD;
uniform sampler2D uCur, uCurD, uHist, uHistD, uMask, uMaskPrev, uPoseC, uPoseP, uRoiC, uRoiD;
uniform vec2  uResI;
uniform vec3  uCamPos, uCamU, uCamV, uCamW;
uniform float uFl;
uniform float uReproj;       // 1 = also try the history reprojected by the head's rigid motion (pose textures)
uniform vec2  uJit;          // this frame's jitter in internal pixels, 0..1
uniform float uScale;        // internal / output
uniform float uInvSpp, uAlpha, uAlphaMiss, uGamma, uReset;
uniform vec4  uRoiBox;       // x0, y0, x1, y1 output px (GL, bottom-up); x1 <= x0: no ROI
uniform float uRoiScale, uRoiFeather;
uniform vec2  uRoiRes;
// body: the camera and the character's root may move between frames.  uMoveOn = 0: today's code path exactly.
uniform float uMoveOn;                      // 1 = camera or root moved since last frame
uniform vec3  uPCamPos, uPCamU, uPCamV, uPCamW;
uniform float uPFl;                         // last frame's camera
uniform float uRootMv;                      // 1 = the root transform changed: try its rigid motion as a candidate
uniform mat3  uRootAiC, uRootAP;            // this frame's world->local, last frame's local->world (linear parts)
uniform vec3  uRootOC, uRootOP;
uniform float uMotionW;                     // extra weight of this frame under fast root motion
vec2 projPrev(vec3 p){
  vec3 v = p - uPCamPos;
  vec2 uvp = vec2(dot(v, uPCamU), dot(v, uPCamV))*uPFl/max(dot(v, uPCamW), 1e-3);
  return (uvp*uRes.y + uRes)*0.5;
}
bool inFrame(vec2 rp){ return all(greaterThan(rp, vec2(0.5))) && all(lessThan(rp, uRes - 0.5)); }
mat3 poseR(sampler2D t){ return mat3(texelFetch(t, ivec2(0,0), 0).xyz, texelFetch(t, ivec2(1,0), 0).xyz, texelFetch(t, ivec2(2,0), 0).xyz); }

// this frame's estimate at output pixel centre o from a layer marched at `sc` (internal px per output px) whose
// pixel (0,0) starts at output px `org`: sample (hit) or bilinear (miss), and the 3x3 variance box
void currentOf(sampler2D tc, sampler2D td, vec2 res, float sc, vec2 org, vec2 o,
               out vec4 cur, out vec4 curD, out vec4 bmin, out vec4 bmax, out bool hit){
  vec2 x = (o - org)*sc;
  vec2 lo = x - 0.5*sc, hi = x + 0.5*sc;
  vec2 jmin = ceil(lo - uJit);
  hit = all(lessThan(jmin + uJit, hi)) || sc >= 0.999;
  ivec2 lim = ivec2(res) - 1;
  ivec2 jc = clamp(sc >= 0.999 ? ivec2(floor(x)) : ivec2(hit ? jmin : floor(x - uJit + 0.5)), ivec2(0), lim);
  vec4 m1 = vec4(0.0), m2 = vec4(0.0), mn = vec4(1e9), mx = vec4(-1e9);
  for(int dy=-1; dy<=1; dy++) for(int dx=-1; dx<=1; dx++){
    vec4 c = texelFetch(tc, clamp(jc + ivec2(dx, dy), ivec2(0), lim), 0)*uInvSpp;
    m1 += c; m2 += c*c; mn = min(mn, c); mx = max(mx, c);
  }
  m1 /= 9.0; m2 /= 9.0;
  vec4 sd = sqrt(max(m2 - m1*m1, 0.0));
  bmin = max(mn, m1 - uGamma*sd); bmax = min(mx, m1 + uGamma*sd);
  if(hit){
    cur = texelFetch(tc, jc, 0)*uInvSpp; curD = texelFetch(td, jc, 0)*uInvSpp;
  } else {
    vec2 uv = (x - uJit + 0.5)/res;              // texel centres = sample positions
    cur = texture(tc, uv)*uInvSpp; curD = texture(td, uv)*uInvSpp;
  }
}

void main(){
  ivec2 op = ivec2(gl_FragCoord.xy);
  vec2 o = gl_FragCoord.xy;
  ivec2 tile = ivec2(floor(o*uScale/8.0));
  if(texelFetch(uMask, tile, 0).g < 0.5){ outC = vec4(0.0); outD = vec4(0.0); return; }
  vec4 cur, curD, bmin, bmax; bool hit;
  currentOf(uCur, uCurD, uResI, uScale, vec2(0.0), o, cur, curD, bmin, bmax, hit);
  if(uRoiBox.z > uRoiBox.x){
    vec2 e = min(o - uRoiBox.xy, uRoiBox.zw - o);
    float f = clamp(min(e.x, e.y)/uRoiFeather, 0.0, 1.0);
    if(f > 0.0){
      vec4 c2, d2, b0, b1; bool h2;
      currentOf(uRoiC, uRoiD, uRoiRes, uRoiScale, uRoiBox.xy, o, c2, d2, b0, b1, h2);
      cur = mix(cur, c2, f); curD = mix(curD, d2, f); bmin = mix(bmin, b0, f); bmax = mix(bmax, b1, f);
      hit = f > 0.5 ? h2 : hit;
    }
  }
  float w = hit ? uAlpha : uAlphaMiss;
  // where was this surface point last frame?  The head is rigid: world -> head space now, head -> world then.
  // Body pixels do not follow the head, so both candidates are tried and the one nearer this frame wins.
  vec2 hpix = o;
  if(uReproj > 0.5 && cur.a > 0.02){
    vec2 uvc = (2.0*o - uRes)/uRes.y;
    vec3 rd = normalize(uvc.x*uCamU + uvc.y*uCamV + uFl*uCamW);
    vec3 p = uCamPos + rd*(curD.r/cur.a);
    vec3 q = poseR(uPoseC)*(p - texelFetch(uPoseC, ivec2(3,0), 0).xyz);
    vec3 pp = transpose(poseR(uPoseP))*q + texelFetch(uPoseP, ivec2(3,0), 0).xyz;
    vec3 v = pp - uCamPos;
    vec2 uvp = vec2(dot(v, uCamU), dot(v, uCamV))*uFl/max(dot(v, uCamW), 1e-3);
    vec2 rp = (uvp*uRes.y + uRes)*0.5;
    if(uMoveOn < 0.5){
      if(all(greaterThan(rp, vec2(0.5))) && all(lessThan(rp, uRes - 0.5))){
        vec4 hs = texelFetch(uHist, op, 0), hr = texture(uHist, rp/uRes);
        if(length(hr - cur) < length(hs - cur)) hpix = rp;
      }
    } else {
      // body: three candidates, each projected with LAST frame's camera: the point static in the world, carried
      // by the head's rigid motion, carried by the root's rigid motion; the history nearest this frame wins
      vec2 rs = projPrev(p);
      float best = 1e9;
      if(inFrame(rs)){ hpix = rs; best = length(texture(uHist, rs/uRes) - cur); }
      rp = projPrev(pp);
      if(inFrame(rp)){ float e = length(texture(uHist, rp/uRes) - cur); if(e < best){ best = e; hpix = rp; } }
      if(uRootMv > 0.5){
        vec2 rr = projPrev(uRootAP*(uRootAiC*(p - uRootOC)) + uRootOP);
        if(inFrame(rr)){ float e = length(texture(uHist, rr/uRes) - cur); if(e < best){ best = e; hpix = rr; } }
      }
    }
  }
  ivec2 htile = ivec2(floor(hpix*uScale/8.0));
  bool valid = uReset < 0.5 && texelFetch(uMaskPrev, tile, 0).g > 0.5 && texelFetch(uMaskPrev, htile, 0).g > 0.5;
  if(!valid){
    outC = cur; outD = vec4(curD.r, 0.0, 0.0, 1.0);
    return;
  }
  vec4 h = texture(uHist, hpix/uRes);
  float hd = texture(uHistD, hpix/uRes).r;
  vec4 hc = clamp(h, bmin, bmax);
  // depth follows the colour: rescale the premultiplied depth by the alpha clip
  float hdc = h.a > 1e-4 ? hd*(hc.a/h.a) : curD.r;
  // how far the history had to move: a big clip means the character changed here, trust this frame more
  float moved = length(hc - h)/(length(bmax - bmin) + 0.02);
  w = clamp(w + 0.5*smoothstep(0.1, 0.6, moved)*(hit ? 1.0 : 0.3), 0.0, 1.0);
  w = max(w, uMotionW);     // body
  outC = mix(hc, cur, w);
  outD = vec4(mix(hdc, curD.r, w), 0.0, 0.0, 1.0);
}
#endif

// ===========================================================================
#ifdef PASS_COMPDOF
// engine/post.glsl PASS_CHARDOF + PASS_COMP in one pass, only inside the R1 tiles.
#define MAXC 7.0
uniform float uTime;
uniform float uP[NPARAMS];
#define hash12 rigHash12
#pragma include common
#undef hash12
#pragma include rig
out vec4 fragColor;
uniform sampler2D uLive, uVol, uChar, uCharD, uMask;
uniform float uFocus, uDofK, uPxScale, uInvSpp, uScale;
uniform int   uFrame;
uniform vec3 uCamPos, uCamU, uCamV, uCamW;
uniform float uFl;
uniform vec3 uSunDir, uSunCol, uRimDir, uRimCol;
uniform float uCharKey;
float coc(float d){ return clamp(uDofK*abs(1.0/uFocus - 1.0/max(d,0.05)), 0.0, MAXC)*uPxScale; }
vec3 rigCamRay(vec2 pix){
  vec2 p=(2.0*pix-uRes)/uRes.y;
  return normalize(p.x*uCamU+p.y*uCamV+uFl*uCamW);
}
vec3 rigNormal(vec3 p){
  const vec2 e=vec2(.003,0.0);
  return normalize(vec3(rigDistance(p+e.xyy)-rigDistance(p-e.xyy),
    rigDistance(p+e.yxy)-rigDistance(p-e.yxy),
    rigDistance(p+e.yyx)-rigDistance(p-e.yyx)));
}
vec4 traceRig(vec3 ro, vec3 rd){
  // Torso and articulated limbs are part of the renderer and share the
  // camera/depth buffer with the head. This pass avoids recompiling the large
  // animal shader when the limb rig changes.
  float t=CHAR_T_NEAR;
  for(int i=0;i<36;i++){
    if(t>CHAR_T_FAR)break;
    float d=rigDistance(ro+rd*t);
    if(d<.005){
      vec3 p=ro+rd*t, n=rigNormal(p);
      float mat; rigPart(p,mat);
      Surf s=surfDefault(n); rigSurface(mat,p,s);
      float diffuse=clamp((dot(n,uSunDir)+s.wrap)/(1.0+s.wrap),0.0,1.0);
      float rim=pow(1.0-clamp(dot(n,-rd),0.0,1.0),2.5);
      vec3 col=s.alb*(vec3(.27,.25,.24)+uSunCol*diffuse*.7)
        +rim*uRimCol*s.sheen*.38;
      return vec4(col*uCharKey,t);
    }
    t+=max(d*.82,.012);
  }
  return vec4(0.0,0.0,0.0,1e6);
}
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec4 bg = texture(uLive, uv);
  vec2 px = 3.0/uRes;      // the volume is 1/3 res: tent filter it
  vec3 vol = 0.25*(texture(uVol, uv + px*vec2(0.75,0.25)).rgb + texture(uVol, uv + px*vec2(-0.25,0.75)).rgb
                 + texture(uVol, uv + px*vec2(-0.75,-0.25)).rgb + texture(uVol, uv + px*vec2(0.25,-0.75)).rgb);
  vec4 ch = vec4(0.0); float dd = 0.0;
  ivec2 tile = ivec2(floor(gl_FragCoord.xy*uScale/8.0));
  if(texelFetch(uMask, tile, 0).g > 0.5){
    vec4 c0 = texture(uChar, uv)*uInvSpp;
    vec4 d0v = texture(uCharD, uv)*uInvSpp;
    float d0 = c0.a > 0.001 ? d0v.r/c0.a : uFocus;
    float cc = max(coc(d0), 0.45*uPxScale);
    const int N = DOF_N;
    float rot0 = hash12(gl_FragCoord.xy + float(uFrame)*0.37)*6.2831;
    vec4 acc = c0*1.5; float wsum = 1.5; float dAcc = d0v.r*1.5;
    for(int i=0;i<N;i++){
      float r = sqrt((float(i)+0.5)/float(N));
      float a = float(i)*2.39996 + rot0;
      vec2 uv2 = uv + vec2(cos(a), sin(a))*r*cc/uRes;
      acc += texture(uChar, uv2)*uInvSpp;
      dAcc += texture(uCharD, uv2).r*uInvSpp; wsum += 1.0;
    }
    ch = acc/wsum; dd = dAcc/wsum;
  }
  float d = mix(bg.a, ch.a > 0.001 ? dd/ch.a : bg.a, ch.a);
  vec3 rgb=bg.rgb*(1.0 - ch.a) + ch.rgb + vol*(1.0 - 0.85*ch.a);
#if RIG_PASS == 1
  vec4 rig=traceRig(uCamPos,rigCamRay(gl_FragCoord.xy));
  if(rig.a<d){rgb=rig.rgb;d=rig.a;}
#endif
  fragColor = vec4(rgb,d);
}
#endif

// ===========================================================================
#ifdef PASS_BGVIEW
// body: the cached background layers for THIS frame's camera and root.  The camera only turns and zooms about
// its fixed position (no parallax), so the bake is a function of the ray direction: inside the rest camera's frame
// the exact bake, outside it the wide panorama (same position, wider lens), feathered.  While the root moves the
// layers baked without the character (its shadow / reflection is drawn live) replace today's, by uRootLive.
layout(location=0) out vec4 outA;
layout(location=1) out vec4 outS;
uniform sampler2D uBgA, uBgS, uBgA0, uBgS0, uPanA, uPanS;
uniform float uRootLive, uCamOn;
uniform vec3  uCamU, uCamV, uCamW;
uniform float uFl;
uniform vec3  uCam0U, uCam0V, uCam0W;
uniform float uCam0Fl, uPanFl;
uniform vec2  uPanRes;
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  if(uCamOn < 0.5){
    outA = mix(texelFetch(uBgA, ip, 0), texelFetch(uBgA0, ip, 0), uRootLive);
    outS = mix(texelFetch(uBgS, ip, 0), texelFetch(uBgS0, ip, 0), uRootLive);
    return;
  }
  vec2 uv = (2.0*gl_FragCoord.xy - uRes)/uRes.y;
  vec3 rd = normalize(uv.x*uCamU + uv.y*uCamV + uFl*uCamW);
  float z = dot(rd, uCam0W);
  vec2 q = vec2(dot(rd, uCam0U), dot(rd, uCam0V))/max(z, 1e-4);
  vec2 px0 = (q*uCam0Fl*uRes.y + uRes)*0.5;
  vec2 e = min(px0, uRes - px0);
  float f = z > 0.0 ? clamp((min(e.x, e.y) - 3.0)/(0.04*uRes.y), 0.0, 1.0) : 0.0;
  vec4 a = vec4(0.0), s = vec4(0.0);
  if(f > 0.0){
    vec2 u0 = px0/uRes;
    a = mix(texture(uBgA, u0), texture(uBgA0, u0), uRootLive);
    s = mix(texture(uBgS, u0), texture(uBgS0, u0), uRootLive);
  }
  if(f < 1.0){
    vec2 uP = (q*uPanFl*uPanRes.y + uPanRes)*0.5/uPanRes;
    a = mix(texture(uPanA, uP), a, f); s = mix(texture(uPanS, uP), s, f);
  }
  outA = a; outS = s;
}
#endif

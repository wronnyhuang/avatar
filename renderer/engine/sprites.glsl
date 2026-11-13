// ---------------------------------------------------------------------------
//  engine/sprites.glsl — cheap rasterized life: dust motes (points) and falling leaves (quads).
//  Compiled as both stages: render.py adds `#define VS` or `#define FS` plus MOTES or LEAVES.
//  Depends only on the set (for light), so they cost almost nothing per frame.
// ---------------------------------------------------------------------------
uniform vec2  uRes;
uniform float uTime;
uniform int   uFrame;
uniform float uP[NPARAMS];
uniform vec3  uCamPos, uCamU, uCamV, uCamW;
uniform float uFl;
uniform vec3  uSunDir, uSunCol;
uniform vec3  uMotesMin, uMotesSize;
uniform float uMoteGain, uFocus, uDofK;
uniform int   uLayer;     // 0 = behind the character, 1 = in front
uniform vec3  uLeafColA, uLeafColB;

#pragma include common
#pragma include set

// world -> clip for the camera model used by the ray passes
vec4 project(vec3 p){
  vec3 v = p - uCamPos;
  float z = dot(v, uCamW);
  float x = dot(v, uCamU)*uFl/max(z, 0.01);
  float y = dot(v, uCamV)*uFl/max(z, 0.01);
  return vec4(x*uRes.y/uRes.x, y, 0.0, 1.0);
}

#ifdef VS
#ifdef MOTES
out float vGlow; out float vSoft;
void main(){
  setInit();
  float T = uTime;
  vec3 h = hash31(float(gl_VertexID)*1.731 + 0.37);
  vec3 h2 = hash31(float(gl_VertexID)*0.913 + 7.1);
  vec3 p = uMotesMin + h*uMotesSize;
  p += 0.35*vec3(sin(T*0.31 + h.x*20.0), 0.6*sin(T*0.23 + h.y*17.0), cos(T*0.27 + h.z*13.0));
  p.y += 0.04*mod(T*(h.x - 0.4) + h2.x*30.0, 30.0) - 0.6;
  float z = dot(p - uCamPos, uCamW);
  float tw = 0.55 + 0.45*sin(T*2.2 + h.y*40.0);
  float coc = min(uDofK*abs(1.0/uFocus - 1.0/max(z, 0.1)), 0.03*uRes.y);     // blur disc, px
  float base = (1.2 + 2.0*h2.y)*uRes.y/1080.0;
  float size = max(base, coc);
  gl_PointSize = 3.0*size + 2.0;
  vSoft = size;
  vGlow = (setVol(p) + 0.18)*tw*(base*base)/(size*size)*uMoteGain;
  gl_Position = project(p);
  if((p.z > 0.0) != (uLayer == 1)) gl_Position = vec4(3.0,3.0,0.0,1.0);
}
#else
out vec2 vUv; out float vLight; out float vPick; out float vSoft;
void main(){
  setInit();
  float T = uTime;
  int id = gl_InstanceID;
  vec3 h = hash31(float(id)*3.71 + 1.3);
  float per = 11.0 + 5.0*h.x;
  float ph = fract((T + h.y*per)/per);
  // leaves fall behind the character only: a near leaf would cross the face as a huge blurred quad
  vec3 c = uMotesMin + vec3(h.z*uMotesSize.x, (1.0 - ph)*(uMotesSize.y + 3.0) - 1.5, h.x*min(uMotesSize.z, -0.9 - uMotesMin.z));
  c.x += 0.8*sin(ph*per*1.3 + h.y*6.0);
  c.z += 0.4*cos(ph*per*1.1 + h.x*6.0);
  float tt = T*(1.6 + h.x) + h.z*6.0;
  vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID>>1) & 1))*2.0 - 1.0;
  vec2 q = corner;
  float zz = dot(c - uCamPos, uCamW);
  float blurW = uDofK*abs(1.0/uFocus - 1.0/max(zz, 0.1))*2.0*zz/(uFl*uRes.y);   // lens blur, world units
  vSoft = blurW;
  q.x *= 0.13 + blurW; q.y *= 0.06*(0.25 + 0.75*abs(cos(tt*0.9))) + blurW;    // tumble: the leaf foreshortens
  q = rot(0.8*sin(tt) + h.z*6.0)*q;
  vec3 p = c + uCamU*q.x + uCamV*q.y;
  vUv = corner;
  vPick = h.y;
  vLight = 0.35 + 0.65*setCookie(c);
  gl_Position = project(p);
  if((c.z > 0.0) != (uLayer == 1)) gl_Position = vec4(3.0,3.0,0.0,1.0);
}
#endif
#endif

#ifdef FS
#ifdef MOTES
in float vGlow; in float vSoft;
out vec4 fragColor;
void main(){
  vec2 d = gl_PointCoord*2.0 - 1.0;
  float g = exp(-dot(d,d)*4.5);
  fragColor = vec4(uSunCol*g*vGlow, 0.0);
}
#else
in vec2 vUv; in float vLight; in float vPick; in float vSoft;
out vec4 fragColor;
void main(){
  float x = vUv.x, y = vUv.y;
  float w = (1.0 - x*x)*(1.0 - 0.35*x);
  float soft = clamp(vSoft*8.0, 0.0, 1.0);
  float a = smoothstep(-0.05 - soft*0.5, 0.20 + soft*0.5, w - abs(y))*mix(1.0, 0.55, soft);
  vec3 alb = mix(uLeafColA, uLeafColB, vPick)*(0.85 + 0.25*smoothstep(0.1, 0.0, abs(y)));
  vec3 col = alb*(uSunCol*0.55*vLight + vec3(0.5,0.45,0.35));
  fragColor = vec4(col*a, a);
}
#endif
#endif

// ---------------------------------------------------------------------------
//  engine/common.glsl — shared maths for every set and every animal.
//  noise, SDF primitives, smooth ops, fur / eye / mouth helpers, the Surf struct.
//  Nothing here reads a uniform except uTime-free pure functions.
// ---------------------------------------------------------------------------
#define PI 3.14159265359

// ------------------------------------------------------------------ noise
float hash11(float p){ p = fract(p*.1031); p *= p+33.33; p *= p+p; return fract(p); }
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3,p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
float hash13(vec3 p3){ p3 = fract(p3*.1031); p3 += dot(p3,p3.zyx+31.32); return fract((p3.x+p3.y)*p3.z); }
vec2  hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); p3 += dot(p3,p3.yzx+33.33); return fract((p3.xx+p3.yz)*p3.zy); }
vec3  hash33(vec3 p3){ p3 = fract(p3*vec3(.1031,.1030,.0973)); p3 += dot(p3,p3.yxz+33.33); return fract((p3.xxy+p3.yxx)*p3.zyx); }
vec3  hash31(float p){ vec3 p3 = fract(vec3(p)*vec3(.1031,.1030,.0973)); p3 += dot(p3,p3.yzx+33.33); return fract((p3.xxy+p3.yzz)*p3.zyx); }

float vnoise2(vec2 p){
  vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.0-2.0*f);
  return mix(mix(hash12(i),hash12(i+vec2(1,0)),u.x), mix(hash12(i+vec2(0,1)),hash12(i+vec2(1,1)),u.x), u.y);
}
float vnoise3(vec3 p){
  vec3 i=floor(p), f=fract(p); vec3 u=f*f*(3.0-2.0*f);
  float a=hash13(i),             b=hash13(i+vec3(1,0,0));
  float c=hash13(i+vec3(0,1,0)), d=hash13(i+vec3(1,1,0));
  float e=hash13(i+vec3(0,0,1)), f1=hash13(i+vec3(1,0,1));
  float g=hash13(i+vec3(0,1,1)), h=hash13(i+vec3(1,1,1));
  return mix(mix(mix(a,b,u.x),mix(c,d,u.x),u.y), mix(mix(e,f1,u.x),mix(g,h,u.x),u.y), u.z);
}
const mat2 M2 = mat2(1.6,1.2,-1.2,1.6);
float fbm2(vec2 p){ float s=0., a=0.5; for(int i=0;i<5;i++){ s+=a*vnoise2(p); p=M2*p; a*=0.5; } return s/0.96875; }
float fbm2_3(vec2 p){ float s=0., a=0.5; for(int i=0;i<3;i++){ s+=a*vnoise2(p); p=M2*p; a*=0.5; } return s/0.875; }
float fbm3(vec3 p){ float s=0., a=0.5; for(int i=0;i<4;i++){ s+=a*vnoise3(p); p=p*2.03+vec3(1.7,9.2,3.1); a*=0.5; } return s/0.9375; }

// ------------------------------------------------------------------ SDF primitives + smooth ops
mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,s,-s,c); }
float smin(float a, float b, float k){ float h=max(k-abs(a-b),0.0)/k; return min(a,b)-h*h*k*0.25; }
float smax(float a, float b, float k){ return -smin(-a,-b,k); }
float sdEllipsoid(vec3 p, vec3 r){ float k0=length(p/r); float k1=length(p/(r*r)); return k0*(k0-1.0)/max(k1,1e-5); }
float sdCapsule(vec3 p, vec3 a, vec3 b, float r){ vec3 pa=p-a, ba=b-a; float h=clamp(dot(pa,ba)/dot(ba,ba),0.,1.); return length(pa-ba*h)-r; }
// linear-radius capsule (cheap, not exact; fine for tufts, snouts, limbs)
float sdTaper(vec3 p, vec3 a, vec3 b, float r1, float r2){ vec3 pa=p-a, ba=b-a; float h=clamp(dot(pa,ba)/dot(ba,ba),0.,1.); return length(pa-ba*h)-mix(r1,r2,h); }
// same but squeezed: `sq` scales the local y of the cross-section (sq>1 = flatter)
float sdTaperSq(vec3 p, vec3 a, vec3 b, float r1, float r2, float sq){
  vec3 s = vec3(1.0, sq, 1.0);
  return sdTaper(p*s, a*s, b*s, r1, r2)/sq;
}
vec2 opU(vec2 a, vec2 b){ return a.x<b.x ? a : b; }

// ------------------------------------------------------------------ surface description
// charSurface()/setSurface() fill this in; scene_main lights it.
struct Surf {
  vec3  alb;      // linear albedo
  vec3  n;        // shading normal
  vec3  T;        // strand direction (fur only)
  float fur;      // 0 = hard surface (plain Blinn), 1 = fur (Kajiya-Kay + asperity rim)
  float spec;     // Blinn strength
  float shin;     // Blinn exponent
  float wrap;     // diffuse wrap 0..1
  float trans;    // back-lit translucency 0..1 (ear edges, leaves)
  vec3  transCol; // tint of the translucency
  float sheen;    // fresnel-ish rim 0..1
  float occ;      // painted occlusion multiplier (fur roots, creases) 0..1
  float mirror;   // 0..1 environment reflection (nose, cornea)
  vec3  emit;     // added radiance (painted catch-lights)
};
Surf surfDefault(vec3 n){
  Surf s;
  s.alb = vec3(0.5); s.n = n; s.T = vec3(0,1,0); s.fur = 0.0; s.spec = 0.04; s.shin = 16.0;
  s.wrap = 0.0; s.trans = 0.0; s.transCol = vec3(1.0,0.45,0.18); s.sheen = 0.0; s.occ = 1.0;
  s.mirror = 0.0; s.emit = vec3(0.0);
  return s;
}

// ------------------------------------------------------------------ character evaluation
// charEval(p) (animal file) fills these; everything the engine marches is derived from them.
float gcBase;   // distance to the fur-covered skin (fur NOT included)
float gcBare;   // distance to bare parts: eyes, nose, teeth, tongue, claws
float gcLen;    // local fur length (0 = bare skin here)
vec3  gcRest;   // rest-pose position of this skin point: fur and paint stick to it

// ------------------------------------------------------------------ FUR helpers
// Fur everywhere flows AWAY from a pole (the nose tip on a face).  Use a rest-pose point `q`
// (so fur does not swim when a jaw or an ear moves).  The strand coordinates are seamless:
//   furCoord(q, pole, K, k): K = strands per radian-ish, k = strands per unit along the flow.
vec3 furCoord(vec3 q, vec3 pole, float K, float k){
  vec3 v = q - pole;
  float r = length(v);
  vec2 w = v.xy + 0.05*sin(v.yx*vec2(5.1,4.3) + vec2(1.3,0.7)) + 0.02*sin(v.yx*vec2(13.0,11.0) + v.z*7.0);  // cowlicks
  vec2 az = w/max(length(w), 1e-3);
  // keep a mild dependence on z so the far side of a sphere does not mirror the near side
  return vec3(az*K, r*k + v.z*0.7*k);
}
// Strand height field 0..1 at a rest point: clumps (K,L) + strands (about 4x finer).  3 noise evals.
// The shell march keeps a sample where this height exceeds the normalized height above the skin.
float furHeight(vec3 rest, vec3 pole, float K, float L){
  float clump = vnoise3(furCoord(rest, pole, K, L));
  vec3 f = furCoord(rest, pole, K*3.9, L*2.2);
  float st = vnoise3(f);
  float st2 = vnoise3(f*vec3(2.3,2.3,1.1) + vec3(4.1,1.7,8.3));
  return clamp(clump*0.60 + st*0.34 + st2*0.26 - 0.06, 0.0, 1.0);
}
// Tuft height 0..1 (2 noise evals) — colour break-up, cheap displacement.
float furTuft(vec3 c){
  float a = vnoise3(c);
  float b = vnoise3(c*vec3(2.3,2.3,1.0) + vec3(4.1,1.7,8.3));
  float ridge = 1.0 - abs(2.0*a - 1.0);
  return clamp(0.20 + 0.80*ridge*(0.45 + 0.85*b), 0.0, 1.0);
}
// Fine strand signal 0..1 for shading only (colour + normal break-up).
float furStrand(vec3 c){ return vnoise3(c)*0.6 + vnoise3(c*vec3(2.7,2.7,1.5) + 3.1)*0.4; }
// Tangent of the flow at surface normal n: strands lie in the surface, pointing away from the pole,
// plus an optional bias (gravity droop).
vec3 furFlow(vec3 v, vec3 n, vec3 bias){
  vec3 d = normalize(v) + bias;
  d -= n*dot(d, n);
  return normalize(d + vec3(1e-4));
}

// ------------------------------------------------------------------ EYE helper
// A real sphere eyeball plus two lid shells cut by planes through the eye centre.
// e = eye-local point (origin = eye centre, +z = forward, +y = up, already tilted),
// R = eyeball radius, th = lid thickness, aU / aL = elevation of the upper / lower lid margin (rad).
float sdEyelids(vec3 e, float R, float th, float aU, float aL){
  float shell = abs(length(e) - (R + th*0.9)) - th;
  vec2 nU = vec2(cos(aU), -sin(aU));   // upper lid keeps dot(e.yz,nU) > 0
  vec2 nL = vec2(-cos(aL), sin(aL));   // lower lid keeps dot(e.yz,nL) > 0
  float up = smax(shell, -dot(e.yz, nU), 0.02);
  float lo = smax(shell, -dot(e.yz, nL), 0.02);
  return min(up, lo);
}
// Eye colour.  e = unit point on the eyeball in GAZE space (+z = through the pupil),
// v = view ray in the same space.  Returns radiance-ready albedo; `irisOut` = iris coverage.
vec3 eyeColour(vec3 e, vec3 v, vec3 irisInner, vec3 irisOuter, vec3 sclera, float rIris, float rPupil, out float irisOut){
  // cornea refraction: slide the iris plane against the view direction
  float inFront = smoothstep(0.1, 0.5, e.z);
  vec2 uv = e.xy + v.xy*0.17*inFront;
  float r = length(uv);
  float a = atan(uv.y, uv.x);
  float fib  = vnoise2(vec2(a*7.0, r*4.0));
  float fib2 = vnoise2(vec2(a*19.0, r*7.0 + 3.0));
  float rn = clamp((r - rPupil)/(rIris - rPupil), 0.0, 1.0);
  vec3 iris = mix(irisInner, irisOuter, smoothstep(0.05, 1.0, rn));
  iris *= 0.72 + 0.50*fib*(0.6 + 0.8*fib2);
  // collarette: a bright ring around the pupil
  iris += irisInner*0.55*smoothstep(0.35, 0.0, rn)*(0.6 + 0.8*fib2);
  // limbal ring
  iris *= 1.0 - 0.88*smoothstep(0.70, 1.0, rn);
  float irisM = smoothstep(rIris + 0.012, rIris - 0.012, r)*step(0.0, e.z);
  float pup   = smoothstep(rPupil + 0.012, rPupil - 0.012, r);
  vec3 col = mix(sclera, iris, irisM);
  col = mix(col, vec3(0.004,0.003,0.003), pup*irisM);
  irisOut = irisM;
  return col;
}

// ------------------------------------------------------------------ MOUTH helpers
// Generic ingredients every animal's mouth reuses.  See animals/fox.glsl for how they combine.
// Rotate a head-space point into the jaw's rest frame: `angle` radians about the x-axis through `hinge`.
vec3 jawSpace(vec3 q, vec3 hinge, float angle){
  vec3 r = q - hinge;
  r.yz = rot(angle)*r.yz;
  return r + hinge;
}
// Mouth hollow carved from a solid, leaving a lip rim.  solid = the muzzle SDF before the lip cut,
// y/yLip = point height and lip height, rim = lip thickness, depth = how far the hollow reaches,
// side = +1 for the roof (hollow above the lip line), -1 for the floor.  Subtract with smax(x, -hollow, k).
float mouthCavity(float solid, float y, float yLip, float rim, float depth, float side){
  float up = (y - yLip)*side;
  return max(max(solid + rim, -up - 0.02), up - depth);
}
// Tooth row: a thin strip following a level set of the muzzle SDF (the dental arch), cut into points.
//   muzzle = muzzle SDF, inset = how far inside the surface, thick = half thickness,
//   y = height from the gum line with tips toward y<0 (pass -y for a lower row),
//   arc = a coordinate along the arch (z + |x| works), len = tooth length, freq = teeth per unit arc.
float toothRow(float muzzle, float y, float arc, float inset, float thick, float len, float freq){
  float tri = abs(fract(arc*freq) - 0.5)*2.0;
  float band = abs(muzzle + inset) - thick;
  float tips = -y - len*(0.25 + 0.75*tri);
  return max(band, max(tips, y - 0.03));
}
// Pointed tooth: cone from root to tip.
float sdFang(vec3 p, vec3 root, vec3 tip, float r){ return sdTaper(p, root, tip, r, 0.0015); }

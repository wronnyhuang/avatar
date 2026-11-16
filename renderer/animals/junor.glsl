// ---------------------------------------------------------------------------
// animals/juno.glsl — render: Juno's stand-in INSIDE the shared SDF scene (docs/CONTRACT--render.md).
//  Juno herself is a rasterized skinned mesh (renderer/mesh/, drawn after the scene and depth-composited); this file
//  only gives the scene what it needs from "the character":
//    * a capsule PROXY (uniform uJCap, from the rig's bone positions) for her soft shadow + contact shade on
//      the set (PASS_LIVE rootShadowLive / lookFloor) and on held props (PASS_CHAR shadow taps).  It is never SEEN:
//      the visible marches (PASS_PRE, PASS_CHAR) and the bake (PASS_BG) get "nothing here".
//    * the grasp frames + head frame the props attach to (junoGrasp / junoHeadFrame, PROPS_GRASP_FN / _HEAD_FN through
//      juno.json "props_renderer"), char-local (CHAR_ROOT, as Muse).
//  Without uniforms set (all radii 0) she is invisible and casts nothing: the bake is the empty set.
// ---------------------------------------------------------------------------
#define JUNO_MESH 1
// names fixed by docs/CONTRACT--rig.md section 2 (RIG): everything CHAR-LOCAL (juno.json char_defines CHAR_ROOT:
// the engine's root wrappers map the world into this space, exactly as for Muse)
#define JCAP_N 18
uniform vec4 uJCap[36];          // 18 capsules x 2: (a.xyz, r), (b.xyz, kind)
uniform mat4 uJGrasp[2];         // grasp_local, side 0 = screen-left (columns: across palm, to fingertips, out of palm, pos)
uniform mat4 uJHead;             // head_local (worn props)
uniform mat4 uJHeadC;            // columns 0-2 = head axes (head_rot_local), 3 = head_center_local

mat3  gHR = mat3(1.0);           // world(local) -> head (the POSE pass and the marks read these)
vec3  gHO = vec3(0.0);

void charInit(){
  gHO = uJHeadC[3].xyz;
  gHR = transpose(mat3(uJHeadC));
}

float junoCapsule(vec3 p, vec3 a, vec3 b, float r){
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba)/max(dot(ba, ba), 1e-8), 0.0, 1.0);
  return length(pa - ba*h) - r;
}
float junoProxy(vec3 p){
  float d = 1e3;
  for(int i=0;i<JCAP_N;i++){
    vec4 a = uJCap[2*i], b = uJCap[2*i + 1];
    if(a.w > 0.0) d = min(d, junoCapsule(p, a.xyz, b.xyz, a.w));
  }
  return d;
}

// the visible character is the mesh: the SDF marches see nothing of her
void charEval(vec3 p){ gcBase = 1e3; gcBare = 1e3; gcLen = 0.0; gcRest = p; }
float charTuft(){ return 0.0; }

//@shadow-begin
float charShadow(vec3 p){
#ifdef PASS_BG
  return 1e3;                    // the bake never sees her (her shadow on the set is live, PASS_LIVE)
#else
  return junoProxy(p);
#endif
}
//@shadow-end

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){ s.alb = vec3(0.5); }

void junoGrasp(int side, out vec3 pos, out mat3 frame){ mat4 G = uJGrasp[side == 0 ? 0 : 1]; frame = mat3(G); pos = G[3].xyz; }
void junoHeadFrame(out vec3 pos, out mat3 frame){ frame = mat3(uJHead); pos = uJHead[3].xyz; }

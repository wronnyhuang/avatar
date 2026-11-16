// mat_cornea.glsl — render-face: the clear wet shell over each eye (cornea_l/r).  MAT_CORNEA, blended
// (premultiplied, alpha 0 = additive): reflection only.  ONE crisp key highlight per eye (the key's GGX lobe on the
// cornea bulge: the same place in both eyes), a very faint broad secondary from the fill, the room dimmed under the
// lids; nothing where the lid shades the eye.
uniform mat3  uHeadR;
uniform float uHeadS;
uniform vec3  uEyeC[2];
uniform float uEyeR;
uniform float uLidTop[26], uLidBot[26];
uniform float uEyeQ;
vec2 corLids(vec3 p){
  int k = distance(p, uEyeC[0]) < distance(p, uEyeC[1]) ? 0 : 1;
  vec3 d = normalize(transpose(uHeadR)*(p - uEyeC[k]));
  float az = degrees(atan(d.x, d.z)), el = degrees(asin(clamp(d.y, -1.0, 1.0)));
  float f = clamp((az + 66.0)/11.0, 0.0, 11.999);
  if(isnan(f)) f = 6.0;          // render6: atan(0, 0) / a degenerate d -> never an out-of-range uniform index
  int i = int(f); float t = fract(f);
  return vec2(mix(uLidTop[k*13 + i], uLidTop[k*13 + i + 1], t) - el, el - mix(uLidBot[k*13 + i], uLidBot[k*13 + i + 1], t));
}
uniform float uCorneaBulge;
uniform vec4 uEyeK2;        // render5 (mesh.face.eye_k2): x window catchlight, y second small catchlight    // extra curvature of the cornea over the iris (0 = the mesh's own shape)
uniform vec4 uEyeK3;        // render8 (mesh.face.eye_k3): the key catchlight's shape -- x roundness (0 the rounded
                             //   rectangle .. 1 a soft disc), y size, z/w its offset in the window plane (place it at 10-11 o'clock)
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}
vec4 matShade(SIn S, inout vec4 fragDOut){   // inout (an out param is undefined until written: NVIDIA gave 0 -> mesh depth layer 0 -> no SSS, no depth)
  vec3 L = uKeyDir, v = S.v;
  // the cornea bulges over the iris (a stylised eye's dome is steeper than the eyeball): tilt the normal away from
  // the apex inside the iris disc so the key's highlight lands ON the iris near the pupil, not up under the lid
  vec2 c = S.uv - 0.5;
  float rr = length(c)/max(uEyeP.x, 1e-4);
  float dome = smoothstep(1.35, 0.6, rr)*uCorneaBulge;
  // render: the dome from the eyeball centre (robust to the cornea's own tangents/normals): the normal of a
  // sphere about the eye centre, tilted further out by `dome` (a steeper stylised cornea)
  int ke = distance(S.p, uEyeC[0]) < distance(S.p, uEyeC[1]) ? 0 : 1;
  vec3 ns0 = normalize(S.p - uEyeC[ke]);
  vec3 gaze = normalize(S.ng);
  vec3 ng = normalize(ns0 + (ns0 - gaze*dot(ns0, gaze))*1.2*dome);
  vec2 lid = corLids(S.p);
  float open = smoothstep(0.5, 4.0, lid.x)*smoothstep(0.0, 2.0, lid.y);
  vec3 r = reflect(-v, ng);
  float Fr = 0.025 + 0.975*pow(1.0 - S.nv, 5.0);
  // the room: dim, only the window reads (a faint secondary); attenuated toward the lids
  vec3 env = envSpec(S.p, r)*Fr*mix(0.35, 1.0, open)*mix(1.0, 0.6, uEyeQ);
  // the key: one crisp highlight (energy-clamped so it reads as a small soft white dot, not a flare)
  // ( render: the catchlight's light is the key pulled toward the camera -- the animation-film cheat: a real
  //  key 40 deg up-left puts its reflection under the upper lid of a big stylised eye; this lands it on the iris,
  //  upper-left of the pupil, in the same place in both eyes; lid shadow from the shadow map is kept, softened)
  vec3 Lc = normalize(mix(L, v, 0.62));
  float hk = specGGX(ng, v, Lc, 0.085)*0.11*PI;     // render3: a little larger / softer catchlight
  vec3 key = S.keyC*mix(0.35, 1.0, S.shw.x)*min(hk, 4.0)*mix(0.3, 1.0, open);
  // render5 (eye_k2.x/y, judge r4 "weak catchlight at bust distance"): the reference's catchlight is a LARGE soft
  // window (a rounded rectangle, upper-left of the pupil) + a second small one lower-right.  An area light seen in the
  // cornea mirror: the reflected ray inside a soft rectangle about the (camera-pulled) key direction.
  if(uEyeK2.x > 0.0 || uEyeK2.y > 0.0){
    vec3 up0 = uHeadR[1];
    vec3 wx = normalize(cross(up0, Lc)), wy = cross(Lc, wx);
    vec3 rl = r/max(dot(r, Lc), 1e-3);
    vec2 wq = vec2(dot(rl, wx), dot(rl, wy));
    float win = smoothstep(0.24, 0.10, abs(wq.x))*smoothstep(0.28, 0.12, abs(wq.y))*step(0.0, dot(r, Lc));
    if(uEyeK3.y > 0.0){                                   // render8: a soft round catchlight, placed
      vec2 wq2 = (wq - uEyeK3.zw)/uEyeK3.y;
      float disc = smoothstep(0.24, 0.13, length(wq2*vec2(1.0, 0.9)));
      float rect = smoothstep(0.24, 0.10, abs(wq2.x))*smoothstep(0.28, 0.12, abs(wq2.y));
      win = mix(rect, disc, uEyeK3.x)*step(0.0, dot(r, Lc));
    }
    key += S.keyC*win*uEyeK2.x*0.55*mix(0.4, 1.0, S.shw.x)*open;
    vec3 L2 = normalize(mix(normalize(uFillDir - 0.6*up0), v, 0.75));
    float d2 = dot(r, L2);
    key += S.keyC*smoothstep(0.9975, 0.9993, d2)*uEyeK2.y*0.45*open;
  }
  vec3 fill = uFillCol*min(specGGX(ng, v, uFillDir, 0.16)*0.02*PI, 0.25)*open;
  fragDOut = vec4(0.0);
  return vec4(env + key + fill, 0.0);
}

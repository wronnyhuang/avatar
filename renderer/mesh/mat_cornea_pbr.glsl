// mat_cornea_pbr.glsl — shade-eyes: the clear wet surface of the eye (cornea_l/r) for a realistic character
// (pbr.py; gated by mesh.pbr.cornea).  Spliced instead of mat_cornea.glsl.  Blended premultiplied (ONE,
// ONE_MINUS_SRC_ALPHA): rgb = what the surface reflects, alpha = its Fresnel (that much of the eyeball behind is not
// transmitted).  The surface is ANALYTIC, not the mesh's: the view ray through the fragment hits the union of the
// eyeball sphere (the tear film over the sclera) and the cornea dome (a sphere cap over the iris, the same dome
// mat_eye_pbr refracts through), normals blended across the limbus.
//   key    an AREA light: the reflected ray inside the key's disc (angular radius = uLightP.x x key_size) -> a crisp
//          catchlight with a soft edge (the film's roughness), the light's radiance = irradiance / solid angle
//   fill   a second, bigger and dimmer disc;  room  the radiance probe at the film's roughness;  Fresnel (F0 0.025)
//   lids   the key through the shadow map (lid + lash shadows), the room's reflection occluded under the upper lid
//          (its underside fills the reflected hemisphere there) and right at both lid margins
uniform mat3  uHeadR;
uniform float uHeadS;
uniform vec3  uEyeC[2];
uniform float uEyeR;
uniform float uLidTop[26], uLidBot[26];
uniform vec3  uEyeGX[2], uEyeGY[2], uEyeGZ[2];
uniform vec4  uEyeIris;
uniform vec4  uPbrEye;      // x ior, y anterior chamber depth (m, for a 6 mm iris)
uniform vec4  uPbrCornea;   // x roughness, y key gain, z room gain, w fill gain
uniform vec4  uPbrCornea4;  // x, y placed catchlight offset (screen), z on
uniform vec4  uPbrCornea3;  // x softbox shape (0 = disc, 1 = rounded rectangle), y aspect (height / width), z corner roundness
uniform vec4  uPbrCornea2;  // x key size (x the key's angular radius), y key pull toward the camera (0 = physical), z fill disc size (rad), w lid occlusion

int pcEye(vec3 p){ return distance(p, uEyeC[0]) < distance(p, uEyeC[1]) ? 0 : 1; }
vec3 pcLoc(vec3 d, int k){ return vec3(dot(d, uEyeGX[k]), dot(d, uEyeGY[k]), dot(d, uEyeGZ[k])); }
vec3 pcWld(vec3 l, int k){ return uEyeGX[k]*l.x + uEyeGY[k]*l.y + uEyeGZ[k]*l.z; }
vec2 pcLids(vec3 p){
  int k = pcEye(p);
  vec3 d = normalize(transpose(uHeadR)*(p - uEyeC[k]));
  float az = degrees(atan(d.x, d.z)), el = degrees(asin(clamp(d.y, -1.0, 1.0)));
  float f = clamp((az + 66.0)/11.0, 0.0, 11.999);
  if(isnan(f)) f = 6.0;
  int i = int(f); float t = fract(f);
  return vec2(mix(uLidTop[k*13 + i], uLidTop[k*13 + i + 1], t) - el, el - mix(uLidBot[k*13 + i], uLidBot[k*13 + i + 1], t));
}
// nearest crossing (t >= 0) of the ray o + t d with a sphere (centre C, radius R), -1 if none
float pcSphere(vec3 o, vec3 d, vec3 C, float R){
  vec3 oc = o - C; float b = dot(oc, d), c = dot(oc, oc) - R*R, h = b*b - c;
  if(h < 0.0) return -1.0;
  h = sqrt(h);
  float t = -b - h;
  return t >= 0.0 ? t : (-b + h >= 0.0 ? -b + h : -1.0);
}
// a disc light seen in a mirror: 1 inside angular radius th, soft edge e
float pcDisc(vec3 r, vec3 l, float th, float e){
  float d = acos(clamp(dot(r, l), -1.0, 1.0));
  return smoothstep(th + e, max(th - e, 0.0), d);
}

// a rounded-rectangle softbox seen in a mirror: half sizes (tan) hw x hh about l, 'up' = the box's vertical
float pcRect(vec3 r, vec3 l, vec3 up, float hw, float hh, float e, float rnd){
  float c = dot(r, l);
  if(c <= 0.0) return 0.0;
  vec3 wx = normalize(cross(up, l)), wy = cross(l, wx);
  vec2 q = abs(vec2(dot(r, wx), dot(r, wy))/c);
  float rr = rnd*min(hw, hh);
  vec2 d = q - vec2(hw, hh) + rr;
  float sd = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - rr;   // signed distance (tan units)
  return smoothstep(e, -e, sd);
}
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}

vec4 matShade(SIn S, inout vec4 fragDOut){
  S.shw = pbrShadow(S.p, S.ng, S.shw);                                   // + the head cascade
  vec3 L = uKeyDir, v = S.v;
  int k = pcEye(S.p);
  vec3 n = normalize(S.p - uEyeC[k]);
  float onDome = 0.0;
  if(uEyeIris.x > 0.0){
    float rI = uEyeIris.x, zL = uEyeIris.y;
    float h = max(uPbrEye.y, 1e-4)/0.006*rI;
    float Rc = (rI*rI + h*h)/(2.0*h), zc = zL + h - Rc;
    float Rb = length(vec2(rI, zL));                        // the ball through the limbus
    vec3 lv = pcLoc(v, k);
    vec3 o = pcLoc(S.p - uEyeC[k], k) + lv*(3.0*Rb);         // start in front of the eye, march toward it
    vec3 d = -lv;
    float tb = pcSphere(o, d, vec3(0.0), Rb);
    float td = pcSphere(o, d, vec3(0.0, 0.0, zc), Rc);
    vec3 nb = tb >= 0.0 ? normalize(o + d*tb) : normalize(pcLoc(n, k));
    vec3 nl = nb;
    if(td >= 0.0){
      vec3 qd = o + d*td;
      float w = smoothstep(zL - 0.02*rI, zL + 0.10*rI, qd.z);  // the dome above the limbus plane (its sulcus soft)
      if(tb < 0.0 || td <= tb + 0.15*rI) nl = normalize(mix(nb, (qd - vec3(0.0, 0.0, zc))/Rc, w));
      onDome = w;
    }
    n = pcWld(nl, k);
  }
  if(dot(n, v) < 0.02) n = normalize(n + v*(0.02 - dot(n, v)));
  float nv = clamp(dot(n, v), 1e-3, 1.0);
  vec3 r = reflect(-v, n);
  float F = 0.025 + 0.975*pow(1.0 - nv, 5.0);
  vec2 lid = pcLids(S.p);
  float open = smoothstep(0.3, 3.0, lid.x)*smoothstep(0.0, 1.5, lid.y);
  // the upper lid's underside fills the reflected sky near it; the lower half reflects the (dim) cheek less strongly
  float up = dot(r, uHeadR[1]);
  float envOcc = mix(1.0, smoothstep(0.0, 16.0, lid.x)*mix(1.0, 0.5, clamp(up, 0.0, 1.0)) + 0.15*smoothstep(-0.2, -0.7, up),
                     uPbrCornea2.w);
  envOcc = clamp(envOcc, 0.0, 1.0);
  float rough = clamp(uPbrCornea.x, 0.005, 0.5);
  vec3 env = uProbeP.x > 0.5 ? envProbe(r, rough)*uAmbGain*uAmbTint : envSpec(S.p, r);
  vec3 col = env*uPbrCornea.z*F*max(envOcc, 0.35);
  // key: an area light (disc) -- radiance = irradiance / solid angle; the catchlight's edge from the film roughness
  // key pull (an eye-light cheat) on the cornea dome only: the tear film over the sclera keeps the physical reflection
  vec3 Lk = normalize(mix(L, v, clamp(uPbrCornea2.y, 0.0, 0.95)*onDome));
  // a placed catchlight (pbr.cornea.place = [x, y] in screen directions from the view ray, e.g. [-.35, .35]
  // = the 10-11 o'clock window of a portrait): the key's reflection direction set in camera space, shadowed as the key
  if(uPbrCornea4.z > 0.5) Lk = normalize(v + uPbrCornea4.x*uCamU + uPbrCornea4.y*uCamV);
  float th = max(atan(uLightP.x)*uPbrCornea2.x, 0.01);
  float e = 0.02 + 0.6*rough;
  float disc = pcDisc(r, Lk, th, e);
  if(uPbrCornea3.x > 0.0){                                            // the key as a softbox (same solid angle)
    float asp = max(uPbrCornea3.y, 0.2);
    float hw = tan(th)*0.886/sqrt(asp), hh = hw*asp;                  // area = the disc's (pi th^2 = 4 hw hh)
    disc = mix(disc, pcRect(r, Lk, uHeadR[1], hw, hh, e, uPbrCornea3.z), uPbrCornea3.x);
  }
  float Fk = 0.025 + 0.975*pow(1.0 - clamp(dot(n, Lk), 0.0, 1.0), 5.0);
  float lumK = 1.0/(th*th + e*e);
  vec3 key = S.keyC*disc*lumK*Fk*mix(0.0, 1.0, S.shw.x)*uPbrCornea.y;
  // + its GGX tail (the film is not a perfect mirror: a faint glow round the catchlight)
  key += S.keyC*S.shw.x*specGGX(n, v, Lk, rough*2.5)*Fk*PI*0.15*uPbrCornea.y;
  // fill: a big soft source -> a large dim disc
  float thf = max(uPbrCornea2.z, 0.02);
  vec3 Lf = uFillDir;
  if(uPbrCornea4.z > 0.5) Lf = normalize(v - 0.6*uPbrCornea4.x*uCamU - 0.9*uPbrCornea4.y*uCamV);   // placed: opposite, lower
  float fd = pcDisc(r, Lf, thf, 0.12 + rough);
  vec3 fill = uFillCol*fd/(thf*thf + 0.0144)*F*0.35*uPbrCornea.w;
  col += (key + fill)*open;
  col *= mix(0.25, 1.0, open);
  fragDOut = vec4(0.0);
  float a = clamp(F*open, 0.0, 1.0);
  return vec4(col, a);
}

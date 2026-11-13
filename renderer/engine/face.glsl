// ---------------------------------------------------------------------------
//  engine/face.glsl — what every shape character shares above the mouth: body pose, eyes, brows.
//  Include it from a character file with `#pragma include face`, after the defines below.
//  docs/SHAPE_GUIDE.md is the recipe.  Nothing here knows the host shape: distances come in as arguments.
//
//  Required defines
//    SHAPE_PIVOT  vec3   the base of the shape, head space.  head_* rotate about it
//    SHAPE_WORLD  vec3   where that base sits in the world
//    EYE_C        vec3   right eye centre, head space (the left mirrors it)
//    EYE_R        float  eyeball radius
//    BROW_C       vec3   middle of the right brow, head space (only x and y matter; depth comes from the host)
//  Optional: see the #ifndef block.
// ---------------------------------------------------------------------------
#ifndef SHAPE_YAW0
#define SHAPE_YAW0 0.0              // resting turn; 0 faces the camera
#endif
#ifndef SHAPE_BOB
#define SHAPE_BOB 0.0               // slow float, world units; a shape on the ground keeps 0
#endif
#ifndef SHAPE_SQUASH
#define SHAPE_SQUASH 0.013          // breath: y stretch at breath=1, volume kept
#endif
#ifndef SHAPE_JIGGLE
#define SHAPE_JIGGLE 0.006          // voice: the same, at voice=1
#endif
#ifndef EYE_LID_TH
#define EYE_LID_TH 0.0095           // lid thickness
#endif
#ifndef EYE_AU
#define EYE_AU 0.40                 // upper lid margin elevation, radians.  It must cut the top of the iris
#endif
#ifndef EYE_AL
#define EYE_AL (-0.62)              // lower lid margin
#endif
#ifndef EYE_AC
#define EYE_AC (-0.30)              // where they meet when closed
#endif
#ifndef EYE_TILT
#define EYE_TILT 0.0                // outer corners up when positive
#endif
#ifndef EYE_SQUINT
#define EYE_SQUINT 0.36             // squint = 1 raises the lower lid margin this far, radians
#endif
#ifndef EYE_IRIS
#define EYE_IRIS 0.82               // iris radius on the unit eyeball; large, so little white shows
#endif
#ifndef EYE_PUPIL
#define EYE_PUPIL 0.50
#endif
#ifndef EYE_COL_A
#define EYE_COL_A vec3(0.52,0.235,0.070)   // iris, the glow opposite the light
#endif
#ifndef EYE_COL_B
#define EYE_COL_B vec3(0.085,0.036,0.018)  // iris, outer ring and top
#endif
#ifndef EYE_COL_W
#define EYE_COL_W vec3(0.86,0.81,0.76)     // the white: warm, never paper
#endif
#ifndef LINER_COL
#define LINER_COL vec3(0.17,0.075,0.070)   // lid margin; warm, never black
#endif
#ifndef BROW_L
#define BROW_L 0.170                // half length
#endif
#ifndef BROW_W
#define BROW_W 0.028                // half width at the fat end
#endif
#ifndef BROW_H
#define BROW_H 0.020                // how far it stands off the host
#endif
#ifndef BROW_RISE
#define BROW_RISE 0.130             // travel at brow=1, up
#endif
#ifndef EYE_GLOW
#define EYE_GLOW vec3(0.0)          // look: light gathered in the lower iris (radiance)
#endif
#ifndef EYE_CATCH
#define EYE_CATCH 0.950             // look: cosine at the key catch-light's edge (smaller = a bigger light)
#endif
#ifndef EYE_GLOSS
#define EYE_GLOSS 1.0               // look: how much of the room the cornea reflects
#endif
#ifndef BROW_COL
#define BROW_COL vec3(0.40,0.205,0.170)
#endif

// ---- pose
mat3  gHR;                  // world -> head
vec3  gHO;                  // world position of the pivot
float gSqY, gSqX;           // squash scale
float gBlinkL, gBlinkR, gBrowL, gBrowR, gKnit, gSquint, gWide;
vec2  gGaze;

mat3 rotX(float a){ float c=cos(a),s=sin(a); return mat3(1,0,0, 0,c,s, 0,-s,c); }
mat3 rotY(float a){ float c=cos(a),s=sin(a); return mat3(c,0,-s, 0,1,0, s,0,c); }
mat3 rotZ(float a){ float c=cos(a),s=sin(a); return mat3(c,s,0, -s,c,0, 0,0,1); }

void faceInit(){
  // yaw>0 toward screen right, pitch>0 nose down, roll>0 crown toward screen right (docs/AUDIO.md)
  // a shape is its own body: body_yaw / body_pitch / body_lean turn it on top of the head columns, body_bounce lifts it
  mat3 F = rotY(SHAPE_YAW0 + uP[P_HEAD_YAW] + 0.35*uP[P_BODY_YAW])*rotX(uP[P_HEAD_PITCH] + 0.15*uP[P_BODY_PITCH])
          *rotZ(-uP[P_HEAD_ROLL] - 0.12*uP[P_BODY_LEAN]);
  gHR = transpose(F);
  gHO = SHAPE_WORLD + vec3(uP[P_HEAD_X], uP[P_HEAD_Y] + SHAPE_BOB*sin(uTime*0.83) + 0.10*uP[P_BODY_BOUNCE], uP[P_HEAD_Z]);
  gSqY = 1.0 + SHAPE_SQUASH*uP[P_BREATH] + SHAPE_JIGGLE*(uP[P_VOICE] - 0.3)*step(0.001, uP[P_VOICE]);
  gSqX = 1.0/sqrt(gSqY);
  gBlinkL = clamp(uP[P_BLINK_L], 0.0, 1.0); gBlinkR = clamp(uP[P_BLINK_R], 0.0, 1.0);
  gGaze = vec2(uP[P_GAZE_X], uP[P_GAZE_Y]);
  gBrowL = uP[P_BROW_L]; gBrowR = uP[P_BROW_R]; gKnit = clamp(uP[P_BROW_KNIT], 0.0, 1.0);
  gSquint = clamp(uP[P_SQUINT], 0.0, 1.0);
  gWide = clamp(uP[P_EYE_WIDE], 0.0, 1.0);
  gBrowL += 0.55*gWide; gBrowR += 0.55*gWide;           // surprise reads in the brows as much as the lids
}

// world -> head space, with the whole shape turning, squashing and stretching about its base
vec3 shapeSpace(vec3 pw){
  vec3 q = gHR*(pw - gHO);
  q /= vec3(gSqX, gSqY, gSqX);
  return q + SHAPE_PIVOT;
}

// ---- outputs of faceEval
float gFEye, gFBrow, gFMU, gFML, gFLid;
float gFOffLips = 1.0;      // 0 on and near the lips .. 1 clear of them.  lipsEval sets it

// skin = host with mouth, returned with the sockets, lids and brows on it.  host = the plain host distance.
float faceEval(vec3 q, float skin, float host){
  vec3 qa = vec3(abs(q.x), q.yz);
  float neg = q.x < 0.0 ? 1.0 : 0.0;
  float blink = mix(gBlinkR, gBlinkL, neg), bu = mix(gBrowR, gBrowL, neg);
  vec3 ea = qa - EYE_C;
  float rr = length(ea);
  gFEye = rr - EYE_R;
  skin = smax(skin, -(rr - (EYE_R + 0.004)), 0.030);
  vec3 el = ea;
  el.xy = rot(-EYE_TILT)*el.xy;
  float aU = mix(EYE_AU - 0.10*gSquint + 0.22*gWide, EYE_AC, blink) + 0.07*clamp(bu, -1.0, 1.0)*(1.0 - blink);
  float aL = mix(EYE_AL + EYE_SQUINT*gSquint - 0.12*gWide, EYE_AC, blink);
  float lids = sdEyelids(el, EYE_R + 0.004, EYE_LID_TH, aU, aL);
  gFLid = lids;
  gFMU = el.y*cos(aU) - el.z*sin(aU);
  gFML = -el.y*cos(aL) + el.z*sin(aL);
  skin = smin(skin, lids, 0.036);
  // a squint is the cheek rising under the eye as much as the lid.  It stays off the lips: a big low eye sits close above them
  vec2 ck = (ea.xy + vec2(0.0, 1.25*EYE_R))/(EYE_R*vec2(1.30,0.55));
  float cb = max(1.0 - dot(ck, ck), 0.0);
  skin -= 0.014*gSquint*(1.0 - blink)*cb*cb*gFOffLips;
#ifdef NOSE_P
  skin = smin(skin, sdEllipsoid(q - NOSE_P, NOSE_S), 0.030);    // optional: a small soft button
#endif

  // brow: an ellipsoid in (x, y, height above the host), so it lies on any surface.  Fat inner end, thin outer.
  vec2 b = qa.xy - BROW_C.xy;
  b.y -= BROW_RISE*bu - 0.050*gKnit;
  b.x += 0.052*gKnit;
#ifdef BROW_SAD_TILT
  // style (opt-in per character): knit with the brows RAISED is worry / sadness, the inner ends go UP, not down
  b = rot(-0.16*bu*0.0 + 0.10 - 0.62*gKnit + BROW_SAD_TILT*gKnit*smoothstep(0.0, 0.3, bu))*b;
#else
  b = rot(-0.16*bu*0.0 + 0.10 - 0.62*gKnit)*b;       // rest: inner ends a little high.  Knit: down and in
#endif
  float bt = clamp((b.x + BROW_L)/(2.0*BROW_L), 0.0, 1.0);
  b.y += 1.05*b.x*b.x;                                // arched
  float bw = BROW_W*(1.0 - 0.62*bt*bt);
  float hd = max(skin, 0.10 - q.z);                   // rides whatever is there, the lid included; never the far side
  float brow = sdEllipsoid(vec3(b, hd - 0.003), vec3(BROW_L, bw, BROW_H));
  gFBrow = brow;
  return smin(skin, brow, 0.010);
}

// paint eyes, lid liner and brows over whatever the character already painted.  toEye = the base camera aim.
void facePaint(vec3 q, vec3 rd, vec3 n, float skinD, inout Surf s){
  vec3 qa = vec3(abs(q.x), q.yz);
  vec3 ea = qa - EYE_C;
  float rr = length(ea);
  float lidOn = smoothstep(0.012, 0.002, gFLid);
  // the lash line: thick on the upper lid, a breath on the lower, the closed seam when they meet
  float linerU = smoothstep(0.017, 0.004, gFMU)*smoothstep(-0.012, -0.002, gFMU)*lidOn;
  float linerL = 0.26*smoothstep(0.009, 0.002, gFML)*smoothstep(-0.008, -0.001, gFML)*lidOn;
  float liner = max(linerU, linerL)*step(rr, EYE_R + 2.2*EYE_LID_TH + 0.012);
  s.alb = mix(s.alb, LINER_COL, 0.86*liner);
  s.spec *= 1.0 - liner;
  // skin right at the eye's edge is liner: a sliver of host between lid and eye must come out dark, never as a speck
  float rimE = smoothstep(0.0075, 0.0020, gFEye);
  s.alb = mix(s.alb, LINER_COL, 0.92*rimE);
  s.emit *= 1.0 - rimE; s.spec *= 1.0 - rimE;
  // crease above the lid
  float crease = smoothstep(0.012, 0.0, abs(rr - (EYE_R + 2.0*EYE_LID_TH + 0.022)))*smoothstep(0.0, 0.03, gFMU - 0.03)*step(0.0, ea.z);
  s.alb *= 1.0 - 0.035*crease;

  // brow
  float bm = smoothstep(0.0060, 0.0010, gFBrow);
  s.alb = mix(s.alb, BROW_COL, bm);
  s.spec = mix(s.spec, 0.12, bm); s.shin = mix(s.shin, 24.0, bm); s.mirror *= 1.0 - bm;
  s.sheen *= 1.0 - bm;
  s.emit = mix(s.emit, BROW_COL*0.08, bm);
  s.emit *= 1.0 - 0.8*liner;

  bool isEye = gFEye < 0.002 && gFEye <= skinD + 0.002;
  if(isEye){
    // gaze space: the eyes look at the camera by default, gaze adds to that
    float sx = q.x < 0.0 ? -1.0 : 1.0;
    vec3 ce = vec3(sx*EYE_C.x, EYE_C.y, EYE_C.z);
    vec3 toCam = rootGaze(gHR*(uCamPos - gHO)) + SHAPE_PIVOT - ce;
    vec3 eyeF = normalize(normalize(toCam) + vec3(gGaze.x*0.28, gGaze.y*0.22, 0.0));
    vec3 ex = normalize(cross(vec3(0,1,0), eyeF)), ey = cross(eyeF, ex);
    mat3 toE = transpose(mat3(ex, ey, eyeF));
    vec3 ee = (q - ce)/EYE_R;
    vec3 e = toE*ee;
    vec3 vv = toE*(gHR*rd);
    // a storybook eye, not a photograph: a smooth iris that glows low, opposite the light, with no fibres
    vec2 uv = e.xy + vv.xy*0.15*smoothstep(0.1, 0.5, e.z);
    float r = length(uv);
    float rn = clamp((r - EYE_PUPIL)/(EYE_IRIS - EYE_PUPIL), 0.0, 1.0);
    vec3 iris = mix(EYE_COL_B, EYE_COL_A, smoothstep(-0.25, -0.85, uv.y/EYE_IRIS - 0.1*uv.x)*0.92 + 0.08);
    iris = mix(iris, EYE_COL_B*0.8, smoothstep(0.62, 1.0, rn));
    float irisM = smoothstep(EYE_IRIS + 0.016, EYE_IRIS - 0.016, r)*step(0.0, e.z);
    float pup = smoothstep(EYE_PUPIL + 0.020, EYE_PUPIL - 0.020, r);
    vec3 c = mix(EYE_COL_W, iris, irisM);
    c = mix(c, vec3(0.008,0.006,0.007), pup*irisM);
    s.alb = c;
    s.occ = mix(0.66, 1.0, smoothstep(0.70, 0.12, ee.y));      // the upper lid's shadow
    // a soft contact line all around the opening.  Keyed to the skin as well as the lids: the inner corner's edge
    // is the socket, and normals there are broken, so a catch-light would fire as white flecks
    float lc = min(smoothstep(0.0, 0.022, gFLid), smoothstep(0.001, 0.016, skinD));
    s.alb *= mix(0.30, 1.0, lc);
    s.sheen = 0.0; s.wrap = 0.0; s.trans = 0.0;
    s.spec = 1.5*smoothstep(0.74, 0.42, ee.y)*lc; s.shin = 900.0;
    s.mirror = 0.10*smoothstep(0.74, 0.40, ee.y)*lc;           // low: a horizon in the set draws a hard line across the eye
    s.n = n;
    vec3 r2 = reflect(rd, n);
    float k1 = smoothstep(0.958, 0.972, dot(r2, normalize(uSunDir + vec3(0.05,0.2,0.3))));
    float k2 = smoothstep(0.986, 0.994, dot(r2, normalize(vec3(0.5,-0.05,0.85))));
    s.emit = vec3(1.0,0.97,0.92)*(3.6*k1 + 1.3*k2)*smoothstep(0.78, 0.52, ee.y)*lc + s.alb*0.06;
#ifdef LOOK_V2
    // look: the eyeball is a sphere, so shade it with its TRUE normal (the marched one carries the fur relief
    // and breaks up at the lid: the catch-lights came out as specks).  A clean glossy bead: one clear catch-light
    // from the key, a small second one, a soft sheet of sky high on the cornea, a little depth low in the iris.
    // Nothing at the limb (a bright edge reads as a ring round the eye: tired), no shade that narrows the eye.
    vec3 nE = normalize(transpose(gHR)*normalize(q - ce));
    vec3 rE = reflect(rd, nE);
    float lidL = smoothstep(0.92, 0.62, ee.y)*lc;
    float c1 = smoothstep(EYE_CATCH, EYE_CATCH + 0.012, dot(rE, normalize(uSunDir + vec3(0.05,0.2,0.3))));
    float c2 = smoothstep(0.978, 0.986, dot(rE, normalize(vec3(0.5,-0.05,0.85) - 0.9*vec3(uSunDir.x, 0.25, 0.0))));
    float sky = smoothstep(0.15, 0.85, rE.y)*smoothstep(0.98, 0.55, length(e.xy));
    s.n = nE;
    s.spec = 0.0; s.mirror = 0.0;
    s.occ = mix(0.80, 1.0, smoothstep(0.90, 0.30, ee.y));
    float glow = smoothstep(-0.10, -0.80, e.y - 0.25*e.x*sign(uSunDir.x))*irisM*(1.0 - 0.6*pup)*smoothstep(1.0, 0.75, length(e.xy));
    s.emit = vec3(1.0,0.98,0.95)*(5.0*c1 + 2.2*c2)*lidL + vec3(0.62,0.72,0.86)*0.030*EYE_GLOSS*sky*lc
           + EYE_GLOW*glow*lc + s.alb*0.06;
#endif
  }
}

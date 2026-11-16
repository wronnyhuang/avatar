// mat_generic.glsl — render-face: teeth, gums, tongue, mouth interior (MAT_MOUTH), buttons, shoes.  MAT_GENERIC.
//  Mouth parts (any fragment near the lip seam, from eyesmouth.py's per-frame lip points): the light that reaches the
//  inside of the mouth falls off with depth behind the lip seam, scaled by how wide the lips are open (a closed-ish
//  mouth is dark right behind the lips; a wide "ah" lets light onto the front teeth and the tongue), plus the upper
//  lip's soft shadow on the upper teeth.  Teeth are wet warm off-white (never chalk), tongue salmon with a sheen.
uniform mat3  uHeadR;
uniform float uHeadS;
uniform vec3  uMouthC, uMouthU, uMouthL;
uniform float uMouthW;
uniform vec4 uMouthK2;        // render7 (mesh.face.mouth_k2): teeth x exposure, y shade toward the corners / back, z upper-lip shade
uniform vec4 uMouthK;         // render4 (mesh.face.mouth_k): x key through the lip opening (0 = shadow map only), y cavity floor gain
void matUV(inout vec2 uv, vec3 v, vec3 T, vec3 B, vec3 ng){}

// x = light reaching here (0 dark .. 1 open air), y = mouth-part weight (0 = not a mouth part)
vec2 mouthLight(vec3 p, vec3 n){
  if(uHeadS <= 0.0 || uMouthW <= 0.0) return vec2(1.0, 0.0);
  mat3 Ri = transpose(uHeadR);
  vec3 q = Ri*(p - uMouthC)/uHeadS;                 // metres, head frame, origin at the lip seam
  float hw = 0.5*uMouthW/uHeadS;
  if(length(q) > 3.0*hw) return vec2(1.0, 0.0);
  vec3 u = Ri*(uMouthU - uMouthC)/uHeadS, l = Ri*(uMouthL - uMouthC)/uHeadS;
  float open = max(u.y - l.y, 0.0);                 // lip gap, metres
  float behind = max(-q.z, 0.0);
  // the opening admits light down to a depth ~ the gap (a cone); the back of the mouth is near black
  float reach = 0.004 + 0.9*open;            // ( render look-dev: a lit warm interior, not a black hole)
  float lit = exp(-behind/reach);
  // lateral: toward the cheeks it is darker (the lip corners shade it)
  lit *= mix(1.0, 0.45, smoothstep(0.35*hw, 1.0*hw, abs(q.x)));
  // the upper lip's shadow on the upper teeth (light comes from above)
  float underU = smoothstep(0.0, 0.004, u.y - q.y)*smoothstep(0.007, 0.0, u.y - q.y);
  lit *= mix(1.0, 0.55, underU*smoothstep(0.0, 0.003, behind + 0.003));
  // the top of the cavity sits in the upper lip / teeth shadow (light comes from above): grade down from the lip
  lit *= mix(1.0, mix(0.3, 1.0, smoothstep(0.0, 0.6*open + 0.002, u.y - q.y)), smoothstep(0.002, 0.008, behind));
  return vec2(clamp(lit, 0.10, 1.0), 1.0);
}

vec4 matShade(SIn S, inout vec4 fragDOut){   // inout (an out param is undefined until written: NVIDIA gave 0 -> mesh depth layer 0 -> no SSS, no depth)
  vec3 L = uKeyDir, n = S.n, v = S.v;
  vec2 ml = mouthLight(S.p, n);
  float mouth = ml.y;
  vec3 alb = S.alb;
  float rough = S.rough;
  float f0 = 0.04*S.specL;
  if(mouth > 0.5){
    // warm the teeth off the chalk white, saturate the tongue / gums a touch
    float bright = dot(alb, vec3(0.333));
    alb *= bright > 0.6 ? mix(vec3(0.90, 0.85, 0.78), vec3(1.0, 0.97, 0.91), clamp(uMouthK.z, 0.0, 1.0)) : vec3(1.0, 0.92, 0.92);   // render5: mouth_k.z bright warm-white teeth
    mat3 Ri = transpose(uHeadR);
    vec3 qm = Ri*(S.p - uMouthC)/uHeadS;
    // render3: no painted seams between the teeth -- the asset has 24 real teeth + gums since v4; only the
    // tongue's midline groove stays procedural
    if(bright <= 0.6) alb *= mix(1.0, 0.88, smoothstep(0.006, 0.0, abs(qm.x)));
    rough = min(rough, 0.32);
  }
  float ndl = dot(n, L);
  float w = uMatP.z*0.3;
  float dif = clamp((ndl + w)/(1.0 + w), 0.0, 1.0);
  float occ = ml.x;
  // render5 (mouth_k.z): the front teeth sit right behind the lips -- in the reference they read bright, clean
  if(mouth > 0.5 && dot(S.alb, vec3(0.333)) > 0.6) occ = mix(occ, max(occ, 0.75), clamp(uMouthK.z, 0.0, 1.0));
  // render7: the v7 full upper row read as stark white pegs: teeth sit back in the mouth -- toward the corners
  // they turn into the cheeks' shade, the upper row sits under the upper lip; and an overall exposure
  float teethK = 1.0;
  if(mouth > 0.5 && dot(S.alb, vec3(0.333)) > 0.6){
    mat3 Ri7 = transpose(uHeadR);
    vec3 q7 = Ri7*(S.p - uMouthC)/uHeadS;
    vec3 u7 = Ri7*(uMouthU - uMouthC)/uHeadS;
    float hw7 = 0.5*uMouthW/max(uHeadS, 1e-6);
    teethK = uMouthK2.x*mix(1.0, 1.0 - uMouthK2.y, smoothstep(0.15*hw7, 0.85*hw7, abs(q7.x)));
    teethK *= mix(1.0, 1.0 - uMouthK2.z, smoothstep(0.0035, 0.0, u7.y - q7.y)*smoothstep(-0.002, 0.001, u7.y - q7.y));
  }
  // render4 (brief item 5, the near-black tongue): inside the mouth the key's shadow map always says "occluded"
  // (the lips / nose / head sit between the key and anything behind the seam), so the mouth model (occ) carries the
  // occlusion alone there and the key reaches the tongue / lower teeth through the lip opening
  float shk = mouth > 0.5 ? mix(S.shw.x, 1.0, uMouthK.x) : S.shw.x;
  vec3 diffuse = alb*(S.keyC*dif*shk*occ + uFillCol*clamp(dot(n, uFillDir)*0.5 + 0.5, 0.0, 1.0)*0.5*occ + S.amb*S.ao*occ);
  if(mouth > 0.5){
    // light scattered inside the mouth is red (soft tissue): the dim parts go warm red-brown, not grey
    float ambL = dot(S.amb, vec3(0.30, 0.59, 0.11));   // render3: level, not the (cool-balanced) room colour
    diffuse += alb*vec3(0.30, 0.08, 0.055)*ambL*smoothstep(0.6, 0.05, occ)*0.8;
  }
  float sp = specGGX(n, v, L, rough);
  vec3 F = F_Schlick(vec3(f0), clamp(dot(v, normalize(v + L)), 0.0, 1.0));
  vec3 r = reflect(-v, n);
  diffuse *= teethK;                                                   // render7
  vec3 col = diffuse*S.ao + S.keyC*F*sp*S.shw.x*PI*occ
           + envSpec(S.p, r)*(f0 + 0.5*pow(1.0 - S.nv, 5.0))*S.specL*S.ao*(1.0 - rough)*occ*occ;
  // render3: a mouth part never drops to the grade's black floor (the tongue under the key's head shadow with
  // occ at its 0.10 clamp read pure black in the open shots): a dim warm floor from the room's level
  if(mouth > 0.5) col = max(col, alb*dot(S.amb, vec3(0.30, 0.59, 0.11))*0.45*uMouthK.y*vec3(1.0, 0.85, 0.8));
#if defined(MAT_MOUTH)
  // render3: the cavity reads warm red-brown and dim, never the grade's black floor (its albedo is a deep red
  // ~0.07 linear: lit by the faint light through the lips it was 0 -> the teal black of the grade at card size)
  col = col*0.6 + alb*(2.2*dot(S.amb, vec3(0.30, 0.59, 0.11)) + 0.25*dot(S.keyC, vec3(0.30, 0.59, 0.11)))*mix(0.35, 1.0, occ)*vec3(1.0, 0.85, 0.8);
  col *= 1.0 + uMouthK.w*vec3(1.0, 0.75, 0.65)*mix(0.5, 1.0, occ);     // render5 (mouth_k.w): a warm, lit interior
#else
  col += rimLight(S)*(1.0 - mouth);
#endif
  return vec4(col*S.alpha, S.alpha);
}

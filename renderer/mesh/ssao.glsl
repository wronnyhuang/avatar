// ---------------------------------------------------------------------------
// mesh/ssao.glsl -- shade-post: contact AO on the mesh layer (GATED: mesh.pbr.ao; character.py builds it only then).
//  Volumetric obscurance with line samples (Loos & Sloan 2010) on the resolved mesh depth (t_md: (t*a, ., ., a), fp16 ray
//  distance): needs no normals (the fp16 depth is far too coarse for per-pixel normals at 1080p), a flat or tilted surface
//  reads exactly open (the line samples cancel), a crease / the neck under the chin / the collar on the neck / the
//  hairline under the hair read occluded.  Half resolution, 12 samples on a per-pixel rotated Vogel disc, then a separable
//  depth-aware blur (2 passes).  The composite (FS_COMP, PBR_AO) multiplies the mesh colour by it.
//    SSAO_MAIN   r = obscurance-derived AO (1 = open), g = view depth (for the blur)
//    SSAO_BLUR   9 taps along uDir, weights by depth similarity
// ---------------------------------------------------------------------------
uniform vec2  uRes;          // this pass's target size
uniform vec2  uFull;         // the mesh layer's size
uniform vec3  uCamPos, uCamU, uCamV, uCamW;
uniform float uFl;
out vec4 fragColor;
vec3 rayAt(vec2 pix){ vec2 uv = (2.0*pix - uFull)/uFull.y; return normalize(uv.x*uCamU + uv.y*uCamV + uFl*uCamW); }

#ifdef SSAO_MAIN
uniform sampler2D uMD;
uniform vec4 uAo;            // x radius (scene units), y strength, z far-occluder cut (x radius), w min radius px
uniform vec2 uAo2;           // x floor (the darkest AO), y hole test (0 = off): a pixel with ~every sample in front of it
                             // (the open-mouth cavity, a nostril) is a hole that models its own occlusion -> left open
float zAt(vec2 pix, out float a){
  vec4 d = texture(uMD, pix/uFull);
  a = d.a;
  if(d.a < 0.5) return 1e4;
  float t = d.r/d.a;
  return t*dot(rayAt(pix), uCamW);
}
void main(){
  vec2 pix = gl_FragCoord.xy*(uFull/uRes);
  float a0;
  float z0 = zAt(pix, a0);
  if(a0 < 0.5){ fragColor = vec4(1.0, 1e4, 0.0, 1.0); return; }
  float R = uAo.x;
  float rpx = R*uFl*uFull.y*0.5/max(z0, 1e-3);
  if(rpx < uAo.w){ fragColor = vec4(1.0, z0, 0.0, 1.0); return; }
  rpx = min(rpx, 0.12*uFull.y);
  float rot = fract(52.9829189*fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))))*6.2831853;
  const int N = 12;
  float occ = 0.0, tot = 0.0, nf = 0.0, nv = 0.0;
  for(int i = 0; i < N; i++){
    float rr = sqrt((float(i) + 0.5)/float(N));
    float an = float(i)*2.39996 + rot;
    vec2 o = vec2(cos(an), sin(an))*rr;
    float h = sqrt(max(1.0 - rr*rr, 0.0))*R;           // the sphere's half chord at this offset
    float a;
    float zs = zAt(pix + o*rpx, a);
    float w = 2.0*h;
    // the line [z0 - h, z0 + h] is solid behind the surface zs; open where nothing (a = 0) or far behind
    float oc = clamp(z0 + h - max(zs, z0 - h), 0.0, w);
    // an occluder far in front of the sphere (a hand / hair lock a hand-width away) is not contact: fade to neutral
    float dz = z0 - zs;
    oc = mix(oc, h, smoothstep(uAo.z*R, 2.0*uAo.z*R, dz));
    if(a < 0.5) oc = 0.0;
    else { nv += 1.0; nf += step(0.3*R, dz); }
    occ += oc; tot += w;
  }
  float f = occ/max(tot, 1e-6);                      // 0.5 = flat, > 0.5 = a crease
  float ao = clamp(1.0 - uAo.y*max(2.0*f - 1.0, 0.0), uAo2.x, 1.0);
  if(uAo2.y > 0.0) ao = mix(ao, 1.0, uAo2.y*smoothstep(0.6, 0.85, nf/max(nv, 1.0)));
  fragColor = vec4(ao, z0, 0.0, 1.0);
}
#endif

#ifdef SSAO_BLUR
uniform sampler2D uSrc;
uniform vec2 uDir;           // texel step (1/size along x or y)
void main(){
  vec2 uv = gl_FragCoord.xy/uRes;
  vec2 c = texture(uSrc, uv).rg;
  if(c.g > 9e3){ fragColor = vec4(1.0, c.g, 0.0, 1.0); return; }
  float s = 0.0, ws = 0.0;
  for(int i = -4; i <= 4; i++){
    vec2 v = texture(uSrc, uv + uDir*float(i)).rg;
    float w = exp(-float(i*i)/8.0)*exp(-abs(v.g - c.g)/(0.01*c.g + 1e-4));
    if(v.g > 9e3) w = 0.0;
    s += v.r*w; ws += w;
  }
  fragColor = vec4(s/max(ws, 1e-6), c.g, 0.0, 1.0);
}
#endif

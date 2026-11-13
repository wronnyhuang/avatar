// ---------------------------------------------------------------------------
//  animals/orb.glsl — a luminous pearl with a face.  The first shape character, and the worked example
//  for docs/SHAPE_GUIDE.md.  Head space: origin at the centre of the pearl, +x screen right, +y up, +z
//  toward the camera.  All mouth and face mechanics live in engine/lips.glsl and engine/face.glsl; this
//  file only says what the host is, where the features sit, and what the pearl is made of.
// ---------------------------------------------------------------------------
const float R = 0.70;                               // the host: a sphere

#define SHAPE_PIVOT vec3(0.0,-0.70,0.0)
#define SHAPE_WORLD vec3(0.0,-0.66,0.0)
#define SHAPE_BOB   0.012
// a storybook face, not a person: high forehead, big flush button eyes low and wide, no nose
#define EYE_R 0.180
#define EYE_C vec3(0.262,0.022,0.510)
#define EYE_AU 0.62
#define EYE_AL (-0.66)
#define EYE_AC (-0.34)
#define EYE_TILT 0.10
#define EYE_IRIS 0.86
#define EYE_PUPIL 0.50
#define EYE_COL_A vec3(0.70,0.36,0.11)
#define EYE_COL_B vec3(0.16,0.070,0.028)
#define BROW_W 0.023
#define BROW_L 0.150
#define BROW_COL vec3(0.47,0.27,0.235)
#define BROW_C vec3(0.262,0.262,0.0)
#define LIP_TILT 0.10                               // aims at the camera; the host normal here leans 0.42
#define LIP_O vec3(0.0,-0.3160,0.6246)              // on the surface: R*(0, -sin 0.468, cos 0.468)
#define LIP_W 0.300                                 // the mouth of a shell, not a person: wide, fine lips, a tall opening
#define LIP_HU (0.054*LIPU)                         // lips like the rim of a shell
#define LIP_HL (0.066*LIPU)
#define LIP_TU (0.020*LIPU)
#define LIP_TL (0.024*LIPU)
#define LIP_BOW 0.30                                // a hint of a bow: any more and a person looks out of the pearl
#define LIP_SQUARE 0.44
#define LIP_OPEN_Y 1.10
#define LIP_PUSH (0.080*LIPU)
#define LIP_SKIN 0.70
#define LIP_BLEND (0.012*LIPU)
#define LIP_TOOTH_Z (-0.090*LIPU)                   // deeper than the default: the tilted frame runs out of the sphere below
#define LIP_TEETH_H (0.062*LIPU)
#define LIP_SCALLOP 1.4                             // a row of seed pearls
#define LIP_GROOVE 0.20
#define LIP_COL_U vec3(0.80,0.42,0.50)
#define LIP_COL_L vec3(0.86,0.49,0.55)
#define LIP_COL_LINE vec3(0.17,0.055,0.11)
#define LIP_LINE 1.9                                // a pale lip on a pale pearl: the closed line has to carry the mouth
#define LIP_LINE_SOFT 0.42
#define LIP_FLOW 0.5
#define LIP_COL_CAVE vec3(0.40,0.17,0.30)           // the inside of the shell: plum, never red
#define LIP_COL_DEEP vec3(0.105,0.040,0.095)
#define LIP_COL_GUM  vec3(0.58,0.32,0.42)
#define LIP_COL_TONGUE vec3(0.82,0.46,0.52)
#define LIP_COL_TOOTH vec3(0.93,0.90,0.84)
#define LIP_SPEC 0.30
#define LIP_SHIN 90.0
#define LIP_MIRROR 0.0
#define LIP_SHEEN 0.10
#define LIP_WRAP 0.25
#define LIP_EMIT vec3(0.16,0.12,0.13)
#pragma include face
#pragma include lips
// real 3-D arms with mitten hands on the contract's hand columns (engine/shape_arms.glsl, docs/RIG.md)
#define SA_SHO vec3(0.600,-0.300,0.050)
#define SA_REST vec3(0.140,-0.360,0.220)
#define SA_LEN 0.800
#define SA_R 0.086
#define SA_HAND 0.170
#define SA_ALB vec3(0.86,0.79,0.74)
#define SA_HALB vec3(0.90,0.77,0.74)
#pragma include shape_arms

float gSkin, gBare;
vec3  gQ;

void charInit(){ faceInit(); lipsInit();  saInit(); }

void orbEval(vec3 pw){
  vec3 q = shapeSpace(pw);
  gQ = q;
  float host = length(q) - R;
  float skin = lipsEval(q, host);
  skin = faceEval(q, skin, host);
  gSkin = skin;
  gBare = min(gLipBare, gFEye);
}
void charEval(vec3 p){
  orbEval(p);
  // bare skin, so: fur length a hair below zero and skin just under the bare parts (see monkey.glsl)
  gcBase = smin(min(gSkin, gBare - 0.0007), saDist(p), SA_SKIN); gcBare = gBare; gcRest = gQ; gcLen = -0.0006;
}
float charTuft(){ return 0.0; }

//@shadow-begin
float charShadowHost(vec3 pw){ return length(gHR*(pw - gHO) + SHAPE_PIVOT) - R*0.965; }
float charShadow(vec3 pw){ return min(charShadowHost(pw), saShadow(pw)); }
//@shadow-end

void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  orbEval(pw);
  if(saDist(pw) < gSkin){ saSurface(pw, rd, s); return; }
  vec3 q = gQ, n = s.n;
  // everything optical follows the HOST's normal, not the shaded one: features must not grow a rainbow fringe
  vec3 nh = normalize(q*gHR);
  float nv = clamp(dot(nh, -rd), 0.0, 1.0);
  float fr = 1.0 - nv;

  // nacre: a thin film.  The colour shifts ACROSS the form as well as toward the rim: gold-rose high on the lit
  // side, cream in the middle, lilac then cool mint low on the shadow side, slowly swirled over the surface
  vec3 qf = q; qf.xy -= lipsFlow();                                // the skin's paint rides with the jaw and the corners
  float sw = vnoise3(qf*2.3 + 1.7)*0.62 + vnoise3(qf*5.1 + 4.0)*0.38;
  float along = dot(nh, normalize(vec3(0.55,-0.70,0.0)));          // -1 lit crown .. +1 shadow side
  float ph = 0.46*fr*fr + 0.30*sw + 0.20*along + 0.20;
  vec3 film = 0.5 + 0.5*cos(6.2832*(ph + vec3(0.00,0.30,0.58)));
  vec3 tint = 1.0 + (0.30 + 0.34*smoothstep(0.10, 0.85, fr))*(film - 0.5)*vec3(1.0,0.74,1.0);
  vec3 pearl = vec3(0.90,0.825,0.765);
  s.alb = pearl*tint;
  // cheeks, a soft rose that rises with `cheek`
  vec2 ck = vec2(abs(qf.x) - 0.435, qf.y + 0.170);
  float blush = exp(-dot(ck*vec2(1.0,1.2), ck*vec2(1.0,1.2))/0.019);
  s.alb = mix(s.alb, vec3(0.86,0.40,0.42), (0.34 + 0.34*gLCheek + 0.12*gLSmile)*blush);

  // weight comes from the falloff: little wrap, and the fill dies away on the side the key cannot reach
  float keyside = dot(nh, normalize(vec3(-0.52,0.62,0.58)));
  s.occ = mix(0.26, 1.0, smoothstep(-0.80, 0.60, keyside));
  s.spec = 0.0; s.mirror = 0.0; s.sheen = 0.0; s.wrap = 0.10; s.fur = 0.0;

  // the shell mirrors the world: rosy sky above the equator, the darker water below it, a bright horizon line
  // between them, and the sun's glow where the rim looks back past the shape
  vec3 r = reflect(rd, nh);
  vec3 sky = setSky(r);
  vec3 water = mix(SP_BOUNCE*0.30, sky*0.30, smoothstep(-0.55, 0.0, r.y));
  vec3 env = mix(water, sky, smoothstep(-0.05, 0.09, r.y));
  float Fr = 0.050 + 0.62*pow(fr, 4.0);
  // a dark grazing band sells the sphere: the last of the rim sees mostly the shell's own shade
  float limb = 1.0 - 0.42*smoothstep(0.80, 1.0, fr);
  // one big soft window of light on the sun side, inside a broad bloom
  float hl = max(dot(r, normalize(vec3(-0.55,0.82,0.15))), 0.0);
  vec3 gloss = vec3(1.0,0.97,0.93)*(0.16*pow(hl, 7.0) + 0.30*pow(hl, 40.0) + 1.05*smoothstep(0.955, 0.988, hl));
  // inner glow, faint and warm, face on; plus the water's cool light rising under the belly
  float belly = smoothstep(0.30, -0.95, nh.y);
  s.alb *= limb*mix(vec3(0.60,0.57,0.66), vec3(1.0), smoothstep(-0.92, 0.25, keyside));   // the shadow side is a colour, not a grey
  s.emit = s.alb*vec3(0.036,0.026,0.028)*(0.4 + 0.6*nv) + env*Fr*tint*limb + gloss
         + vec3(0.052,0.050,0.066)*belly*(0.4 + 0.6*fr);

  lipsPaint(n, s);
  // the lips are the shell's own nacre, a turn further along the same film: they catch the sky off their own form
  float lipN = gPLip*(1.0 - gPLine);
  if(lipN > 0.0){
    vec3 rl = reflect(rd, n);
    float pl = ph + 0.42*(1.0 - dot(n, nh)) + 0.16;
    vec3 fl = 0.5 + 0.5*cos(6.2832*(pl + vec3(0.00,0.30,0.58)));
    s.alb *= mix(vec3(1.0), 1.0 + 0.62*(fl - 0.5), lipN);
    float hlL = max(dot(rl, normalize(vec3(-0.40,0.86,0.32))), 0.0);
    float crest = smoothstep(0.15, 0.55, gLE/max(gLH, 1e-3))*smoothstep(1.05, 0.70, gLE/max(gLH, 1e-3));
    s.emit += lipN*(setSky(rl)*(0.05 + 0.40*pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 3.0))*(0.6 + 0.8*fl)
                    + vec3(1.0,0.96,0.92)*0.34*smoothstep(0.90, 0.985, hlL)*crest);
  }
  facePaint(q, rd, n, gSkin, s);
}

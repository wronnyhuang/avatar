// Shared 3-D torso, two-bone arms and articulated paws.  The geometry adapts
// the round-3 fox/bear/rabbit armSolve: shoulder follows the hand, elbow bends
// around a pole with fixed bone lengths, wrist stays outside the body, and the
// digits open separately.  This is a small raymarch pass for the live K620.
float rigPart(vec3 p, out float material){
  material = 0.0;
  vec3 bound=max(abs(p-vec3(0.0,-0.68,0.0))-vec3(1.55,1.27,1.25),0.0);
  if(dot(bound,bound)>0.01)return length(bound);
  float d = 100.0;
  vec3 q = p;
  q.xz = rot(-0.38*uP[P_BODY_YAW])*(q.xz-vec2(0.0,-0.18))+vec2(0.0,-0.18);
  q.x -= 0.13*uP[P_BODY_LEAN];
  q.y -= 0.11*uP[P_BODY_BOUNCE]*sin(uTime*7.0);
  // The chest is behind the head and receives the same depth test as the limbs.
  float torso = sdEllipsoid(q-vec3(0.0,-1.22,-0.33),vec3(0.56,0.76,0.42));
  d = torso; material = 0.5;
  for(int i=0;i<2;i++){
    float side=i==0?-1.0:1.0;
    float raise=clamp(i==0?uP[P_ARM_L]:uP[P_ARM_R],0.0,1.0);
    float wave=clamp(i==0?uP[P_WAVE_L]:uP[P_WAVE_R],0.0,1.0);
    float point=clamp(i==0?uP[P_POINT_L]:uP[P_POINT_R],0.0,1.0);
    float open=clamp(i==0?uP[P_PAW_OPEN_L]:uP[P_PAW_OPEN_R],0.0,1.0);
    float up=max(max(raise,wave),max(point,open));
    up=max(up, uP[P_THUMBS_UP]*step(0.0,side));
    float clap=clamp(uP[P_CLAP],0.0,1.0);
    vec3 shoulder=vec3(side*(0.58+0.045*up),-0.65+0.09*up*up,-0.08+0.04*up);
    vec3 wrist=mix(vec3(side*0.83,-1.17,0.23),vec3(side*1.00,0.05,0.46),up);
    wrist.x+=side*wave*0.11*sin(uTime*16.0);
    wrist=mix(wrist,vec3(side*0.16,-0.55,0.72),clap*(0.5+0.5*sin(uTime*12.0)));
    // Fixed-length two-bone IK, adapted from the round-3 animal arms.  A soft
    // reach cap avoids an elbow snap and the pole keeps it out of the chest.
    const float upperLen=0.60, foreLen=0.57;
    vec3 reach=wrist-shoulder;
    float rawLen=max(length(reach),0.001);
    vec3 dir=reach/rawLen;
    float len=clamp(rawLen,0.20,0.97*(upperLen+foreLen));
    wrist=shoulder+dir*len;
    vec3 pole=vec3(side*(0.50+0.27*up*up),-0.78+0.20*up,-0.34+0.23*up);
    vec3 perp=normalize(pole-dir*dot(pole,dir));
    float along=(upperLen*upperLen-foreLen*foreLen+len*len)/(2.0*len);
    float bend=sqrt(max(upperLen*upperLen-along*along,0.0));
    vec3 elbow=shoulder+dir*along+perp*bend;
    float upper=sdTaper(q,shoulder,elbow,0.155,0.125);
    float lower=sdTaper(q,elbow,wrist,0.125,0.103);
    float arm=smin(upper,lower,0.075);
    if(arm<d){d=arm;material=lower<upper?1.55:1.0;}
    vec3 palm=wrist+normalize(wrist-elbow)*0.08;
    float paw=sdEllipsoid(q-palm,vec3(0.155,0.171,0.107));
    if(paw<d){d=paw;material=2.0;}
    if(point>0.05){
      float finger=sdTaper(q,wrist+vec3(side*0.08,0.04,0.025),
                          wrist+vec3(side*0.29,0.16,0.07),0.065,0.033)-0.015*point;
      if(finger<d){d=finger;material=2.0;}
    }
    if(open>0.05 || wave>0.05){
      for(int j=0;j<4;j++){
        float spread=float(j)-1.5;
        float fan=(0.065+0.08*open)*spread;
        vec3 root=palm+vec3(fan,0.12,0.028);
        vec3 knuckle=root+vec3(fan*0.27,0.105,0.015);
        vec3 tip=knuckle+vec3(fan*0.18,0.083,-0.025*(1.0-open));
        float finger=min(sdTaper(q,root,knuckle,0.044,0.034),
                         sdTaper(q,knuckle,tip,0.032,0.018));
        if(finger<d){d=finger;material=2.0;}
      }
    }
    if(uP[P_THUMBS_UP]>0.05 && side>0.0){
      float thumb=sdTaper(q,wrist+vec3(-0.07,0.06,0.045),
          wrist+vec3(-0.07,0.30,0.065),0.060,0.030);
      if(thumb<d){d=thumb;material=2.0;}
    }
  }
#if RIG_FUR == 1
  float wag=clamp(uP[P_TAIL_WAG],-1.0,1.0), lift=clamp(uP[P_TAIL_RAISE],0.0,1.0);
  float tail=sdTaper(q,vec3(0.48,-1.35,-0.35),
     vec3(1.04+0.20*wag*sin(uTime*11.0),-1.18+0.36*lift,-0.31),0.18,0.09);
  if(tail<d){d=tail;material=1.0;}
#endif
  return d;
}
float rigDistance(vec3 p){float mat;return rigPart(p,mat);}
void rigSurface(float material, vec3 p, inout Surf s){
  if(material<0.75){
    float center=1.0-smoothstep(0.13,0.47,abs(p.x));
    s.alb=mix(RIG_ARM,RIG_CHEST,center);
  }else{
    s.alb=material>1.3?RIG_PAW:RIG_ARM;
  }
  float grain=0.94+0.10*vnoise3(p*18.0);
  s.alb*=grain;
  s.spec=RIG_FUR==1?0.025:0.09;
  s.shin=RIG_FUR==1?14.0:28.0;
  s.wrap=0.18;
  s.sheen=RIG_FUR==1?0.35:0.12;
  s.fur=RIG_FUR==1?0.25:0.0;
  s.occ=0.9;
  s.mirror=0.0;
  s.T=normalize(vec3(0.1,0.9,0.05));
}

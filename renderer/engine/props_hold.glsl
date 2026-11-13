// ---------------------------------------------------------------------------
//  engine/props_hold.glsl — props, the part a character's HAND code reads (included BEFORE the animal by
//  scene_main.glsl; the models / attach / painter are engine/props.glsl, included after it).  propsPre() runs at the
//  start of charInit, before the arm solve.  Generated tables (engine/props_defs.py): PROPS_N, PR_PX[k] (column of
//  prop{k}_x; hl = +6, hr = +7), PR_PHH[k], PR_PS[k], PR_SIZE[k], PR_GR[k] (grip radius, unit frame), PR_GRIP[k].
//  Without a props file PROPS_N is 0 and nothing here changes a character.
// ---------------------------------------------------------------------------
#ifndef PROPS_N
#define PROPS_N 0
#endif
#define PROPS_HOLD 1

// per side (0 = screen left, _l): what the paw holds and the counting column.  A character's hand code may read these
// to close its digits round what it holds (gHoldR = radius of the part it closes on, scene units; gHoldG 0 power,
// 1 pinch, 2 two-hand) and to raise digits for counting (gCount 0..5: index, middle, ring, pinky, thumb).
float gHoldW[2];
float gHoldR[2];
int   gHoldG[2];
float gCount[2];

void propsPre(){
  for(int s=0;s<2;s++){ gHoldW[s] = 0.0; gHoldR[s] = 0.1; gHoldG[s] = 0; }
  gCount[0] = clamp(uP[P_FINGERS_L], 0.0, 5.0);
  gCount[1] = clamp(uP[P_FINGERS_R], 0.0, 5.0);
#if PROPS_N > 0
  for(int k=0;k<PROPS_N;k++){
    float wl = clamp(uP[PR_PX[k] + 6], 0.0, 1.0), wr = clamp(uP[PR_PX[k] + 7], 0.0, 1.0);
    int g = (wl > 0.05 && wr > 0.05) ? 2 : PR_GRIP[k];       // two paws on it: palms on its sides
    float r = PR_GR[k]*PR_SIZE[k];
    if(wl > gHoldW[0]){ gHoldW[0] = wl; gHoldR[0] = r; gHoldG[0] = g; }
    if(wr > gHoldW[1]){ gHoldW[1] = wr; gHoldR[1] = r; gHoldG[1] = g; }
  }
#endif
}

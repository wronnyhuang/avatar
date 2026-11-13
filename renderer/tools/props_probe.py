#!/usr/bin/env python3
"""props: read back the renderer's own geometry through a tiny GLSL pass (the same scene_main.glsl + animal +
props code the frame passes compile), on the CPU backend by default.

  python tools/props_probe.py height --character fox            # H (lowest point -> top of the head at the midline)
  python tools/props_probe.py grasp  --character fox --pose 'hand_l_x=0.5 hand_l_y=0.2'   # grasp points + frames
  python tools/props_probe.py props  --character fox --pose '...'  # every prop's centre (world) for a pose

Library use: Probe(character).grasp(vectors) -> array (N, 2 sides, 4 rows: pos, x, y, z of the frame)
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
from anim.schema import IDX, NPARAMS  # noqa: E402

HEIGHT_MAIN = """
out vec4 fragOut;
uniform vec2 uGrid;       // columns: x in [-uGrid.x, uGrid.x], z in [-uGrid.y, uGrid.y]
uniform vec2 uGridN;
float probeD(vec3 p){ charEvalA(p); return min(gcBare, gcBase - gcLen); }
void main(){
  setInit(); charInit();
  vec2 ij = floor(gl_FragCoord.xy);
  vec2 f = ij/max(uGridN - 1.0, vec2(1.0));
  float x = mix(-uGrid.x, uGrid.x, f.x), z = mix(-uGrid.y, uGrid.y, f.y);
  // from above
  float y = 3.5, top = -1e3;
  for(int i=0;i<400;i++){ float d = probeD(vec3(x, y, z)); if(d < 0.002){ top = y; break; } y -= max(d*0.7, 0.003); if(y < -6.0) break; }
  float yb = -6.0, bot = 1e3;
  for(int i=0;i<400;i++){ float d = probeD(vec3(x, yb, z)); if(d < 0.002){ bot = yb; break; } yb += max(d*0.7, 0.003); if(yb > 3.5) break; }
  fragOut = vec4(top, bot, x, z);
}
"""

GRASP_MAIN = """
out vec4 fragOut;
uniform float uSlot;
void main(){
  setInit(); charInit();
  int c = int(gl_FragCoord.x) % 8;
  int side = c/4, r = c % 4;
  vec3 pos; mat3 fr;
  PROPS_GRASP_FN(side, pos, fr);
  pos = rootFwd(pos); fr = gRootR*fr;
  fragOut = r == 0 ? vec4(pos, 1.0) : vec4(fr[r - 1], 0.0);
}
"""

PROPS_MAIN = """
out vec4 fragOut;
void main(){
  setInit(); charInit();
  int k = int(gl_FragCoord.x);
#if PROPS_N > 0
  fragOut = k < PROPS_N ? vec4(gPrC[k], gPrArt[k]) : vec4(0.0);
#else
  fragOut = vec4(0.0);
#endif
}
"""


def parse_pose(spec):
    v = np.zeros(NPARAMS, "f4")
    for tok in (spec or "").split():
        k, x = tok.split("=")
        v[IDX[k]] = float(x)
    return v


class Probe:
    def __init__(self, character, backend="cpu", threads=8):
        import render
        from render import Shaders, load_cfg, backend_env, VS_FULL, DEFAULT_QUALITY
        idx = backend_env(backend, threads)
        import moderngl
        self.mgl = moderngl
        self.cfg = cfg = load_cfg(character)
        q = dict(DEFAULT_QUALITY); q.update(cfg.get("quality", {}))
        cfg2 = dict(cfg); cfg2["quality"] = q
        self.sh = Shaders(cfg2)
        self.sh.head += "#define DOF_N 8\n#define VOL_NV 8\n"
        self.ctx = moderngl.create_standalone_context(backend="egl", device_index=idx)
        base = self.sh.frag("scene_main.glsl", "PASS_NONE")
        self.VS = VS_FULL
        self.src = base
        self.progs = {}
        self.quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        cam = cfg["camera"]
        self.cam_pos = np.array(cam["pos"], float)

    def prog(self, name, main):
        if name not in self.progs:
            p = self.ctx.program(vertex_shader=self.VS, fragment_shader=self.src + main)
            self.progs[name] = (p, self.ctx.vertex_array(p, [(self.quad, "2f", "in_pos")]))
        return self.progs[name]

    def _setP(self, p, vec):
        if "uP" in p:
            u = p["uP"]
            p["uP"].write(np.array(vec[:u.array_length], dtype="f4").tobytes())
        if "uCamPos" in p:
            p["uCamPos"].value = tuple(self.cam_pos)

    def height(self, gx=1.6, gz=1.6, n=(65, 33), vec=None):
        p, vao = self.prog("height", HEIGHT_MAIN)
        self._setP(p, np.zeros(NPARAMS, "f4") if vec is None else vec)
        p["uGrid"].value = (gx, gz); p["uGridN"].value = (float(n[0]), float(n[1]))
        tex = self.ctx.texture(n, 4, dtype="f4"); fb = self.ctx.framebuffer([tex])
        fb.use(); self.ctx.viewport = (0, 0, n[0], n[1])
        vao.render(self.mgl.TRIANGLE_STRIP)
        a = np.frombuffer(fb.read(components=4, dtype="f4"), "f4").reshape(n[1], n[0], 4)
        return a

    def grasp(self, vecs):
        """vecs (N, NPARAMS) -> (N, 2, 4, 3): per side pos, frame x, y, z (world = root applied)."""
        p, vao = self.prog("grasp", GRASP_MAIN)
        N = len(vecs)
        tex = self.ctx.texture((8*N, 1), 4, dtype="f4"); fb = self.ctx.framebuffer([tex])
        fb.use()
        for i, v in enumerate(vecs):
            self._setP(p, v)
            self.ctx.viewport = (8*i, 0, 8, 1)
            vao.render(self.mgl.TRIANGLE_STRIP)
        a = np.frombuffer(fb.read(components=4, dtype="f4", viewport=(0, 0, 8*N, 1)), "f4").reshape(N, 2, 4, 4)
        return a[..., :3]

    def props(self, vec):
        p, vao = self.prog("props", PROPS_MAIN)
        self._setP(p, vec)
        tex = self.ctx.texture((16, 1), 4, dtype="f4"); fb = self.ctx.framebuffer([tex])
        fb.use(); self.ctx.viewport = (0, 0, 16, 1)
        vao.render(self.mgl.TRIANGLE_STRIP)
        return np.frombuffer(fb.read(components=4, dtype="f4"), "f4").reshape(16, 4)


def measure_height(pr):
    a = pr.height()
    top, bot, xs, zs = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    mid = np.argmin(np.abs(xs[0] - 0.0))
    tops = top[:, mid]
    head_top = float(tops[tops > -1e2].max())
    b = bot[bot < 1e2]
    low = float(b.min())
    return dict(top=head_top, low=low, H=head_top - low, all_top=float(top[top > -1e2].max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["height", "grasp", "props"])
    ap.add_argument("--character", default="fox")
    ap.add_argument("--pose", default="")
    ap.add_argument("--backend", default="cpu")
    a = ap.parse_args()
    pr = Probe(a.character, a.backend)
    if a.what == "height":
        print(json.dumps(measure_height(pr)))
    elif a.what == "grasp":
        g = pr.grasp([parse_pose(a.pose)])[0]
        for s in range(2):
            print("side", "lr"[s], "pos", g[s, 0].round(4).tolist(), "x", g[s, 1].round(3).tolist(),
                  "y", g[s, 2].round(3).tolist(), "z", g[s, 3].round(3).tolist())
    else:
        print(pr.props(parse_pose(a.pose)).round(4))


if __name__ == "__main__":
    main()

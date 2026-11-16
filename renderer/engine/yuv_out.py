"""render-perf (PERF2): the engine's frame read back as I420 (planar YUV 4:2:0), converted on the GPU.

The live H.264 path used to read back RGBA8 (w*h*4 bytes) and convert on ONE CPU core in the encoder thread
(cv2 RGBA->I420, ~5-6 ms at 1080p, plus the 8.3 MB map/copy).  Here one tiny fragment pass writes the I420 bytes
straight into an R8 target of w x 1.5h (Y rows, then the U plane, then the V plane, exactly the memory layout of
cv2's COLOR_*2YUV_I420 / ffmpeg yuv420p), and the PBO readback moves 3.1 MB instead of 8.3 MB.

Colour: BT.601 limited range with cv2's coefficients (Y 16..235, Cb/Cr 16..240).  Chroma = the mean of each 2x2
block (cv2 takes the top-left pixel; the mean is the better 4:2:0 filter: less chroma crawl on hair / knit).

Use (an engine with f_out / pbo / w / h, i.e. RTEngine and subclasses):
    yo = YuvOut(eng)            # swaps eng.f_out's renderbuffer for a texture (same pixels), compiles the pass
    yo.on = True                # following frames: eng.render() returns uint8 (1.5h, w) I420 instead of (h, w, 3) RGB
rt_engine._draw / _fetch call yo.read(k) / yo.fetch(k) (two marked hunks); off = the RGB path, unchanged.
"""
import numpy as np

FS_I420 = """#version 330
uniform sampler2D uSrc;
uniform ivec2 uSize;
out vec4 fragColor;
float Yof(vec3 c){ return dot(c, vec3(0.256788, 0.504129, 0.097906)) + 16.0/255.0; }
float Uof(vec3 c){ return dot(c, vec3(-0.148223, -0.290993, 0.439216)) + 128.0/255.0; }
float Vof(vec3 c){ return dot(c, vec3(0.439216, -0.367788, -0.071427)) + 128.0/255.0; }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  int w = uSize.x, h = uSize.y;
  if (p.y < h) {                                   // Y plane: memory row r = image row r (top first) = GL row h-1-r
    fragColor = vec4(Yof(texelFetch(uSrc, ivec2(p.x, h - 1 - p.y), 0).rgb), 0.0, 0.0, 1.0);
    return;
  }
  int hw = w/2, hh = h/2, q = hw*hh;
  int i = (p.y - h)*w + p.x;                       // linear index into the chroma area (U plane, then V plane)
  int plane = i >= q ? 1 : 0;
  i -= plane*q;
  if (i >= q) { fragColor = vec4(0.5); return; }
  int cy = i / hw, cx = i - cy*hw;
  int x = 2*cx, y = h - 1 - 2*cy;                  // the 2x2 block's top-left (image) in GL rows
  vec3 c = texelFetch(uSrc, ivec2(x, y), 0).rgb + texelFetch(uSrc, ivec2(x + 1, y), 0).rgb
         + texelFetch(uSrc, ivec2(x, y - 1), 0).rgb + texelFetch(uSrc, ivec2(x + 1, y - 1), 0).rgb;
  c *= 0.25;
  fragColor = vec4(plane == 0 ? Uof(c) : Vof(c), 0.0, 0.0, 1.0);
}
"""


class YuvOut:
    def __init__(self, eng):
        import moderngl
        from render import VS_FULL
        self.eng = eng
        ctx = eng.ctx
        w, h = eng.w, eng.h
        if w % 2 or h % 2:
            raise ValueError("I420 needs even width and height")
        self.w, self.h = w, h
        # the post pass's target as a TEXTURE (the renderbuffer cannot be sampled); the same RGBA8 pixels
        self.t_out = ctx.texture((w, h), 4)
        self.t_out.filter = (moderngl.NEAREST, moderngl.NEAREST)
        old = eng.rb_out
        eng.f_out = ctx.framebuffer([self.t_out])
        eng.rb_out = self.t_out
        try:
            old.release()
        except Exception:
            pass
        self.rows = h*3//2
        self.t_yuv = ctx.texture((w, self.rows), 1)
        self.f_yuv = ctx.framebuffer([self.t_yuv])
        self.prog = ctx.program(vertex_shader=VS_FULL, fragment_shader=FS_I420)
        self.prog["uSrc"].value = 0
        self.prog["uSize"].value = (w, h)
        self.vao = ctx.vertex_array(self.prog, [(eng.quad, "2f", "in_pos")])
        self.pbo = [ctx.buffer(reserve=w*self.rows) for _ in range(len(eng.pbo))]
        self.slot = {}
        self.on = False

    def read(self, k):
        """Called where the engine reads f_out back: when on, convert + read I420 into this slot; -> did it."""
        self.slot[k] = bool(self.on)
        if not self.on:
            return False
        import moderngl
        ctx = self.eng.ctx
        self.f_yuv.use(); ctx.viewport = (0, 0, self.w, self.rows)
        self.t_out.use(0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.f_yuv.read_into(self.pbo[k], components=1, alignment=1)
        return True

    def has(self, k):
        return self.slot.get(k, False)

    def fetch(self, k):
        self.slot[k] = False
        return np.frombuffer(self.pbo[k].read(), dtype=np.uint8).reshape(self.rows, self.w)

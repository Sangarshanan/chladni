"""
Shared GPU helpers: blend modes, per-context state, and an immediate-mode
drawer for plain vertex geometry (soft round points, triangles, lines).
"""

import moderngl
import numpy as np

_MODES = {
    "fan": moderngl.TRIANGLE_FAN,
    "triangles": moderngl.TRIANGLES,
    "lines": moderngl.LINE_LOOP,
    "points": moderngl.POINTS,
}

_VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec3 in_color;
uniform float point_size;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    gl_PointSize = point_size;
    v_color = in_color;
}
"""

_FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
uniform bool soft;
uniform float opacity;
out vec4 f_color;
void main() {
    float a = opacity;
    if (soft) {
        float d = length(gl_PointCoord * 2.0 - 1.0);
        a *= 1.0 - smoothstep(0.5, 1.0, d);
        if (a <= 0.0) discard;
    }
    f_color = vec4(v_color * a, a);
}
"""


def set_blend(ctx, mode):
    """Select how the next draw composites: replace, alpha, add or screen."""
    if mode in (None, "replace", "none"):
        ctx.disable(moderngl.BLEND)
        return
    ctx.enable(moderngl.BLEND)
    if mode == "alpha":
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
    elif mode == "add":
        ctx.blend_func = moderngl.ONE, moderngl.ONE
    elif mode == "screen":
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_COLOR
    else:
        raise ValueError(f"unknown blend mode {mode!r} (replace, alpha, add, screen)")


class _Draw:
    """Draws position+colour vertex arrays through one persistent program and buffer."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER)
        self._vbo = None
        self._vao = None

    def draw(self, positions, colors, mode, size=1.0, opacity=1.0, pixel_scale=1.0):
        count = len(positions)
        if count == 0:
            return
        data = np.hstack([positions, colors]).astype("f4").tobytes()
        if self._vbo is None or self._vbo.size < len(data):
            self.release()
            self._vbo = self.ctx.buffer(reserve=max(len(data) * 2, 1 << 16), dynamic=True)
            self._vao = self.ctx.vertex_array(
                self.program, [(self._vbo, "2f 3f", "in_pos", "in_color")]
            )
        self._vbo.write(data)
        self.program["point_size"].value = float(size) * pixel_scale
        self.program["opacity"].value = float(opacity)
        self.program["soft"].value = mode == "points"
        self._vao.render(_MODES.get(mode, moderngl.TRIANGLES), vertices=count)

    def release(self):
        for res in (self._vao, self._vbo):
            if res is not None:
                res.release()
        self._vao = self._vbo = None


class GfxState:
    """Per-context shared state: the vertex drawer and the logical-to-framebuffer pixel ratio."""

    def __init__(self, ctx):
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.draw = _Draw(ctx)
        self.pixel_scale = 1.0


def state(ctx):
    """Return the `GfxState` for `ctx`, creating it on first use."""
    if not isinstance(ctx.extra, GfxState):
        ctx.extra = GfxState(ctx)
    return ctx.extra


def draw_geometry(ctx, geometry):
    """Render one geometry (custom `render` or plain `vertices()`) with its blend mode."""
    if hasattr(geometry, "render"):
        geometry.render(ctx)
        return
    positions, colors, mode = geometry.vertices()
    if len(positions) == 0:
        return
    gfx = state(ctx)
    set_blend(ctx, getattr(geometry, "blend", "alpha"))
    gfx.draw.draw(
        positions, colors, mode,
        size=float(getattr(geometry, "size", 1.0)),
        opacity=float(getattr(geometry, "opacity", 1.0)),
        pixel_scale=gfx.pixel_scale,
    )
    set_blend(ctx, None)

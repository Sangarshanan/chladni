"""
2D geometries whose parameters can be plain numbers or live Signal, plus
the Output sink that the runner renders to screen.
"""

import moderngl
import numpy as np

from .core import Node
from .gfx import set_blend, state
from .post import DEFAULT_SETTINGS
from .signals import evaluate


class Param:
    """A geometry attribute that transparently evaluates Signals on read."""

    def __init__(self, default):
        self.default = default

    def __set_name__(self, owner, name):
        self.attr = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return evaluate(getattr(obj, self.attr, self.default))

    def __set__(self, obj, value):
        setattr(obj, self.attr, value)


class Geometry(Node):
    """Base for drawable shapes. Subclasses implement `vertices()` (or a custom `render(ctx)`)."""

    blend = "alpha"
    opacity = Param(1.0)

    def vertices(self):
        raise NotImplementedError


_CIRCLE_VERTEX = """
#version 330
in vec2 in_pos;
in vec2 in_local;
out vec2 v_local;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_local = in_local;
}
"""

_CIRCLE_FRAGMENT = """
#version 330
in vec2 v_local;
uniform vec3 color;
uniform float opacity;
uniform float softness;
uniform float edge;
out vec4 f_color;
void main() {
    float d = length(v_local);
    float inner = max(1.0 - softness, 0.0);
    float a = 1.0 - smoothstep(inner - edge, 1.0 + edge, d);
    a *= opacity;
    if (a <= 0.0) discard;
    f_color = vec4(color * a, a);
}
"""


class Circle(Geometry):
    """
    An anti-aliased disc. `x`, `y`, `radius`, `color` and `softness` may be Signals.
    """

    x = Param(0.0)
    y = Param(0.0)
    radius = Param(0.2)
    color = Param((1.0, 1.0, 1.0))
    softness = Param(0.0)

    def __init__(self, x=0.0, y=0.0, radius=0.2, color=(1.0, 1.0, 1.0), softness=0.0, segments=48):
        super().__init__()
        self.x, self.y, self.radius, self.color = x, y, radius, color
        self.softness = softness
        self.segments = segments
        self._prog = self._vbo = self._vao = None

    def vertices(self):
        angles = np.linspace(0, 2 * np.pi, self.segments, endpoint=False)
        cx, cy, r = self.x, self.y, self.radius
        rim = np.column_stack([cx + np.cos(angles) * r, cy + np.sin(angles) * r])
        positions = np.vstack([[cx, cy], rim, rim[:1]]).astype("f4")
        colors = np.tile(np.array(self.color, dtype="f4"), (len(positions), 1))
        return positions, colors, "fan"

    def render(self, ctx):
        if self._prog is None:
            self._prog = ctx.program(vertex_shader=_CIRCLE_VERTEX, fragment_shader=_CIRCLE_FRAGMENT)
            self._vbo = ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
            self._vao = ctx.vertex_array(self._prog, [(self._vbo, "2f 2f", "in_pos", "in_local")])
        width, height = ctx.fbo.size
        cx, cy, r = float(self.x), float(self.y), max(float(self.radius), 1e-4)
        rx = r * height / width
        radius_px = r * height / 2.0
        pad = 1.0 + 2.0 / max(radius_px, 1.0)
        x0, x1 = cx - rx * pad, cx + rx * pad
        y0, y1 = cy - r * pad, cy + r * pad
        quad = np.array([
            x0, y0, -pad, -pad,  x1, y0, pad, -pad,  x1, y1, pad, pad,
            x0, y0, -pad, -pad,  x1, y1, pad, pad,  x0, y1, -pad, pad,
        ], dtype="f4")
        self._vbo.write(quad.tobytes())
        color = tuple(float(c) for c in self.color)
        self._prog["color"].value = color[:3]
        self._prog["opacity"].value = float(self.opacity)
        self._prog["softness"].value = min(max(float(self.softness), 0.0), 1.0)
        self._prog["edge"].value = 1.0 / max(radius_px, 1.0)
        set_blend(ctx, self.blend)
        self._vao.render(moderngl.TRIANGLES)
        set_blend(ctx, None)

    def close(self):
        for res in (self._vao, self._vbo, self._prog):
            if res is not None:
                res.release()
        self._prog = self._vbo = self._vao = None


class Bars(Geometry):
    """A row of bars, one per entry in `values` (each may be a Signal in [0, 1])."""

    def __init__(self, values=(0.0, 0.0, 0.0), x=0.0, y=-0.9, width=1.8, gap=0.1,
                 color=(1.0, 1.0, 1.0)):
        super().__init__()
        self.values = list(values)
        self.x, self.y, self.width, self.gap = x, y, width, gap
        self.color = color

    def vertices(self):
        n = len(self.values)
        slot = self.width / n
        bar_width = slot * (1 - self.gap)
        quads = []
        for i, value in enumerate(self.values):
            height = max(0.0, min(1.0, evaluate(value))) * 1.8
            left = self.x - self.width / 2 + i * slot + (slot - bar_width) / 2
            right = left + bar_width
            bottom, top = self.y, self.y + height
            quads += [
                [left, bottom], [right, bottom], [right, top],
                [left, bottom], [right, top], [left, top],
            ]
        positions = np.array(quads, dtype="f4")
        colors = np.tile(np.array(self.color, dtype="f4"), (len(positions), 1))
        return positions, colors, "triangles"


def _lattice_hash2(ix, iy):
    """Vectorized deterministic pseudo-random float in [0, 1) for a 2D lattice point."""
    h = (ix * 374761393 + iy * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return h / 0xFFFFFFFF


def _noise2(x, y, t, period=None):
    """
    2D gradient (Perlin) noise with quintic fade, scrolled over time by `t`.
    """
    x = x + t
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    x0i, y0i = x0.astype(np.int64), y0.astype(np.int64)
    x1i, y1i = x0i + 1, y0i + 1
    if period:
        x0i, x1i, y0i, y1i = x0i % period, x1i % period, y0i % period, y1i % period

    def corner(ix, iy, dx, dy):
        angle = _lattice_hash2(ix, iy) * (2 * np.pi)
        return np.cos(angle) * dx + np.sin(angle) * dy

    n00 = corner(x0i, y0i, fx, fy)
    n10 = corner(x1i, y0i, fx - 1, fy)
    n01 = corner(x0i, y1i, fx, fy - 1)
    n11 = corner(x1i, y1i, fx - 1, fy - 1)
    a = n00 + (n10 - n00) * u
    b = n01 + (n11 - n01) * u
    return a + (b - a) * v


def _curl(x, y, t, eps=0.02, period=None):
    """
    Divergence-free velocity field

    https://www.youtube.com/watch?v=rB83DpBJQsE
    """
    dpsi_dy = (_noise2(x, y + eps, t, period) - _noise2(x, y - eps, t, period)) / (2 * eps)
    dpsi_dx = (_noise2(x + eps, y, t, period) - _noise2(x - eps, y, t, period)) / (2 * eps)
    return dpsi_dy, -dpsi_dx


class Particles(Geometry):
    """
    A pool of particles drifting through a divergence-free curl-noise flow
    """

    blend = "add"
    count = Param(500)
    speed = Param(0.3)
    size = Param(4.0)
    color = Param((1.0, 1.0, 1.0))
    turbulence = Param(1.0)
    variation = Param(0.6)

    def __init__(self, max_count=500, count=500, speed=0.3, size=4.0,
                 color=(1.0, 1.0, 1.0), turbulence=1.0, scale=1.5,
                 color_by=None, variation=0.6, lifetime=4.0, seed=None):
        super().__init__()
        self.max_count = max_count
        self.count, self.speed, self.size, self.color, self.turbulence = (
            count, speed, size, color, turbulence
        )
        # The field must tile across the wrapped [-1, 1] domain, so the number of
        # noise cells spanning it is snapped to an integer (scale = cells / 2).
        self._period = max(1, round(2.0 * scale))
        self.scale = self._period / 2.0  # spatial frequency of the noise field
        self.color_by = color_by  # None (default), "speed", or "age"
        self.variation = variation
        self.lifetime = lifetime
        rng = np.random.default_rng(seed)
        self._pos = rng.uniform(-1.0, 1.0, size=(max_count, 2)).astype("f4")
        self._vel_mag = np.zeros(max_count, dtype="f4")
        self._age = rng.uniform(0.0, lifetime, size=max_count).astype("f4")
        self._t = 0.0

    def _velocity(self, x, y, t):
        base_x, base_y = _curl(x * self.scale, y * self.scale, t * 0.1, period=self._period)
        detail_x, detail_y = _curl(
            x * self.scale * 3.0, y * self.scale * 3.0, t * 0.3, period=self._period * 3
        )
        turbulence = self.turbulence
        return base_x + turbulence * detail_x, base_y + turbulence * detail_y

    def update(self, dt):
        self._t += dt
        x, y = self._pos[:, 0], self._pos[:, 1]
        # Midpoint (RK2) integration, subdivided when the step is large relative to
        # the noise detail: plain Euler spirals particles out of vortex cores and into
        # clumps, which defeats the point of a divergence-free field.
        step = self.speed * dt
        substeps = int(min(4, max(1, np.ceil(abs(step) * (1.0 + abs(self.turbulence)) / 0.02))))
        step /= substeps
        px, py = x, y
        for _ in range(substeps):
            vx, vy = self._velocity(px, py, self._t)
            vx, vy = self._velocity(px + vx * step * 0.5, py + vy * step * 0.5, self._t)
            px, py = px + vx * step, py + vy * step
        self._pos[:, 0] = (px + 1.0) % 2.0 - 1.0
        self._pos[:, 1] = (py + 1.0) % 2.0 - 1.0
        self._vel_mag = np.hypot(vx, vy)
        self._age = (self._age + dt) % max(self.lifetime, 1e-6)

    def vertices(self):
        n = max(0, min(int(self.count), self.max_count))
        positions = self._pos[:n]
        base = np.array(self.color, dtype="f4")
        if self.color_by == "speed":
            metric = np.clip(self._vel_mag[:n], 0.0, 1.0)
        elif self.color_by == "age":
            metric = self._age[:n] / max(self.lifetime, 1e-6)
        else:
            metric = None
        if metric is None:
            colors = np.tile(base, (n, 1))
        else:
            brightness = 1.0 + (metric - 0.5) * 2.0 * self.variation
            colors = np.clip(base[None, :] * brightness[:, None], 0.0, 1.0).astype("f4")
        return positions, colors, "points"


class Output(Node):
    """Sink: geometries connected with `>>` are rendered by the runner each frame.

    Besides the clear colour, an Output carries the post-processing settings
    (the final "Level / Bloom / Feedback" stage). Every setting may be a Signal,
    e.g. `Output(bloom=audio.low.map(0.2, 1.2), trails=0.85)`:

        render_scale    supersampling factor for anti-aliasing (1 = native, 2 = 4x samples)
        bloom           glow intensity; `bloom_threshold` is where glow starts
        trails          feedback decay (0 = off, ~0.9 = long trails); `trails_zoom` drifts them
        exposure, contrast, saturation, hue, vignette
        tonemap         "none", "aces" or "reinhard"
        palette         list of RGB stops; blended in by `palette_mix` as a gradient map
        dither          remove 8-bit gradient banding
    """

    background = Param((0.02, 0.02, 0.03))
    bloom = Param(DEFAULT_SETTINGS["bloom"])
    bloom_threshold = Param(DEFAULT_SETTINGS["bloom_threshold"])
    trails = Param(DEFAULT_SETTINGS["trails"])
    trails_zoom = Param(DEFAULT_SETTINGS["trails_zoom"])
    exposure = Param(DEFAULT_SETTINGS["exposure"])
    contrast = Param(DEFAULT_SETTINGS["contrast"])
    saturation = Param(DEFAULT_SETTINGS["saturation"])
    hue = Param(DEFAULT_SETTINGS["hue"])
    vignette = Param(DEFAULT_SETTINGS["vignette"])
    palette_mix = Param(DEFAULT_SETTINGS["palette_mix"])

    def __init__(self, background=(0.02, 0.02, 0.03), render_scale=1.0, tonemap="none",
                 palette=None, dither=True, **post):
        super().__init__()
        unknown = set(post) - set(DEFAULT_SETTINGS)
        if unknown:
            raise TypeError(f"Output got unexpected settings: {sorted(unknown)}")
        self.background = background
        self.render_scale = render_scale
        self.tonemap = tonemap
        self.palette = palette
        self.dither = dither
        for key, value in post.items():
            setattr(self, key, value)

    @property
    def layers(self):
        return self.inputs

    def settings(self):
        """Resolve every post-processing setting to plain numbers for this frame."""
        values = {key: evaluate(getattr(self, key, default)) for key, default in DEFAULT_SETTINGS.items()}
        values["render_scale"] = min(max(float(values["render_scale"]), 1.0), 4.0)
        return values

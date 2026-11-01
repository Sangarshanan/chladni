"""
2D geometries whose parameters can be plain numbers or live Signal, plus
the Output sink that the runner renders to screen.
"""

import numpy as np

from .core import Node
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
    """Base for drawable shapes. Subclasses implement `vertices()`."""

    def vertices(self):
        raise NotImplementedError


class Circle(Geometry):
    """A filled circle. `x`, `y`, `radius`, and `color` may be Signals."""

    x = Param(0.0)
    y = Param(0.0)
    radius = Param(0.2)
    color = Param((1.0, 1.0, 1.0))

    def __init__(self, x=0.0, y=0.0, radius=0.2, color=(1.0, 1.0, 1.0), segments=48):
        super().__init__()
        self.x, self.y, self.radius, self.color = x, y, radius, color
        self.segments = segments

    def vertices(self):
        angles = np.linspace(0, 2 * np.pi, self.segments, endpoint=False)
        cx, cy, r = self.x, self.y, self.radius
        rim = np.column_stack([cx + np.cos(angles) * r, cy + np.sin(angles) * r])
        positions = np.vstack([[cx, cy], rim, rim[:1]]).astype("f4")
        colors = np.tile(np.array(self.color, dtype="f4"), (len(positions), 1))
        return positions, colors, "fan"


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


def _noise2(x, y, t):
    """2D value noise (smoothstep-interpolated lattice), scrolled over time by `t`."""
    x = x + t
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    x0i, y0i = x0.astype(np.int64), y0.astype(np.int64)
    x1i, y1i = x0i + 1, y0i + 1
    v00 = _lattice_hash2(x0i, y0i)
    v10 = _lattice_hash2(x1i, y0i)
    v01 = _lattice_hash2(x0i, y1i)
    v11 = _lattice_hash2(x1i, y1i)
    a = v00 + (v10 - v00) * fx
    b = v01 + (v11 - v01) * fx
    return a + (b - a) * fy


def _curl(x, y, t, eps=0.02):
    """Divergence-free velocity field: the curl (rotated gradient) of a scalar
    noise potential. Unlike reading a direction straight off the noise, this is
    guaranteed to neither converge nor diverge particles over time - the same
    trick curl-noise-based particle systems (Houdini, TouchDesigner, Notch) use
    to keep flow-field particles organically swirling instead of clumping.
    """
    dpsi_dy = (_noise2(x, y + eps, t) - _noise2(x, y - eps, t)) / (2 * eps)
    dpsi_dx = (_noise2(x + eps, y, t) - _noise2(x - eps, y, t)) / (2 * eps)
    return dpsi_dy, -dpsi_dx


class Particles(Geometry):
    """A pool of particles drifting through a divergence-free curl-noise flow
    field (no clumping, no voids - particles stay spread out indefinitely),
    rendered as GL_POINTS. `count`, `speed`, `size`, `color`, `turbulence`,
    and `variation` may all be Signals, e.g.
    `particles.speed = audio.mid.map(0.1, 0.6)`.
    `turbulence` layers in a faster, finer-grained noise octave on top of the
    base flow (0 = smooth swirl, higher = more chaotic detail).

    By default every particle is the single `color`. Set `color_by="speed"`
    or `color_by="age"` to instead brighten/dim each particle individually by
    its current flow speed or its position in its `lifetime` cycle, scaled by
    `variation` (0 = no per-particle variation, same as the default).
    """

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
        self.scale = scale  # spatial frequency of the noise field
        self.color_by = color_by  # None (default), "speed", or "age"
        self.variation = variation
        self.lifetime = lifetime
        rng = np.random.default_rng(seed)
        self._pos = rng.uniform(-1.0, 1.0, size=(max_count, 2)).astype("f4")
        self._vel_mag = np.zeros(max_count, dtype="f4")
        self._age = rng.uniform(0.0, lifetime, size=max_count).astype("f4")
        self._t = 0.0

    def update(self, dt):
        self._t += dt
        x, y = self._pos[:, 0], self._pos[:, 1]
        base_x, base_y = _curl(x * self.scale, y * self.scale, self._t * 0.1)
        detail_x, detail_y = _curl(x * self.scale * 3.0, y * self.scale * 3.0, self._t * 0.3)
        vx = base_x + self.turbulence * detail_x
        vy = base_y + self.turbulence * detail_y
        step = self.speed * dt
        self._pos[:, 0] = (x + vx * step + 1.0) % 2.0 - 1.0
        self._pos[:, 1] = (y + vy * step + 1.0) % 2.0 - 1.0
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
    """Sink: geometries connected with `>>` are rendered by the runner each frame."""

    def __init__(self, background=(0.02, 0.02, 0.03)):
        super().__init__()
        self.background = background

    @property
    def layers(self):
        return self.inputs

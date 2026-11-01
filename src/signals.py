"""
Reactive values: Signal is anything whose .value() can change over time.

Assign a Signal to a geometry parameter (e.g. circle.radius = audio.low)
and it gets re-evaluated every frame.
"""

import math
import operator
import random

from .core import Node


class Signal:
    """Base class for values that change over time and combine with arithmetic."""

    def value(self):
        raise NotImplementedError

    def __call__(self):
        return self.value()

    def map(self, lo, hi):
        """Rescale a signal assumed to be in [0, 1] into [lo, hi]."""
        return _BinOp(_BinOp(self, hi - lo, operator.mul), lo, operator.add)

    def __add__(self, other):
        return _BinOp(self, other, operator.add)

    def __radd__(self, other):
        return _BinOp(other, self, operator.add)

    def __sub__(self, other):
        return _BinOp(self, other, operator.sub)

    def __rsub__(self, other):
        return _BinOp(other, self, operator.sub)

    def __mul__(self, other):
        return _BinOp(self, other, operator.mul)

    def __rmul__(self, other):
        return _BinOp(other, self, operator.mul)

    def __truediv__(self, other):
        return _BinOp(self, other, operator.truediv)

    def __rtruediv__(self, other):
        return _BinOp(other, self, operator.truediv)

    def __pow__(self, other):
        return _BinOp(self, other, operator.pow)

    def __neg__(self):
        return _BinOp(self, -1, operator.mul)


class _Const(Signal):
    """Wraps a plain number as a Signal so it can be combined with others."""

    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _BinOp(Signal):
    """Lazily applies op to two signals every time .value() is read."""

    def __init__(self, a, b, op):
        self._a = a if isinstance(a, Signal) else _Const(a)
        self._b = b if isinstance(b, Signal) else _Const(b)
        self._op = op

    def value(self):
        return self._op(self._a.value(), self._b.value())


def evaluate(value):
    """Resolve a plain number, a Signal, or a tuple/list mixing both, to plain numbers."""
    if isinstance(value, Signal):
        return value.value()
    if isinstance(value, (tuple, list)):
        return type(value)(evaluate(item) for item in value)
    return value


def _lattice_hash(i):
    """Deterministic pseudo-random float in [0, 1) for an integer lattice point."""
    i = (i % 1_000_000_007) ^ 0x5DEECE66D
    i = (i * 2654435761 + 1) & 0x7FFFFFFF
    return i / 0x7FFFFFFF


def _value_noise(t):
    """Smoothly interpolated 1D value noise (random lattice + smoothstep)."""
    i = math.floor(t)
    f = t - i
    f = f * f * (3 - 2 * f)
    a, b = _lattice_hash(i), _lattice_hash(i + 1)
    return a + (b - a) * f


class Noise(Node, Signal):
    """Smoothly wandering pseudo-random signal in [0, 1], advanced each tick."""

    def __init__(self, freq=0.5, seed=None):
        super().__init__()
        self.freq = freq
        self._t = random.Random(seed).uniform(0, 1000)
        self._value = _value_noise(self._t)

    def update(self, dt):
        self._t += dt * self.freq
        self._value = _value_noise(self._t)

    def value(self):
        return self._value


class GaussianNoise(Node, Signal):
    """Stochastic, mean-reverting random signal (Ornstein-Uhlenbeck process).

    Unlike Noise (deterministic, smoothly wandering), this is genuinely
    random each run: it diffuses around mean with roughly std standard
    deviation, pulled back over smoothing seconds. Good for layering
    unpredictable jitter/turbulence on top of other signals, e.g.
    particles.turbulence = GaussianNoise(mean=1.0, std=0.5).
    """

    def __init__(self, mean=0.0, std=1.0, smoothing=0.2, seed=None):
        super().__init__()
        self.mean = mean
        self.std = std
        self.smoothing = smoothing
        self._rng = random.Random(seed)
        self._value = mean

    def update(self, dt):
        theta = 1.0 / max(self.smoothing, 1e-6)
        drift = (self.mean - self._value) * theta * dt
        diffusion = self._rng.gauss(0.0, self.std) * math.sqrt(2 * theta * dt)
        self._value += drift + diffusion

    def value(self):
        return self._value


class Smooth(Node, Signal):
    """Exponentially smooths another signal towards its current value over time."""

    def __init__(self, source, time=0.2):
        super().__init__()
        self.source = source if isinstance(source, Signal) else _Const(source)
        self.time = time
        self._value = self.source.value()

    def update(self, dt):
        alpha = 1.0 - math.exp(-dt / max(self.time, 1e-6))
        self._value += (self.source.value() - self._value) * alpha

    def value(self):
        return self._value


class Impulse(Node, Signal):
    """Fires a quick decaying pulse (0 -> 1 -> 0) whenever source spikes above
    its own recent moving average - a simple onset/transient ("kick hit")
    detector. Good for a quick flash on top of a continuous mapping, e.g.
    particles.size = audio.high.map(2.0, 6.0) + Impulse(audio.low) * 20.
    """

    def __init__(self, source, sensitivity=1.5, decay=0.15, average_time=1.0):
        super().__init__()
        self.source = source if isinstance(source, Signal) else _Const(source)
        self.sensitivity = sensitivity
        self.decay = decay
        self.average_time = average_time
        self._average = self.source.value()
        self._value = 0.0

    def update(self, dt):
        level = self.source.value()
        if level > self._average * self.sensitivity and level > 1e-3:
            self._value = 1.0
        else:
            decay_alpha = 1.0 - math.exp(-dt / max(self.decay, 1e-6))
            self._value -= self._value * decay_alpha
        average_alpha = 1.0 - math.exp(-dt / max(self.average_time, 1e-6))
        self._average += (level - self._average) * average_alpha

    def value(self):
        return self._value

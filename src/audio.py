"""
Background PortAudio input stream
"""

import math
import threading

import numpy as np
import sounddevice as sd

from .core import Node
from .signals import Signal

# (name, low_hz, high_hz)
_BANDS = (("low", 20, 250), ("mid", 250, 2000), ("high", 2000, 8000))


class _AudioStream:
    """Input stream computing low/mid/high band energy."""

    _instance = None

    def __init__(self, samplerate=44100, blocksize=1024, device=None):
        self.samplerate = samplerate
        self._lock = threading.Lock()
        self._bands = {"low": 0.0, "mid": 0.0, "high": 0.0}
        freqs = np.fft.rfftfreq(blocksize, 1.0 / samplerate)
        self._masks = {name: (freqs >= lo) & (freqs < hi) for name, lo, hi in _BANDS}
        self._window = np.hanning(blocksize)
        self._stream = sd.InputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            channels=1,
            device=device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        spectrum = np.abs(np.fft.rfft(indata[:, 0] * self._window))
        bands = {}
        for name, mask in self._masks.items():
            magnitude = spectrum[mask].mean() if mask.any() else 0.0
            bands[name] = float(np.clip(np.log1p(magnitude) / 8.0, 0.0, 1.0))
        with self._lock:
            self._bands = bands

    def read(self):
        with self._lock:
            return self._bands

    @classmethod
    def instance(cls, **kwargs):
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance


class _Band(Signal):
    """A live `Signal` reading one band's current value off an `AudioBands` node."""

    def __init__(self, owner, name):
        self._owner = owner
        self._name = name

    def value(self):
        return self._owner._values[self._name]


class AudioBands(Node):
    """Exposes `.low`, `.mid`, `.high` as live Signals from the default input device."""

    def __init__(self, samplerate=44100, blocksize=1024, smoothing=0.1, device=None):
        super().__init__()
        self._stream = _AudioStream.instance(
            samplerate=samplerate, blocksize=blocksize, device=device
        )
        self.smoothing = smoothing
        self._values = {"low": 0.0, "mid": 0.0, "high": 0.0}
        self.low = _Band(self, "low")
        self.mid = _Band(self, "mid")
        self.high = _Band(self, "high")

    def update(self, dt):
        alpha = 1.0 - math.exp(-dt / max(self.smoothing, 1e-6))
        for name, target in self._stream.read().items():
            self._values[name] += (target - self._values[name]) * alpha

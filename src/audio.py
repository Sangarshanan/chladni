"""
Background PortAudio input stream, band analysis, and auto-ranged AudioBands signals.
"""

import math
import threading

import numpy as np
import sounddevice as sd

from .core import Node
from .signals import Signal

_BANDS = (("low", 20, 250), ("mid", 250, 2000), ("high", 2000, 15000))
_FLOOR_DB = -100.0


class _AudioStream:
    """
    Input stream reporting per-band energy in dB.
    """

    _instance = None

    def __init__(self, samplerate=44100, blocksize=1024, fft_size=4096, device=None):
        self.samplerate = samplerate
        self._lock = threading.Lock()
        self._db = {name: _FLOOR_DB for name, _, _ in _BANDS}
        self._db["level"] = _FLOOR_DB
        freqs = np.fft.rfftfreq(fft_size, 1.0 / samplerate)
        self._masks = {name: (freqs >= lo) & (freqs < hi) for name, lo, hi in _BANDS}
        self._window = np.hanning(fft_size)
        self._scale = 2.0 / self._window.sum()  # a full-scale sine peaks at 1.0
        self._ring = np.zeros(fft_size, dtype="f4")
        self._stream = sd.InputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            channels=1,
            device=device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        block = indata[:, 0]
        n = min(len(block), len(self._ring))
        self._ring = np.roll(self._ring, -n)
        self._ring[-n:] = block[-n:]
        spectrum = np.abs(np.fft.rfft(self._ring * self._window)) * self._scale
        power = spectrum * spectrum / 2.0  # per-bin power of the equivalent sine
        db = {}
        for name, mask in self._masks.items():
            db[name] = 10.0 * math.log10(float(power[mask].sum()) + 1e-10)
        rms = float(np.sqrt(np.mean(np.square(block)))) if len(block) else 0.0
        db["level"] = 20.0 * math.log10(rms + 1e-5)
        with self._lock:
            self._db = db

    def read(self):
        with self._lock:
            return dict(self._db)

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


class _Tracker:

    def __init__(self, floor_db, min_range_db, min_peak_db, hi_fall_db_s, lo_rise_db_s):
        self.floor = floor_db
        self.min_range = min_range_db
        self.min_peak = min_peak_db
        self.hi_fall = hi_fall_db_s
        self.lo_rise = lo_rise_db_s
        self.lo = (
            None  # seeded from the first reading, which is assumed to be room noise
        )
        self.hi = max(floor_db + min_range_db, min_peak_db)

    def normalise(self, db, dt):
        if self.lo is None:
            self.lo = max(db, self.floor)
        self.lo = max(min(db, self.lo + self.lo_rise * dt), self.floor)
        self.hi = max(db, self.hi - self.hi_fall * dt, self.min_peak)
        return (db - self.lo) / max(self.hi - self.lo, self.min_range)


class AudioBands(Node):
    """
    Exposes `.low`, `.mid`, `.high` and overall `.level` as live Signals in 0..1.
    """

    def __init__(
        self,
        samplerate=44100,
        blocksize=1024,
        smoothing=None,
        device=None,
        attack=0.03,
        release=0.25,
        auto_gain=True,
        gain=1.0,
        gate=0.05,
        floor_db=-70.0,
        min_range_db=25.0,
        min_peak_db=-45.0,
        peak_fall_db_s=2.0,
        floor_rise_db_s=0.5,
    ):
        self._stream = _AudioStream.instance(
            samplerate=samplerate, blocksize=blocksize, device=device
        )
        super().__init__()
        self.attack = attack
        self.release = (
            release if smoothing is None else smoothing
        )  # `smoothing` is the old name
        self.auto_gain = auto_gain
        self.gain = gain
        self.gate = gate
        self.floor_db = floor_db
        self._trackers = {
            name: _Tracker(
                floor_db, min_range_db, min_peak_db, peak_fall_db_s, floor_rise_db_s
            )
            for name in ("low", "mid", "high", "level")
        }
        self._values = {name: 0.0 for name in self._trackers}
        self.low = _Band(self, "low")
        self.mid = _Band(self, "mid")
        self.high = _Band(self, "high")
        self.level = _Band(self, "level")

    def _normalise(self, name, db, dt):
        if self.auto_gain:
            value = self._trackers[name].normalise(db, dt)
        else:
            value = (db - self.floor_db) / (0.0 - self.floor_db)
        value = value * self.gain
        gate = min(max(self.gate, 0.0), 0.95)
        value = (value - gate) / (1.0 - gate)
        return min(max(value, 0.0), 1.0)

    def update(self, dt):
        for name, db in self._stream.read().items():
            target = self._normalise(name, db, dt)
            current = self._values[name]
            lag = self.attack if target > current else self.release
            alpha = 1.0 - math.exp(-dt / max(lag, 1e-4))
            self._values[name] = current + (target - current) * alpha

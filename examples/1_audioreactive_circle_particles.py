"""
Audio Reactive Circle and Particles
"""

from src import (
    Pipeline, AudioBands, Noise, GaussianNoise, Impulse, Circle, Bars,
    Particles, Output,
)

with Pipeline() as pipeline:
    audio = AudioBands()
    noise = Noise(freq=0.3)
    jitter = GaussianNoise(mean=1.0, std=0.6, smoothing=0.15)
    kick = Impulse(audio.low, sensitivity=1.4, decay=0.15)

    circle = Circle(color=(0.2, 0.6, 1.0))
    circle.radius = audio.low.map(0.12, 0.5)
    circle.x = (noise - 0.5) * 0.8
    circle.color = (audio.low, audio.mid, audio.high)

    bars = Bars(values=[audio.low, audio.mid, audio.high], color=(1.0, 0.4, 0.2))

    particles = Particles(max_count=1500, color=(0.6, 0.8, 1.0), color_by="speed")
    particles.count = audio.low.map(400, 1500)
    particles.speed = audio.mid.map(0.1, 1.5)
    particles.size = audio.high.map(4.0, 6.0)
    particles.turbulence = jitter

    output = Output(background=(0.26, 0.1, 0.02), bloom=0.5, bloom_threshold=0.6, render_scale=1.5)

    circle >> output
    bars >> output
    particles >> output

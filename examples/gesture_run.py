"""Gesture + audio example.

Hold your hand in front of the camera:
  - Move hand left/right/up/down -> moves the circle
  - Pinch (thumb+index) -> controls circle size
  - Open/close fist -> controls particle turbulence
  - Number of raised fingers -> controls particle count
  - Audio low/mid/high still drive colors
"""

from src import (
    Pipeline, AudioBands, GestureInput, Noise, Impulse,
    Circle, Bars, Particles, Output,
)

with Pipeline() as pipeline:
    audio = AudioBands()
    gesture = GestureInput()
    noise = Noise(freq=0.3)
    kick = Impulse(audio.low, sensitivity=1.4, decay=0.15)

    circle = Circle(color=(0.2, 0.6, 1.0))
    circle.x = gesture.hand_x                        # hand position controls x
    circle.y = gesture.hand_y                        # hand position controls y
    circle.radius = gesture.pinch.map(0.05, 0.45)   # pinch controls radius
    circle.color = (audio.low, audio.mid, audio.high)

    bars = Bars(values=[audio.low, audio.mid, audio.high], color=(1.0, 0.4, 0.2))

    particles = Particles(max_count=2000, color=(0.6, 0.8, 1.0), color_by="speed")
    particles.count = gesture.fingers.map(100, 2000)      # fingers control count
    particles.speed = audio.mid.map(0.1, 0.6)             # audio drives speed
    particles.size = audio.high.map(2.0, 6.0)
    particles.turbulence = gesture.openness.map(0.2, 3.0) # openness = calm vs chaotic

    output = Output(background=(0.03, 0.03, 0.05))
    gesture.overlay >> output
    circle >> output
    bars >> output
    particles >> output

"""
Audio-reactive infinite zoom kaleidoscope with dynamic color cycling.
"""

from src import Pipeline, AudioBands, GestureInput, CameraOverlay, ISFFilter, Output

with Pipeline() as pipeline:
    audio = AudioBands(device="VB-Cable")
    gesture = GestureInput()

    camera = CameraOverlay(gesture, skeleton=False, saturation=1.2)

    # Radial symmetry kaleidoscope filter
    kaleidoscope = ISFFilter("kaleidoscope")

    # Raised fingers (0-5) set mirror symmetry segments (3 to 16)
    kaleidoscope.segments = gesture.fingers.map(3.0, 16.0)
    # Melody smoothly rotates the mandala
    kaleidoscope.rotation = audio.mid.map(0, 1)

    output = Output(
        background=(0.0, 0.0, 0.0),
        # Infinite zoom feedback: decay controls trail length, zoom pushes outward
        trails=0.93,
        trails_zoom=audio.low.map(1.02, 1.08),
        # Audio-reactive color transformations
        hue=audio.mid.map(0.0, 6.28),          # Melody cycles the full color spectrum
        saturation=audio.high.map(1.2, 2.2),   # High frequencies punch up color saturation
        bloom=audio.low.map(0.6, 1.6),         # Bass kicks trigger glowing HDR bursts
        bloom_threshold=0.5,
        render_scale=1.5,
        tonemap="aces",
    )

    kaleidoscope >> camera >> output

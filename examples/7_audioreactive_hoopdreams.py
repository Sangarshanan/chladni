"""
Audio-reactive geometric hoop kaleidoscope using shaders/hoopdreams.fs
with camera overlay and gesture-controlled color shifts.
"""

from src import Pipeline, AudioBands, ISFGenerator, Output

with Pipeline() as pipeline:
    audio = AudioBands(device="VB-Cable")

    hoops = ISFGenerator("shaders/hoopdreams.fs")

    # Audio-reactive parameter mappings
    hoops.zoom = audio.low.map(1.5, 2)
    hoops.animate = audio.mid.map(0.2, 0.6)
    hoops.size = audio.low.map(0.2, 0.3)
    hoops.rotate = audio.mid.map(-0.5, 0.5)
    hoops.blur = audio.high.map(0.004, 0.01)

    output = Output(
        background=(0.0, 0.0, 0.0),
        palette=[
            (0.02, 0.01, 0.08),
            (0.1, 0.2, 0.8),
            (0.0, 0.9, 0.9),
            (1.0, 0.2, 0.6),
            (1.0, 0.95, 0.9),
        ],
        palette_mix=0.9,
    )

    hoops >> output

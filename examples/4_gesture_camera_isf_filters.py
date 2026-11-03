"""
Camera effects with chained ISF filters.
"""

from src import Pipeline, AudioBands, GestureInput, CameraOverlay, ISFFilter, Output

with Pipeline() as pipeline:
    audio = AudioBands()
    gesture = GestureInput()

    camera = CameraOverlay(gesture, skeleton=False)

    # There are pre-available ISF filters in the ISFFilter class
    # You can also create your own custom ISF filters

    # Mosaic pixelation
    pixelate = ISFFilter("pixelate")
    pixelate.pixels = gesture.pinch.map(16.0, 160.0)

    # Glitch
    glitch = ISFFilter("glitch")
    glitch.amount = audio.low.map(0.01, 0.08)
    glitch.speed = audio.mid.map(2.0, 8.0)

    output = Output(
        background=(0.0, 0.0, 0.0),
        bloom=0.5,
        vignette=0.2,
        render_scale=1.5,
    )

    # Camera -> Pixelate -> Glitch -> Screen
    camera >> pixelate >> glitch >> output

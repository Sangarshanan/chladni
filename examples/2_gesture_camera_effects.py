"""
Camera effects with zoom tunnel and thermal color mapping
"""

from src import Pipeline, AudioBands, GestureInput, CameraOverlay, Output

with Pipeline() as pipeline:
    audio = AudioBands()
    gesture = GestureInput()

    # High-contrast monochrome feed
    camera = CameraOverlay(
        gesture,
        skeleton=True,
        skeleton_color=(0.1, 1.0, 0.5),
        saturation=0.0,
        contrast=1.6,
    )

    output = Output(
        background=(0.0, 0.0, 0.0),
        trails=0.9,
        trails_zoom=1.1,
        bloom=1.2,
        bloom_threshold=1,
        tonemap="aces",
        palette=[
            (0.0, 0.0, 0.1),
            (0.1, 0.4, 0.9),
            (0.9, 0.1, 0.5),
            (1.0, 0.9, 0.2),
        ],
        palette_mix=0.85,
    )

    camera >> output

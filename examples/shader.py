"""
ISF ColorDiffusionFlow driven by audio and hand gestures over a live camera feed.
"""

from src import Pipeline, AudioBands, GestureInput, CameraOverlay, ISFGenerator, Output

with Pipeline() as pipeline:
    audio = AudioBands()
    gesture = GestureInput()

    warp = ISFGenerator("shaders/color_diffusion_flow.fs")

    # Audio drives flow speed and depth
    warp.rate1 = audio.mid.map(0.8, 2.5)
    warp.depthX = audio.mid.map(0.4, 0.9)
    warp.rate2 = audio.low.map(0.2, 1.4)
    warp.depthY = audio.low.map(0.1, 0.5)
    warp.drift = audio.high.map(0.05, 0.4)

    # Finger movement and curl control colour balance and phase
    warp.color1 = gesture.fingers.map(-2.0, 2.0)
    warp.color2 = gesture.openness.map(-1.0, 1.0)
    warp.cycle1 = gesture.fingers.map(0.5, 2.8)
    warp.cycle2 = gesture.openness.map(-0.4, 0.4)

    # Pinch controls shader alpha over the camera feed
    warp.blend_mode = "alpha"
    warp.alpha = gesture.pinch.map(0.1, 0.85)

    output = Output(
        background=(0.0, 0.0, 0.0),
        render_scale=1.5,
        tonemap="aces",
        exposure=0.4,
        bloom=1,
        bloom_threshold=0.8,
    )

    camera = gesture.overlay
    camera.alpha = 0.55
    camera >> output
    warp >> output
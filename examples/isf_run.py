"""ISF example: ColorDiffusionFlow driven by audio, over a live camera background.

  - Audio mid  (melody)    -> primary flow speed and horizontal depth
  - Audio high (harmonics) -> colour phase and fine drift
  - Audio low  (warmth)    -> counter-current, vertical depth, blue/amber balance
  - Audio level            -> shader opacity: the camera shows through when it is quiet

Flow *rates* are integrated on the CPU (INTEGRATE inputs in the shader header), so
audio changes the speed of the motion instead of making the picture jump.
Set USE_CAMERA = False to run the shader on its own.
"""

from src import (
    Pipeline, AudioBands, GestureInput, CameraOverlay, ISFGenerator,
    Output,
)

USE_CAMERA = True

with Pipeline() as pipeline:
    audio = AudioBands()

    warp = ISFGenerator("shaders/ColorDiffusionFlow.fs")

    warp.rate1 = audio.mid.map(0.8, 2.5)         # [-3, 3]
    warp.depthX = audio.mid.map(0.4, 0.9)        # [0.001, 0.9]
    warp.drift = audio.high.map(0.05, 0.4)       # [-1, 1]
    warp.color1 = audio.high.map(-1.5, 1.5)      # [-2.5, 2.5]
    warp.cycle1 = audio.high.map(0.5, 2.8)       # [0.01, 3.14]
    warp.rate2 = audio.low.map(0.2, 1.4)         # [-3, 3]
    warp.depthY = audio.low.map(0.1, 0.5)        # [0.001, 0.9]
    warp.color2 = audio.low.map(-0.8, 1.0)       # [-1.25, 1.125]
    warp.cycle2 = audio.low.map(-0.4, 0.4)       # [-0.497, 0.497]
    # loopcycle is left at its default: it offsets the whole field, so driving it
    # from audio would shift the picture instead of animating it.

    output = Output(
        background=(0.0, 0.0, 0.0),
        render_scale=1.5,        # supersample the 10-fold warp to remove shimmer
        tonemap="aces", exposure=0.8, saturation=1.2,
        bloom=0.3, bloom_threshold=0.8,
        vignette=0.3,
    )

    camera = None
    if USE_CAMERA:
        try:
            camera = CameraOverlay(GestureInput(), skeleton=False, saturation=0.8)
        except RuntimeError as exc:
            print(f"[isf_run] no camera ({exc}); running the shader on its own")

    if camera is not None:
        warp.blend_mode = "alpha"
        warp.alpha = audio.level.map(0.35, 0.9)
        camera >> output

    warp >> output

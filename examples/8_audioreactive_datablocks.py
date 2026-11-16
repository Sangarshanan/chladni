from src import Pipeline, AudioBands, ISFGenerator, Output
with Pipeline() as pipeline:
    audio = AudioBands()
    blocks = ISFGenerator("shaders/Pattern_DataBlocks_03.fs")

    # Audio reactive modulations
    # blocks.rate = audio.mid.map(-0.04, -0.16)                  # Gentle acceleration with melody
    # blocks.density = audio.low.map(180.0, 340.0)               # Subtle breathing in data density
    # blocks.shift = audio.low.map(0.15, 0.32)                   # Block thickness & slice spacing
    # blocks.offset = audio.high.map(0.05, 0.45)    # Subtle RGB chromatic split
    curve_val = audio.low.map(0.12, 0.38)
    blocks.curve = (curve_val, curve_val)

    # blocks.polar_on = True

    # Grid dimensions (keep stable for clean visual structure)
    blocks.grid = (45.0, 30.0)
    blocks.grid2 = (6.0, 1000.0)

    output = Output(
        background=(0.01, 0.01, 0.02),

        # Phosphorescent feedback trails & slight outward drift
        # trails=0.84,
        # trails_zoom=1.01,

        # Luminous glow
        # bloom=0.1,
        # bloom_threshold=0.1,
        # tonemap="aces",

        palette=[
            (0.2, 0.01, 0.04),
            (0.05, 0.4, 0.9),
            (0.7, 0.15, 0.9),
            (1.0, 0.6, 0.15),
            (1.0, 0.98, 0.9),
        ],
        palette_mix=0.1,
        hue=audio.mid.map(0.0, 3.14),
        render_scale=1.5,
    )

    blocks >> output

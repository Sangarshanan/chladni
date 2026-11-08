"""
Audio-reactive tunnel visual synthesizer
"""

from src import Pipeline, AudioBands, Impulse, ISFGenerator, Particles, Output

with Pipeline() as pipeline:
    audio = AudioBands(device="VB-Cable")

    # triggers a spike on an impulse (Bass or Kick?)
    kick = Impulse(audio.low, sensitivity=1.5, decay=0.2)

    # 3D tunnel generator
    tunnel = ISFGenerator("tunnel")
    tunnel.speed = audio.mid.map(0.8, 3.0)
    tunnel.twist = audio.high.map(0.5, 3.5)
    tunnel.color = (0.2, 0.9, 0.7, 1.0)

    # Additive particle dust swirling in the tunnel
    particles = Particles(max_count=1200, color=(1.0, 0.4, 0.8), color_by="speed")
    particles.speed = audio.mid.map(0.2, 1.2)
    particles.size = kick * 6.0 + 2.0

    output = Output(
        background=(0.0, 0.0, 0.0),
        bloom=1.2,
        bloom_threshold=0.6,
        trails=0.85,
        trails_zoom=1.02,
        tonemap="aces",
    )

    tunnel >> output
    particles >> output
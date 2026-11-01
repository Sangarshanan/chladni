"""
Post-processing chain to render the final scene.
scene (float16) -> Feedback (trails) -> Bloom -> Level/Lookup/Tone map -> dithered screen
"""

import moderngl
import numpy as np

DEFAULT_SETTINGS = {
    "render_scale": 1.0,       # >1 supersamples the scene (anti-aliasing)
    "bloom": 0.0,              # bloom intensity
    "bloom_threshold": 0.7,    # brightness where bloom starts
    "trails": 0.0,             # feedback decay per 1/60 s (0 = off, 0.9 = long trails)
    "trails_zoom": 1.0,        # >1 pushes trails outward, <1 pulls them in
    "exposure": 1.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "hue": 0.0,                # hue rotation in radians
    "vignette": 0.0,
    "tonemap": "none",         # "none", "aces" or "reinhard"
    "palette": None,           # list of RGB stops used as a gradient map (Lookup TOP)
    "palette_mix": 0.0,
    "dither": True,
}

_TONEMAPS = {"none": 0, "aces": 1, "reinhard": 2}
_BLOOM_LEVELS = 5

_VERTEX = """
#version 330
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FEEDBACK = """
#version 330
uniform sampler2D scene;
uniform sampler2D prev;
uniform float decay;
uniform float zoom;
in vec2 uv;
out vec4 f_color;
void main() {
    vec3 fresh = texture(scene, uv).rgb;
    vec3 old = texture(prev, (uv - 0.5) / zoom + 0.5).rgb * decay;
    f_color = vec4(max(fresh, old), 1.0);
}
"""

_PREFILTER = """
#version 330
uniform sampler2D source;
uniform vec2 texel;
uniform float threshold;
in vec2 uv;
out vec4 f_color;
void main() {
    vec3 c = texture(source, uv + texel * vec2(-1.0, -1.0)).rgb
           + texture(source, uv + texel * vec2( 1.0, -1.0)).rgb
           + texture(source, uv + texel * vec2(-1.0,  1.0)).rgb
           + texture(source, uv + texel * vec2( 1.0,  1.0)).rgb;
    c *= 0.25;
    float br = max(c.r, max(c.g, c.b));
    float knee = 0.5 * threshold + 1e-4;
    float soft = clamp(br - threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float w = max(soft, br - threshold) / max(br, 1e-4);
    f_color = vec4(c * w, 1.0);
}
"""

_DOWN = """
#version 330
uniform sampler2D source;
uniform vec2 texel;
in vec2 uv;
out vec4 f_color;
void main() {
    vec3 c = texture(source, uv).rgb * 4.0;
    c += texture(source, uv + texel * vec2(-1.0, -1.0)).rgb;
    c += texture(source, uv + texel * vec2( 1.0, -1.0)).rgb;
    c += texture(source, uv + texel * vec2(-1.0,  1.0)).rgb;
    c += texture(source, uv + texel * vec2( 1.0,  1.0)).rgb;
    f_color = vec4(c * 0.125, 1.0);
}
"""

_UP = """
#version 330
uniform sampler2D source;
uniform vec2 texel;
in vec2 uv;
out vec4 f_color;
void main() {
    vec3 c = texture(source, uv + texel * vec2(-2.0,  0.0)).rgb
           + texture(source, uv + texel * vec2( 2.0,  0.0)).rgb
           + texture(source, uv + texel * vec2( 0.0, -2.0)).rgb
           + texture(source, uv + texel * vec2( 0.0,  2.0)).rgb;
    c += 2.0 * (texture(source, uv + texel * vec2(-1.0, -1.0)).rgb
              + texture(source, uv + texel * vec2( 1.0, -1.0)).rgb
              + texture(source, uv + texel * vec2(-1.0,  1.0)).rgb
              + texture(source, uv + texel * vec2( 1.0,  1.0)).rgb);
    f_color = vec4(c / 12.0, 1.0);
}
"""

_COMPOSITE = """
#version 330
uniform sampler2D scene;
uniform sampler2D bloom_tex;
uniform sampler2D palette_tex;
uniform vec2 scene_size;
uniform float render_scale;
uniform float bloom;
uniform float exposure;
uniform float contrast;
uniform float saturation;
uniform float hue;
uniform float vignette;
uniform float palette_mix;
uniform int tonemap;
uniform float dither;
uniform float seed;
in vec2 uv;
out vec4 f_color;

vec3 sample_scene(vec2 p) {
    if (render_scale <= 1.01) return texture(scene, p).rgb;
    vec2 o = 0.25 * render_scale / scene_size;
    return 0.25 * (texture(scene, p + vec2( o.x,  o.y)).rgb
                 + texture(scene, p + vec2(-o.x,  o.y)).rgb
                 + texture(scene, p + vec2( o.x, -o.y)).rgb
                 + texture(scene, p + vec2(-o.x, -o.y)).rgb);
}

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

float rand(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec3 c = sample_scene(uv) + texture(bloom_tex, uv).rgb * bloom;
    c *= exposure;

    const vec3 k = vec3(0.57735026);
    float ca = cos(hue), sa = sin(hue);
    c = c * ca + cross(k, c) * sa + k * dot(k, c) * (1.0 - ca);

    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(luma), c, saturation);
    c = max((c - 0.5) * contrast + 0.5, 0.0);

    if (tonemap == 1) c = aces(c);
    else if (tonemap == 2) c = c / (1.0 + c);

    if (palette_mix > 0.0) {
        float l = clamp(dot(c, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
        c = mix(c, texture(palette_tex, vec2(l, 0.5)).rgb, palette_mix);
    }

    c *= 1.0 - vignette * smoothstep(0.35, 1.0, length(uv - 0.5) * 1.4142);

    if (dither > 0.0) {
        vec2 p = gl_FragCoord.xy + seed;
        c += (rand(p) + rand(p + 17.17) - 1.0) / 255.0;
    }
    f_color = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""


def _tex(ctx, size):
    tex = ctx.texture(size, 4, dtype="f2")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = False
    return tex


class PostChain:
    """Owns the float scene buffer and the full-screen passes that turn it into the final image."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.size = None
        self.scale = None
        self._scene = None
        self._feedback = None
        self._feedback_valid = False
        self._bloom = []
        self._palette_tex = None
        self._palette_key = None
        self._frame = 0

        self._progs = {
            name: ctx.program(vertex_shader=_VERTEX, fragment_shader=src)
            for name, src in (
                ("feedback", _FEEDBACK), ("prefilter", _PREFILTER), ("down", _DOWN),
                ("up", _UP), ("composite", _COMPOSITE),
            )
        }
        self._vaos = {name: ctx.vertex_array(prog, []) for name, prog in self._progs.items()}

    def _build(self, fb_size, scale):
        self._release_targets()
        size = (max(1, int(fb_size[0] * scale)), max(1, int(fb_size[1] * scale)))
        self.size, self.scale = size, scale

        tex = _tex(self.ctx, size)
        self._scene = (tex, self.ctx.framebuffer(color_attachments=[tex]))

        tex_a, tex_b = _tex(self.ctx, size), _tex(self.ctx, size)
        self._feedback = [
            (tex_a, self.ctx.framebuffer(color_attachments=[tex_a])),
            (tex_b, self.ctx.framebuffer(color_attachments=[tex_b])),
        ]
        self._feedback_valid = False

        w, h = size
        for _ in range(_BLOOM_LEVELS):
            w, h = max(1, w // 2), max(1, h // 2)
            t = _tex(self.ctx, (w, h))
            self._bloom.append((t, self.ctx.framebuffer(color_attachments=[t])))

    def _release_targets(self):
        pairs = ([self._scene] if self._scene else []) + (self._feedback or []) + self._bloom
        for tex, fbo in pairs:
            fbo.release()
            tex.release()
        self._scene, self._feedback, self._bloom = None, None, []

    def begin(self, fb_size, scale):
        """Bind and return the float scene framebuffer, (re)allocating it on size changes."""
        target = (max(1, int(fb_size[0] * scale)), max(1, int(fb_size[1] * scale)))
        if self._scene is None or self.size != target or self.scale != scale:
            self._build(fb_size, scale)
        self._scene[1].use()
        return self._scene[1]

    def _fullscreen(self, name, target, **uniforms):
        prog = self._progs[name]
        for key, value in uniforms.items():
            if key in prog:
                prog[key].value = value
        target.use()
        self._vaos[name].render(moderngl.TRIANGLES, vertices=3)

    def _update_palette(self, stops):
        key = tuple(tuple(s) for s in stops)
        if key == self._palette_key:
            return
        self._palette_key = key
        xs = np.linspace(0.0, 1.0, len(stops))
        grid = np.linspace(0.0, 1.0, 256)
        data = np.stack(
            [np.interp(grid, xs, [s[i] for s in stops]) for i in range(3)] + [np.ones(256)], axis=1
        ).astype("f4")
        if self._palette_tex is not None:
            self._palette_tex.release()
        self._palette_tex = self.ctx.texture((256, 1), 4, data.tobytes(), dtype="f4")
        self._palette_tex.repeat_x = self._palette_tex.repeat_y = False
        self._palette_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def finish(self, settings, dt, screen, fb_size):
        """Run feedback, bloom and the final composite from the scene into `screen`."""
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)
        source = self._scene[0]

        trails = float(settings["trails"])
        if trails > 0.0:
            cur, prev = self._feedback
            if not self._feedback_valid:
                prev[1].use()
                ctx.clear(0.0, 0.0, 0.0, 1.0)
            decay = min(trails, 0.999) ** (min(dt, 0.1) * 60.0)
            self._scene[0].use(0)
            prev[0].use(1)
            self._fullscreen(
                "feedback", cur[1], scene=0, prev=1, decay=decay,
                zoom=float(settings["trails_zoom"]),
            )
            self._feedback.reverse()
            self._feedback_valid = True
            source = cur[0]
        else:
            self._feedback_valid = False

        bloom = max(float(settings["bloom"]), 0.0)
        if bloom > 0.0:
            self._run_bloom(source, max(float(settings["bloom_threshold"]), 0.0))
        bloom_tex = self._bloom[0][0]

        palette_mix = float(settings["palette_mix"])
        palette = settings["palette"]
        if palette and palette_mix > 0.0:
            self._update_palette(palette)
        else:
            palette_mix = 0.0
            if self._palette_tex is None:
                self._update_palette([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

        source.use(0)
        bloom_tex.use(1)
        self._palette_tex.use(2)
        self._frame += 1
        screen.use()
        ctx.viewport = (0, 0, fb_size[0], fb_size[1])
        self._fullscreen(
            "composite", screen,
            scene=0, bloom_tex=1, palette_tex=2,
            scene_size=(float(self.size[0]), float(self.size[1])),
            render_scale=float(self.scale),
            bloom=bloom,
            exposure=float(settings["exposure"]),
            contrast=float(settings["contrast"]),
            saturation=float(settings["saturation"]),
            hue=float(settings["hue"]),
            vignette=float(settings["vignette"]),
            palette_mix=palette_mix,
            tonemap=_TONEMAPS.get(str(settings["tonemap"]).lower(), 0),
            dither=1.0 if settings["dither"] else 0.0,
            seed=float(self._frame % 61),
        )

    def _run_bloom(self, source, threshold):
        ctx = self.ctx
        source.use(0)
        _, first_fbo = self._bloom[0]
        self._fullscreen(
            "prefilter", first_fbo, source=0, threshold=threshold,
            texel=(1.0 / self.size[0], 1.0 / self.size[1]),
        )
        for i in range(1, len(self._bloom)):
            prev_tex, _ = self._bloom[i - 1]
            _, fbo = self._bloom[i]
            prev_tex.use(0)
            self._fullscreen(
                "down", fbo, source=0, texel=(1.0 / prev_tex.width, 1.0 / prev_tex.height)
            )
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self._bloom) - 1, 0, -1):
            low_tex, _ = self._bloom[i]
            _, high_fbo = self._bloom[i - 1]
            low_tex.use(0)
            self._fullscreen(
                "up", high_fbo, source=0, texel=(1.0 / low_tex.width, 1.0 / low_tex.height)
            )
        ctx.disable(moderngl.BLEND)

    def release(self):
        self._release_targets()
        for res in list(self._vaos.values()) + list(self._progs.values()):
            res.release()
        if self._palette_tex is not None:
            self._palette_tex.release()
        self._vaos, self._progs, self._palette_tex = {}, {}, None

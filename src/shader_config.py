import re

# Built-in ISF Shader Presets
_PRESET_PLASMA = """/*{
    "DESCRIPTION": "Multi-frequency Liquid Plasma Generator",
    "TYPE": "generator",
    "INPUTS": [
        { "NAME": "speed", "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.0, "MAX": 5.0 },
        { "NAME": "scale", "TYPE": "float", "DEFAULT": 3.5, "MIN": 1.0, "MAX": 12.0 },
        { "NAME": "color1", "TYPE": "color", "DEFAULT": [0.1, 0.4, 1.0, 1.0] },
        { "NAME": "color2", "TYPE": "color", "DEFAULT": [1.0, 0.2, 0.6, 1.0] }
    ]
}*/
void main() {
    vec2 p = (isf_FragNormCoord - 0.5) * scale;
    float t = TIME * speed;
    float v1 = sin(p.x * 2.0 + t);
    float v2 = sin(p.y * 3.0 - t * 0.8);
    float v3 = sin((p.x + p.y) * 2.5 + t * 1.2);
    float v4 = sin(length(p) * 4.0 - t * 1.5);
    float v = (v1 + v2 + v3 + v4) * 0.25;

    vec3 col = mix(color1.rgb, color2.rgb, 0.5 + 0.5 * v);
    col += vec3(0.08 / (length(p) + 0.15));
    gl_FragColor = vec4(col * alpha, 1.0);
}"""

_PRESET_WAVES = """/*{
    "DESCRIPTION": "Neon Audio-reactive Sine Waves Generator",
    "TYPE": "generator",
    "INPUTS": [
        { "NAME": "speed", "TYPE": "float", "DEFAULT": 1.5, "MIN": 0.0, "MAX": 5.0 },
        { "NAME": "frequency", "TYPE": "float", "DEFAULT": 8.0, "MIN": 1.0, "MAX": 25.0 },
        { "NAME": "amplitude", "TYPE": "float", "DEFAULT": 0.25, "MIN": 0.0, "MAX": 0.8 },
        { "NAME": "glow", "TYPE": "color", "DEFAULT": [0.2, 0.8, 1.0, 1.0] }
    ]
}*/
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * speed;
    float y = uv.y - 0.5;

    float d1 = abs(y - sin(uv.x * frequency + t) * amplitude);
    float d2 = abs(y - cos(uv.x * (frequency * 0.7) - t * 1.2) * (amplitude * 0.75));
    float d = min(d1, d2);

    float intensity = 0.015 / (d + 0.005);
    gl_FragColor = vec4(glow.rgb * intensity * alpha, 1.0);
}"""

_PRESET_TUNNEL = """/*{
    "DESCRIPTION": "Infinite Swirling Tunnel Generator",
    "TYPE": "generator",
    "INPUTS": [
        { "NAME": "speed", "TYPE": "float", "DEFAULT": 1.2, "MIN": 0.0, "MAX": 5.0 },
        { "NAME": "twist", "TYPE": "float", "DEFAULT": 1.5, "MIN": 0.0, "MAX": 5.0 },
        { "NAME": "color", "TYPE": "color", "DEFAULT": [0.3, 0.9, 0.7, 1.0] }
    ]
}*/
void main() {
    vec2 p = (isf_FragNormCoord - 0.5) * 2.0;
    float r = length(p);
    float a = atan(p.y, p.x);
    float u = 0.15 / (r + 0.002) + TIME * speed;
    float v = (a / 3.14159265) + sin(r * 4.0 - TIME * speed) * twist;
    float c = sin(u * 16.0) * sin(v * 10.0);
    gl_FragColor = vec4(color.rgb * clamp(c, 0.0, 1.0) * (r * 1.6) * alpha, 1.0);
}"""

_PRESET_GLITCH = """/*{
    "DESCRIPTION": "RGB Shift & Scanline Jitter Filter",
    "TYPE": "filter",
    "INPUTS": [
        { "NAME": "inputImage", "TYPE": "image" },
        { "NAME": "amount", "TYPE": "float", "DEFAULT": 0.03, "MIN": 0.0, "MAX": 0.2 },
        { "NAME": "speed", "TYPE": "float", "DEFAULT": 3.0, "MIN": 0.0, "MAX": 10.0 }
    ]
}*/
void main() {
    vec2 uv = isf_FragNormCoord;
    float t = TIME * speed;

    float jitter = sin(uv.y * 35.0 + t * 4.0) * cos(uv.y * 12.0 - t * 2.0);
    if (abs(jitter) > 0.65) {
        uv.x += (jitter * 0.015 * amount * 20.0);
    }

    vec2 offset = vec2(amount, 0.0);
    float r = texture2D(inputImage, uv + offset).r;
    float g = texture2D(inputImage, uv).g;
    float b = texture2D(inputImage, uv - offset).b;
    float a = texture2D(inputImage, uv).a;

    gl_FragColor = vec4(r, g, b, a * alpha);
}"""

_PRESET_KALEIDOSCOPE = """/*{
    "DESCRIPTION": "Radial Kaleidoscope Mirror Filter",
    "TYPE": "filter",
    "INPUTS": [
        { "NAME": "inputImage", "TYPE": "image" },
        { "NAME": "segments", "TYPE": "float", "DEFAULT": 6.0, "MIN": 2.0, "MAX": 24.0 },
        { "NAME": "rotation", "TYPE": "float", "DEFAULT": 0.0, "MIN": 0.0, "MAX": 6.283 }
    ]
}*/
void main() {
    vec2 uv = isf_FragNormCoord - 0.5;
    float r = length(uv);
    float a = atan(uv.y, uv.x) + rotation;

    float seg = 6.28318530718 / max(segments, 1.0);
    a = mod(a, seg);
    if (a > seg * 0.5) {
        a = seg - a;
    }

    vec2 new_uv = vec2(cos(a), sin(a)) * r + 0.5;
    new_uv = clamp(new_uv, 0.0, 1.0);
    gl_FragColor = texture2D(inputImage, new_uv) * alpha;
}"""

_PRESET_RGB_SHIFT = """/*{
    "DESCRIPTION": "Chromatic Aberration / RGB Split Filter",
    "TYPE": "filter",
    "INPUTS": [
        { "NAME": "inputImage", "TYPE": "image" },
        { "NAME": "offset_x", "TYPE": "float", "DEFAULT": 0.015, "MIN": -0.1, "MAX": 0.1 },
        { "NAME": "offset_y", "TYPE": "float", "DEFAULT": 0.0, "MIN": -0.1, "MAX": 0.1 }
    ]
}*/
void main() {
    vec2 uv = isf_FragNormCoord;
    vec2 off = vec2(offset_x, offset_y);
    float r = texture2D(inputImage, uv + off).r;
    float g = texture2D(inputImage, uv).g;
    float b = texture2D(inputImage, uv - off).b;
    float a = texture2D(inputImage, uv).a;
    gl_FragColor = vec4(r, g, b, a * alpha);
}"""

_PRESET_PIXELATE = """/*{
    "DESCRIPTION": "Mosaic Pixelation Filter",
    "TYPE": "filter",
    "INPUTS": [
        { "NAME": "inputImage", "TYPE": "image" },
        { "NAME": "pixels", "TYPE": "float", "DEFAULT": 40.0, "MIN": 4.0, "MAX": 200.0 }
    ]
}*/
void main() {
    vec2 uv = isf_FragNormCoord;
    float n = max(pixels, 2.0);
    vec2 p = floor(uv * n) / n;
    gl_FragColor = texture2D(inputImage, p) * alpha;
}"""

SHADER_PRESETS = {
    "plasma": _PRESET_PLASMA,
    "waves": _PRESET_WAVES,
    "tunnel": _PRESET_TUNNEL,
    "glitch": _PRESET_GLITCH,
    "kaleidoscope": _PRESET_KALEIDOSCOPE,
    "rgb_shift": _PRESET_RGB_SHIFT,
    "pixelate": _PRESET_PIXELATE,
}

# Header & Shader Parsing

_HEADER_REGEX = re.compile(r"/\*\{(.*?)\}\s*\*/", re.DOTALL)

_VERTEX_SHADER = """#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 isf_FragNormCoord;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    isf_FragNormCoord = in_uv;
}
"""

_FRAGMENT_PREAMBLE = """#version 330
precision highp float;

uniform vec2 RENDERSIZE;
uniform float TIME;
uniform float TIMEDELTA;
uniform int FRAMEINDEX;
uniform vec4 DATE;
uniform float alpha;

in vec2 isf_FragNormCoord;
out vec4 chladni_FragColor;

#define gl_FragColor chladni_FragColor
#define texture2D texture
#define IMG_NORM_PIXEL(sampler, uv) texture(sampler, uv)
#define IMG_PIXEL(sampler, px) texture(sampler, (px) / RENDERSIZE)
#define IMG_THIS_NORM_PIXEL(sampler) texture(sampler, isf_FragNormCoord)
#define IMG_THIS_PIXEL(sampler) texture(sampler, gl_FragCoord.xy / RENDERSIZE)
"""


_SHADER_SUFFIXES = (".fs", ".frag", ".glsl", ".isf")

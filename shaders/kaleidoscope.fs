/*{
    "DESCRIPTION": "Radial Kaleidoscope Mirror Filter",
    "CREDIT": "chladni",
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
}

/*{
    "DESCRIPTION": "RGB Shift & Scanline Jitter Filter",
    "CREDIT": "chladni",
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
}

/*{
    "DESCRIPTION": "Neon Audio-reactive Sine Waves Generator",
    "CREDIT": "chladni",
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
}

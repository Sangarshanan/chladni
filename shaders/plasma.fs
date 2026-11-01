/*{
    "DESCRIPTION": "Liquid Plasma Generator",
    "CREDIT": "chladni",
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
}

#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/normal.hpp>

#include "sceneStructs.h"
#include "utilities.h"

__device__
float noise2D(glm::vec2 p) {
    return glm::fract(glm::sin(glm::dot(p, glm::vec2(127.1, 311.7))) *
        43758.5453);
}

__device__
float interpNoise2D(float x, float y) {
    int intX = int(floor(x));
    float fractX = glm::fract(x);
    int intY = int(floor(y));
    float fractY = glm::fract(y);

    float v1 = noise2D(glm::vec2(intX, intY));
    float v2 = noise2D(glm::vec2(intX + 1, intY));
    float v3 = noise2D(glm::vec2(intX, intY + 1));
    float v4 = noise2D(glm::vec2(intX + 1, intY + 1));

    float i1 = glm::mix(v1, v2, fractX);
    float i2 = glm::mix(v3, v4, fractX);
    return glm::mix(i1, i2, fractY);
}

__device__
float fbm(glm::vec2 p) {
    float total = 0;
    float persistence = 0.5f;
    int octaves = 8;
    float freq = 2.f;
    float amp = 0.5f;
    for (int i = 1; i <= octaves; i++) {
        freq *= 2.f;
        amp *= persistence;

        total += interpNoise2D(p.x * freq,
            p.y * freq) * amp;
    }
    return total;
}

__device__
float pattern(glm::vec2 p)
{
    glm::vec2 q = glm::vec2(fbm(p + glm::vec2(0.0, 0.0)),
        fbm(p + glm::vec2(5.2, 1.3)));

    glm::vec2 r = glm::vec2(fbm(p + 4.0f * q + glm::vec2(1.7, 9.2)),
        fbm(p + 4.0f * q + glm::vec2(8.3, 2.8)));

    return fbm(p + 4.0f * r);
}
__device__
glm::vec3 ProcColorValue2(double u, double v, const glm::vec3 p) {
    glm::vec3 odd(0.98, 0.98, 0.98);
    glm::vec3  even(0.1, 0.1, 0.1);
    float pac = pattern(glm::vec2(u, v)) * 2;


    if (pac > 0.6)
    {
        return glm::vec3(0.25f, 0.25f, 0.85f);
    }
    if (pac > 0.4)
    {
        return glm::vec3(0.85f, 0.2f, 0.3f);
    }

    // return yellow
    return glm::vec3(0.85, 0.67, 0.35);
}



__device__
glm::vec3 ProcColorValue(double u, double v, const glm::vec3 p) {
    glm::vec3 odd(0.2, 0.3, 0.1);
    glm::vec3  even(0.9, 0.9, 0.9);
    auto sines = sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
    if (sines < 0)
        return odd;
    else
        return even;
}

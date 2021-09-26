#pragma once


#include "glm/glm.hpp"
#include "../utilities.h"
#include "../sceneStructs.h"

enum MaterialAdvType {
    BLINN_PHONG,
    COOK_TOLERANCE,
};

struct MaterialAdvance {
    inline glm::vec3 sampleOut(glm::vec3 in, glm::vec3 normal) const;
    inline glm::vec3 evalBSDF(glm::vec3 in, glm::vec3 normal, glm::vec3 out) const;
    inline float evalPDF(glm::vec3 in, glm::vec3 normal, glm::vec3 out) const;

    inline glm::vec4 getDiffuseTexture(glm::vec2 uv) const;
    inline glm::vec3 getNormalTexture(glm::vec2 uv) const;

    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    glm::vec4* dev_diffuseTexture = nullptr;
    glm::ivec2 diffuseTextureSize;
    glm::vec3* dev_normalTexture = nullptr;
    glm::ivec2 normalTextureSize;

    enum MaterialAdvType materialType = BLINN_PHONG;
};

inline glm::vec3 MaterialAdvance::sampleOut(glm::vec3 in, glm::vec3 normal) const
{
    return glm::vec3();
}

inline glm::vec3 MaterialAdvance::evalBSDF(glm::vec3 in, glm::vec3 normal, glm::vec3 out) const
{
    return glm::vec3();
}

inline float MaterialAdvance::evalPDF(glm::vec3 in, glm::vec3 normal, glm::vec3 out) const
{
    return 0.0f;
}

inline glm::vec4 MaterialAdvance::getDiffuseTexture(glm::vec2 uv) const
{
    return glm::vec4();
}

inline glm::vec3 MaterialAdvance::getNormalTexture(glm::vec2 uv) const
{
    return glm::vec3();
}


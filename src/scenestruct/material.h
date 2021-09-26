#pragma once

#include <thrust/random.h>
#include "glm/glm.hpp"
#include "../utilities.h"
#include "texture.h"

enum class MaterialType : ui8 {
    PHONG,
    COOK_TOLERANCE,
};

struct MonteCarloPair {
    GLM_FUNC_QUALIFIER MonteCarloPair(glm::vec3 out = glm::vec3(), glm::vec3 bsdfTimesCosSlashPDF = glm::vec3())
        : out(out)
        , bsdfTimesCosSlashPDF(bsdfTimesCosSlashPDF) {}
        //, bsdfTimesCosSlashPDF(glm::clamp(bsdfTimesCosSlashPDF, glm::vec3(0.f), glm::vec3(1.f))) {}

    glm::vec3 out;
    glm::vec3 bsdfTimesCosSlashPDF;
};

struct Material {
    GLM_FUNC_QUALIFIER MonteCarloPair sampleScatter(glm::vec3 in, glm::vec3 normal, glm::vec2 uv, thrust::default_random_engine& rng) const;

    GLM_FUNC_QUALIFIER glm::vec4 sampleOutWithPDF(glm::vec3 in, glm::vec3 normal, thrust::default_random_engine& rng) const;
    GLM_FUNC_QUALIFIER glm::vec3 evalBSDF(glm::vec3 in, glm::vec3 normal, glm::vec3 out, glm::vec2 uv, float prob = 0.f) const;

    // Albedo
    GLM_FUNC_QUALIFIER glm::vec3 getDiffuse(glm::vec2 uv) const;
    GLM_FUNC_QUALIFIER glm::vec3 getSpecular(glm::vec2 uv) const;
    GLM_FUNC_QUALIFIER glm::vec3 getNormal(glm::vec2 uv) const;

    glm::vec3 color;
    struct {
        float exponent = 0.f;
        glm::vec3 color;
    } specular;
    ui8 hasReflective = 0;
    ui8 hasRefractive = 0;
    float indexOfRefraction = 0.f;
    float roughness = 0.f;
    float emittance = 0.f;

    Texture2D<glm::vec3> diffuseTexture;
    Texture2D<glm::vec3> specularTexture;
    Texture2D<glm::vec3> normalTexture;

    MaterialType materialType = MaterialType::PHONG;

protected:
    GLM_FUNC_QUALIFIER float fresnel(const glm::vec3& in, const glm::vec3& normal, float ior);

    GLM_FUNC_QUALIFIER MonteCarloPair Phong_sampleScatter_Uniform(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const;
    GLM_FUNC_QUALIFIER glm::vec4 Phong_sampleOutWithPDF_Uniform(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const;

    GLM_FUNC_QUALIFIER MonteCarloPair Phong_sampleScatter_CosWeighted_Phong(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const;
    GLM_FUNC_QUALIFIER glm::vec4 Phong_sampleOutWithPDF_CosWeighted(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const;
    GLM_FUNC_QUALIFIER glm::vec4 Phong_sampleOutWithPDF_Phong(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const;

    GLM_FUNC_QUALIFIER glm::vec3 Phong_evalBSDF(const glm::vec3& in, const glm::vec3& normal, const glm::vec3& out, const glm::vec2& uv, float prob = 0.f) const;
};

GLM_FUNC_QUALIFIER MonteCarloPair Material::sampleScatter(glm::vec3 in, glm::vec3 normal, glm::vec2 uv, thrust::default_random_engine& rng) const
{
    switch (materialType) {
    case MaterialType::PHONG:
    {
        return Phong_sampleScatter_CosWeighted_Phong(in, normal, uv, rng);
        //return sampleScatter_Uniform(in, normal, uv, rng);
    }
    case MaterialType::COOK_TOLERANCE:
        return MonteCarloPair();
    }
    return MonteCarloPair();
}

GLM_FUNC_QUALIFIER glm::vec4 Material::sampleOutWithPDF(glm::vec3 in, glm::vec3 normal, thrust::default_random_engine& rng) const {
    switch (materialType) {
    case MaterialType::PHONG:
        return Phong_sampleOutWithPDF_Uniform(in, normal, rng);
    case MaterialType::COOK_TOLERANCE:
        return glm::vec4();//TODO
    }
    return glm::vec4();
}

GLM_FUNC_QUALIFIER glm::vec3 Material::evalBSDF(glm::vec3 in, glm::vec3 normal, glm::vec3 out, glm::vec2 uv, float prob) const {
    switch (materialType) {
    case MaterialType::PHONG:
        return Phong_evalBSDF(in, normal, out, uv, prob);
    case MaterialType::COOK_TOLERANCE:
        return glm::vec3();//TODO
    }
    return glm::vec3();
}

GLM_FUNC_QUALIFIER glm::vec3 Material::getDiffuse(glm::vec2 uv) const {
    return diffuseTexture.isReadable() ? diffuseTexture.getPixelByUVBilinear(uv.x, uv.y) : color;
}

GLM_FUNC_QUALIFIER glm::vec3 Material::getSpecular(glm::vec2 uv) const {
    return specularTexture.isReadable() ? specularTexture.getPixelByUVBilinear(uv.x, uv.y) : specular.color;
}

GLM_FUNC_QUALIFIER glm::vec3 Material::getNormal(glm::vec2 uv) const {
    return normalTexture.isReadable() ? glm::normalize(normalTexture.getPixelByUVBilinear(uv.x, uv.y)) : glm::vec3(0.f, 0.f, 1.f);
}

#include "material.inl"

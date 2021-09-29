#pragma once

#include <thrust/random.h>
#include "glm/glm.hpp"
#include "../utilities.h"
#include "texture.h"

enum class MaterialType : ui8 {
    PHONG,
    DIELECTRIC,
    MICROFACET_GGX,
};

struct MonteCarloReturn {
    GLM_FUNC_QUALIFIER MonteCarloReturn(glm::vec3 out = glm::vec3(), glm::vec3 bsdfTimesCosSlashPDF = glm::vec3(), ui8 penetrate = 0)
        : out(out)
        , bsdfTimesCosSlashPDF(bsdfTimesCosSlashPDF)
        //, bsdfTimesCosSlashPDF(glm::clamp(bsdfTimesCosSlashPDF, glm::vec3(0.f), glm::vec3(1.f)))
        , penetrate(penetrate) {}

    glm::vec3 out;
    glm::vec3 bsdfTimesCosSlashPDF;
    ui8 penetrate;
};

struct Material {
    GLM_FUNC_QUALIFIER MonteCarloReturn sampleScatter(glm::vec3 in, glm::vec3 normal, glm::vec2 uv, thrust::default_random_engine& rng) const;

    // Albedo
    GLM_FUNC_QUALIFIER glm::vec3 getDiffuse(glm::vec2 uv) const;
    GLM_FUNC_QUALIFIER glm::vec3 getSpecular(glm::vec2 uv) const;
    GLM_FUNC_QUALIFIER glm::vec3 getNormal(glm::vec2 uv) const;

    glm::vec3 color;
    struct {
        float exponent = 0.f;
        glm::vec3 color;
    } specular;
    union {
        float hasReflective = 0.f;
        float metallic;
    };
    float hasRefractive = 0.f;
    float indexOfRefraction = 0.f;
    float emittance = 0.f;

    Texture2D<glm::vec3> diffuseTexture;
    Texture2D<glm::vec3> specularTexture;
    Texture2D<glm::vec3> normalTexture;

    MaterialType materialType = MaterialType::PHONG;

protected:
    GLM_FUNC_QUALIFIER static float fresnel(const glm::vec3& in, const glm::vec3& normal, float ior, float* normalSgn = nullptr);

    //GLM_FUNC_QUALIFIER MonteCarloReturn Phong_sampleScatter_Uniform(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const;
    //GLM_FUNC_QUALIFIER glm::vec4 Phong_sampleOutWithPDF_Uniform(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const;

    // Start Phong
    GLM_FUNC_QUALIFIER MonteCarloReturn Phong_sampleScatter_CosWeighted_Phong(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const;
    GLM_FUNC_QUALIFIER glm::vec4 Phong_sampleOutWithPDF_CosWeighted(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const;
    GLM_FUNC_QUALIFIER glm::vec4 Phong_sampleOutWithPDF_PhongSpecular(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const;

    GLM_FUNC_QUALIFIER glm::vec3 Phong_evalBSDF(const glm::vec3& in, const glm::vec3& normal, const glm::vec3& out, const glm::vec2& uv, float prob = 0.f) const;

    // End Phong

    // Start Dielectric
    GLM_FUNC_QUALIFIER MonteCarloReturn Dielectric_sampleScatter(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const;
    GLM_FUNC_QUALIFIER glm::vec4 Dielectric_sampleOutWithPDF(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const;

    GLM_FUNC_QUALIFIER glm::vec3 Dielectric_evalBSDF(const glm::vec3& in, const glm::vec3& normal, const glm::vec3& out, const glm::vec2& uv, float prob = 0.f) const;
    // End Dielectric

    // Start Cook-Torrence
    GLM_FUNC_QUALIFIER float getRoughness() const { return 1.f - metallic; }
    
    GLM_FUNC_QUALIFIER static float MicrofacetGGX_NormalDistribution(float cosH, float roughness, float tanH);
    GLM_FUNC_QUALIFIER static float MicrofacetGGX_GeometrySchlick(const glm::vec3& v, const glm::vec3& n, float k);
    GLM_FUNC_QUALIFIER static float MicrofacetGGX_GeometrySmithApproximation(const glm::vec3& in, const glm::vec3& mfNormal, const glm::vec3& out, float roughness);
    GLM_FUNC_QUALIFIER static float MicrofacetGGX_Geometry(const glm::vec3& in, const glm::vec3& mfNormal, const glm::vec3& out, float roughness);
    GLM_FUNC_QUALIFIER static glm::vec3 MicrofacetGGX_FresnelSchlick(const glm::vec3& F0, float cosIM);

    GLM_FUNC_QUALIFIER static glm::vec3 calculateGGXRandomDirectionInHemisphere(
        glm::vec3 normal, float roughness, thrust::default_random_engine& rng, float* pdf = nullptr);
    
    GLM_FUNC_QUALIFIER MonteCarloReturn MicrofacetGGX_sampleScatter(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const;
    GLM_FUNC_QUALIFIER glm::vec4 MicrofacetGGX_sampleMicrofacetNormalWithPDF(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const;

    //GLM_FUNC_QUALIFIER glm::vec3 MicrofacetGGX_evalBSDF(const glm::vec3& in, const glm::vec3& normal, const glm::vec3& out, const glm::vec2& uv, float prob = 0.f) const;

    // End Cook-Torrence
};
#include "geometry.h"
#include "material.inl"
#include "materialPhong.inl"
#include "materialPerfectDielectric.inl"
#include "materialMicrofacetGGX.inl"

#include "material.h"

// Start Microfacet-GGX

GLM_FUNC_QUALIFIER float Material::MicrofacetGGX_NormalDistribution(float cosH, float roughness, float tanH) {
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float cos2H = cosH * cosH;
    float a2Minus1 = a2 - 1.f;
    float sqrtDe = glm::max(cos2H * a2Minus1 + 1.f, EPSILON);
    //float a2AddTan2H = a2 + tanH * tanH;
    //float sqrtDe = glm::max(cos2H * a2AddTan2H, EPSILON);

    return a2 / (PI * sqrtDe * sqrtDe);
}

GLM_FUNC_QUALIFIER float Material::MicrofacetGGX_GeometrySchlick(const glm::vec3& v, const glm::vec3& n, float k) {
    float cosNV = glm::max(0.f, glm::dot(n, v));
    return cosNV / (glm::max(cosNV, EPSILON) * (1.f - k) + k);
}

GLM_FUNC_QUALIFIER float Material::MicrofacetGGX_GeometrySmithApproximation(const glm::vec3& in, const glm::vec3& mfNormal, const glm::vec3& out, float roughness) {
    float k = roughness * roughness * 0.5f;//(roughness + 1) * (roughness + 1) / 8.f;
    float schlickIn = MicrofacetGGX_GeometrySchlick(-in, mfNormal, k);
    float schlickOut = MicrofacetGGX_GeometrySchlick(out, mfNormal, k);
    return schlickIn * schlickOut;
}

GLM_FUNC_QUALIFIER float Material::MicrofacetGGX_Geometry(const glm::vec3& in, const glm::vec3& mfNormal, const glm::vec3& out, float roughness) {
    float cosOM = glm::max(0.f, glm::dot(mfNormal, out));
    float nu = 2.f * cosOM;
    float a2 = roughness * roughness;
    a2 *= a2;
    float de = cosOM + sqrt(a2 + (1.f - a2) * cosOM * cosOM);
    return nu / glm::max(de, EPSILON);
}

GLM_FUNC_QUALIFIER glm::vec3 Material::MicrofacetGGX_FresnelSchlick(const glm::vec3& F0, float cosIM) {
    float cosFactor = (1. - cosIM) * (1. - cosIM);
    cosFactor *= cosFactor * (1. - cosIM);

    return F0 + (1.f - F0) * cosFactor;
}

GLM_FUNC_QUALIFIER glm::vec3 Material::calculateGGXRandomDirectionInHemisphere(glm::vec3 normal, float roughness, thrust::default_random_engine& rng, float* pdf) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float si1 = u01(rng);
    float up = 0.f; // cos(H)

    float tanH = roughness * roughness * sqrt(si1) / glm::max(sqrt(1.f - si1), FLT_EPSILON);
    up = 1.f / sqrt((tanH * tanH) + 1.f);

    float over = sqrt(1.f - up * up); // sin(H)
    float around = u01(rng) * TWO_PI;
    glm::vec3 dir(cos(around) * over, sin(around) * over, up);
    if (pdf) {
        *pdf = MicrofacetGGX_NormalDistribution(up, roughness, tanH) * up;
    }
    return tangentSpaceToWorldSpace(dir, normal);
}

GLM_FUNC_QUALIFIER MonteCarloReturn Material::MicrofacetGGX_sampleScatter(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const {
    float roughness = getRoughness();

    float cosINRaw = glm::dot(-in, normal);
    glm::vec3 nrm = cosINRaw < 0.f ? -normal : normal;

    glm::vec4 sample = MicrofacetGGX_sampleMicrofacetNormalWithPDF(in, nrm, rng);
    glm::vec3 mfNormal(sample);
    glm::vec3 out = glm::reflect(in, mfNormal);
    if (glm::dot(out, nrm) < 0.f && glm::dot(-in, mfNormal) < 0.f) {
        return MonteCarloReturn(mfNormal, glm::vec3(0.f), 0);
    }
    //while (glm::dot(out, nrm) < 0.f && glm::dot(-in, mfNormal) < 0.f) {
    //    //out = mfNormal; // Can it deal with black spots?
    //    //mfNormal = glm::normalize(out - in);
    //    sample = MicrofacetGGX_sampleMicrofacetNormalWithPDF(in, nrm, rng);
    //    mfNormal = glm::vec3(sample);
    //    out = glm::reflect(in, mfNormal);
    //}

    float geom = //MicrofacetGGX_Geometry(in, mfNormal, out, roughness);
        MicrofacetGGX_GeometrySmithApproximation(in, mfNormal, out, roughness);
    geom = glm::max(geom, 0.f);

    float cosIMRaw = glm::dot(-in, mfNormal);
    cosINRaw = glm::dot(-in, nrm);
    float cosMNRaw = glm::dot(mfNormal, nrm);

    float cosIM = glm::max(0.f, cosIMRaw);
    float cosIN = glm::max(0.f, cosINRaw);
    float cosMN = glm::max(0.f, cosMNRaw);

    //if (cosIN * cosMN < EPSILON) { // Otherwise there are so many white spots with roughness->1. But there are black spots...
    //    return MonteCarloReturn(out, glm::vec3(0.f), 0);
    //    //printf("< 0.f???: cosIM = %f, cosIN = %f, geom = %f, cosMN = %f\n", cosIM, cosIN, geom, cosMN);
    //}

    glm::vec3 colorWithFresnel = MicrofacetGGX_FresnelSchlick(getDiffuse(uv), cosIM);
    float factor = (cosIM / glm::max(cosIN, EPSILON) * geom / glm::max(cosMN, EPSILON));
    //float factor = (cosIM * geom / glm::max(cosIN * cosMN, EPSILON));
    //float cosLight = glm::max(0.f, glm::dot(out, mfNormal));
    //factor = glm::clamp(factor * cosLight, 0.f, 1.f);
    
    factor = glm::clamp(factor, 0.f, 1.f);
    
    //if (factor > 1.f / EPSILON || factor < 0.0001f) {
    //    //factor = 1.f / EPSILON;
    //    printf("factor = %f: cosIM = %f, cosIN = %f, geom = %f, cosMN = %f\n", factor, cosIMRaw, cosINRaw, geom, cosMNRaw);
    //}
    return MonteCarloReturn(out, colorWithFresnel * factor, 0);
}

GLM_FUNC_QUALIFIER glm::vec4 Material::MicrofacetGGX_sampleMicrofacetNormalWithPDF(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const {
    float pdf;
    glm::vec3 mfNormal = glm::normalize(calculateGGXRandomDirectionInHemisphere(normal, getRoughness(), rng, &pdf));
    return glm::vec4(mfNormal, pdf);
}

//GLM_FUNC_QUALIFIER glm::vec3 Material::MicrofacetGGX_evalBSDF(const glm::vec3& in, const glm::vec3& normal, const glm::vec3& out, const glm::vec2& uv, float prob) const {
//    
//    return GLM_FUNC_QUALIFIER glm::vec3();
//}

// End Microfacet-GGX


GLM_FUNC_QUALIFIER MonteCarloReturn Material::sampleScatter(glm::vec3 in, glm::vec3 normal, glm::vec2 uv, thrust::default_random_engine& rng) const
{
    switch (materialType) {
    case MaterialType::PHONG:
    {
        return Phong_sampleScatter_CosWeighted_Phong(in, normal, uv, rng);
        //return sampleScatter_Uniform(in, normal, uv, rng);
    }
    case MaterialType::DIELECTRIC:
        return Dielectric_sampleScatter(in, normal, uv, rng);
    case MaterialType::MICROFACET_GGX:
        return MicrofacetGGX_sampleScatter(in, normal, uv, rng);
    }
    return MonteCarloReturn();
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

GLM_FUNC_QUALIFIER float Material::fresnel(const glm::vec3& in, const glm::vec3& normal, float ior, float* normalSgn) {
    float cosi = glm::clamp(glm::dot(in, normal), -1.f, 1.f);
    float etai = 1.f, etat = ior;
    if (normalSgn) {
        *normalSgn = 1.f;
    }
    if (cosi > 0) {
        thrust::swap(etai, etat); 
        if (normalSgn) {
            *normalSgn = -1.f; // Flip the normal for shading.
        }
    }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(glm::max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1.f) {
        return 1.f;
    }
    else {
        float cost = sqrtf(glm::max(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);

        float etatcosi = etat * cosi, etaicost = etai * cost;
        float etaicosi = etai * cosi, etatcost = etat * cost;
        float RsDe = (etatcosi)+(etaicost), RpDe = (etaicosi)+(etatcost);

        float Rs = fabs(RsDe) < FLT_EPSILON ? 0.f : ((etatcosi) - (etaicost)) / (RsDe);
        float Rp = fabs(RpDe) < FLT_EPSILON ? 0.f : ((etaicosi) - (etatcost)) / (RpDe);
        return (Rs * Rs + Rp * Rp) * 0.5f;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
}

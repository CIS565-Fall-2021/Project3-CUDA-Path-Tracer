
// Start Dielectric
#define USE_IN_TO_EVAL 1

GLM_FUNC_QUALIFIER MonteCarloReturn Material::Dielectric_sampleScatter(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const {
    glm::vec4 sample;
    float probSpecular = 0.f;
    ui8 refractionThisSample = 0;
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    glm::vec3 albedoForRefraction = color;

#if USE_IN_TO_EVAL
    glm::vec3 inForBSDF(in);
#endif // USE_IN_TO_EVAL
    ui8 filpNormal = glm::dot(in, normal) > 0.f;
    glm::vec3 nrmForSample = filpNormal ? -normal : normal;
    float ior = filpNormal ? indexOfRefraction : 1.f / indexOfRefraction;
    float kr = indexOfRefraction > 0.f ? fresnel(in, nrmForSample, ior) : 1.f;

    if (hasRefractive) { 
        float probReflection = u01(rng);
        if (probReflection >= kr) {
            // Refraction instead.
            refractionThisSample = 1;
#if USE_IN_TO_EVAL
            inForBSDF = glm::refract(inForBSDF, nrmForSample, ior);
            inForBSDF = glm::reflect(inForBSDF, nrmForSample);
            nrmForSample = -nrmForSample;
#endif // USE_IN_TO_EVAL
        }
    }

#if !USE_IN_TO_EVAL
    sample = Phong_sampleOutWithPDF_PhongSpecular(in, nrmForSample, rng);
    glm::vec3 out(sample);
    glm::vec3 outForBSDF(sample);
    float cosLight = glm::max(0.f, glm::dot(outForBSDF, nrmForSample));
    glm::vec3 color = Phong_evalBSDF(in, nrmForSample, outForBSDF, uv, probSpecular) * cosLight;
    if (refractionThisSample) {
        out = glm::reflect(out, -nrmForSample);
        out = glm::refract(out, nrmForSample, 1.f / ior);
        color = (ior * ior) * (albedoForRefraction - color);
    }
#else // USE_IN_TO_EVAL
    sample = Dielectric_sampleOutWithPDF(inForBSDF, nrmForSample, rng);
    glm::vec3 out(sample);

    albedoForRefraction = Dielectric_evalBSDF(inForBSDF, nrmForSample, out, uv, probSpecular);

    float cosLight = glm::max(0.f, glm::dot(out, nrmForSample));
    glm::vec3 color = albedoForRefraction * (refractionThisSample ? 1.f : cosLight);//refractionThisSample ? (1.f - kr) * albedoForRefraction : kr * albedoForRefraction;// Phong_evalBSDF(inForBSDF, nrmForSample, out, uv, probSpecular)* (specular.exponent == 0.f ? 1.f : cosLight);
    //printf("%f %f %f\n", color.r, color.g, color.b);
    if (refractionThisSample) {
        color = (ior * ior) * color;
    }
#endif // USE_IN_TO_EVAL
    return MonteCarloReturn(out, color / (sample.w), refractionThisSample);
}

GLM_FUNC_QUALIFIER glm::vec4 Material::Dielectric_sampleOutWithPDF(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const {
    float pdf;
    glm::vec3 out = glm::normalize(calculateCosWeightedRandomDirectionInPhongSpecularRegion(glm::reflect(in, normal), rng, specular.exponent, &pdf));
    return glm::vec4(out, pdf);
}

GLM_FUNC_QUALIFIER glm::vec3 Material::Dielectric_evalBSDF(const glm::vec3& in, const glm::vec3& normal, const glm::vec3& out, const glm::vec2& uv, float prob) const {
    //glm::vec3 diffuseColor = getDiffuse(uv);

    //if (!hasReflective) { // Diffuse
    //    return diffuseColor / PI;
    //}

    //glm::vec3 specularColor = getSpecular(uv);
    //float diffuseWeight = glm::max(diffuseColor.r, glm::max(diffuseColor.g, diffuseColor.b));
    //float specularWeight = glm::max(specularColor.r, glm::max(specularColor.g, specularColor.b));
    //float weightedProb = prob * (diffuseWeight + specularWeight);
    //if (weightedProb < diffuseWeight) { // Diffuse
    //    return diffuseColor / PI;
    //}
    //else { // Specular
    //    glm::vec3 reflect = glm::reflect(in, normal);
    //    float cosRI = glm::max(0.f, glm::dot(out, reflect));
    //    glm::vec3 outColor = specularColor * (powf(cosRI, specular.exponent) * (specular.exponent + 2.f) / (2.f));
    //    return outColor / PI;
    //}

    return getSpecular(uv);
}

// End Dielectric

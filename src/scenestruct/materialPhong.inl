
// Start Phong

//GLM_FUNC_QUALIFIER MonteCarloReturn Material::Phong_sampleScatter_Uniform(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const {
//    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
//    float prob = u01(rng);
//    glm::vec4 sample = Phong_sampleOutWithPDF_Uniform(in, normal, rng);
//    glm::vec3 out(sample);
//    return MonteCarloReturn(out, Phong_evalBSDF(in, normal, out, uv, prob) * (glm::max(0.f, glm::dot(out, normal)) / sample.w));
//    //glm::vec3 out = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
//    //glm::vec3 half = glm::normalize(in + out);
//    //float cosH = glm::max(0.f, glm::dot(half, normal));
//    //glm::vec3 outColor = getDiffuse(uv);
//    //if (hasReflective && specular.exponent >= 0.f) {
//    //    outColor += specular.color * (powf(cosH, specular.exponent) * (specular.exponent + 8.f) / (8.f));
//    //}
//
//    //return MonteCarloReturn(out, outColor * 2.f);
//}
//
//GLM_FUNC_QUALIFIER glm::vec4 Material::Phong_sampleOutWithPDF_Uniform(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const {
//    float pdf;
//    glm::vec3 out = glm::normalize(calculateUniformRandomDirectionInHemisphere(normal, rng, &pdf));
//    return glm::vec4(out, pdf);
//}

#if 0 // Seems to be bad to implement the refraction within Phong...
GLM_FUNC_QUALIFIER MonteCarloReturn Material::Phong_sampleScatter_CosWeighted_Phong(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const {
    glm::vec4 sample;
    float probSpecular = 0.f;
    ui8 refractionThisSample = 0;
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    glm::vec3 albedoForRefraction = color;

    glm::vec3 nrmForSample = glm::dot(in, normal) > 0.f ? -normal : normal;
    float ior = indexOfRefraction;
    if (hasRefractive && indexOfRefraction > 0.f) { // TODO: Seems not correct...
        //ior = glm::dot(in, normal) > 0.f ? 1.f / indexOfRefraction : indexOfRefraction; // if > 0, from inside to outside so inverse the ior.
        float nrmSgn;
        float kr = fresnel(in, normal, ior, &nrmSgn);
        float probReflection = u01(rng);
        if (probReflection >= kr) {
            // Refraction instead.
            refractionThisSample = 1;
            //nrmForSample = nrmSgn * normal;
        }
    }

    if (!hasReflective) {
        sample = Phong_sampleOutWithPDF_CosWeighted(in, nrmForSample, rng);
    }
    else {
        // Reference: https://www.cs.princeton.edu/courses/archive/fall16/cos526/papers/importance.pdf
        // But why k_d + k_s <= u then contribution is 0? So I change the rule.
        probSpecular = u01(rng);

        float diffuseWeight = glm::max(color.r, glm::max(color.g, color.b));
        float specularWeight = glm::max(specular.color.r, glm::max(specular.color.g, specular.color.b));
        float weightedProb = probSpecular * (diffuseWeight + specularWeight);
        if (weightedProb < diffuseWeight) {
            sample = Phong_sampleOutWithPDF_CosWeighted(in, nrmForSample, rng);
        }
        else {
            sample = Phong_sampleOutWithPDF_PhongSpecular(in, nrmForSample, rng);
            albedoForRefraction = specular.color;
        }
    }
    glm::vec3 out(sample);
    glm::vec3 outForBSDF(sample);
    //float cosLight = glm::max(0.f, (1.f - refraction * 2.f) * glm::dot(outForBSDF, nrmForSample));
    float cosLight = glm::max(0.f, glm::dot(outForBSDF, nrmForSample));
    glm::vec3 color = Phong_evalBSDF(in, nrmForSample, outForBSDF, uv, probSpecular) * cosLight;
    if (refractionThisSample) {
        out = glm::reflect(out, -nrmForSample);
        out = glm::refract(out, nrmForSample, 1.f / ior);
        color = ior * ior * (albedoForRefraction - color);
    }
    return MonteCarloReturn(out, color / sample.w, refractionThisSample);
}
#endif // 0

GLM_FUNC_QUALIFIER MonteCarloReturn Material::Phong_sampleScatter_CosWeighted_Phong(const glm::vec3& in, const glm::vec3& normal, const glm::vec2& uv, thrust::default_random_engine& rng) const {
    glm::vec4 sample;
    float probSpecular = 0.f;

    ui8 filpNormal = glm::dot(in, normal) > 0.f;
    glm::vec3 nrmForSample = filpNormal ? -normal : normal;

    if (!hasReflective) {
        sample = Phong_sampleOutWithPDF_CosWeighted(in, nrmForSample, rng);
    }
    else {
        // Reference: https://www.cs.princeton.edu/courses/archive/fall16/cos526/papers/importance.pdf
        // But why k_d + k_s <= u then contribution is 0? So I change the rule.
        thrust::uniform_real_distribution<float> u01(0.f, 1.f);
        probSpecular = u01(rng);

        float diffuseWeight = glm::max(color.r, glm::max(color.g, color.b));
        float specularWeight = glm::max(specular.color.r, glm::max(specular.color.g, specular.color.b));
        float weightedProb = probSpecular * (diffuseWeight + specularWeight);
        if (weightedProb < diffuseWeight) {
            sample = Phong_sampleOutWithPDF_CosWeighted(in, nrmForSample, rng);
        }
        else {
            sample = Phong_sampleOutWithPDF_PhongSpecular(in, nrmForSample, rng);
        }
    }
    glm::vec3 out(sample);
    float cosLight = glm::max(0.f, glm::dot(out, nrmForSample));
    glm::vec3 color = Phong_evalBSDF(in, nrmForSample, out, uv, probSpecular) * cosLight;
    return MonteCarloReturn(out, color / sample.w, 0);
}

GLM_FUNC_QUALIFIER glm::vec4 Material::Phong_sampleOutWithPDF_CosWeighted(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const {
    float pdf;
    glm::vec3 out = glm::normalize(calculateCosWeightedRandomDirectionInHemisphere(normal, rng, &pdf));
    return glm::vec4(out, pdf);
}

GLM_FUNC_QUALIFIER glm::vec4 Material::Phong_sampleOutWithPDF_PhongSpecular(const glm::vec3& in, const glm::vec3& normal, thrust::default_random_engine& rng) const {
    float pdf;
    glm::vec3 out = glm::normalize(calculateCosWeightedRandomDirectionInPhongSpecularRegion(glm::reflect(in, normal), rng, specular.exponent, &pdf));
    return glm::vec4(out, pdf);
}

GLM_FUNC_QUALIFIER glm::vec3 Material::Phong_evalBSDF(const glm::vec3& in, const glm::vec3& normal, const glm::vec3& out, const glm::vec2& uv, float prob) const {
    glm::vec3 diffuseColor = getDiffuse(uv);

    if (!hasReflective) { // Diffuse
        return diffuseColor / PI;
    }

    glm::vec3 specularColor = getSpecular(uv);
    float diffuseWeight = glm::max(diffuseColor.r, glm::max(diffuseColor.g, diffuseColor.b));
    float specularWeight = glm::max(specularColor.r, glm::max(specularColor.g, specularColor.b));
    float weightedProb = prob * (diffuseWeight + specularWeight);
    if (weightedProb < diffuseWeight) { // Diffuse
        return diffuseColor / PI;
    }
    else { // Specular
        glm::vec3 reflect = glm::reflect(in, normal);
        float cosRI = glm::max(0.f, glm::dot(out, reflect));
        glm::vec3 outColor = specularColor * (powf(cosRI, specular.exponent) * (specular.exponent + 2.f) / (2.f));
        return outColor / PI;
    }
    ////glm::vec3 half = glm::normalize(-in + out);
    ////float cosH = glm::max(0.f, glm::dot(half, normal));
    //glm::vec3 reflect = glm::reflect(-out, normal);
    //float cosRI = glm::max(0.f, glm::dot(in, reflect));
    //glm::vec3 outColor = getDiffuse(uv);
    //if (hasReflective && specular.exponent >= 0.f) {
    //    //outColor += specular.color * (powf(cosH, specular.exponent) * (specular.exponent + 8.f) / (8.f)); // Hard to do the importance sample?
    //    outColor += specular.color * (powf(cosRI, specular.exponent) * (specular.exponent + 2.f) / (2.f));
    //}
    //return outColor / PI;
}

// End Phong

#pragma once

#include "intersections.h"

#define SWITCH_IN_OUT_RAY 0

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRaySimple(
        PathSegment & pathSegment, 
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // DONE: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    if (pathSegment.remainingBounces < 0) {
        return;
    }
    glm::vec3 in = glm::normalize(pathSegment.ray.direction);
    glm::vec3 multColor(1.0f);

    thrust::uniform_real_distribution<float> u01(0, 1);
    float prob = u01(rng);
    glm::vec3 scatterDir;
#if SWITCH_IN_OUT_RAY
    float cosNI = glm::dot(normal, -in);
#else // SWITCH_IN_OUT_RAY
#endif // SWITCH_IN_OUT_RAY
    if (!m.hasReflective || prob < 0.5f) {
        scatterDir = glm::normalize(calculateCosWeightedRandomDirectionInHemisphere(normal, rng));
        multColor *= m.color;
    }
    else {
        //TODO: what about m.specular.exponent?
        scatterDir = glm::reflect(in, normal);
#if SWITCH_IN_OUT_RAY
        multColor *= m.specular.color / PI;
        cosNI = max(cosNI, 0.f);
        multColor *= cosNI;
#else // SWITCH_IN_OUT_RAY
        multColor *= m.specular.color / PI;
        float cosNO = glm::dot(normal, scatterDir);
        cosNO = max(cosNO, 0.f);
        multColor *= cosNO;

        //multColor *= m.specular.color * powf(cosNI, m.specular.exponent) * TWO_PI;
        //multColor *= m.specular.color * powf(cosNI, m.specular.exponent) * 0.5f; // m.color / PI / (0.5f / PI);
#endif // SWITCH_IN_OUT_RAY
    }
#if SWITCH_IN_OUT_RAY
    //cosNI = max(cosNI, 0.f);
    //multColor *= cosNI;
#else // SWITCH_IN_OUT_RAY
    //float cosNO = glm::dot(normal, scatterDir);

    ////if (cosNO < 0.5f) {
    ////    printf("cosNO = %f\n", cosNO);
    ////}

    //cosNO = max(cosNO, 0.f);
    //multColor *= cosNO;
#endif // SWITCH_IN_OUT_RAY

    if (multColor.r > 1.0f || multColor.g > 1.0f || multColor.b > 1.0f) {
        printf("multColor = %f, %f, %f\n", multColor.r, multColor.g, multColor.b);
    }

    pathSegment.color *= multColor;
    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = scatterDir;
    --pathSegment.remainingBounces;
}

#undef SWITCH_IN_OUT_RAY
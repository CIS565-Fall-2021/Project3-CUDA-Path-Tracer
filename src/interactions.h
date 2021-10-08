#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
float reflectance(float indexRefract, float cosTheta)
{
    float r0 = (1.f - indexRefract) / (1.f + indexRefract);
    r0 = r0 * r0;

    return r0 + (1.f - r0) * glm::pow(1 - cosTheta, 5.f);
}

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
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    // DONE: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 incomingRayDirection = pathSegment.ray.direction;

    float totalProb = m.hasReflective + m.hasRefractive;
    float random = u01(rng);

    if (random < m.hasReflective)
    {
        // Color the ray according to surface's material encountered
        pathSegment.color *= m.color;

        // Perform reflection about surface normal of surface
        pathSegment.ray.direction = glm::normalize(glm::reflect(incomingRayDirection, normal));
        pathSegment.ray.origin = intersect;
    }
    else if (random < m.hasReflective + m.hasRefractive)
    {
        float uniformSample = u01(rng);

        bool isOutside = glm::dot(incomingRayDirection, normal) < 0;
        float eta = isOutside ? 1.f / m.indexOfRefraction : m.indexOfRefraction;

        float cosTheta = glm::min(1.f, glm::dot(-incomingRayDirection, normal));
        float sinTheta = glm::sqrt(1.f - cosTheta * cosTheta);

        float refractiveProbability = reflectance(eta, cosTheta);
        bool cannotRefract = eta * sinTheta > 1.f;

        if (cannotRefract || refractiveProbability > uniformSample)
        {
            // Reflect the ray back
            pathSegment.ray.direction = glm::normalize(glm::reflect(incomingRayDirection, normal * (isOutside ? 1.f : -1.f)));
            pathSegment.color *= m.color;

            pathSegment.ray.origin = intersect + 0.001f * normal * (isOutside ? 1.f : -1.f);
        }
        else
        {
            // Refract the ray about the surface normal
            pathSegment.ray.direction = glm::normalize(glm::refract(incomingRayDirection, normal * (isOutside ? 1.f : -1.f), eta));
            pathSegment.color *= glm::vec3(1.f); // *= m.color;

            pathSegment.ray.origin = intersect - 0.001f * normal * (isOutside ? 1.f : -1.f);
        }
    }
    else
    {
        // Color the ray according to surface's material encountered
        pathSegment.color *= m.color;

        // Perform random scattering from surface normal of surface
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.ray.origin = intersect;
    }
}

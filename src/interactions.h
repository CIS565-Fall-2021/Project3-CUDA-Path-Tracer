#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
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
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
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
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) 
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float rand = u01(rng);

    if (!m.hasRefractive && !m.hasReflective) { // pure diffuse surface
        glm::vec3 randDir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + (0.001f * normal);
        pathSegment.ray.direction = glm::normalize(randDir);
        pathSegment.color *= m.color;
    } 
    else if (!m.hasRefractive && m.hasReflective) { // reflective surface
        glm::vec3 reflected = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect + (0.001f * normal);
        pathSegment.ray.direction = glm::normalize(reflected);
        pathSegment.color *= m.specular.color;
    }
    else if (m.hasRefractive) { // refractive surface
        // Schlick's approximation implementation for the process described in PBRT 8.2
        // https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission

        // incident ray has component opposite normal so ray outside and need to swap
        float eta = glm::dot(pathSegment.ray.direction, normal) < 0 ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;

        float cosTheta = glm::dot(-pathSegment.ray.direction, normal) / 
            (glm::length(pathSegment.ray.direction) * (glm::length(normal)));

        float r0 = glm::pow((1.f - eta) / (1.f + eta), 2);
        float schlick = r0 + (1.f - r0) * glm::pow(1 - cosTheta, 5);

        if (schlick < rand) { // refract
            pathSegment.ray.direction = glm::normalize(glm::refract(pathSegment.ray.direction, normal, eta));
            pathSegment.color *= m.color;

            pathSegment.ray.origin = intersect - 0.001f * normal;           
        }
        else { //reflect
            pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
            pathSegment.color *= m.color;

            pathSegment.ray.origin = intersect + 0.001f * normal;
        }
    }
}

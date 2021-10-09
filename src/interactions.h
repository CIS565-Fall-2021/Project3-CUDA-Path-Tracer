#pragma once

#include "intersections.h"
#include <thrust/random.h>

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine &rng) {
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
    glm::vec3 perpendicularDirection1 = glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 = glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// https://blog.demofox.org/2017/01/09/raytracing-reflection-refraction-fresnel-total-internal-reflection-and-beers-law/
__host__ __device__
float FresnelReflectAmount(float n1, float n2, glm::vec3 normal, glm::vec3 incident) {
    // Schlick aproximation
    float r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    float cosX = -glm::dot(normal, incident);

    if (n1 > n2) {
        float n = n1 / n2;
        float sinT2 = n * n * (1.0 - cosX * cosX);

        // Total internal reflection
        if (sinT2 > 1.0) {
            return 1.0;
        }
        cosX = glm::sqrt(1.0 - sinT2);
    }

    float x = 1.0 - cosX;
    float ret = r0 + (1.0 - r0) * x * x * x * x * x;

    // adjust reflect multiplier for object reflectivity
    //ret = (OBJECT_REFLECTIVITY + (1.0 - OBJECT_REFLECTIVITY) * ret);

    return ret;
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
 * combining other types of materials (such as refractive).
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
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &material,
    thrust::default_random_engine &rng,
    bool outside
) {
    if (material.emittance > 0.0f) {
        pathSegment.color *= material.emittance;
        pathSegment.remainingBounces = 0;
    } else {
        pathSegment.ray.origin = intersect;
        pathSegment.remainingBounces -= 1;
        pathSegment.color *= material.color;

        if (material.hasReflective) {
            pathSegment.ray.direction = pathSegment.ray.direction - 2.f * (glm::dot(pathSegment.ray.direction, normal) * normal);
        } else if (material.hasRefractive) {
            float n1 = outside ? 1.0 : material.indexOfRefraction;
            float n2 = outside ? material.indexOfRefraction : 1.0;

            float r = FresnelReflectAmount(n1, n2, normal, pathSegment.ray.direction);
            thrust::uniform_real_distribution<float> u01(0, 1);

            if (u01(rng) < r) {
                pathSegment.ray.direction = pathSegment.ray.direction - 2.f * (glm::dot(pathSegment.ray.direction, normal) * normal);
            } else {
                float eta = outside ? 1.0 / material.indexOfRefraction : material.indexOfRefraction;
                pathSegment.ray.direction = glm::normalize(glm::refract(pathSegment.ray.direction, normal, eta));
            }
        } else {
            pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        }
    }
}


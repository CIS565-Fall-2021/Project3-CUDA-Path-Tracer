#pragma once

#include "intersections.h"

#define ONE_OVER_PI 0.318309886183790671537767526745028724f

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

// adapted from https://raytracing.github.io/books/RayTracingInOneWeekend.htm  
__host__ __device__
float reflectance(float cosine, float ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
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
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 n_pathDirection = glm::normalize(pathSegment.ray.direction);
    glm::vec3 n_normal = glm::normalize(normal);

    // If the material indicates that the object was a light, "light" the ray
    if (m.emittance > 0.0f) {
        pathSegment.color *= (m.color * m.emittance);
        pathSegment.remainingBounces = 0;
    }
    else {
        // find the new direction of the ray based on material (BSDF)
        pathSegment.ray.origin = intersect;

        bool divideColorInHalf = false;
        if (m.hasReflective > 0.0)
        {
            // reflective
            glm::vec3 diffuse = calculateRandomDirectionInHemisphere(normal, rng);
            glm::vec3 reflect = glm::reflect(n_pathDirection,
                n_normal);
            float result = u01(rng);

            pathSegment.ray.direction = diffuse;
            pathSegment.ray.direction = (result > 0.5f) ? diffuse : reflect;
            divideColorInHalf = true;
        }
        else if (m.hasRefractive > 0.0)
        {
            // refractive
            // adapted from https://raytracing.github.io/books/RayTracingInOneWeekend.htm 

            // Flip the refraction ratio depending if we are entering the object or leaving it
            float refractionRatio = !pathSegment.insideObject ? (1.0 / m.indexOfRefraction) : m.indexOfRefraction; 

            // Determine if we should reflect or refract
            float cosTheta = min(glm::dot(-n_pathDirection, n_normal), 1.f);
            float sinTheta = sqrt(1.f - cosTheta * cosTheta);
            bool cannotRefract = refractionRatio * sinTheta > 1.f;

            // Schlick's Approximation
            bool reflect = reflectance(cosTheta, refractionRatio) > u01(rng);

            glm::vec3 direction;
            if (cannotRefract || reflect)
            {
                direction = glm::reflect(n_pathDirection, n_normal);
            }
            else
            {
                direction = glm::refract(n_pathDirection, n_normal, refractionRatio);
            }

            pathSegment.ray.direction = direction;
            pathSegment.ray.origin += direction * 0.01f;
        }
        else
        {
            // diffuse
            pathSegment.ray.direction =
                calculateRandomDirectionInHemisphere(normal, rng); // TODO: improve basic implementation
        }

        pathSegment.color *= divideColorInHalf ? (m.color / 2.f) : m.color;
        pathSegment.remainingBounces--;
        if (pathSegment.remainingBounces == 0)
            pathSegment.color = glm::vec3(0.0f); // if didn't reach light, terminate
    }
}

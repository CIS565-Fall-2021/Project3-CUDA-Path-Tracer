#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
    glm::vec3
    calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng));      // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal + cos(around) * over * perpendicularDirection1 + sin(around) * over * perpendicularDirection2;
}

#define SMALL_OFFSET 0.001f
#define OFFSET_VECTOR(newDir) SMALL_OFFSET *newDir

__host__ __device__ float reflectance(float cosine, float ior)
{
    float r0 = (1 - ior) / (1 + ior);
    r0 *= r0;
    float tmp = 1 - cosine;
    float tmp5 = tmp * tmp * tmp * tmp * tmp;
    return r0 + (1 - r0) * tmp5;
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
__host__ __device__ void scatterRay(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // TODO: split probability

    glm::vec3 cacheDir = glm::normalize(pathSegment.ray.direction);
    glm::vec3 colorAcc(0.f);
    glm::vec3 origAcc(0.f);
    glm::vec3 dirAcc(0.f);
    thrust::uniform_real_distribution<float> u01(0, 1);
    // This lets us probabilistically choose refractive, reflective, and diffuse
    float randFloat(u01(rng));

    // if (m.hasRefractive > 0.f) // will bend light but this also applies glass stuff
    if (randFloat < m.hasRefractive)
    {
        // TODO: When to normalize

        float dProd = glm::dot(cacheDir, normal);
        bool leaving = dProd > 0.f;
        float negateOnLeave = leaving ? -1.f : 1.f;
        float eta = !leaving ? (1.f / m.indexOfRefraction) : m.indexOfRefraction;

        float cosTheta = min(dProd * -1.f, 1.f);
        float sinTheta = sqrt(1.f - cosTheta * cosTheta);
        bool willReflect = (eta * sinTheta > 1.f) || reflectance(cosTheta, eta) > u01(rng);
        glm::vec3 offsetOrigin = intersect + OFFSET_VECTOR(normal * negateOnLeave * (willReflect ? 1.f : -1.f));
        glm::vec3 refractedDir = glm::normalize(
            willReflect
                ? glm::reflect(cacheDir, normal * negateOnLeave)
                : glm::refract(cacheDir, normal * negateOnLeave, eta));
        origAcc = offsetOrigin;
        dirAcc = refractedDir;
        colorAcc = willReflect ? m.specular.color : glm::vec3(1.f);
    }
    // else if (m.hasReflective > 0.f) // Shiny
    else if (randFloat < m.hasReflective)
    {
        colorAcc = m.specular.color;
        origAcc = intersect + OFFSET_VECTOR(normal);
        dirAcc = glm::reflect(cacheDir, normal);
    }
    else // Else lambort
    {
        colorAcc = m.color;
        origAcc = intersect + OFFSET_VECTOR(normal);
        dirAcc = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    }

    // Set the color
    pathSegment.color *= colorAcc;
    pathSegment.ray.origin = origAcc;
    pathSegment.ray.direction = dirAcc;
}

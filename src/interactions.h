#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine &rng)
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

__host__ __device__ __forceinline__ float reflectance(float cosine, float ior)
{
    float r0 = (1.f - ior) / (1.f + ior);
    r0 *= r0;
    float tmp = 1.f - cosine;
    float tmp5 = tmp * tmp * tmp * tmp * tmp;
    return r0 + (1.f - r0) * tmp5;
}

__host__ __device__ __forceinline__ void dielectricScatter(glm::vec3 &color, glm::vec3 &orig, glm::vec3 &direc, glm::vec3 intersect, glm::vec3 normal, Material const &m, glm::vec3 rayDir, float randomFloat)
{
    float dProd = glm::dot(rayDir, normal);
    bool leaving = dProd > 0.f;
    float negateOnLeave = leaving ? -1.f : 1.f;
    float eta = !leaving ? (1.f / m.indexOfRefraction) : m.indexOfRefraction;

    float cosTheta = min(dProd * -1.f, 1.f);
    float sinTheta = sqrt(1.f - cosTheta * cosTheta);
    bool willReflect = (eta * sinTheta > 1.f) || reflectance(cosTheta, eta) > randomFloat;
    orig = intersect + OFFSET_VECTOR(normal * negateOnLeave * (willReflect ? 1.f : -1.f));
    direc = glm::normalize(
        willReflect
            ? glm::reflect(rayDir, normal * negateOnLeave)
            : glm::refract(rayDir, normal * negateOnLeave, eta));
    color *= willReflect ? m.specular.color : glm::vec3(1.f);
}

__host__ __device__ __forceinline__ void specularScatter(glm::vec3 &color, glm::vec3 &orig, glm::vec3 &direc, glm::vec3 intersect, glm::vec3 normal, Material const &m, glm::vec3 rayDir)
{
    color *= m.specular.color;
    orig = intersect + OFFSET_VECTOR(normal);
    direc = glm::normalize(glm::reflect(rayDir, normal));
}

__host__ __device__ __forceinline__ void diffuseScatter(glm::vec3 &color, glm::vec3 &orig, glm::vec3 &direc, glm::vec3 intersect, glm::vec3 normal, Material const &m, thrust::default_random_engine &rng)
{
    color *= m.color;
    orig = intersect + OFFSET_VECTOR(normal);
    direc = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
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
__host__ __device__ void scatterRay(PathSegment &pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material &m, thrust::default_random_engine &rng)
{
    glm::vec3 cacheDir = glm::normalize(pathSegment.ray.direction);
    thrust::uniform_real_distribution<float> u01(0, 1);
    // This lets us probabilistically choose refractive, reflective, and diffuse
    float randFloat = (u01(rng));
    float randomFloat = (u01(rng));

    // if (m.hasRefractive > 0.f) // will bend light but this also applies glass stuff
    if (randFloat < m.hasRefractive)
    {
        dielectricScatter(pathSegment.color, pathSegment.ray.origin, pathSegment.ray.direction, intersect, normal, m, cacheDir, randomFloat);
    }
    // else if (m.hasReflective > 0.f) // Shiny
    else if (randFloat < m.hasReflective)
    {
        specularScatter(pathSegment.color, pathSegment.ray.origin, pathSegment.ray.direction, intersect, normal, m, cacheDir);
    }
    else // Else lambort
    {
        diffuseScatter(pathSegment.color, pathSegment.ray.origin, pathSegment.ray.direction, intersect, normal, m, rng);
    }
}

#pragma once

#include "intersections.h"

#define STRATIFIED_SAMPLING

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

__host__ __device__
    glm::vec3
    calculateStratifiedDirectionInHemisphere(glm::vec3 normal, thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec2 samples = glm::vec2(u01(rng), u01(rng));

    // map square to concentric disc using Peter Shirley's method
    // https://pdfslide.net/documents/a-low-distortion-map-between-disk-and-square.html

    float phi, r, u, v;

    // remap from -1 to 1
    float a = 2 * samples.x - 1;
    float b = 2 * samples.y - 1;

    if (a > -b)
    {
        if (a > b)
        {
            r = a;
            phi = (PI / 4.f) * (b / a);
        }
        else
        {
            r = b;
            phi = (PI / 4.f) * (2 - (a / b));
        }
    }
    else
    {
        if (a < b)
        {
            r = -a;
            phi = (PI / 4) * (4 + (b / a));
        }
        else
        {
            r = -b;
            if (b != 0)
            {
                phi = (PI / 4) * (6 - (a / b));
            }
            else
            {
                phi = 0;
            }
        }
    }

    u = r * glm::cos(phi);
    v = r * glm::sin(phi);

    float w = glm::sqrt(1 - u * u - v * v);
    glm::vec3 dir(u, v, w);

    // convert direction from tangent to world space
    glm::vec3 tangent, bitangent;
    if (std::abs(normal.x) > std::abs(normal.y))
    {

        tangent = glm::vec3(-normal.z, 0, normal.x) / std::sqrt(normal.x * normal.x + normal.z * normal.z);
    }
    else
    {
        tangent = glm::vec3(0, normal.z, -normal.y) / std::sqrt(normal.y * normal.y + normal.z * normal.z);
    }
    bitangent = glm::cross(normal, tangent);
    glm::mat3 tangentToWorld;
    for (int i = 0; i < 3; i++) {
        tangentToWorld[0][i] = tangent[i];
        tangentToWorld[1][i] = bitangent[i];
        tangentToWorld[2][i] = normal[i];
    }

    return glm::normalize(tangentToWorld * dir);
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
    thrust::uniform_real_distribution<float> u01(0, 1);
    float sample = u01(rng);
    if (m.hasReflective > sample)
    {
        // reflective
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect + 0.001f * normal;
        pathSegment.color *= m.specular.color;
    }
    else if (m.hasRefractive > sample)
    {
        // refractive
        float cosTheta = glm::dot(normal, -pathSegment.ray.direction);
        float sinTheta = glm::sqrt(1.0f - cosTheta * cosTheta);
        bool entering = cosTheta > 0;
        float eta = entering ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;

        glm::vec3 dir = glm::refract(pathSegment.ray.direction, normal, eta);
        glm::vec3 ori = intersect + 0.001f * dir;
        // cannot refract
        if (sinTheta * eta > 1.0f)
        {
            dir = glm::reflect(pathSegment.ray.direction, normal);
            ori = intersect + 0.001f * normal;
        }

        pathSegment.ray.direction = dir;
        pathSegment.ray.origin = ori;
        pathSegment.color *= m.specular.color;
    }
    else
    {
        // diffuse

#ifdef STRATIFIED_SAMPLING
        pathSegment.ray.direction = calculateStratifiedDirectionInHemisphere(normal, rng);
#else
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
#endif
        //     dir = glm::dot(dir, normal) > 0 ? dir : -dir;
        //    pathSegment.ray.direction = dir;
        pathSegment.ray.origin = intersect + 0.001f * normal;
        pathSegment.color *= m.color;
    }
}

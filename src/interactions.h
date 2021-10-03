#pragma once

#include "intersections.h"
#include <thrust/random.h>

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

__device__ glm::vec3 random_spherical(thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(-1, 1);
    thrust::uniform_real_distribution<float> u03(-1, 1);
    thrust::uniform_real_distribution<float> u02(-1, 1);
    glm::vec3 random_vec = glm::normalize(glm::vec3(u01(rng), u02(rng), u03(rng)));
    return random_vec;
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
        thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float random_num = u01(rng);
    if (m.hasReflective == 0 && m.hasRefractive == 0) {
        // my implementation
        //glm::vec3 random_vec = random_spherical(rng);
        //pathSegments.ray.direction = glm::normalize(normal + random_vec);
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.ray.origin = intersect + EPSILON * normal;
        pathSegment.color *= m.color;
        pathSegment.color = glm::clamp(pathSegment.color, glm::vec3(0.0f), glm::vec3(1.0f));
        
    }
    else if (m.hasReflective > random_num){
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect + EPSILON * normal;
        pathSegment.color *= m.specular.color;
        pathSegment.color = glm::clamp(pathSegment.color, glm::vec3(0.0f), glm::vec3(1.0f));
    }
    else if (m.hasRefractive > random_num) {
        glm::vec3 ray_direction = glm::normalize(pathSegment.ray.direction);
        bool from_inside = glm::dot(ray_direction, normal) > 0.0f;
        glm::vec3 refract_ray_direction;
        float refraction_ratio = m.indexOfRefraction;
        if (from_inside) {
            refract_ray_direction = glm::normalize(glm::refract(pathSegment.ray.direction, glm::normalize(-1.0f * normal), m.indexOfRefraction));
        }
        else {
            float refraction_ratio = 1.0f / m.indexOfRefraction;
            refract_ray_direction = glm::normalize(glm::refract(pathSegment.ray.direction, normal, 1.0f / m.indexOfRefraction));
        }
        float cos_theta = fmin(glm::dot(ray_direction, normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
        bool cannot_refract = (refraction_ratio * sin_theta) > 1.0;

        if (cannot_refract) {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.origin = intersect + EPSILON * normal;;
            pathSegment.color *= m.specular.color;
        }
        else {
            pathSegment.ray.direction = refract_ray_direction;
            pathSegment.ray.origin = intersect + 0.001f * refract_ray_direction;
            pathSegment.color *= m.specular.color;
        }
        
    }
    pathSegment.remainingBounces--;
    pathSegment.color = glm::clamp(pathSegment.color, glm::vec3(0.0f), glm::vec3(1.0f));
    
    
    
    
}

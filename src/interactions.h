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

// Calculate the direction reflected by a mirror-like surface
__host__ __device__
glm::vec3 calculate_reflection_direction(glm::vec3 normal, glm::vec3 incident) {
    return glm::reflect(incident, normal);
}

// Calculate the transmission direction based on Snell's Law
__host__ __device__
glm::vec3 calculate_refraction_direction(glm::vec3 normal, glm::vec3 incident, float index) {
    return glm::refract(incident, normal, 1.f / index);
}

// Calculate the direction mixed of reflection and transmission
// The ratio between these two depends on Schlick's approximation
__host__ __device__
glm::vec3 calculate_reflection_and_refraction_direction(glm::vec3 normal, glm::vec3 incident, float index, thrust::default_random_engine &rng, bool* is_refract) {
    // Determine whether the light goes from material to air
    if (glm::dot(normal, incident) > 0) {
        normal = -normal;
        index = 1.f / index;
    }

    // Decide reflection or transmission
    thrust::uniform_real_distribution<float> u01(0, 1);
    float R_0 = powf((1.f - index) / (1.f + index), 2.f);
    float ref_coeff = R_0 + (1 - R_0) * powf(1.f + glm::dot(normal, incident), 5.f); 

    glm::vec3 emergent;
    if (u01(rng) < ref_coeff) {
        emergent = calculate_reflection_direction(normal, incident);
    }
    else {
        emergent = calculate_refraction_direction(normal, incident, index);
        *is_refract = true;

        // Totally reflection over critical angle
        if (glm::length(emergent) == 0.f) {
            emergent = calculate_reflection_direction(normal, incident);
            *is_refract = false;
        }
    }

    return emergent;
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
        glm::vec3 texture,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // Determine whether refraction orrurs
    bool is_refraction = false;

    if (pathSegment.remainingBounces > 0) {
        glm::vec3 bounce_ray;

        // Both refractive and reflective
        if (m.hasRefractive) {
            // Shade rays
            pathSegment.color *= m.specular.color;
            // Shoot new ray
            bounce_ray = calculate_reflection_and_refraction_direction(normal, pathSegment.ray.direction, m.indexOfRefraction, rng, &is_refraction);
        }
        // Perfect specular
        else if (m.hasReflective) {
            // Shade rays
            pathSegment.color *= m.specular.color;
            // Shoot new ray
            bounce_ray = calculate_reflection_direction(normal, pathSegment.ray.direction);
        }
        // Ideal diffuse
        else {
            if (texture[0] < 0.f) {
                // Shade rays
                pathSegment.color *= m.color;
            }
            else {
                // Use texture if exists
                pathSegment.color *= texture;
            }

            // Use this to test normal
            //pathSegment.color = (normal + 1.f) / 2.f;
            
            // Shoot new ray
            bounce_ray = calculateRandomDirectionInHemisphere(normal, rng);
        }
        
        // Add small offset to avoid hitting intersection
        bounce_ray = glm::normalize(bounce_ray);
        if (is_refraction) {
            pathSegment.ray.origin = intersect + .0002f * pathSegment.ray.direction;
        }
        else {
            pathSegment.ray.origin = intersect;
        } 
        pathSegment.ray.direction = bounce_ray;

        pathSegment.remainingBounces--;
    }
}

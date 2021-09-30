#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__device__
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


__device__ float reflectance(double cosine, double ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__
glm::vec3 DielectricScatter(
    PathSegment& pathSegment, glm::vec3 intersect, glm::vec3 normal, const Material& m, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    bool  front_face = glm::dot(pathSegment.ray.direction, normal);
    double refraction_ratio = front_face ? (1.0 / m.hasRefractive) : m.hasRefractive;

    glm::vec3 unit_direction = glm::normalize(pathSegment.ray.direction);
    double cos_theta = fmin(glm::dot(-unit_direction, normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    glm::vec3 direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > u01(rng))
        direction = glm::reflect(unit_direction, normal);
    else
        direction = glm::refract(unit_direction, normal, u01(rng));

    return direction;
}

inline  bool near_zero(glm::vec3 vec) {
    // Return true if the vector is close to zero in all dimensions.
    const auto s = 1e-8;
    return (fabs(vec[0]) < EPSILON) && (fabs(vec[1]) < EPSILON) && (fabs(vec[2]) < EPSILON);
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
__device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 scatter_direction;

    if (m.hasReflective > 0)
    {
        scatter_direction = glm::reflect(pathSegment.ray.direction, normal);
    }
    else if (m.hasRefractive > 0)
    {
        scatter_direction = DielectricScatter(pathSegment, intersect, normal, m, rng);
    }
    else
    {
        scatter_direction = DielectricScatter(pathSegment, intersect, normal, m, rng);
        //scatter_direction = calculateRandomDirectionInHemisphere(normal, rng);
    }

    /*if (near_zero(scatter_direction))
        scatter_direction = normal;*/
    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = scatter_direction;

}

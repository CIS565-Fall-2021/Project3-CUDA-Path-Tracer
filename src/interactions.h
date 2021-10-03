#pragma once

#include <thrust/random.h>

#include "intersections.h"
#include "static_config.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    const glm::vec3 normal, thrust::default_random_engine &rng) {
  thrust::uniform_real_distribution<float> u01(0, 1);

  float up     = sqrt(u01(rng));     // cos(theta)
  float over   = sqrt(1 - up * up);  // sin(theta)
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

  return up * normal + cos(around) * over * perpendicularDirection1 +
         sin(around) * over * perpendicularDirection2;
}

/**
 * @brief Use Schlick's approximation to calculate Fresnel factor based on given
 * incident angle & index of refraction. Assumes ray always shoots from air to
 * material (n_1 = 1).
 *
 * Reference: https://en.wikipedia.org/wiki/Schlick's_approximation
 *
 * @param cos_theta:    cos(theta), theta is incident angle
 * @param idx_refract:  index refraction of material
 * @return float
 */
__host__ __device__ float schlicks(const float cos_theta,
                                   const float idx_refract) {
  float r0 = powf((1.0f - idx_refract) / (1.0f + idx_refract), 2.0f);
  return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
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
 *   - This way is inefficient, but serves as a good starting point --- it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * You may need to change the parameter list for your purposes!
 *
 * @return  This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 */
__host__ __device__ void scatterRay(PathSegment &pathSegment,
                                    const glm::vec3 intersect,
                                    const glm::vec3 normal, const Material m,
                                    thrust::default_random_engine &rng) {
  if (m.emittance > 0.0f) {
    pathSegment.color *= (m.color * m.emittance);
    pathSegment.remainingBounces = 0;
  } else {
    thrust::uniform_real_distribution<float> u01(0, 1);
    const float random_sample = u01(rng);

    // check if the current ray is in material
    glm::vec3 in_direction = glm::normalize(pathSegment.ray.direction);
    bool is_inMaterial     = glm::dot(in_direction, normal) > 0.0f;
    glm::vec3 true_normal  = (is_inMaterial) ? -1.0f * normal : normal;

    if (random_sample < m.hasRefractive) {
      float idx_refraction_ratio =
          (is_inMaterial) ? m.indexOfRefraction : (1.0f / m.indexOfRefraction);

      // get the incident angle
      float cos_angle = glm::abs(glm::dot(in_direction, normal));
      // calculate Fresnel factor
      float fresnel_factor = schlicks(cos_angle, m.indexOfRefraction);
      // As refraction is determined by both reflection & transmission,
      // probabilistically determine whether the next ray is reflective ray or
      // transmission ray
      glm::vec3 out_refracted_dir = glm::refract(
          in_direction, glm::normalize(true_normal), idx_refraction_ratio);
      const float prob = u01(rng);
      if (prob < fresnel_factor || glm::length(out_refracted_dir) < EPS) {
        pathSegment.ray.direction = glm::reflect(in_direction, true_normal);
        pathSegment.ray.origin    = intersect + EPS * true_normal;
      } else {
        pathSegment.ray.direction = out_refracted_dir;
        pathSegment.ray.origin    = intersect - EPS * true_normal;
      }
      pathSegment.color *= m.specular.color;
    } else if (random_sample < m.hasReflective) {
      pathSegment.ray.direction = glm::reflect(in_direction, true_normal);
      pathSegment.color *= m.specular.color;
      pathSegment.ray.origin = intersect + EPS * true_normal;
    } else {
      pathSegment.ray.direction =
          calculateRandomDirectionInHemisphere(true_normal, rng);
      pathSegment.ray.origin = intersect + EPS * true_normal;
    }

    --pathSegment.remainingBounces;
    pathSegment.color *= m.color;
    pathSegment.color =
        glm::clamp(pathSegment.color, glm::vec3(0.f), glm::vec3(1.0f));
  }
}

#pragma once


#include "intersections.h"
#include "cuda_runtime.h"

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
 * Computes the imperfect specular ray direction.
 * Based on: https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
 */
__host__ __device__
glm::vec3 calculateImperfectSpecularDirection(
    glm::vec3 normal, glm::vec3 reflect, glm::vec4 tangent,
    thrust::default_random_engine& rng, 
    float roughness, float shininess) {

    thrust::uniform_real_distribution<float> u01(0, roughness);
    float x1 = 1 + u01(rng);
    float x2 = u01(rng);
  
    float theta = acos(1 / powf(x1, shininess + 1));
    float phi = 2 * PI * x2;
  
    glm::vec3 dir;
    dir.x = cos(phi) * sin(theta);
    dir.y = sin(phi) * sin(theta);
    dir.z = cos(theta);
    
    // Transform dir from specular-space to tangent-space
    float c = glm::dot(normal, reflect);
    float s = glm::length(glm::cross(normal, reflect));
    dir = glm::mat3(c, 0.0f, s, 0.0f, 1.0f, 0.0f, -s, 0.0f, c) * dir;

    // Transform dir from tangent-space to world-space
    glm::vec3 t(tangent);
    glm::vec3 b = glm::cross(normal, t) * tangent.w;
    dir = t * dir.x + b * dir.y + normal * dir.z;

    return dir;
}

__device__
void sampleTexture(Color& color, cudaTextureObject_t texObj, const glm::vec2 uv) {
  // NOTE: cudaReadModeNormalizedFloat will convert uchar4 to float4
  float4 rgba = tex2D<float4>(texObj, uv.x, uv.y);
  color = Color(rgba.x, rgba.y, rgba.z);
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
    const ShadeableIntersection& i,
    const Material& m,
    cudaTextureObject_t* textures,
    thrust::default_random_engine &rng) {
    
    glm::vec3 intersect = getPointOnRay(pathSegment.ray, i.t);

    Color color;
    glm::vec3 newDir;
    int txId = -1;

    txId = m.pbrMetallicRoughness.baseColorTexture.index;
    if (txId < 0) {
      color = m.pbrMetallicRoughness.baseColorFactor;
    }
    else {
      sampleTexture(color, textures[txId], i.uv);
    }

    float pM, pR;  // metallic and roughness parameters

    txId = m.pbrMetallicRoughness.metallicRoughnessTexture.index;
    if (txId < 0) {
      pM = m.pbrMetallicRoughness.metallicFactor;
      pR = m.pbrMetallicRoughness.roughnessFactor;
    }
    else {
      Color pbr;
      sampleTexture(pbr, textures[txId], i.uv);
      pM = pbr.b * m.pbrMetallicRoughness.metallicFactor;
      pR = pbr.g * m.pbrMetallicRoughness.roughnessFactor;
    }

    thrust::uniform_real_distribution<float> u01(0, 1);

    if (u01(rng) < pM) {
      // Specular
      glm::vec3 reflect = glm::reflect(pathSegment.ray.direction, i.surfaceNormal);
      newDir = calculateImperfectSpecularDirection(i.surfaceNormal, reflect, i.tangent, rng, pR, m.specular.exponent);
      color *= pM;
    }
    else {
      // Diffuse
      newDir = calculateRandomDirectionInHemisphere(i.surfaceNormal, rng);
      color *= (1.0f - pM);
    }

    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = glm::normalize(newDir);
    pathSegment.color *= color;
}


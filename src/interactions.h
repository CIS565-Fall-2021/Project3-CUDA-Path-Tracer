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

__host__ __device__ float getFresnelCoefficient(float eta, float cosTheta, float matIOF) {
    // handle total internal reflection
    float sinThetaI = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
    float sinThetaT = eta * sinThetaI;
    float fresnelCoeff = 1.f;

    cosTheta = abs(cosTheta);
    if (sinThetaT < 1) {
        // calc fresnel coefficient
        float R0 = ((1.f - matIOF) / (1.f + matIOF));
        R0 = R0 * R0;

        fresnelCoeff = R0 + (1.f - R0) * (1.f - cosTheta);

        /*float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));

        float rparl = ((m.indexOfRefraction * cosTheta) - (cosThetaT)) / ((m.indexOfRefraction * cosTheta) + (cosThetaT));
        float rperp = ((cosTheta) - (m.indexOfRefraction * cosThetaT)) / ((cosTheta) + (m.indexOfRefraction * cosThetaT));
        fresnelCoeff = (rparl * rparl + rperp * rperp) / 2.0;*/
    }
    return fresnelCoeff;
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

    glm::vec3 newDir;
    // QUESTION: should we worry about this now? cornell.txt doesn't have two materials, right?
    // specular surface
    if (m.hasReflective) {
        newDir = glm::reflect(pathSegment.ray.direction, normal);
    }
    else if (m.hasRefractive) {
        const glm::vec3& wi = pathSegment.ray.direction;

        float cosTheta = dot(normal, wi);

        // incoming direction should be opposite normal direction if entering medium
        bool entering = cosTheta < 0;
        glm::vec3 faceForwardN = !entering ? -normal : normal;

        // if entering, divide air iof (1.0) by the medium's iof
        float eta = entering ? 1.f / m.indexOfRefraction : m.indexOfRefraction;
        float fresnelCoeff = getFresnelCoefficient(eta, cosTheta, m.indexOfRefraction);

        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) < fresnelCoeff) {
            newDir = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            // book refract function
            /*float cosThetaI = dot(faceForwardN, wi);
            float sin2ThetaI = max(0.f, 1.f - cosThetaI * cosThetaI);
            float sin2ThetaT = eta * eta * sin2ThetaI;
            float cosThetaT = sqrt(max(0.f, 1.f - sin2ThetaT));

            newDir = eta * wi + (eta * cosThetaI - cosThetaT) * faceForwardN;*/

            /*if (sin2ThetaT >= 1) {
                newDir = glm::reflect(pathSegment.ray.direction, normal);
            }*/
            newDir = glm::refract(wi, faceForwardN, eta);
        }

        pathSegment.ray.origin = intersect + (entering ? 0.0002f : -0.0002f) * pathSegment.ray.direction;
        pathSegment.ray.direction = newDir;
        return;
        
    }
    // diffuse surface
    else {
        newDir = calculateRandomDirectionInHemisphere(normal, rng);    
    }

    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = newDir;  
}

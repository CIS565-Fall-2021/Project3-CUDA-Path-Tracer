
__host__ __device__ inline
glm::vec3 tangentSpaceToWorldSpace(const glm::vec3& dir, const glm::vec3& normal) {
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

    return dir.z * normal
        + dir.x * perpendicularDirection1
        + dir.y * perpendicularDirection2;
}

__host__ __device__ inline
glm::vec3 calculateUniformRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine &rng, float* pdf = nullptr) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = u01(rng);
    float over = sqrt(1.f - up * up);
    float around = u01(rng) * TWO_PI;
    glm::vec3 dir(cos(around) * over, sin(around) * over, up);
    if (pdf) {
        *pdf = 1.f / TWO_PI;
    }
    return tangentSpaceToWorldSpace(dir, normal);
}

__host__ __device__ inline
glm::vec3 calculateCosWeightedRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine &rng, float* pdf = nullptr) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1.f - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;
    glm::vec3 dir(cos(around) * over, sin(around) * over, up);
    if (pdf) {
        *pdf = up / PI;
    }
    return tangentSpaceToWorldSpace(dir, normal);
}

__host__ __device__ inline
glm::vec3 calculateCosWeightedRandomDirectionInPhongSpecularRegion(
    glm::vec3 normal, thrust::default_random_engine &rng, float specex = 0.f, float* pdf = nullptr) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Reference: http://vclab.kaist.ac.kr/cs580/slide16-Path_tracing(2).pdf
    float up = specex <= 0.f ? 1.f : powf(u01(rng), 1.f / (specex + 1.f)); // cos(alpha)
    float over = sqrt(1.f - up * up); // sin(alpha)
    float around = u01(rng) * TWO_PI;
    glm::vec3 dir(cos(around) * over, sin(around) * over, up);
    if (pdf) {
        *pdf = specex <= 0.f ? 1.f : (specex + 1) * powf(up, specex) / TWO_PI;
    }
    return tangentSpaceToWorldSpace(dir, normal);
}


#if BUILD_BVH_FOR_TRIMESH
template<>
GLM_FUNC_QUALIFIER void BoundingVolumeHierarchy<TriMesh>::buildBVH(TriMesh* dev_trimesh) {

}
#endif // BUILD_BVH_FOR_TRIMESH

template<>
GLM_FUNC_QUALIFIER BBox BBox::getLocalBoundingBox(const Triangle& geom) {
    return BBox{
        glm::min(geom.pos[0], glm::min(geom.pos[1], geom.pos[2])),
        glm::max(geom.pos[0], glm::max(geom.pos[1], geom.pos[2])),
        1
    };
}


template<typename TVec>
GLM_FUNC_QUALIFIER TVec barycentricInterpolation(const glm::vec3& bary, const TVec& v0, const TVec& v1, const TVec& v2) {
    return v0 * bary.x + v1 * bary.y + v2 * bary.z;
}

GLM_FUNC_QUALIFIER glm::vec4 getBarycentric(const glm::vec2& p, const glm::vec2& v0, const glm::vec2& v1, const glm::vec2& v2) {
    glm::vec2 e0 = p - v0, e1 = p - v1, e2 = p - v2;
    float alpha = e0.x * e1.y - e0.y * e1.x, beta = e1.x * e2.y - e1.y * e2.x, gamma = e2.x * e0.y - e2.y * e0.x;
    float sum = fabs(alpha + beta + gamma);
    if (sum < FLT_EPSILON) {
        return glm::vec4(1.f / 3.f, 1.f / 3.f, 1.f / 3.f, 0.f);
    }
    alpha /= sum;
    beta /= sum;
    gamma /= sum;
    return glm::vec4(alpha, beta, gamma, 1.f);
}

GLM_FUNC_QUALIFIER glm::vec4 getBarycentric(const glm::vec3& p, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) {
    glm::vec3 e0 = p - v0, e1 = p - v1, e2 = p - v2;
    glm::vec3 posAxis = glm::normalize(glm::cross(v1 - v0, v2 - v0));
    glm::vec3 c01 = glm::cross(e0, e1), c12 = glm::cross(e1, e2), c20 = glm::cross(e2, e0);
    float alpha = glm::dot(c12, posAxis), beta = glm::dot(c20, posAxis), gamma = glm::dot(c01, posAxis);
    float sum = fabs(alpha + beta + gamma);
    if (sum < FLT_EPSILON) {
        return glm::vec4(1.f / 3.f, 1.f / 3.f, 1.f / 3.f, 0.f);
    }
    alpha /= sum;
    beta /= sum;
    gamma /= sum;
    return glm::vec4(alpha, beta, gamma, 1.f);
}

GLM_FUNC_QUALIFIER float TriMesh::localIntersectionTest(Ray q, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal, int& triangleId) {
#if BUILD_BVH_FOR_TRIMESH

#else // BUILD_BVH_FOR_TRIMESH
    float tnear = -1.f;
    for (int i = 0; i < triangleNum; ++i) {
        tnear = triangles[i].triangleLocalIntersectionTest(q, intersectionPoint, intersectionBarycentric, normal);
        if (tnear > 0.f) {
            triangleId = i;
            return tnear;
        }
    }
    return -1.f;
#endif // BUILD_BVH_FOR_TRIMESH
}

GLM_FUNC_QUALIFIER float Triangle::triangleLocalIntersectionTest(Ray q, glm::vec3& intersectionPoint, glm::vec3& intersectionBarycentric, glm::vec3& normal) {
    glm::vec3 e1 = pos1 - pos0, e2 = pos2 - pos0;
    glm::vec3 s = q.origin - pos0;
    glm::vec3 s1 = glm::cross(q.direction, e2), s2 = glm::cross(s, e1);

    float s1_dot_e1 = glm::dot(s1, e1);
    float s2_dot_e2 = glm::dot(s2, e2);
    float s1_dot_s = glm::dot(s1, s);
    float s2_dot_dir = glm::dot(s2, q.direction);

    if (fabs(s2_dot_dir) < EPSILON) {
        return -1.f;
    }

    if (!twoSided && s2_dot_dir < 0.f) {
        return -1.f;
    }

    float tnear = s2_dot_e2 / s1_dot_e1;
    float u = s1_dot_s / s1_dot_e1;
    float v = s2_dot_dir / s1_dot_e1;
    float w = 1.f - u - v;

    if (tnear < EPSILON || u < 0.f || v < 0.f || w < 0.f) {
        return -1.f;
    }

    intersectionBarycentric.x = u;
    intersectionBarycentric.y = v;
    intersectionBarycentric.z = w;

    intersectionPoint = barycentricInterpolation(intersectionBarycentric, pos0, pos1, pos2);
    return tnear;
}


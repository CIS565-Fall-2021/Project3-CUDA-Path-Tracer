#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// Determine whther a ray intersects the bounding box of a mesh
// Similar to boxIntersectionTest
__host__ __device__ float bound_box_intersection_test(AABB bound_box, Ray r,
    glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q = r;

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (bound_box.bottom_left[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (bound_box.upper_right[xyz] - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }

        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// Get bilinear interpolated value on grid according to uv coordinates
__host__ __device__ glm::vec3 bilinear_interpolation(glm::vec2 uv_coords, glm::ivec2 grid_shape, glm::vec3 *unrolled_grid) {
    float u = uv_coords.x * (grid_shape[0] - 1);
    float v = (1 - uv_coords.y) * (grid_shape[1] - 1);
    int u_int = int(u);
    int v_int = int(v);
    int u_ceil = glm::min(u_int + 1, grid_shape[0] - 1);
    int v_ceil = glm::min(v_int + 1, grid_shape[1] - 1);
    float u_fract = u - u_int;
    float v_fract = v - v_int;

    glm::vec3 interp_x1 = (1 - u_fract) * unrolled_grid[u_int + v_int * grid_shape[0]] + u_fract * unrolled_grid[u_ceil + v_int * grid_shape[0]];
    glm::vec3 interp_x2 = (1 - u_fract) * unrolled_grid[u_int + v_ceil * grid_shape[0]] + u_fract * unrolled_grid[u_ceil + v_ceil * grid_shape[0]];

    return (1 - v_fract) * interp_x1 + v_fract * interp_x2;
}

// Determine whther a ray intersects a mesh by naively checking all triangles of it
__host__ __device__ float mesh_triangle_intersection_test(Geom mesh, Ray r, glm::vec3 &intersectionPoint, 
    glm::vec3 &texture, glm::vec3 &normal, bool &outside, Triangle *meshes, glm::vec3 *textures, glm::vec3 *normals, bool use_bound_box) {

    // Convert to unit coords
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    // Check intersection with bounding box first if toggled
    if (use_bound_box) {
        float bound_test = bound_box_intersection_test(mesh.bound_box, rt, intersectionPoint, normal, outside);
        if (bound_test < 0.f) {
            return bound_test;
        }
    }

    // Find first intersection with all triangle meshes
    glm::vec3 barycentric_t;
    glm::vec3 norm;
    float t_min = FLT_MAX;
    int res = -1;
    Triangle target_tri;
    glm::vec3 target_barycentric_t;

    for (int i = 0; i < mesh.mesh_num; i++) {
        Triangle test_tri = meshes[i + mesh.mesh_offset];
        res = glm::intersectRayTriangle(rt.origin, rt.direction, test_tri.pos[0], test_tri.pos[1], test_tri.pos[2], barycentric_t);
        if (res) {
            // barycentric_t = (u, v, t)
            float t = barycentric_t[2];
            if (t_min > t)
            {
                // Update closest intersection
                t_min = t;
                target_tri = test_tri;
                target_barycentric_t = barycentric_t;               
            }
        }   
    }

    // No intersection
    if (res < 0.f) {
        return res;
    }

    // Barycentric interpolation for normal
    norm = glm::normalize(target_barycentric_t[0] * target_tri.normal[1] + target_barycentric_t[1] * target_tri.normal[2] + (1 - target_barycentric_t[0] - target_barycentric_t[1]) * target_tri.normal[0]);

    // Use face normal to improve speed but lose fidelity
    //norm = glm::normalize((test_tri.normal[1] + test_tri.normal[2] + test_tri.normal[0]) / 3.f);

    texture = glm::vec3(-1.f, -1.f, -1.f);

    if (mesh.texture_offset >= 0) {
        glm::vec2 uv = target_barycentric_t[0] * target_tri.uv[1] + target_barycentric_t[1] * target_tri.uv[2] + (1 - target_barycentric_t[0] - target_barycentric_t[1]) * target_tri.uv[0];
        texture = bilinear_interpolation(uv, mesh.texture_shape, textures + mesh.texture_offset);
    }

    if (mesh.normal_offset >= 0) {
        glm::vec2 uv = target_barycentric_t[0] * target_tri.uv[1] + target_barycentric_t[1] * target_tri.uv[2] + (1 - target_barycentric_t[0] - target_barycentric_t[1]) * target_tri.uv[0];
        norm = bilinear_interpolation(uv, mesh.normal_shape, normals + mesh.normal_offset);
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t_min);

    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(norm, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

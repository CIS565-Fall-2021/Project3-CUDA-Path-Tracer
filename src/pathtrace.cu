#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "math_constants.h"

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define SORT_MATERIAL 1
#define CACHE_INTERSECTION 0
#define DEPTH_OF_FIELD 1
#define MESH_BOUND_CHECK 1
#define ANTI_ALIASING 0

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_cache_intersections = NULL;
//Triangles for testing arbitrary obj file intersection
static TriangleGeom* dev_triangles = NULL;

void pathtraceInit(Scene* scene) {
    hst_scene = scene;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(TriangleGeom));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(TriangleGeom), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_cache_intersections);
    cudaFree(dev_triangles);
    checkCUDAError("pathtraceFree");
}



__host__ __device__ glm::vec2 concentricSampleDisk(thrust::default_random_engine& rng) {

    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec2 u = glm::vec2(u01(rng), u01(rng));
    glm::vec2 uOffset = 2.f * u - glm::vec2(1.f, 1.f);

    if (uOffset.x == 0 && uOffset.y == 0) {
        return glm::vec2(0.f, 0.f);
    }
    float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = CUDART_PIO4_F * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = CUDART_PIO2_F - CUDART_PIO4 * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(std::cos(theta), std::sin(theta));
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);

#if ANTI_ALIASING
        thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
        );
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

#if DEPTH_OF_FIELD
        cam.focalDistance = 5.f;
        cam.lensRadius = 0.2f;

        if (cam.lensRadius > 0) {
            glm::vec2 randomSample{ 0 };
            glm::vec2 pLens = cam.lensRadius * concentricSampleDisk(rng);

            float ft = glm::abs((cam.focalDistance) / segment.ray.direction.z);
            glm::vec3 pFocus = ft * segment.ray.direction;

            segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0.f);
            segment.ray.direction = glm::normalize(pFocus - glm::vec3(pLens.x, pLens.y, 0.f));
        }
#endif

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__host__ __device__ bool checkWithinMeshBoundingBox(Geom& object, 
    glm::vec3 min, glm::vec3 max, Ray& r) {
    //Basically the same thing as boxIntersectionTest
    Ray q;
    q.origin = multiplyMV(object.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(object.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        if (glm::abs(qdxyz) > 0.00001f) {
            float t1 = (min[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (max[xyz] - q.origin[xyz]) / qdxyz;
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
        return true;
    }
    return false;
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment* pathSegments
    , Geom* geoms
    , int geoms_size
    , TriangleGeom* triangles
    , int triangles_size
    , glm::vec3 triangle_bound_min 
    , glm::vec3 triangle_bound_max
    , ShadeableIntersection* intersections
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == CUSTOM_OBJ) {
                //Compute the minimum t (a ray might intersect multiple triangle (front and back) in a custom obj
#if MESH_BOUND_CHECK
                //Only spawn the rays which hits inside the bounding box
                if (checkWithinMeshBoundingBox(geom, triangle_bound_min, triangle_bound_max, pathSegment.ray)) {
                    float closest_dist = FLT_MAX;
                    for (int j = 0; j < triangles_size; j++) {
                        TriangleGeom& triangle = triangles[j];
                        float triangle_inter = triangleIntersectionTest(geom, pathSegment.ray,
                            tmp_intersect, triangle.vertex1, triangle.vertex2, triangle.vertex3,
                            triangle.normal1, triangle.normal2, triangle.normal3, tmp_normal, outside);
                        if (triangle_inter != -1) {
                            closest_dist = glm::min(closest_dist, triangle_inter);
                        }
                    }
                    t = closest_dist;
                }
#else
                float closest_dist = FLT_MAX;
                for (int j = 0; j < triangles_size; j++) {
                    TriangleGeom& triangle = triangles[j];
                    float triangle_inter = triangleIntersectionTest(geom, pathSegment.ray,
                        tmp_intersect, triangle.vertex1, triangle.vertex2, triangle.vertex3,
                        triangle.normal1, triangle.normal2, triangle.normal3, tmp_normal, outside);
                    if (triangle_inter != -1) {
                        closest_dist = glm::min(closest_dist, triangle_inter);
                    }
                }
                t = closest_dist;
#endif
            }
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 
                pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].remainingBounces = 0;
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), 
                    intersection.surfaceNormal, material, rng);
                //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                //pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].remainingBounces = 0;
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

struct needCompact
{
    __host__ __device__ bool operator()(const PathSegment& seg)
    {
        return seg.remainingBounces;
    }
};

struct cmpMaterials
{
    //overwrite sort_by_key compares key objects using operator<.
    __host__ __device__ bool operator()(ShadeableIntersection& intersect1, const ShadeableIntersection& intersect2)
    {
        return intersect1.materialId < intersect2.materialId;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete) {
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
       
#if CACHE_INTERSECTION
        
    if (iter == 1 || depth != 0) {
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        glm::vec3 check = hst_scene->triangle_bound_min;
        glm::vec3 check2 = hst_scene->triangle_bound_max;

        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_triangles
            , hst_scene->triangles.size()
            , hst_scene->triangle_bound_min
            , hst_scene->triangle_bound_max
            , dev_intersections
            );
        checkCUDAError("trace one bounce");

        cudaDeviceSynchronize();
        if (depth == 0) {
            // store first time bounce into cache
            cudaMemcpy(dev_cache_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToHost);
            cudaDeviceSynchronize();
        }
    }
    else {
        cudaMemcpy(dev_intersections, dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToHost);
    }
        
#else 
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_triangles
            , hst_scene->triangles.size()
            , hst_scene->triangle_bound_min
            , hst_scene->triangle_bound_max
            , dev_intersections
            );
        checkCUDAError("trace one bounce");

        cudaDeviceSynchronize();
#endif
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
      // evaluating the BSDF.
      // Start off with just a big kernel that handles all the different
      // materials you have in the scenefile.
      // TODO: compare between directly shading the path segments and shading
      // path segments that have been reshuffled to be contiguous in memory.

#if SORT_MATERIAL
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, cmpMaterials());
#endif

        shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );

        // compaction using thrust::partition
        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, needCompact());
        num_paths = dev_path_end - dev_paths;

        if (num_paths == 0) {
            iterationComplete = true; // TODO: should be based off stream compaction results.
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");

}

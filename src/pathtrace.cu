#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#if ENABLE_ADVANCED_PIPELINE
#define shadePipeline shadeAndScatter
#else // ENABLE_ADVANCED_PIPELINE
#define shadePipeline shadeFakeMaterial
#endif // ENABLE_ADVANCED_PIPELINE

#define DEBUG_COLLISION 0

#if DEBUG_COLLISION
#undef shadePipeline
#define shadePipeline shadeDebugCollision
__global__ void shadeDebugCollision(
    int iter
    , int depth
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
    , int traceDepth
    , int recordDepth
    , Background background
);
#endif // DEBUG_COLLISION

#define ERRORCHECK 1

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
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
    _CrtDbgBreak();
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
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = nullptr;
static glm::vec3 * dev_image = nullptr;
static Geom * dev_geoms = nullptr;
static Material * dev_materials = nullptr;
static PathSegment * dev_paths = nullptr;
static ShadeableIntersection * dev_intersections = nullptr;

// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
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

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    if(dev_image) cudaFree(dev_image);  // no-op if dev_image is null
    if(dev_paths) cudaFree(dev_paths);
    if(dev_geoms) cudaFree(dev_geoms);
    if(dev_materials) cudaFree(dev_materials);
    if(dev_intersections) cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
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
        PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // DONE: implement antialiasing by jittering the ray
        float dx = 0.f, dy = 0.f;
#if JITTER_ANTI_ALIASING
        thrust::random::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::random::uniform_real_distribution<float> ud(-0.5f, 0.5f);
        dx = ud(rng);
        dy = ud(rng);
#else // JITTER_ANTI_ALIASING
#endif // JITTER_ANTI_ALIASING
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + dx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + dy - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// DONE:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment * pathSegments
    , Geom * geoms
    , int geoms_size
    , ShadeableIntersection * intersections
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec3 barycentric;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;
        int triangleId = -1;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        glm::vec3 tmp_barycentric;
        int tmp_triangleId = -1;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom & geom = geoms[i];

            if (geom.type == GeomType::CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == GeomType::SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == GeomType::TRI_MESH)
            {
                t = trimeshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_barycentric, tmp_normal, tmp_triangleId);
            }
            // LOOK: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                barycentric = tmp_barycentric;
                triangleId = tmp_triangleId;
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
            if (geoms[hit_geom_index].type == GeomType::TRI_MESH) {
                Triangle triangle = geoms[hit_geom_index].trimeshRes.triangles[triangleId];
                intersections[path_index].uv = barycentricInterpolation(barycentric, triangle.uv00, triangle.uv01, triangle.uv02);
            }
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
__global__ void shadeFakeMaterial (
    int iter
    , int depth
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    , int traceDepth
    , int recordDepth
    , Background background
    ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces >= 0) {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.getDiffuse(intersection.uv);
            intersection.surfaceNormal = glm::normalize(intersection.surfaceNormal);

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                glm::vec3 emitColor = materialColor * material.emittance;
                if (recordDepth >= 0) {
                    if (traceDepth - pathSegments[idx].remainingBounces < recordDepth + 1) {
                    //if (traceDepth - pathSegments[idx].remainingBounces != recordDepth) {
                        emitColor *= 0.f;
                    }
                    //else {
                    //    if (recordDepth > 1 && (pathSegments[idx].color.r > 0.5f || pathSegments[idx].color.g > 0.5f || pathSegments[idx].color.b > 0.5f)) {
                    //        printf("cur idx color : %f, %f, %f\n", pathSegments[idx].color.r, pathSegments[idx].color.g, pathSegments[idx].color.b);
                    //    }
                    //}
                }
                pathSegments[idx].color *= emitColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::FIND_EMIT_SOURCE;
            }
            else {
                // Otherwise, do some pseudo-lighting computation. This is actually more
                // like what you would expect from shading in a rasterizer like OpenGL.
                // DONE: replace this! you should be able to start with basically a one-liner
                glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
                scatterRaySimple(
                    pathSegments[idx],
                    intersect,
                    intersection.surfaceNormal,
                    material,
                    rng);
                //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                //pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
        } 
        else {
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
            glm::vec3 backgroundColor = background.getBackgroundColor(pathSegments[idx].ray.direction);
            if (backgroundColor == glm::vec3(0.0f)) {
                pathSegments[idx].color *= backgroundColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::OUT_OF_SCENE;
                shadeableIntersections[idx].materialId = INT_MAX;
                
            }
            else {
                pathSegments[idx].color *= backgroundColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::FIND_EMIT_SOURCE;
                shadeableIntersections[idx].materialId = INT_MAX;
            }
        }
    }
}

////////////////////////////// ADVANCED

__host__ __device__
void scatterRayGeneric(
    PathSegment & pathSegment, 
    float t,
    glm::vec3 normal,
    glm::vec2 uv, 
    const Material &m,
    thrust::default_random_engine &rng) {
    // DONE: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    if (pathSegment.remainingBounces < 0) {
        return;
    }
    glm::vec3 in = glm::normalize(pathSegment.ray.direction);
    glm::vec3 multColor(1.0f);

    MonteCarloReturn mcret = m.sampleScatter(in, normal, uv, rng);
    glm::vec3 scatterDir = mcret.out;
    multColor *= mcret.bsdfTimesCosSlashPDF;
    
    glm::vec3 intersect = pathSegment.ray.origin + pathSegment.ray.direction * t;
        //mcret.penetrate ? getPointOnRayPenetrate(pathSegment.ray, t) : getPointOnRay(pathSegment.ray, t);

    Ray newRay;
    pathSegment.color *= multColor;
    newRay.origin = intersect;
    newRay.direction = scatterDir;
    updateOriginWithBias(newRay);
    pathSegment.ray = newRay;
    --pathSegment.remainingBounces;
}

__global__ void shadeAndScatter (
    int iter
    , int depth
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    , int traceDepth
    , int recordDepth
    , Background background
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces >= 0) {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
                                     // Set up the RNG
                                     // LOOK: this is how you use thrust's RNG! Please look at
                                     // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            //printf("emitColor[%f,%f]=...\n", intersection.uv.x, intersection.uv.y);
            glm::vec3 materialColor = material.getDiffuse(intersection.uv);
            //printf("emitColor[%f,%f]=<%f,%f,%f>\n", intersection.uv.x, intersection.uv.y, materialColor.x, materialColor.y, materialColor.z);///TEST
            intersection.surfaceNormal = glm::normalize(intersection.surfaceNormal);
            // TODO: Normal map?

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                glm::vec3 emitColor = materialColor * material.emittance;
                if (recordDepth >= 0) {
                    if (traceDepth - pathSegments[idx].remainingBounces < recordDepth + 1) {
                        //if (traceDepth - pathSegments[idx].remainingBounces != recordDepth) {
                        emitColor *= 0.f;
                    }
                    //else {
                    //    if (recordDepth > 1 && (pathSegments[idx].color.r > 0.5f || pathSegments[idx].color.g > 0.5f || pathSegments[idx].color.b > 0.5f)) {
                    //        printf("cur idx color : %f, %f, %f\n", pathSegments[idx].color.r, pathSegments[idx].color.g, pathSegments[idx].color.b);
                    //    }
                    //}
                }
                pathSegments[idx].color *= emitColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::FIND_EMIT_SOURCE;
            }
            else {
                scatterRayGeneric(
                    pathSegments[idx],
                    intersection.t,
                    intersection.surfaceNormal,
                    intersection.uv,
                    material,
                    rng);
            }
        } 
        else {
            glm::vec3 backgroundColor = background.getBackgroundColor(pathSegments[idx].ray.direction);
            if (backgroundColor == glm::vec3(0.0f)) {
                pathSegments[idx].color *= backgroundColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::OUT_OF_SCENE;
                shadeableIntersections[idx].materialId = INT_MAX;

            }
            else {
                pathSegments[idx].color *= backgroundColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::FIND_EMIT_SOURCE;
                shadeableIntersections[idx].materialId = INT_MAX;
            }
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths, int iter)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        //image[iterationPath.pixelIndex] += iterationPath.color;
        glm::vec3 color = iterationPath.remainingBounces != RayRemainingBounce::FIND_EMIT_SOURCE ? glm::vec3(0.f) : iterationPath.color;
        color = glm::pow(color, glm::vec3(1.f / 2.2f));

        image[iterationPath.pixelIndex] += color;
        //float alpha = 1.f / iter;
        //image[iterationPath.pixelIndex] = alpha * color + (1.f - alpha) * image[iterationPath.pixelIndex];
    }
}

struct StreamCompactionPredicate {
    __host__ __device__ bool operator()(const PathSegment& p1) const {
    //inline bool operator()(const PathSegment& p1) const {
        return //p1.remainingBounces < 0 || 
            p1.remainingBounces == RayRemainingBounce::OUT_OF_SCENE ||
            (p1.color.r < FLT_EPSILON &&
                p1.color.g < FLT_EPSILON &&
                p1.color.b < FLT_EPSILON);
    }
} streamCompactionPredicate;

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const int recordDepth = hst_scene->state.recordDepth;

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

    generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    long long num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    
    bool iterationComplete = false;
    while (!iterationComplete) {
        //printf("num_paths = %d\n", num_paths);


        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

#if ENABLE_SORTING
        // Sort
        thrust::device_ptr<PathSegment> thrust_dev_paths(dev_paths);
#if !ADVANCED_PIPELINE
        thrust::device_ptr<ShadeableIntersection> thrust_dev_intersection(dev_intersections);
        thrust::sort_by_key(thrust_dev_intersection, thrust_dev_intersection + num_paths, thrust_dev_paths);
#else // ADVANCED_PIPELINE
#endif // ADVANCED_PIPELINE
        checkCUDAError("sort");

#endif // ENABLE_SORTING

        // DONE:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        
        //shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
        shadePipeline<<<numblocksPathSegmentTracing, blockSize1d>>> (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            traceDepth,
            recordDepth,
            hst_scene->background);
        checkCUDAError("shade one bounce");

#if ENABLE_COMPACTION
        // Stream compaction
        // TODO: Use stable_partition instead.
        thrust::device_ptr<PathSegment> thrust_dev_paths_end = thrust::remove_if(thrust_dev_paths, thrust_dev_paths + num_paths, streamCompactionPredicate);
        dev_path_end = thrust_dev_paths_end.get();
        num_paths = dev_path_end - dev_paths;
        checkCUDAError("compaction");
#endif // ENABLE_COMPACTION

        iterationComplete = depth >= traceDepth || num_paths == 0;
        //iterationComplete = true; // TODO: should be based off stream compaction results.
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths, iter);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

////////////////// START DEBUG COLLISION /////////////////////////

__global__ void shadeDebugCollision (
    int iter
    , int depth
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    , int traceDepth
    , int recordDepth
    , Background background
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths && pathSegments[idx].remainingBounces >= 0) {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
                                     // Set up the RNG
                                     // LOOK: this is how you use thrust's RNG! Please look at
                                     // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.getDiffuse(intersection.uv);
            intersection.surfaceNormal = glm::normalize(intersection.surfaceNormal);
            // TODO: Normal map?

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                glm::vec3 emitColor = materialColor * material.emittance;
                if (recordDepth >= 0) {
                    if (traceDepth - pathSegments[idx].remainingBounces < recordDepth + 1) {
                        //if (traceDepth - pathSegments[idx].remainingBounces != recordDepth) {
                        emitColor *= 0.f;
                    }
                    //else {
                    //    if (recordDepth > 1 && (pathSegments[idx].color.r > 0.5f || pathSegments[idx].color.g > 0.5f || pathSegments[idx].color.b > 0.5f)) {
                    //        printf("cur idx color : %f, %f, %f\n", pathSegments[idx].color.r, pathSegments[idx].color.g, pathSegments[idx].color.b);
                    //    }
                    //}
                }
                pathSegments[idx].color *= emitColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::FIND_EMIT_SOURCE;
            }
            else {
                glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
                pathSegments[idx].color *= materialColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::FIND_EMIT_SOURCE;
            }
        } 
        else {
            glm::vec3 backgroundColor = background.getBackgroundColor(pathSegments[idx].ray.direction);
            if (backgroundColor == glm::vec3(0.0f)) {
                pathSegments[idx].color *= backgroundColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::OUT_OF_SCENE;
                shadeableIntersections[idx].materialId = INT_MAX;

            }
            else {
                pathSegments[idx].color *= backgroundColor;
                pathSegments[idx].remainingBounces = RayRemainingBounce::FIND_EMIT_SOURCE;
                shadeableIntersections[idx].materialId = INT_MAX;
            }
        }
    }
}

////////////////// END DEBUG COLLISION /////////////////////////
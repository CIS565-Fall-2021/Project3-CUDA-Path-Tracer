#include <cuda.h>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdio>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "interactions.h"
#include "intersections.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "static_config.h"
#include "stream_compaction.h"
#include "utilities.h"

#define ERRORCHECK 1

#define FILENAME \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
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
#ifdef _WIN32
  getchar();
#endif
  exit(EXIT_FAILURE);
#endif
}

__host__ __device__ thrust::default_random_engine makeSeededRandomEngine(
    int iter, int index, int depth) {
  int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
  return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution, int iter,
                               glm::vec3 *image) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index     = x + (y * resolution.x);
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

static Scene *hst_scene                         = NULL;
static glm::vec3 *dev_image                     = NULL;
static Geom *dev_geoms                          = NULL;
static Material *dev_materials                  = NULL;
static PathSegment *dev_paths                   = NULL;
static ShadeableIntersection *dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static int *dev_materialIDs        = NULL;
static int *dev_materialIDBuffers  = NULL;
static glm::vec3 *dev_image_buffer = NULL;

// first-bounce intersection caching
#ifdef CACHE_INTERSECTIONS
static ShadeableIntersection *dev_intersections_cache = NULL;
static int *dev_materialIDs_cache                     = NULL;
#endif

void pathtraceInit(Scene *scene) {
  hst_scene            = scene;
  const Camera &cam    = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_paths, ANTIALIAS_FACTOR * pixelcount * sizeof(PathSegment));

  cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(),
             scene->materials.size() * sizeof(Material),
             cudaMemcpyHostToDevice);

  cudaMalloc(&dev_intersections,
             ANTIALIAS_FACTOR * pixelcount * sizeof(ShadeableIntersection));
  cudaMemset(dev_intersections, 0,
             ANTIALIAS_FACTOR * pixelcount * sizeof(ShadeableIntersection));

  // TODO: initialize any extra device memeory you need
  cudaMalloc((void **)&dev_materialIDs,
             ANTIALIAS_FACTOR * pixelcount * sizeof(int));
  cudaMalloc((void **)&dev_materialIDBuffers,
             ANTIALIAS_FACTOR * pixelcount * sizeof(int));
  cudaMalloc(&dev_image_buffer, pixelcount * sizeof(glm::vec3));
#ifdef CACHE_INTERSECTIONS
  cudaMalloc((void **)&dev_intersections_cache,
             ANTIALIAS_FACTOR * pixelcount * sizeof(ShadeableIntersection));
  cudaMalloc((void **)&dev_materialIDs_cache,
             ANTIALIAS_FACTOR * pixelcount * sizeof(int));
#endif

  checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
  cudaFree(dev_image);  // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_geoms);
  cudaFree(dev_materials);
  cudaFree(dev_intersections);
  // TODO: clean up any extra device memory you created
  cudaFree(dev_materialIDs);
  cudaFree(dev_materialIDBuffers);
  cudaFree(dev_image_buffer);
#ifdef CACHE_INTERSECTIONS
  cudaFree(dev_intersections_cache);
  cudaFree(dev_materialIDs_cache);
#endif

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth,
                                      PathSegment *pathSegments) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < cam.resolution.x && y < cam.resolution.y) {
    int index = x + (y * cam.resolution.x);

    // primary ray per pixel
    PathSegment &segment     = pathSegments[index];
    segment.ray.origin       = cam.position;
    segment.color            = glm::vec3(1.0f, 1.0f, 1.0f);
    segment.pixelIndex       = index;
    segment.remainingBounces = traceDepth;
    segment.ray.direction =
        glm::normalize(cam.view -
                       cam.right * cam.pixelLength.x *
                           ((float)x - (float)cam.resolution.x * 0.5f) -
                       cam.up * cam.pixelLength.y *
                           ((float)y - (float)cam.resolution.y * 0.5f));

    // implement antialiasing by jittering the ray
    // sub-sampled extra rays per pixel
    int pixelcount = cam.resolution.x * cam.resolution.y;
    for (int i = 1; i < ANTIALIAS_FACTOR; ++i) {
      PathSegment &extra_segment     = pathSegments[i * pixelcount + index];
      extra_segment.ray.origin       = cam.position;
      extra_segment.color            = glm::vec3(1.0f, 1.0f, 1.0f);
      extra_segment.pixelIndex       = index;
      extra_segment.remainingBounces = traceDepth;
      thrust::default_random_engine rng =
          makeSeededRandomEngine(iter, index, i);
      thrust::uniform_real_distribution<float> u01(0, 1);
      extra_segment.ray.direction = glm::normalize(
          cam.view -
          cam.right * cam.pixelLength.x *
              ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f) -
          cam.up * cam.pixelLength.y *
              ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f));
    }
  }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(const int depth, const int num_paths,
                                     const PathSegment *pathSegments,
                                     const Geom *geoms, const int geoms_size,
                                     ShadeableIntersection *intersections,
                                     int *materialIDs) {
  int path_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_index < num_paths) {
    PathSegment pathSegment = pathSegments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min        = FLT_MAX;
    int hit_geom_index = -1;
    bool outside       = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    // naive parse through global geoms

    for (int i = 0; i < geoms_size; i++) {
      const Geom &geom = geoms[i];

      if (geom.type == CUBE) {
        t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                tmp_normal, outside);
      } else if (geom.type == SPHERE) {
        t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                   tmp_normal, outside);
      }
      // TODO: add more intersection tests here... triangle? metaball? CSG?

      // Compute the minimum t from the intersection tests to determine what
      // scene geometry object was hit first.
      if (t > 0.0f && t_min > t) {
        t_min           = t;
        hit_geom_index  = i;
        intersect_point = tmp_intersect;
        normal          = tmp_normal;
      }
    }

    if (hit_geom_index == -1) {
      intersections[path_index].t = -1.0f;
      materialIDs[path_index]     = -1;
    } else {
      // The ray hits something
      int material_id                      = geoms[hit_geom_index].materialid;
      intersections[path_index].t          = t_min;
      intersections[path_index].materialId = material_id;
      intersections[path_index].surfaceNormal = normal;
      materialIDs[path_index]                 = material_id;
    }
  }
}

/**
 * Shade all the intersections according to materials using BSDF.
 *
 * @return  updates pathSegments
 */
__global__ void shadeMaterial(
    int iter, int depth, int num_paths,
    const ShadeableIntersection *shadeableIntersections,
    const Material *materials, PathSegment *pathSegments) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
    const ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment path_segment                 = pathSegments[idx];
    if (intersection.t > 0.0f) {  // if the intersection exists
      // Set up an random number generator
      thrust::default_random_engine rng =
          makeSeededRandomEngine(iter, idx, depth);

      const Material material = materials[intersection.materialId];
      glm::vec3 intersect_pos = getPointOnRay(path_segment.ray, intersection.t);
      scatterRay(path_segment, intersect_pos, intersection.surfaceNormal,
                 material, rng);
    }
    // If there was no intersection, color the ray black, terminates bouncing
    else {
      path_segment.color            = glm::vec3(0.0f);
      path_segment.remainingBounces = 0;
    }
    pathSegments[idx] = path_segment;
  }
}

// Add the current iteration's output to the image buffer
__global__ void finalGather(int nPaths, glm::vec3 *img_buffer,
                            const PathSegment *iterationPaths) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < nPaths) {
    PathSegment iterationPath = iterationPaths[index];
    atomicAdd(&img_buffer[iterationPath.pixelIndex][0], iterationPath.color[0]);
    atomicAdd(&img_buffer[iterationPath.pixelIndex][1], iterationPath.color[1]);
    atomicAdd(&img_buffer[iterationPath.pixelIndex][2], iterationPath.color[2]);
  }
}

// Average the accumulative subpixel values in image buffer & add it to final
// image
__global__ void addToImage(int pixelcount, glm::vec3 *image,
                           const glm::vec3 *img_buffer) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < pixelcount) {
    image[index] += (img_buffer[index] / (1.0f * ANTIALIAS_FACTOR));
  }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
  const int traceDepth = hst_scene->state.traceDepth;
  const Camera &cam    = hst_scene->state.camera;
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
  ///////////////////////////////////////////////////////////////////////////

  generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth,
                                                          dev_paths);
  checkCUDAError("generate camera ray");

  int depth            = 0;
  int num_active_paths = ANTIALIAS_FACTOR * pixelcount;

  // --- PathSegment Tracing Stage ---
  // Shoot ray into scene, bounce between objects, push shading chunks
  while (num_active_paths > 0) {
    // clean shading chunks
    cudaMemset(dev_intersections, 0,
               ANTIALIAS_FACTOR * pixelcount * sizeof(ShadeableIntersection));

    // --- Tracing Stage ---
    dim3 numblocksPathSegmentTracing =
        (num_active_paths + blockSize1d - 1) / blockSize1d;

#ifdef CACHE_INTERSECTIONS
    if (depth == 0 && iter > 1) {
      cudaMemcpy(dev_intersections, dev_intersections_cache,
                 pixelcount * sizeof(ShadeableIntersection),
                 cudaMemcpyDeviceToDevice);
      cudaMemcpy(dev_materialIDs, dev_materialIDs_cache,
                 pixelcount * sizeof(int), cudaMemcpyDeviceToDevice);
      if (num_active_paths - pixelcount > 0) {
        dim3 numBlocksAntialiasTracing =
            (num_active_paths - pixelcount + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numBlocksAntialiasTracing, blockSize1d>>>(
            depth, num_active_paths - pixelcount, dev_paths + pixelcount,
            dev_geoms, hst_scene->geoms.size(), dev_intersections + pixelcount,
            dev_materialIDs + pixelcount);
        checkCUDAError("anti-alias extra rays trace one bounce");
        cudaDeviceSynchronize();
      }
    } else {
      computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
          depth, num_active_paths, dev_paths, dev_geoms,
          hst_scene->geoms.size(), dev_intersections, dev_materialIDs);
      checkCUDAError("trace one bounce");
      cudaDeviceSynchronize();

      if (depth == 0 && iter == 1) {
        cudaMemcpy(dev_intersections_cache, dev_intersections,
                   pixelcount * sizeof(ShadeableIntersection),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_materialIDs_cache, dev_materialIDs,
                   pixelcount * sizeof(int), cudaMemcpyDeviceToDevice);
      }
    }
#else
    computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
        depth, num_active_paths, dev_paths, dev_geoms, hst_scene->geoms.size(),
        dev_intersections, dev_materialIDs);
    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();
#endif

    depth++;

    // --- Shading Stage ---
    // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.
    //
    // a) sort intersections & paths with ray material ID
    if (static_config::enable_radixSort) {
      cudaMemcpy(dev_materialIDBuffers, dev_materialIDs,
                 num_active_paths * sizeof(int), cudaMemcpyDeviceToDevice);
      thrust::sort_by_key(thrust::device, dev_materialIDs,
                          dev_materialIDs + num_active_paths,
                          dev_intersections);
      thrust::sort_by_key(thrust::device, dev_materialIDBuffers,
                          dev_materialIDBuffers + num_active_paths, dev_paths);
    }

    // b) Use BSDF to shade & advance each path segment
    shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
        iter, depth, num_active_paths, dev_intersections, dev_materials,
        dev_paths);
    checkCUDAError("shade material");
    cudaDeviceSynchronize();

    // c) Stream-compact & discard the terminated path segments
    num_active_paths =
        stream_compaction::rayCompaction(dev_paths, num_active_paths);
    checkCUDAError("ray compaction");
  }

  // Assemble this iteration and apply it to the image
  cudaMemset(dev_image_buffer, 0, pixelcount * sizeof(glm::vec3));
  dim3 numBlocksSubPixels =
      (ANTIALIAS_FACTOR * pixelcount + blockSize1d - 1) / blockSize1d;
  finalGather<<<numBlocksSubPixels, blockSize1d>>>(
      ANTIALIAS_FACTOR * pixelcount, dev_image_buffer, dev_paths);
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
  addToImage<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image,
                                               dev_image_buffer);

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter,
                                                   dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  checkCUDAError("pathtrace");
}

#include <cuda.h>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <cmath>
#include <cstdio>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
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

// Send Normal vector in GBuffer for visualization
__global__ void gbufferNormalToPBO(uchar4 *pbo, glm::ivec2 resolution,
                                   GBufferPixel *gBuffer) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);

    pbo[index].w = 0;
    pbo[index].x = fabs(gBuffer[index].normal.x * 255.0);
    pbo[index].y = fabs(gBuffer[index].normal.y * 255.0);
    pbo[index].z = fabs(gBuffer[index].normal.z * 255.0);
  }
}

// Send Position vector in GBuffer for visualization
__global__ void gbufferPositionToPBO(uchar4 *pbo, glm::ivec2 resolution,
                                     GBufferPixel *gBuffer,
                                     glm::vec3 scene_min_xyz,
                                     glm::vec3 scene_max_xyz) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index          = x + (y * resolution.x);
    glm::vec3 position = gBuffer[index].position;
    float hit          = (float)(gBuffer[index].t > 0);

    pbo[index].w = 0;
    pbo[index].x = 255.0 * hit * (position.x - scene_min_xyz.x) /
                   (scene_max_xyz.x - scene_min_xyz.x);
    pbo[index].y = 255.0 * hit * (position.y - scene_min_xyz.y) /
                   (scene_max_xyz.y - scene_min_xyz.y);
    pbo[index].z = 255.0 * hit * (position.z - scene_min_xyz.z) /
                   (scene_max_xyz.z - scene_min_xyz.z);
  }
}

// Send denoised weights for visualization
__global__ void gbufferWeightToPBO(uchar4 *pbo, glm::ivec2 resolution,
                                   float *weights) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index    = x + (y * resolution.x);
    float weight = weights[index];

    pbo[index].w = 0;
    pbo[index].x = 255.0 * weight;
    pbo[index].y = 255.0 * weight;
    pbo[index].z = 255.0 * weight;
  }
}

static Scene *hst_scene                         = NULL;
static glm::vec3 *dev_image                     = NULL;
static Geom *dev_geoms                          = NULL;
static Material *dev_materials                  = NULL;
static PathSegment *dev_paths                   = NULL;
static ShadeableIntersection *dev_intersections = NULL;
static GBufferPixel *dev_gBuffer                = NULL;
static int *dev_materialIDs                     = NULL;
static int *dev_materialIDBuffers               = NULL;
static glm::vec3 *dev_image_buffer              = NULL;

// first-bounce intersection caching
#ifdef CACHE_INTERSECTIONS
static ShadeableIntersection *dev_intersections_cache = NULL;
static int *dev_materialIDs_cache                     = NULL;
#endif

// denoising parameters
static glm::ivec2 *dev_kernelOffset         = NULL;
static glm::vec3 *dev_image_denoised        = NULL;
static glm::vec3 *dev_image_denoised_buffer = NULL;
static float *dev_weights                   = NULL;  // for debug
static float *dev_posWeights                = NULL;  // for debug
static float *dev_norWeights                = NULL;  // for debug
static float *dev_colorWeights              = NULL;  // for debug

// 5x5 Gaussian kernel for image denoising
static const std::array<float, KERNEL_SIZE> kernel = {
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765, 0.015019, 0.059912,
    0.094907, 0.059912, 0.015019, 0.023792, 0.094907, 0.150342, 0.094907,
    0.023792, 0.015019, 0.059912, 0.094907, 0.059912, 0.015019, 0.003765,
    0.015019, 0.023792, 0.015019, 0.003765};
__constant__ float cdev_kernel[KERNEL_SIZE];

// 5x5 offset for A-trous convolution
static std::vector<glm::ivec2> kernelOffset;

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

  cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

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

  // ----- Denoising variables init -----
  // construct debugging buffers
  cudaMalloc((void **)&dev_weights, pixelcount * sizeof(float));
  cudaMalloc((void **)&dev_posWeights, pixelcount * sizeof(float));
  cudaMalloc((void **)&dev_colorWeights, pixelcount * sizeof(float));
  cudaMalloc((void **)&dev_norWeights, pixelcount * sizeof(float));
  checkCUDAError(
      "cudaMalloc dev_weights, dev_posWeights, dev_colorWeights, "
      "dev_norWeights failed");
  // construct denoised image buffer
  cudaMalloc((void **)&dev_image_denoised, pixelcount * sizeof(glm::vec3));
  cudaMalloc((void **)&dev_image_denoised_buffer,
             pixelcount * sizeof(glm::vec3));
  checkCUDAError(
      "cudaMalloc dev_image_denoised, dev_image_denoised_buffer failed");
  cudaMemset(dev_image_denoised, 0, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image_denoised_buffer, 0, pixelcount * sizeof(glm::vec3));
  checkCUDAError("cudaMemset dev_image_denoised failed");
  // construct kernel
  cudaMemcpyToSymbol(cdev_kernel, kernel.data(), KERNEL_SIZE * sizeof(float));
  checkCUDAError("cudaMemcpyToSymbol to cdev_kernel failed");
  // construct convolution offsets
  for (int i = -KERNEL_WIDTH / 2; i <= KERNEL_WIDTH / 2; ++i) {
    for (int j = -KERNEL_WIDTH / 2; j <= KERNEL_WIDTH / 2; ++j) {
      kernelOffset.emplace_back(i, j);
    }
  }
  assert(kernelOffset.size() == KERNEL_SIZE);
  cudaMalloc((void **)&dev_kernelOffset, KERNEL_SIZE * sizeof(glm::ivec2));
  checkCUDAError("cudaMalloc dev_kernelOffset failed");
  cudaMemcpy(dev_kernelOffset, kernelOffset.data(),
             KERNEL_SIZE * sizeof(glm::ivec2), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy to dev_kernelOffset failed");

  checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
  cudaFree(dev_image);  // no-op if dev_image is null
  cudaFree(dev_paths);
  cudaFree(dev_geoms);
  cudaFree(dev_materials);
  cudaFree(dev_intersections);
  cudaFree(dev_gBuffer);
  cudaFree(dev_materialIDs);
  cudaFree(dev_materialIDBuffers);
  cudaFree(dev_image_buffer);
#ifdef CACHE_INTERSECTIONS
  cudaFree(dev_intersections_cache);
  cudaFree(dev_materialIDs_cache);
#endif
  cudaFree(dev_kernelOffset);
  cudaFree(dev_image_denoised);
  cudaFree(dev_image_denoised_buffer);
  cudaFree(dev_weights);
  cudaFree(dev_posWeights);
  cudaFree(dev_norWeights);
  cudaFree(dev_colorWeights);

  checkCUDAError("pathtraceFree");
}

#ifdef DEPTH_OF_FIELD
/**
 * Given a sampled point at [-1,1]x[-1,1], uniformly map to some values on disk
 *  in concentric style
 *
 * @return  (x,y) point on a unit disk
 */
__device__ glm::vec2 concentricSampleDisk(const glm::vec2 &u) {
  // Map uniform random numbers to [-1, 1]x[-1, 1]
  glm::vec2 offset = 2.f * u - glm::vec2(1, 1);

  // Handle degeneracy at the origin
  if (offset.x == 0 && offset.y == 0) return glm::vec2(0, 0);

  // Apply concentric mapping to point
  float theta, r;
  if (std::abs(offset.x) > std::abs(offset.y)) {
    r     = offset.x;
    theta = PI_4 * (offset.y / offset.x);
  } else {
    r     = offset.y;
    theta = PI_2 - PI_4 * (offset.x / offset.y);
  }
  return r * glm::vec2(std::cos(theta), std::sin(theta));
}

/**
 * Updates the origin & direction of each generated ray based on thin-lens
 * camera model Reference:
 * https://www.pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models
 *
 * @return  Ray&  ray
 */
__device__ void updateRayOnLens(Camera cam, int iter, int ray_index, int depth,
                                Ray &ray) {
  glm::mat4 view_mat         = glm::lookAt(cam.position, cam.lookAt, cam.up);
  glm::mat4 view_mat_inverse = glm::inverse(view_mat);

  // 1. sample point on len disk
  thrust::default_random_engine rng =
      makeSeededRandomEngine(iter, ray_index, depth);
  thrust::uniform_real_distribution<float> u01(0, 1);
  glm::vec3 pt_on_lens =
      cam.lensRadius *
      glm::vec3(concentricSampleDisk(glm::vec2(u01(rng), u01(rng))), 0.0f);

  // 2. compute intersection of pinhole ray with plane of focus (in camera local
  // coordinates)
  glm::vec3 origin_local = glm::vec3(view_mat * glm::vec4(ray.origin, 1.0f));
  glm::vec3 dir_local    = glm::vec3(view_mat * glm::vec4(ray.direction, 0.0f));
  float ft               = glm::abs(cam.focalDistance / dir_local.z);
  glm::vec3 pt_on_focus_local = origin_local + ft * dir_local;
  glm::vec3 pt_on_focus =
      glm::vec3(view_mat_inverse * glm::vec4(pt_on_focus_local, 1.0f));

  // 3. update ray's origin & direction on point
  glm::vec3 origin_new =
      glm::vec3(view_mat_inverse * glm::vec4(pt_on_lens, 1.0f));
  glm::vec3 dir_new = glm::normalize(pt_on_focus - origin_new);

  // 4. return
  ray.origin    = origin_new;
  ray.direction = dir_new;
}
#endif

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
#ifdef DEPTH_OF_FIELD
    if (cam.lensRadius > 0) {
      updateRayOnLens(cam, iter, index, INT_MAX, segment.ray);
    }
#endif

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
#ifdef DEPTH_OF_FIELD
      if (cam.lensRadius > 0) {
        updateRayOnLens(cam, iter, index, i, extra_segment.ray);
      }
#endif
    }
  }
}

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
      } else if (geom.type == TRIANGLE) {
        t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect,
                                     tmp_normal, outside);
      }

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

__global__ void generateGBuffer(int num_paths,
                                ShadeableIntersection *shadeableIntersections,
                                PathSegment *pathSegments,
                                GBufferPixel *gBuffer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
    gBuffer[idx].position =
        getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
    gBuffer[idx].t      = shadeableIntersections[idx].t;
    gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
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

__global__ void atrousDenoiser(glm::vec3 *image_denoised, float *weights,
                               float *posWeights, float *colorWeights,
                               float *norWeights, const glm::vec3 *image,
                               const glm::ivec2 resolution, const float c_phi,
                               const float n_phi, const float p_phi,
                               const int stepwidth,
                               const glm::ivec2 *kernel_offset,
                               const GBufferPixel *gBuffer) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int pix_index = x + (y * resolution.x);
    glm::vec3 sum{0.f, 0.f, 0.f};
    float sum_weight = 0.f;

    // for debug visualizations
    float sum_w_normal = 0.f;
    float sum_w_pos    = 0.f;
    float sum_w_color  = 0.f;

    GBufferPixel pix_gBuffer = gBuffer[pix_index];
    glm::vec3 pix_color      = image[pix_index];
    glm::vec3 pix_normal     = pix_gBuffer.normal;
    glm::vec3 pix_pos        = pix_gBuffer.position;

    for (int i = 0; i < KERNEL_SIZE; ++i) {
      glm::ivec2 adj_xy =
          glm::clamp(glm::ivec2(x, y) + stepwidth * kernel_offset[i],
                     glm::ivec2(0, 0), resolution - glm::ivec2(1, 1));
      int adj_index   = adj_xy.x + (adj_xy.y * resolution.x);
      GBufferPixel gb = gBuffer[adj_index];

      glm::vec3 color  = image[adj_index];
      float color_dist = glm::length(pix_color - color);
      float w_color = glm::min(glm::exp(-(color_dist) / (c_phi * c_phi)), 1.0f);

      glm::vec3 normal  = gb.normal;
      float normal_dist = glm::length(pix_normal - normal);
      float w_normal =
          glm::min(glm::exp(-(normal_dist) / (n_phi * n_phi)), 1.0f);

      glm::vec3 pos  = gb.position;
      float pos_dist = glm::length(pix_pos - pos);
      float w_pos    = glm::min(glm::exp(-(pos_dist) / (p_phi * p_phi)), 1.0f);

      float weight = w_color * w_normal * w_pos;
      sum += color * weight * cdev_kernel[i];

      // for debug visualizations
      sum_weight += weight * cdev_kernel[i];
      sum_w_pos += w_pos;
      sum_w_normal += w_normal;
      sum_w_color += w_color;
    }
    image_denoised[pix_index] = sum / sum_weight;

    // for debug visualizations
    weights[pix_index]      = sum_weight;
    posWeights[pix_index]   = sum_w_pos / KERNEL_SIZE;
    norWeights[pix_index]   = sum_w_normal / KERNEL_SIZE;
    colorWeights[pix_index] = sum_w_color / KERNEL_SIZE;
  }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iteration) {
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
  //   * Stream compact away all of the terminated paths.
  //     You may use either your implementation or `thrust::remove_if` or its
  //     cousins.
  //     * Note that you can't really use a 2D kernel launch any more - switch
  //       to 1D.
  //   * Shade the rays that intersected something or didn't bottom out.
  //     That is, color the ray by performing a color computation according
  //     to the shader, then generate a new ray to continue the ray path.
  //     We recommend just updating the ray's PathSegment in place.
  //     Note that this step may come before or after stream compaction,
  //     since some shaders you write may also cause a path to terminate.
  // * Finally:
  //     * if not denoising, add this iteration's results to the image
  //     * if denoising, run kernels that take both the raw pathtraced
  //        result and the gbuffer, and put the result in the "pbo" from opengl
  ///////////////////////////////////////////////////////////////////////////

  generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(
      cam, iteration, traceDepth, dev_paths);
  checkCUDAError("generate camera ray");

  int depth            = 0;
  int num_active_paths = ANTIALIAS_FACTOR * pixelcount;

  // Empty gbuffer
  cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

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
    if (depth == 0 && iteration > 1) {
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

      if (depth == 0 && iteration == 1) {
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

    if (depth == 0) {
      dim3 numBlocksGBuffer = (pixelcount + blockSize1d - 1) / blockSize1d;
      generateGBuffer<<<numBlocksGBuffer, blockSize1d>>>(
          pixelcount, dev_intersections, dev_paths, dev_gBuffer);
    }

    depth++;

    // --- Shading Stage ---
    // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // Compare between directly shading the path segments and shading
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
        iteration, depth, num_active_paths, dev_intersections, dev_materials,
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

  checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something
// that you can visualize for debugging.
void showGBufferNormal(uchar4 *pbo) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for
  // visualization
  gbufferNormalToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution,
                                                       dev_gBuffer);
}

void showGBufferPosition(uchar4 *pbo) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  gbufferPositionToPBO<<<blocksPerGrid2d, blockSize2d>>>(
      pbo, cam.resolution, dev_gBuffer, hst_scene->boundary.min_xyz,
      hst_scene->boundary.max_xyz);
}

void showGBufferWeights(uchar4 *pbo) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  gbufferWeightToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution,
                                                       dev_weights);
}

void showGBufferPositionWeights(uchar4 *pbo) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  gbufferWeightToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution,
                                                       dev_posWeights);
}

void showGBufferNormalWeights(uchar4 *pbo) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  gbufferWeightToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution,
                                                       dev_norWeights);
}

void showGBufferColorWeights(uchar4 *pbo) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  gbufferWeightToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution,
                                                       dev_colorWeights);
}

void showImage(uchar4 *pbo, int iter) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter,
                                                   dev_image);
  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from dev_image to scene");
}

void denoiseImage(int filter_width, float c_phi, float n_phi, float p_phi) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  cudaMemcpy(dev_image_denoised, dev_image, pixelcount * sizeof(glm::vec3),
             cudaMemcpyDeviceToDevice);

  int stepwidth    = 1;
  int kernel_width = KERNEL_WIDTH;
  while (kernel_width < filter_width) {
    atrousDenoiser<<<blocksPerGrid2d, blockSize2d>>>(
        dev_image_denoised_buffer, dev_weights, dev_posWeights,
        dev_colorWeights, dev_norWeights, dev_image_denoised, cam.resolution,
        c_phi, n_phi, p_phi, stepwidth, dev_kernelOffset, dev_gBuffer);
    std::swap(dev_image_denoised, dev_image_denoised_buffer);
    stepwidth++;
    kernel_width = (KERNEL_WIDTH - 1) * stepwidth + 1;
    c_phi /= 2.0f;
  }
}

void showDenoisedImage(uchar4 *pbo, int iter) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
      (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
      (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter,
                                                   dev_image_denoised);
  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image_denoised,
             pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from dev_image_denoised to scene");
}

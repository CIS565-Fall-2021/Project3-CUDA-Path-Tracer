#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <vector>
#include <iostream>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define TIMING 0
#define ERRORCHECK 1
#define CACHE_FIRST_BOUNCE 1
#define RAY_SORTING 1
#define USE_GBUFFER 1
#define COMPACT_GBUFFER 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer, int mode) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);

    switch (mode) {
    case GBUFFER_TIME:
      float timeToIntersect = gBuffer[index].t * 256.0;

      pbo[index].w = 0;
      pbo[index].x = timeToIntersect;
      pbo[index].y = timeToIntersect;
      pbo[index].z = timeToIntersect;
      break;

    case GBUFFER_POSITION:
      if (COMPACT_GBUFFER) {
        float z = abs(gBuffer[index].z) * 256.0f;
        pbo[index].w = 0;
        pbo[index].x = z;
        pbo[index].y = z;
        pbo[index].z = z;
      }
      else {
        glm::vec3 pos = 0.1f * gBuffer[index].p * 256.0f;
        pbo[index].w = 0;
        pbo[index].x = abs(pos.x);
        pbo[index].y = abs(pos.y);
        pbo[index].z = abs(pos.z);
      }
      break;

    case GBUFFER_NORMAL:
      glm::vec3 n = gBuffer[index].n;
      pbo[index].w = 0;
      pbo[index].x = abs((int)(n.x * 255.0));
      pbo[index].y = abs((int)(n.y * 255.0));
      pbo[index].z = abs((int)(n.z * 255.0));
      break;
    }
  }
}

//Static variables for device memory, any extra info you need, etc
static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Mesh * dev_meshes = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;  
static PathSegment * dev_final_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_first_intersections = NULL;

// Texture Data
static cudaTextureObject_t * dev_texObjs = NULL;
static std::vector<cudaArray_t> dev_texArrays;
static std::vector<cudaTextureObject_t> texObjs;

// Mesh Data for the GPU
static PrimData dev_prim_data;

// Denoising
static glm::vec3* dev_denoised_image = NULL;
static GBufferPixel* dev_gBuffer = NULL;

#if TIMING
static cudaEvent_t startEvent = NULL;
static cudaEvent_t endEvent = NULL;
#endif


template <class T>
void mallocAndCopy(T* &d, std::vector<T> &h) {
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
}

/**
* Initialize texture objects
* Based on: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
*/
void textureInit(const Texture& tex, int i) {
    // Allocate CUDA array in device memory
    cudaTextureObject_t texObj;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&dev_texArrays[i], &channelDesc, tex.width, tex.height);

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    // const size_t spitch = tex.width * sizeof(unsigned char);
    // Copy texture image in host memory to device memory
    cudaMemcpyToArray(dev_texArrays[i], 0, 0, tex.image, tex.width * tex.height * tex.components * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_texArrays[i];

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    cudaMemcpy(dev_texObjs+i, &texObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    checkCUDAError("textureInit failed");

    texObjs.push_back(texObj);
}

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    dev_final_paths = dev_paths;

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    mallocAndCopy<Geom>(dev_geoms, scene->geoms);
    mallocAndCopy<Material>(dev_materials, scene->materials);
    mallocAndCopy<Mesh>(dev_meshes, scene->meshes);

    // Mesh GPU data malloc
    mallocAndCopy<Primitive>(dev_prim_data.primitives, scene->primitives);
    mallocAndCopy<uint16_t>(dev_prim_data.indices, scene->mesh_indices);
    mallocAndCopy<glm::vec3>(dev_prim_data.vertices, scene->mesh_vertices);
    mallocAndCopy<glm::vec3>(dev_prim_data.normals, scene->mesh_normals);
    mallocAndCopy<glm::vec2>(dev_prim_data.uvs, scene->mesh_uvs);
    mallocAndCopy<glm::vec4>(dev_prim_data.tangents, scene->mesh_tangents);

    // Create Texture Memory
    texObjs.clear(); dev_texArrays.clear();
    cudaMalloc(&dev_texObjs, scene->textures.size()*sizeof(cudaTextureObject_t));
    dev_texArrays.resize(scene->textures.size());
    for (int i = 0; i < scene->textures.size(); i++)
      textureInit(scene->textures[i], i);

#if CACHE_FIRST_BOUNCE
    cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

#if TIMING
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
#endif

    // Denoising
    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    cudaFree(dev_gBuffer);
    cudaFree(dev_denoised_image);

    // Mesh GPU data free
    dev_prim_data.free();

    for (int i = 0; i < texObjs.size(); i++) {
      cudaDestroyTextureObject(texObjs[i]);
      cudaFreeArray(dev_texArrays[i]);
    }

#if CACHE_FIRST_BOUNCE
    cudaFree(dev_first_intersections);
#endif
    
#if TIMING
    if (startEvent != NULL)
      cudaEventDestroy(startEvent);
    if (endEvent != NULL)
      cudaEventDestroy(endEvent);
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment & segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = Color(1.0);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment * pathSegments
    , Geom * geoms
    , Mesh * meshes
    , int geoms_size
    , PrimData mesh_data
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
        glm::vec2 uv;
        glm::vec4 tangent;
        int materialId = -1;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        // TODO: Maybe just create a temp ShadeableIntersection object
        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        glm::vec4 tmp_tangent;
        int tmp_materialId = -1;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom & geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_materialId = geom.materialid;
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_materialId = geom.materialid;
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, meshes[geom.meshid], mesh_data, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, tmp_materialId);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                materialId = tmp_materialId;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                tangent = tmp_tangent;
            }
        }

        ShadeableIntersection& intersection = intersections[path_index];
        if (hit_geom_index == -1)
        {
            intersection.t = -1.0f;
        }
        else
        {
            //The ray hits something
            intersection.t = t_min;
            intersection.materialId = materialId;
            intersection.surfaceNormal = normal;
            intersection.uv = uv;
            intersection.tangent = tangent;
        }
    }
}

__global__ void shadeBSDF(
  int iter
  , int num_paths
  , ShadeableIntersection* shadeableIntersections
  , PathSegment* pathSegments
  , Material* materials
  , cudaTextureObject_t* textures) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment& pathSegment = pathSegments[idx];
    if (intersection.t > 0.0f) {

      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      Color materialColor = material.pbrMetallicRoughness.baseColorFactor;

      Color emissiveColor = material.emissiveFactor;
      if (material.emissiveTexture.index >= 0) {
        emissiveColor *= sampleTexture(textures[material.emissiveTexture.index], intersection.uv);
      }

      // If the material indicates that the object was a light, "light" the ray
      if (glm::length(emissiveColor) > 0.0f) {
        pathSegment.color *= emissiveColor;
        pathSegment.remainingBounces = 0;
      }
      else {
        scatterRay(pathSegment, intersection, material, textures, rng);
        --pathSegment.remainingBounces;
      }
    }
    else {
      pathSegment.color = Color(0.0f);
      pathSegment.remainingBounces = 0;
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
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = Color(0.0f);
    }
  }
}

__global__ void generateGBuffer(
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
  PathSegment* pathSegments,
  GBufferPixel* gBuffer,
  glm::mat4 camView) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    gBuffer[idx].t = shadeableIntersections[idx].t;
    gBuffer[idx].n = shadeableIntersections[idx].surfaceNormal;
    glm::vec3 point = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
    gBuffer[idx].p = point;
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// Calculate the weight for gBuffer data
__device__ inline float calculateWeight(glm::vec3& a, glm::vec3& b, float phi) {
  glm::vec3 t = a - b;
  float dist2 = glm::dot(t, t);
  return min(exp(-(dist2) / (phi + 0.0001f)), 1.f);
}

// Calculate the weight for the guassian filter
__device__ inline float gaussianWeight(int x, int y, float s) {
  return (1.0f / (2 * PI * s * s)) * exp(-(x * x + y * y) / (2 * s * s));
}

__global__ void normalizeImage(int width, int height, glm::vec3* image, int iter) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < width && y < height) {
    int index = x + (y * width);
    glm::vec3 pix = image[index];

    pix.x /= iter;
    pix.y /= iter;
    pix.z /= iter;

    image[index] = pix;
  }
}

// Denoise Kernel
__global__ void kernDenoise(int width, int height, glm::vec3* image,
  int filterSize, GBufferPixel* gBuffer, int stepWidth, glm::mat4 camView, glm::mat4 camProj,
  float colorWeight, float normalWeight, float positionWeight) {

  // 5x5 B3-spline filter
  float kernel[5][5] = {
    0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
    0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
    0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375,
    0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
    0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625 };

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * width);

  if (index < width * height) {

    glm::vec3 c0 = image[index];
    glm::vec3 c1 = glm::vec3(0.f);
    glm::vec3 dSum = glm::vec3(0.f);

    float k = 0.f;
    for (int i = -2; i <= 2; i++) {
      for (int j = -2; j <= 2; j++) {
        int x0 = x + i * stepWidth;
        int y0 = y + j * stepWidth;
        // Check if the x and y are within bound
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height)
        {
          int idx = x0 + y0 * width;

          float weight = 1.f;

#if USE_GBUFFER
          float c_w = calculateWeight(image[index], image[idx], colorWeight);
          float n_w = calculateWeight(gBuffer[index].n, gBuffer[idx].n, normalWeight);
          float p_w = calculateWeight(gBuffer[index].p, gBuffer[idx].p, positionWeight);
          weight = c_w * n_w * p_w;
#endif

          float ker = kernel[i + 2][j + 2];
          c1 += weight * ker * image[idx];
          k += weight * ker;
        }
      }
    }

    image[index] = c1 / k;
  }
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, bool denoise, int filterSize, int filterPasses, float colorWeight, float normalWeight, float positionWeight) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int N = pixelcount;
    int num_paths = N;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    ShadeableIntersection* intersections = NULL;

    while (!iterationComplete) {

      dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE
      if (depth == 0 && iter != 1)
        intersections = dev_first_intersections;
#endif

      if (intersections == NULL) {
        // Clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
          depth
          , num_paths
          , dev_paths
          , dev_geoms
          , dev_meshes
          , hst_scene->geoms.size()
          , dev_prim_data
          , dev_intersections
          );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

#if CACHE_FIRST_BOUNCE
        // NOTE: Copy before sorting since dev_first_intersections should map to unsorted dev_paths
        if(depth == 0 && iter == 1)
          cudaMemcpy(dev_first_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
#endif

#if RAY_SORTING
        // sort the intersections and rays based on material types
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareIntersections());
#endif

        intersections = dev_intersections;
      }

      if (depth == 0) {
        generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>> (num_paths, intersections, dev_paths, dev_gBuffer, cam.viewMat);
      }

      depth++;

#if TIMING
      cudaEventRecord(startEvent);
#endif

      // TODO:
      // --- Shading Stage ---
      // Shade path segments based on intersections and generate new rays by
      // evaluating the BSDF.
      // Start off with just a big kernel that handles all the different
      // materials you have in the scenefile.
      // TODO: compare between directly shading the path segments and shading
      // path segments that have been reshuffled to be contiguous in memory.
      shadeBSDF <<<numblocksPathSegmentTracing, blockSize1d>>> (
        iter,
        num_paths,
        intersections,
        dev_paths,
        dev_materials,
        dev_texObjs
        );
      checkCUDAError("shadeBSDF failed");

      // partition (stream compaction) the buffer based on whether the ray path is completed
      dev_paths = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, isPathCompleted());

      num_paths = dev_path_end - dev_paths;

      iterationComplete = num_paths == 0;

      intersections = NULL;

#if TIMING
      cudaEventRecord(endEvent);
      cudaEventSynchronize(endEvent);
      float ms;
      cudaEventElapsedTime(&ms, startEvent, endEvent);
      if (depth == 2 && iter % 10 == 0) {
        std::cout << iter;
        std::cout << " " << depth;
        std::cout << " " << ms;
        std::cout << " " << num_paths << endl;
      }
#endif
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(N, dev_image, dev_final_paths);

    // Reset dev_paths to point to first element
    dev_paths = dev_final_paths;  

    if (denoise) {
      cudaMemcpy(dev_denoised_image, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
      normalizeImage << <blocksPerGrid2d, blockSize2d >> > (cam.resolution.x, cam.resolution.y, dev_denoised_image, iter);
      
      for (int i = 0; i < filterPasses; i++) {
        int stepWidth = 1;
        while (4 * stepWidth <= filterSize) {
          kernDenoise<<<blocksPerGrid2d, blockSize2d>>>(
            cam.resolution.x,
            cam.resolution.y,
            dev_denoised_image, filterSize,
            dev_gBuffer, stepWidth, cam.viewMat, cam.projMat,
            colorWeight, normalWeight, positionWeight);
          stepWidth <<= 1;
        }
      }

      cudaMemcpy(hst_scene->state.image.data(), dev_denoised_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }
    else {
      cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }

    checkCUDAError("pathtrace");
}


// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo, int mode) {
  const Camera& cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
    (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
  gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer, mode);
}

void showImage(uchar4* pbo, int iter, bool denoise) {
  const Camera& cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
    (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // Send results to OpenGL buffer for rendering
  sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, denoise? 1 : iter, denoise ? dev_denoised_image : dev_image);
}
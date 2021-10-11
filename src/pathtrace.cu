#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>


#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "../stream_compaction/efficient.cu"
#include "../denoise/oidn/apps/utils/image_io.h"

#include <iostream>

#include <OpenImageDenoise/oidn.hpp>

#include "options.h"

#define ERRORCHECK 1



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

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...


static oidn::DeviceRef oidn_device;
static oidn::FilterRef oidn_filter;
static int img_width, img_height;
static std::shared_ptr<oidn::ImageBuffer> denoised_img;

static glm::vec3 * dev_albedo_image = NULL;
static glm::vec3 * dev_normal_image = NULL;

static std::shared_ptr<oidn::ImageBuffer> albedo_image;
static std::shared_ptr<oidn::ImageBuffer> normal_image;

static Triangle * dev_tris = NULL;
static std::vector<int> geom_tris_starts;
static std::vector<int> geom_tris_ends;
static int * dev_tris_starts;
static int * dev_tris_ends;


void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    // pre calculate all of the offsets where the start of triangles are located, for each geom
    int tri_offset_tmp = 0;
    for(const auto &geom: scene->geoms){
      if (geom.type == MESH){        
        geom_tris_starts.push_back(tri_offset_tmp);
        tri_offset_tmp += geom.num_tris;
        geom_tris_ends.push_back(tri_offset_tmp);
      }else{
        geom_tris_starts.push_back(-1);
        geom_tris_ends.push_back(-1);
      }
    }

    cudaMalloc(&dev_tris_starts, geom_tris_starts.size() * sizeof(int));
    cudaMemcpy(dev_tris_starts, geom_tris_starts.data(), geom_tris_starts.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_tris_ends, geom_tris_ends.size() * sizeof(int));
    cudaMemcpy(dev_tris_ends, geom_tris_ends.data(), geom_tris_ends.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_tris, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_tris, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // oidn
    img_width = cam.resolution.x;
    img_height = cam.resolution.y;
    oidn_device = oidn::newDevice();
    oidn_device.set("numThreads", OIDN_THREADS);
    oidn_device.commit();
    denoised_img = std::make_shared<oidn::ImageBuffer>(img_width, img_height, 3);
    albedo_image = std::make_shared<oidn::ImageBuffer>(img_width, img_height, 3);
    normal_image = std::make_shared<oidn::ImageBuffer>(img_width, img_height, 3);

    oidn_filter = oidn_device.newFilter("RT"); // generic ray tracing filter
    oidn_filter.setImage("color",  denoised_img->data(),  oidn::Format::Float3, img_width, img_height);
    oidn_filter.setImage("albedo", albedo_image->data(), oidn::Format::Float3, img_width, img_height);
    oidn_filter.setImage("normal", normal_image->data(), oidn::Format::Float3, img_width, img_height);
    oidn_filter.setImage("output", denoised_img->data(), oidn::Format::Float3, img_width, img_height);
    oidn_filter.set("hdr", true);
    oidn_filter.set("cleanAux", true);
    oidn_filter.commit();


    // TODO: initialize any extra device memeory you need

  cudaMalloc(&dev_albedo_image, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_normal_image, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_albedo_image);
    cudaFree(dev_normal_image);

//    for(auto dev_ptr: dev_tris_l){
//      cudaFree(dev_ptr);
//    }
  cudaFree(dev_tris);

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


        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );

      thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
      thrust::uniform_real_distribution<float> u01(0, 0.01 * ANTIALIAS_MULTIPLIER);
      auto ray_offset = calculateRandomDirectionInHemisphere2(glm::normalize(glm::cross(cam.up, cam.right)), rng);
      segment.ray.direction = glm::normalize(glm::vec3(segment.ray.direction.x + u01(rng), segment.ray.direction.y + u01(rng), segment.ray.direction.z + u01(rng)));

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
    , int geoms_size
    , Triangle * tris
    , int * g_tris_starts
    , int * g_tris_ends
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
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom & geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }else if(geom.type == MESH){

              #if CHECK_MESH_BOUNDING_BOXES
              // check bounding box first
              if (boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside) < 0.0f){
                continue;
              }
              #endif

              // iterate through triangles of that mesh
              for(int j=g_tris_starts[i]; j<g_tris_ends[i]; j++){
                auto tri = tris[j];
                t =  triIntersectionTest(tri, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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

              continue;
                //t = triIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

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
            intersections[path_index].t = t_min - EPSILON;
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
//__global__ void shadeFakeMaterial (
//  int iter
//  , int num_paths
//    , ShadeableIntersection * shadeableIntersections
//    , PathSegment * pathSegments
//    , Material * materials
//    )
//{
//  int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  if (idx < num_paths)
//  {
//    ShadeableIntersection intersection = shadeableIntersections[idx];
//    if (intersection.t > 0.0f) { // if the intersection exists...
//      // Set up the RNG
//      // LOOK: this is how you use thrust's RNG! Please look at
//      // makeSeededRandomEngine as well.
//      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
//      thrust::uniform_real_distribution<float> u01(0, 1);
//
//      Material material = materials[intersection.materialId];
//      glm::vec3 materialColor = material.color;
//
//      // If the material indicates that the object was a light, "light" the ray
//      if (material.emittance > 0.0f) {
//        pathSegments[idx].color *= (materialColor * material.emittance);
//      }
//      // Otherwise, do some pseudo-lighting computation. This is actually more
//      // like what you would expect from shading in a rasterizer like OpenGL.
//      // TODO: replace this! you should be able to start with basically a one-liner
//      else {
//        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
//        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
//        pathSegments[idx].color *= u01(rng); // apply some noise because why not
//      }
//    // If there was no intersection, color the ray black.
//    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
//    // used for opacity, in which case they can indicate "no opacity".
//    // This can be useful for post-processing and image compositing.
//    } else {
//      pathSegments[idx].color = glm::vec3(0.0f);
//    }
//  }
//}

__global__ void shadeRealMaterial (
        int iter,
        int depth
        , int num_paths
        , ShadeableIntersection * shadeableIntersections
        , PathSegment * pathSegments
        , Material * materials
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {

    pathSegments[idx].remainingBounces -= 1;
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.

      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
      //thrust::uniform_real_distribution<float> u01(-1.0, 1.0);


      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        pathSegments[idx].remainingBounces = 0;
      }
        // Otherwise, do some pseudo-lighting computation. This is actually more
        // like what you would expect from shading in a rasterizer like OpenGL.
        // TODO: replace this! you should be able to start with basically a one-liner
      else {

        //pathSegments[idx].color *= materialColor;

        glm::vec3 new_origin = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;

        scatterRay(
                pathSegments[idx],
                new_origin,
                intersection.surfaceNormal,
                material, rng);



/*
        How I did it before discovering that interactions.h existed.
        //glm::vec3 new_direction = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
        glm::vec3 new_direction = intersection.surfaceNormal;
        new_direction = glm::clamp(glm::vec3(new_direction.x + u01(rng), new_direction.y + u01(rng), new_direction.z + u01(rng)), -1.0f, 1.0f);
        //new_direction = new_direction + u01(rng);

        pathSegments[idx].ray.origin = new_origin;
        pathSegments[idx].ray.direction = new_direction;

 */

//        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
//        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
//        pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
      // If there was no intersection, color the ray black.
      // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
      // used for opacity, in which case they can indicate "no opacity".
      // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f, 0.0f, 0.0f);
      pathSegments[idx].remainingBounces = 0;
    }
  }
}

__global__ void kern_saveAuxDenoiseData (
         int num_paths
        , ShadeableIntersection * shadeableIntersections,
         PathSegment * iterationPaths
        , Material * materials,
        glm::vec3 * albedo,
         glm::vec3 * normal
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment iterationPath = iterationPaths[idx];

    if (intersection.t > 0.0f) {
      Material material = materials[intersection.materialId];
      albedo[iterationPath.pixelIndex] = material.color;
      normal[iterationPath.pixelIndex] = intersection.surfaceNormal;
    }else{
      albedo[iterationPath.pixelIndex] = glm::vec3(0.0f, 0.0f, 0.0f);
      normal[iterationPath.pixelIndex] = glm::vec3(0.0f, 0.0f, 0.0f);
    }
  }
}



struct is_not_zero_remaining_bounces
{
    __host__ __device__
    bool operator()(const PathSegment& path)
    {
      return (path.remainingBounces > 0);
    }
};

struct sort_path_by_mat
{
    __host__ __device__
    bool operator()(const PathSegment& path)
    {
      return (path.remainingBounces > 0);
    }
};

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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
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
    int num_paths = dev_path_end - dev_paths;
    int original_num_paths = num_paths;

  thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);

  thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);

  //thrust::device_ptr<PathSegment> dev_thrust_paths_read(dev_paths_read);
  //thrust::device_ptr<PathSegment> dev_thrust_paths_write(dev_paths_write);
  //thrust::copy(dev_thrust_paths, dev_thrust_paths+num_paths, dev_thrust_paths_read);

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
    while (!iterationComplete) {

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
        , dev_tris
        , dev_tris_starts
        , dev_tris_ends
        , dev_intersections
        );
    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();



    // TODO:
    // --- Shading Stage ---
    // Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

  shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
    frame, depth,
            num_paths,
    dev_intersections,
            dev_paths,
    dev_materials
  );



  // populate denoise images
#if ENABLE_OIDN
  if(depth == 0){
    kern_saveAuxDenoiseData<<<numblocksPathSegmentTracing, blockSize1d>>> (
                    num_paths,
                    dev_intersections,
                            dev_paths,
                    dev_materials,
                    dev_albedo_image,
                    dev_normal_image
    );
  }
#endif

//std::cout << "num paths at iter " << depth << " is " << num_paths  << '\n';
      //std::cout << num_paths  << '\n';

  // use thrust to remove rays with zero bounces remaining
  auto compact_end = thrust::partition(dev_thrust_paths, dev_thrust_paths+num_paths,
          is_not_zero_remaining_bounces());

  num_paths = compact_end - dev_thrust_paths;
//




#if ENABLE_MATERIAL_SORTING
  // use thrust to in place sort the remaining paths with respect to material ID
  thrust::sort_by_key(dev_thrust_intersections, dev_thrust_intersections+num_paths, dev_thrust_paths);
#endif


//
//  auto flip1 = dev_paths_write;
//  auto flip2 = dev_thrust_paths_write;
//  dev_paths_write = dev_paths;
//  dev_thrust_paths_write = dev_thrust_paths;
//  dev_paths = flip1;
//  dev_thrust_paths = flip2;

  if(num_paths == 0){
    iterationComplete = true;
  }

      depth++;
   // TODO: should be based off stream compaction results.

  // guard
//  if(depth > 10){
//    std::cout << "GUARD CALLED " << '\n';
//    iterationComplete = true;
//  }
    }

  //std::cout << "ITER \n";


  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(original_num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

  #if ENABLE_OIDN
    // Retrieve image from GPU (pre denoise)
    cudaMemcpy(denoised_img->data(), dev_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    // retrieve aux images from GPU
    cudaMemcpy(albedo_image->data(), dev_albedo_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_image->data(), dev_normal_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);


    // Filter the image
    oidn_filter.execute();

    // Check for errors
    const char* errorMessage;
    if (oidn_device.getError(errorMessage) != oidn::Error::None){
      std::cout << "Error: " << errorMessage << std::endl;
    }else{
      // put denoised image back on gpu
      cudaMemcpy(dev_image, denoised_img->data(),
                 pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    }
  #endif
  // debug
//  cudaMemcpy(dev_image, albedo_image->data(),
//             pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);


  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

  // Retrieve image from GPU
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

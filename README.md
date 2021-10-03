CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xuntong Liang
  * [LinkedIn](https://www.linkedin.com/in/xuntong-liang-406429181/), [GitHub](https://github.com/PacosLelouch), [twitter](https://twitter.com/XTL90234545).
* Tested on: Windows 10, i7-10750H @ 2.60GHz 16GB, RTX 2070 Super with Max-Q 8192MB



![Result1](img/output_arch/cornell_garage_kit_JAA_ADVPIPE_depth12.2021-10-01_23-22-10z.6031samp.png)

![Result2](img/output_arch/cornell_garage_kit_JAA_ADVPIPE_depth12.2021-10-01_02-50-18z.5933samp.png)



## Feature

![Overall](img/readme/overall.png)



### Overall

- Implemented BSDF for ideal diffuse and specular-reflective surfaces.
- Used stream compaction `thrust::partition` after shading to reduce the number of rays for further bounce.
- Used sorting by material type before shading.
- Implemented a toggleable option to cache the first bounce for re-using.

- Implemented refraction, and non-perfect specular surfaces with Phong material and microfacet material with GGX normal distribution. :two:
- Implemented importance sampling of scattered ray with different materials. :three:
- Implemented `OBJ` mesh loading with [tinyObj](https://github.com/syoyo/tinyobjloader). :four:
- Implemented stochastic anti-aliasing. :two:
- Implemented post processing such as outlining by stencil, and a simple version of ramp shading with a ramp texture. :three:
- Implemented texture import and mapping with triangle meshes, and HDR image sky sphere. :five:
- Implemented bounding volume hierarchy as a hierarchical spatial data structure for triangle meshes. :six:



Some features are shown in the overall picture, and some details and analysis will be shown below. 



### Partition and Sorting

Partition and sorting are some methods to optimize the path tracing. Partition is to reduce the number of paths after some bounces, while sorting is to make threads in every warp do similar amount of computation with sampling and BSDF evaluation. 

The benefit of partition depends on the rate of the surface area of the background and emitted objects in the viewport, as well as the colors of the objects. If the colors are bright, or the surface area of the background and emitted objects is small, there are few paths that can be omitted, thus to make the performance even worse, counting the cost of the partition.

Because I haven't implemented any complex material, such as subsurface scattering surface, sorting seems to make the performance worse in my project. 

Here are some analysis with three scenes. The first scene is relatively simple. The second scene contains a few triangle meshes and more specular. The third scene is closed so that the rays have no chance to reach the background.

//TODO



Based on the result, I would enable partition but disable sorting in further analysis.



### First Bounce Cache

Because stochastic anti-aliasing varies the first intersection for every iteration, the cache is not usable if the random range is large. However, the effect of anti-aliasing is not significant if the random range is too small. I turned off the stochastic anti-aliasing when performed the first bounce cache.

Here are some analysis for the third scene above.

//TODO



There is a slight performance improvement with first bounce cache. 



### Stochastic Anti-Aliasing

Anti-aliasing is actually a task to make our sampling of color in the screen satisfying Nyquist–Shannon sampling theorem as much as possible. For example, if we create all the rays with the same origin and direction in a pixel, it means that the spatial sampling rate of color in the screen is one sample per pixel. This will cause signal overlapping at the area that the signal varies rapidly in both x and y directions, for example, at any diagonal edges. Even though we cannot eliminate the signal overlapping thoroughly, we can still increase the spatial sampling rate to make the signal smoother. 

In a Monte-Carlo based path tracer, we must have many rays created for each pixel in the first bounce. A simple way to increase the spatial sampling rate is to make the origin of the rays different but still close to each other, and a simple way to implement this thought is, randomly assigning an offset, as the figure below shows. 

![SAA](img/readme/antialiasing.png)



The next two pictures show that, the diagonal in the second picture, which is processed by anti-aliasing, is smoother than the sawtooth-like diagonal in the first picture. 

<img src="profiles/anti_aliasing/PA_PartitionSorting3_woAA.png" alt="woAA" style="zoom: 250%;" />

<img src="profiles/anti_aliasing/PA_PartitionSorting3_wiAA.png" alt="wiAA" style="zoom:250%;" />







### Materials and Sampling Methods

I implemented several simple materials, including:

- Phong material, which contains diffuse term and specular term.
- Perfect dielectric material, which can appear as perfect reflection material or perfect refraction material.
- Microfacet material with GGX normal distribution, which is a kind of physically-based material.

These materials have different //TODO



### Post Processing

The post processing methods apply on the final image, which means that the post processing has no influence on the path tracing process. What we have to consider is how to get the other input source.

In a rasterization-based renderer, we can store some extra data, such as base color, normal, stencil, material ID, and object ID, into a G-buffer. In path tracer, we can also get these data in the first bounce, and store these data into our G-buffer as well. When we apply post processing, the G-buffer can be one of the inputs.

I implemented some simple post processing, such as drawing outlines by stencil values, and simple ramp shading by final colors and a ramp texture. In fact, I also apply the gamma correction in the post processing pipeline but it should not be a kind of post processing in my opinion.

Here are some effects of post processing. It seems weird of the ramp shading result, maybe because the ramp texture is not so suitable for this scene.

Outlines:

![Outlines](profiles/postprocess/PA_Outline_JAA_ADVPIPE_depth16_PP1--PARTITION--BVH.2021-10-02_17-20-04z.128samp.png)



Ramp shading:

![Ramp Shading](profiles/postprocess/PA_Outline_JAA_ADVPIPE_depth16_PP0--PARTITION--BVH.2021-10-02_17-20-26z.128samp.png)



Ramp shading with outlines:

![Ramp Shading](profiles/postprocess/PA_Outline_JAA_ADVPIPE_depth16_PP01--PARTITION--BVH.2021-10-02_17-21-21z.128samp.png)



### Mesh Loading and Texture Loading

On the CPU side, mesh loading can be easily done by `tinyObj` and texture loading can be easily done by `stbImage`. However, it should be considered that how to pass, and when to pass these data into GPU. We should pass these data into GPU after CUDA initialization, otherwise there would be a CUDA error.

In this situation, we should keep the relationship between geometries and model files, and the relationship between materials and texture files. After CUDA initialization, we can load these files, parse them, and move them into GPU. 

I also implemented HDR image loading for sky sphere, which means that I have to implement spherical mapping. The core idea is converting the ray direction in world space to spherical coordinate system, and mapping angles into texture coordinates.

I apply bilinear interpolation when reading a pixel from a texture by texture coordinate. 



### Bounding Volume Hierarchy

Bounding volume hierarchy (BVH) is a hierarchical spatial structure based on bounding volume, which the most common representation is bounding box. Compared to other structures such as the octree and the KD-tree, the BVH is more suitable for static triangle meshes in my opinion, because the BVH makes no difficulties for us to deal with many corner cases, such as objects overlapping at some axes. 

The structure of the BVH can be shown by the figure below. each node contains a bounding box, and the subtree is divided based on two smaller bounding boxes. In leaf nodes, there should be some target objects. Using the BVH decreases the average time complexity of ray-triangle intersection from O(N) to O(logN). 

![BVH](img/readme/fig03-bvh.png)

| 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| N1   | N2   | N3   | N4   | N5   | N6   | N7   | O1   |

| 8    | 9    | 10   | 11   | 12   | 13   | 14   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| O2   | O3   | O4   | O5   | O6   | O7   | O8   |



I build the BVH on CPU and simply pass it to GPU. Because I store the triangle (or geometry) index in leaves, and store node index for left and right children, there is no gap between CPU and GPU, compared to the pointer representation. Since we can set the maximum depth of the BVH or it is obvious that the depth cannot exceeds 32 if we build a half-divided BVH, we can allocate a stack with fixed capacity for depth-first traversal, instead of recursive traversal. In addition, the binary tree can be represented in an array, as the table above shown. In this case, the children of i-th node is the (i\*2+1)-th and (i\*2+2)-th. 

I haven't use any heuristic method, such as surface area heuristic, in this project, because it is more difficult and needs to store more data to build an imbalanced binary tree for GPU, and the improvement of half-divided BVH is good enough. Here are some analysis. 





## Questions and Answers





## Scene File Format

To support more features, I modified the original scene file format. 





## Resource and Reference

1. [Simulating Depth of Field Blurring](http://paulbourke.net/miscellaneous/raytracing/)
2. [Importance Sampling of the Phong Reflectance Model](https://www.cs.princeton.edu/courses/archive/fall16/cos526/papers/importance.pdf)
3. [Microfacet Models for Refraction through Rough Surfaces](http://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf)
4. [Bounding Volume Hierarchy](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)
5. [tinyObj](https://github.com/syoyo/tinyobjloader)
6. [sIBL Archive](http://www.hdrlabs.com/sibl/archive.html)


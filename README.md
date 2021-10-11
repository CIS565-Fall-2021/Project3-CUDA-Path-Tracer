CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xiao Wei
* Tested on: Windows 10, i7-9750h @ 2.6GHz 16.0GB, RTX 2070 with MAX-Q Design, 16GB
* 1 late day applied



Feature
======================
* Ideal Diffuse surfaces
* Perfectly specular-reflective (mirrored) surfaces
* Path continuation/termination using Stream Compaction
* Caching the first bounce intersections for re-use
* Mesh Loading with bounding volume intersection culling
* Refraction
* Stochastic Sampled Antialiasing
* Physically-based depth-of-field

Rendering Results
======================

## Ideal Diffuse
![noanti](https://user-images.githubusercontent.com/66859615/136673557-24e26ab2-9a1e-4304-9a09-87f7eda68cd4.png)


## Perfect specular with mesh loaded teapot
![specular teapot](https://user-images.githubusercontent.com/66859615/136673548-8e275359-bb45-43e5-81a6-0187debafab5.png)

## Refractions
 Refraction with Frensel effects using Schlick's approximation
![refract1](https://user-images.githubusercontent.com/66859615/136673533-ad4f9364-603a-4293-9254-7dec0bc53e96.png)
![refract2](https://user-images.githubusercontent.com/66859615/136673535-7ea1c7bd-6c9f-42db-adea-fe298c5158f4.png)
## Antialiasing
For Antialiasing, I divided the pixels into grids and the points from which the rays are shoot is generated randomly inside that grid, and at each iteration we will choose one grid to use and we will use all of them one by one. The first picture is rendered WITHOUT antialiasing and the second is rendered with antialiasing. You can notice the aliasing on the left and right edge of red box is significantly mitigated in the second picture.

![noanti](https://user-images.githubusercontent.com/66859615/136673564-f1e90853-cc2a-48ee-9a18-244e4d9cbc1a.png)
![anti](https://user-images.githubusercontent.com/66859615/136674065-54fc0d35-4b73-4e92-a2ad-589e69bccf32.png)


## Depth of Field
Depth of Field is implemented according to PBRT
![dof](https://user-images.githubusercontent.com/66859615/136673562-edfa4ca5-fdf3-4232-97e9-73670eaf3749.png)

Performance analysis
======================
## Mesh Bounding Box
Time Comparison of rendering the following teapot with and without Bounding Box (200 iterations and depth)

![cornell 2021-10-09_21-32-53z 200samp](https://user-images.githubusercontent.com/66859615/136674312-d80bf5ed-e48a-42f9-a7fb-046a63027de3.png)

Without Bounding Box | With Bounding Box
------------ | -------------
1min 42sec 68ms | 1min 13sec 43ms

As you can see, we can get about 28.4% performance improvement with bounding box in my case

## Material sort and First Bounce Cache

The performance of Material sort is measured with the following rendering with 300 iterations
![cornell 2021-10-09_21-52-39z 300samp](https://user-images.githubusercontent.com/66859615/136674721-4458c1b5-3075-49b3-b070-305caf1342b7.png)


![20211009100655](https://user-images.githubusercontent.com/66859615/136674835-3b092577-49b1-427e-9404-091a65b3d408.png)


Surprisingly, the material sort actually makes the performance worse in my case. I think it is mostly because there are not so many materials in the scene and the sorting consumes more time than what we save from the better memory placement.

## First Bounce Cache
The performance of Cache Bounce is measured with the following rendering with 300 iterations

![cornell 2021-10-09_22-20-37z 300samp](https://user-images.githubusercontent.com/66859615/136675214-58feb48e-1778-4efd-bfa5-bef7797c01e7.png)
![20211009102803](https://user-images.githubusercontent.com/66859615/136675217-966a761f-1298-473e-a147-2927bd79a540.png)
With no surprise, by caching the first bounce, we improved the performance a lot. I initially used some simple scene to test but the improvement was not obvious because the first bounce doesn't take much time. Then I switched to this scene with mesh.




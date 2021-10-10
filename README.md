CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Matt Elser
  * [LinkedIn](https://www.linkedin.com/in/matt-elser-97b8151ba/), [twitter](twitter.com/__mattelser__)
* Tested on: Tested on: Windows 10, i3-10100F @ 3.6GHz 16GB, GeForce 1660 Super 6GB

### Features
![nice image or two showing off some features]()

This is a GPU based forward path tracer, which renders scenes by calculating "camera rays" bouncing around the scene,
simulating individual light rays. The renderer supports the following features:
- Arbitrary mesh loading using .obj file format
- Multiple shader BSDFs, including refraction
- Anti-aliasing
- Depth of Field
- Minor optimizations
- Adaptive Sampling* (not fully implemented)
- Bloopers

### Arbitrary mesh loading 
The renderer supports loading arbitrary meshes via .obj files. 

One issue discovered was that the triangle intersection detection function
initially used (`glm::intersectRayTriangle()`) does not compute intersections
with the "back" of faces. This caused problems for open meshes like the Newell Teapot
or for meshes assigned a refraction shader, as can be seen here:
![newell teapot hole image]()
![incomplete refraction image]()
A bounding box is calculated at load time and used to optimize ray intersection
detection. The bounding box is a mesh itself, consisting of tris. Each ray in
the scene is initially tested for  intersection with these tris, and only if an
intersection is found will the ray be checked against the mesh's tris.

Performance impacts:
- no spatial optimizations are made (other than the bounding box), so each
ray that hits the bounding box is checked against every triangle in the mesh.
As a result, large meshes can have a significant impact on render times.

Known limitations:
- only objs consisting entirely of triangles are supported. Quads are
interpreted with inconsistant results by tinyobjloader
- vertex normals must be included in the .obj file. Missing normals will
inherit whatever memory value the normals were initialized to.
### Multiple shaders 
BSDFs are implemented to allow for pure diffuse objects, objects with diffuse
and reflections, as well as objects with both reflection and refraction. The
Fresnel effect is calculated using Schlick's approximation
![image showing off different materials]()

Since physically correct models do not always provide the preferred result, the
Fresnel effect is tuneable via a user parameter. Note this is separate from the
index of refraction (also tuneable), this is an additional parameter which controls
the power used in Schlick's approximation. 
![image showing different Fresnel powers]() 

Performance impacts:
- TODO: compare scene performance with/without some shader types

Known limitations:
- objects with refractions are assumed to have reflection. An object can be reflective without
refraction, but not vice-versa. 

### Anti-aliasing
anti-aliasing was accomplished by jittering the origin of camera rays for the initial bounce.
![image without anti-aliasing]()
![image with anti-aliasing]()

Performance impacts:
- An unnoticeable impact to the time it takes each pixel to converge as a result of adding some small randomness.
Known limitations:
- This can not be combined with the "first bounce cache" optimization as it depends on 
slightly varied camera ray origins each iteration. 
### Depth of Field
Depth of field can optionally be simulated, with tuneable parameters for aperture size, lens radius,
and focal distance. 
![image showing off depth of field]()

Performance impacts:
- Using DOF requires a greater number of iterations to produce a clean image. The blur is a result 
of a stochastic process, and as a result the greater the blur the larger the variance of each blurred pixel
Known limitations:
- This can not be combined with the "first bounce cache" optimization as it depends on 
slightly varied camera rays each iteration. 
### Minor optimizations
- first bounce cache
An option is provided to cache the first bounce of each camera ray from iteration 1, and use that cache
for each subsequent iteration (until the camera is moved, generating a new iteration 1 and a new cache).
- sort by materials
In order to decrease divergence (i.e. multiple threads in a warp taking
different code paths as a result of conditionals), rays can optionally be
sorted by their material id. This manimizes the number of warps with different
materials, which may take different amounts of time as a result of calculated
differing BSDFs.
- cull dead bounces
Bounces that do not hit an object (i.e. which go off into space) are culled every iteration. 

Performance impacts:
- first bounce cache provides a noticeable improvement (TODO add a metric for this)
- Sorting materials is noteably worse. (TODO provide a metric)
- culling dead bounces (I think?) has a relatively neutral impact (TODO confirm and add metric)
Known limitations:
- As noted above, first bounce cache cannot be combined with DOF or anti-aliasing.
### Adaptive Sampling* (incomplete)
(note, this feature is incomplete. All that is described is implemented, but a
bug prevents it from working properly)
Adaptive sampling is the process of determinig whether a pixel needs further
iterations, and only sampling further if so. To do this, each pixel's variance
is  tracked, and updated each time the pixel color is updated (at the "final
gather" stage). After a user-defined minimum number of samples, this variance
is compared with a user-defined value. Once the pixel's variance falls below
this threshold, its color is considered "converged" and it is marked for
culling.  Converged pixels are culled and rays for these pixels are no longer
cast, saving potentially significant resources.
As a result of this process, pixels take varying numbers of iterations. The
number of iterations needed by a pixel can be useful information, and so with
each saved image, a "heatmap" is saved alongside it. This heatmap shows the
number of iterations taken for each pixel, and can therefore be used to display 
areas where greater or fewer resources are spent. 

How this feature is incomplete:
As stated above, all of the aspects of adaptive sampling described are
implemented.  However, some misalignment exists between the culling of pixels
and the casting of rays in the following iteration. As a result, the last
pixels of the image are always culled instead of the pixels which have
converged. This is likely due to a sorting mismatch,  or using the wrong number
of paths when calling some relevant function. This has not been fixed in time.

![accurate heatmap showing incorrect sampling]()
### Bloopers
![several]()
![bloopers]()
![here]()
![with]()
![explanations]()


CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Paul (San) Jewell
    * [LinkedIn](https://www.linkedin.com/in/paul-jewell-2aba7379), [work website](
      https://www.biociphers.org/paul-jewell-lab-member), [personal website](https://gitlab.com/inklabapp), [twitter](https://twitter.com/inklabapp).
* Tested on: Linux pop-os 5.11.0-7614-generic, i7-9750H CPU @ 2.60GHz 32GB, GeForce GTX 1650 Mobile / Max-Q 4GB


## Reverse Light Ray Bouncy Thing

< this is where the good rendered images will go >

This project represents the concept of a _path tracer_ along with various improvements 
to demonstrate viable use cases for this type of software. A path tracer conceptually 
is a program designed to mimic the behaviour of physically based, real world light 
interactions with various common materials.  This is accomplished, at a high level, 
by choosing rays from the camera plane, and following their bounces backwards toward 
potential sources of light.

### General Implementation details

The path tracer is programmed primarily in C++ and CUDA, prepared with the common 
_CMake_ build help tool. Tested to work with at least CUDA compute capability 5. At least
one GB of GPU RAM and 4GB system RAM are recommended. In order to get the most out of the
de-noising component, at least four CPU cores are recommended. 

### Program usage

For best runtime efficiency, the majority of options are applied at compile time, not run
time. Please edit the file "src/options.h" in order to adjust the desired settings, then 
re-compile. 

At run time, the program may be invoked by simply calling it with a single argument to 
the scene file to use. For example:

`$ ./cis565_path_tracer scenes/cornell.txt`

`$ ./cis565_path_tracer scenes/cornell2/cornell2.obj`

Once the program is launched, you can observe the beauty! You may use the mouse left click + drag to rotate the camera, 
right click and drag up/down to zoom. Pressing the 's' key will capture an image of the render
at the current state. 

### Features!

#### Anti-Aliasing 

An arbitrary value of antialias may be applied to soften edges. To enable, simply
change the value of `ANTIALIAS_MULTIPLIER` to any value greater than zero. Decimal numbers
are acceptable. I would recommend starting with a value between 0.5 and 1.5

Note: It may not be desirable to use antialiasing along with the denoise feature -- experimentally,
it seems that the model was likely trained on non-AA data, which can produce artifacts
along the edges where the antialiasing usually has the greatest impact.

| no AA      | AA 1/2| AA 1|
| ----------- | -----------| -----------|
| ![](img/perf/aa/noaa.png)      | ![](img/perf/aa/aa0.5.png)   | ![](img/perf/aa/aa1.png)   |

#### Model-based Denoising

We make use of the 'Intel® Open Image Denoise' library, which is purpose built to very
quickly converge ray-traced images based on low-iteration (fast) tracing output, combined 
with the information about normals and albedo of the viewed geometry. 

To use this feature, change `ENABLE_OIDN` from `0` to `1` and select a reasonable value
for `OIDN_THREADS` (ideally not more CPUs than your machine possesses).

When using this feature, you may notice some slight model-based 'fog' when scenes contain
very little geometry, but in general, you should be able to achieve an acceptable render
with very few iterations relative to non-denoised.

| Demo Scene      |  time difference | 10 iterations without denoise| 10 iterations with denoise |
| ----------- | -----------| -----------| -----------|
|Default Cornell|1.315s vs 5.406s|![](img/perf/denoise/cornell.png)|![](img/perf/denoise/cornelld.png)|
|cornell2|1.183s vs 5.300s|![](img/perf/denoise/cornell2.png)|![](img/perf/denoise/cornell2d.png)|
|airboat|1.730s vs 5.557s|![](img/perf/denoise/airboat.png)|![](img/perf/denoise/airboatd.png)|
|room|26.412s vs 32.339s|![](img/perf/denoise/room.png)|![](img/perf/denoise/roomd.png)|
|sanura|31.424s vs 35.217s|![](img/perf/denoise/sanura.png)|![](img/perf/denoise/sanurad.png)|
#### Material sorting

When there relatively even usage of materials across many objects, it can help to 
have the algorithm sort them as a processing step on each 'bounce' step on each iteration.
This option can be enabled by setting `ENABLE_MATERIAL_SORTING` to `1`'. On most scenes,
it should help slightly with performance. However, in the scenes I assembled, seemingly 
there were not enough materials in general to see any benefit from this feature.

(listed in order of increasing numbers of triangles)
![](img/perf/matsort.png)

As can be seen, there was only a tiny improvement for one of the more complicated models,
but for most others, the sorting is wasted. 

#### Mesh / scene loading from wavefront .OBJ files

OBJ files are a long standing 'human readable/editable' text object format with common
support among many 3d editor programs such as Blender and Maya. 

There are a few performance options (recommend you leave them enabled) when loading 
from .OBJ files:

- when `CHECK_MESH_BOUNDING_BOXES` is set to `1` the program will only attempt to bounce rays from triangles inside a mesh 
in the case that the original ray lands somewhere in the objects bounding box. In order to 
get the best efficiency from this check, it helps to break down meshes in OBJ files into 
many objects, instead of having one huge mesh. 
- when `TRIANGLE_BACK_FACE_CULLING` is set to `1` the program will ignore any interactions 
with mesh triangles with a normal surface pointing away from the ray. This will only improve
efficiency with most 3d objects, but may produce unexpected results when there are 2d surfaces 
in your mesh. -- If they face the wrong way, they will not be rendered. 

Because the Camera and its relevant options are not included as part of the .OBJ standard,
there is a feature implemented to auto-calculate the best ideal position / distance based
on the requested FOV and Resolution in the compile options. Additionally, there are other 
camera based options available in options.h.

Note that at this time, only fully triangulated based .OBJ files are supported (faces may
only be triangles) -- this is the default type that is output by blender, for example.

obj material files support the Kd (diffuse color) and Ks (specular color) options. We use
the Ke (emissive) color to determine the magnitude of light emitted, so be sure to have 
at least one emissive object in the scene. 

#### Stream compaction

While mostly implicitly obvious, compacting away rays which have hit a light, or a void
can account for a significant speedup for some scene types. 

| farthest      | med1| med2| closest| enclosed (back wall added) |
| ----------- | -----------| -----------| -----------| ----------- |
| ![](img/perf/scompact/farthest.png)      | ![](img/perf/scompact/med1.png)   | ![](img/perf/scompact/med2.png)   | ![](img/perf/scompact/closest.png)   | ![](img/perf/scompact/enclosed.png)        |

![](img/perf/scompact/perf1.png)

Note due to stream compaction, some rays are killed each sub-iteration. With an open box,
while some rays may hit the light source and terminate, the vast majority will terminate into
the void at a fairly high rate even after the first iteration. However, for the enclosed
system, the light source is the only available termination point, so almost all rays remain 
alive through all sub-iterations. 

-----------------------

### OutTakes >w>

Behold some interesting artwork I've made, totally on purpose! 
All credit goes to random typos in my code...

#### Everything's a Mirror~
![](img/outtakes/mirrorbox2.png)

#### Everything's a Mirror (Funky reverse blacklight)~
![](img/outtakes/mirrorbox.png)

#### Radioactive Desaturation~
![](img/outtakes/desaturation.png)

#### Shadow Play~
![](img/outtakes/shadowachne.png)

#### Hammered~
![](img/outtakes/trippydenoise.png)

#### Reflections of Water~
![](img/outtakes/wallsofwater.png)

#### Chaos Rave~
![](img/outtakes/chaosrave.png)

#### Budget Barriers~
![](img/outtakes/budgetwalls.gif)

#### Formless Edges~
![](img/outtakes/formlessedges.png)

### Credits / Acknowledgements 

(note that any specific code usages are also included inline with the source)

- For help with OS directory processing: sehe @ https://stackoverflow.com/a/8518855
- For help with ray-triangle intersection: Scratchapixel @ https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
- The 'tiny object' library for parsing wavefront OBJ file formats: <Many contributors> @ https://github.com/tinyobjloader/tinyobjloader
- The 'Intel® Open Image Denoise' software/library for denoising ray-traced images: <Many Contributors> @ https://github.com/OpenImageDenoise/oidn
- https://people.sc.fsu.edu/~jburkardt/data/obj/obj.html -- for sample OBJ files

//
// Created by pjewell on 10/5/21.
//

#ifndef CIS565_PATH_TRACER_OPTIONS_H
#define CIS565_PATH_TRACER_OPTIONS_H

/*
 * !!! SEE HERE, compile-time options for features !!!
 */

// enable the denoiser
#define ENABLE_OIDN 1
#define OIDN_THREADS 12  // note: CPU threads to use, NOT GPU

#define ENABLE_MATERIAL_SORTING 0

// only relevant for mesh/obj loading
#define CHECK_MESH_BOUNDING_BOXES 1
#define TRIANGLE_BACK_FACE_CULLING 1
#define CAMERA_FOV 45.0f
#define CAMERA_ZOOM_PADDING 2.0  // Just a little buffer added to the auto-calculated camera distance
#define CAMERA_RES_X 800
#define CAMERA_RES_Y 800
#define CAMERA_ITERATIONS 5000

// this can be zero, to turn off, or numbers higher than one for more blur effect
// I would recommend to DISABLE THIS, when using OIDN
#define ANTIALIAS_MULTIPLIER 0.3

// When loading .OBJ files, max bounce count for each ray is not embedded, adjust this value to set your preference
#define MAX_BOUNCES 8


/*
 * !!! End San's options !!!
 */


#endif //CIS565_PATH_TRACER_OPTIONS_H

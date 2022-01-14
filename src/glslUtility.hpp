#pragma once

// GLSL Utility: A utility class for loading GLSL shaders
// Written by Varun Sampath, Patrick Cozzi, and Karl Li.
// Copyright (c) 2012 University of Pennsylvania

#include <GL/glew.h>

namespace glsl_util {

GLuint create_default_prog(const char *attributeLocations[], GLuint numberOfLocations);

GLuint create_prog(const char *vertexShaderPath, const char *fragmentShaderPath,
	const char *attributeLocations[], GLuint numberOfLocations);
}

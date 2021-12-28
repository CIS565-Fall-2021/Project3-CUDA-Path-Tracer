// GLSL Utility: A utility class for loading GLSL shaders
// Written by Varun Sampath, Patrick Cozzi, and Karl Li.
// Copyright (c) 2012 University of Pennsylvania

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include "glslUtility.hpp"

using std::vector;

namespace glsl_util {


// embedded passthrough shaders so that default passthrough shaders don't need to be loaded
static char *passthroughVS =
"attribute vec4 Position; \n"
"attribute vec2 Texcoords; \n"
"varying vec2 v_Texcoords; \n"
"\n"
"void main(void) { \n"
"    v_Texcoords = Texcoords; \n"
"    gl_Position = Position; \n"
"}";

static char *passthroughFS =
"varying vec2 v_Texcoords; \n"
"\n"
"uniform sampler2D u_image; \n"
"\n"
"void main(void) { \n"
"    gl_FragColor = texture2D(u_image, v_Texcoords); \n"
"}";

struct Shaders {
	GLuint vertex;
	GLuint fragment;
	GLint geometry;
};

vector<char> load_file(const char *fname)
{
	std::ifstream file(fname, std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open()) {
		unsigned int size = (unsigned int) file.tellg();
		
		vector<char> buf(size);
		file.seekg(0, std::ios::beg);
		file.read(buf.data(), size);
		file.close();

		printf("file %s loaded\n", fname);
		return buf;
	}

	printf("Unable to open file %s\n", fname);
	exit(EXIT_FAILURE);
}

// print_shader_InfoLog
// From OpenGL Shading Language 3rd Edition, p215-216
// Display (hopefully) useful error messages if shader fails to compile
void print_shader_InfoLog(GLint shader)
{
	int info_log_len = 0;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_len);

	if (info_log_len > 1) {
		vector<GLchar> info_log(info_log_len);
		// error check for fail to allocate memory omitted
		glGetShaderInfoLog(shader, info_log_len, NULL, info_log.data());
		printf("InfoLog:\n%s\n", info_log.data());
	}
}

void print_link_InfoLog(GLint prog)
{
	int info_log_len = 0;

	glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &info_log_len);

	if (info_log_len > 1) {
		vector<GLchar> info_log(info_log_len);
		// error check for failure to allocate memory omitted
		glGetProgramInfoLog(prog, info_log_len, NULL, info_log.data());
		printf("InfoLog:\n%s\n", info_log.data());
	}
}

/* returns the compiled shader */
GLint compile_shader(const char *shaderName, const char *shaderSource, GLenum shaderType)
{
	GLint s = glCreateShader(shaderType);
	int lens[1] = {(int) std::strlen(shaderSource)};

	glShaderSource(s, 1, &shaderSource, lens);

	glCompileShader(s);

	GLint compiled;
	glGetShaderiv(s, GL_COMPILE_STATUS, &compiled);
	if (!compiled)
		printf("%s did not compile\n", shaderName);

	print_shader_InfoLog(s);

	return s;
}

Shaders load_default_shaders()
{
	Shaders out;

	out.vertex = compile_shader("Passthrough Vertex", passthroughVS, GL_VERTEX_SHADER);
	out.fragment = compile_shader("Passthrough Fragment", passthroughFS, GL_FRAGMENT_SHADER);

	return out;
}

Shaders load_shaders(const char *vert_path, const char *frag_path, const char *geom_path = nullptr)
{
	Shaders out;

	// load shaders & get length of each
	vector<char> vertex_src, fragment_src, geometry_src;

	vertex_src = load_file(vert_path);
	out.vertex = compile_shader("Vertex", vertex_src.data(), GL_VERTEX_SHADER);

	fragment_src = load_file(frag_path);
	out.fragment = compile_shader("Fragment", fragment_src.data(), GL_FRAGMENT_SHADER);

	if (geom_path) {
		geometry_src = load_file(geom_path);
		out.geometry = compile_shader("Geometry", geometry_src.data(), GL_GEOMETRY_SHADER);
	}

	return out;
}

void attach_and_link_prog(GLuint program, Shaders shaders)
{
	glAttachShader(program, shaders.vertex);
	glAttachShader(program, shaders.fragment);

	glLinkProgram(program);

	GLint linked;
	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	if (!linked)
		printf("Program did not link\n");

	print_link_InfoLog(program);
}

GLuint create_default_prog(const char *attributeLocations[], GLuint numberOfLocations)
{
	Shaders shaders = load_default_shaders();

	GLuint program = glCreateProgram();

	for (GLuint i = 0; i < numberOfLocations; ++i) {
		glBindAttribLocation(program, i, attributeLocations[i]);
	}

	attach_and_link_prog(program, shaders);

	return program;
}

GLuint create_prog(const char *vertexShaderPath, const char *fragmentShaderPath,
	const char *attributeLocations[], GLuint numberOfLocations)
{
	Shaders shaders = load_shaders(vertexShaderPath, fragmentShaderPath);

	GLuint program = glCreateProgram();

	for (GLuint i = 0; i < numberOfLocations; ++i) {
		glBindAttribLocation(program, i, attributeLocations[i]);
	}

	attach_and_link_prog(program, shaders);

	return program;
}


} /* namespace glsl_util */

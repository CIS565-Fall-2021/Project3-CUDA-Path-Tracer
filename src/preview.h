#pragma once

#include <GL/glew.h>

#include <string>

extern GLuint pbo;

std::string currentTimeString();
bool init();
void mainLoop();

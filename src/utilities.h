#pragma once

#include "glm/glm.hpp"
#include <istream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

namespace utilities {

float clamp(float f, float min, float max);
glm::vec3 clampRGB(glm::vec3 color);
bool epsilon_eq(float a, float b);
std::vector<std::string> tokenize_string(std::string str);
glm::mat4 make_transform_matrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
std::istream &safe_getline(std::istream &is, std::string &t);

}

#pragma once

#include <istream>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#define PI                3.1415926535897932384626422832795028841971f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f


namespace util {

template <typename T>
T max(T v)
{
	return v;
}

template <typename T, typename... U>
T max(T v1, T v2, U ... vs)
{
	return max(v1 > v2 ? v1 : v2, vs...);
}

template <typename T>
T min(T v) {
	return v;
}

template <typename T, typename... U>
T min(T v1, T v2, U ... vs)
{
	return min(v1 < v2 ? v1 : v2, vs...);
}


template <typename T>
T clamp(T val, T min, T max)
{
	return val < min ? min : (val > max ? max : val);
}

glm::vec3 clampRGB(glm::vec3 color);


bool epsilon_eq(float a, float b);

std::vector<std::string> tokenize_string(std::string str);

glm::mat4 make_transform_matrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);

std::string getline(std::istream &is);

}

#include <sstream>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>

#include "util.h"


using std::string;
using std::vector;

using glm::vec3;
using glm::mat4;
using glm::translate;
using glm::rotate;
using glm::scale;

namespace util {

vec3 clampRGB(vec3 color)
{
	return vec3(clamp(color.x, 0.f, 255.f), clamp(color.y, 0.f, 255.f), clamp(color.z, 0.f, 255.f));
}

bool epsilon_eq(float a, float b)
{
	return a < b ? (b - a < EPSILON) : (a - b < EPSILON);
}

vector<string> tokenize_string(string str)
{
	std::stringstream strstr(str);
	std::istream_iterator<string> it(strstr);
	std::istream_iterator<string> end;
	vector<string> results(it, end);
	return results;
}

mat4 make_transform_matrix(vec3 translation, vec3 rotation, vec3 scaling)
{
	return translate(mat4(), translation) *
		rotate(mat4(), rotation.x * PI / 180.f, vec3(1, 0, 0)) *
		rotate(mat4(), rotation.y * PI / 180.f, vec3(0, 1, 0)) *
		rotate(mat4(), rotation.z * PI / 180.f, vec3(0, 0, 1)) *
		scale(mat4(), scaling);
}


string getline(std::istream &is)
{
	string t;
	std::getline(is, t);
	return t;
}

}


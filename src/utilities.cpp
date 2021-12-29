//  UTILITYCORE- A Utility Library by Yining Karl Li
//  This file is part of UTILITYCORE, Copyright (c) 2012 Yining Karl Li
//
//  File: utilities.cpp
//  A collection/kitchen sink of generally useful functions

#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "utilities.h"

float utilities::clamp(float f, float min, float max)
{
	return  f < min ? min :
		f > max ? max :
		f;
}

glm::vec3 utilities::clampRGB(glm::vec3 color)
{
	if (color[0] < 0)
		color[0] = 0;
	else if (color[0] > 255)
		color[0] = 255;

	if (color[1] < 0)
		color[1] = 0;
	else if (color[1] > 255)
		color[1] = 255;

	if (color[2] < 0)
		color[2] = 0;
	else if (color[2] > 255)
		color[2] = 255;

	return color;
}

bool utilities::epsilon_eq(float a, float b)
{
	return fabs(fabs(a) - fabs(b)) < EPSILON;
}

glm::mat4 utilities::make_transform_matrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale)
{
	return glm::translate(glm::mat4(), translation) *
		glm::rotate(glm::mat4(), rotation.x * (float) PI / 180, glm::vec3(1, 0, 0)) *
		glm::rotate(glm::mat4(), rotation.y * (float) PI / 180, glm::vec3(0, 1, 0)) *
		glm::rotate(glm::mat4(), rotation.z * (float) PI / 180, glm::vec3(0, 0, 1)) *
		glm::scale(glm::mat4(), scale);
}

std::vector<std::string> utilities::tokenize_string(std::string str)
{
	std::stringstream strstr(str);
	std::istream_iterator<std::string> it(strstr);
	std::istream_iterator<std::string> end;
	std::vector<std::string> results(it, end);
	return results;
}

std::istream &utilities::safe_getline(std::istream &is, std::string &t)
{
	//Thanks to http://stackoverflow.com/a/6089413
	t.clear();

	// The characters in the stream are read one-by-one using a std::streambuf.
	// That is faster than reading them one-by-one using the std::istream.
	// Code that uses streambuf this way must be guarded by a sentry object.
	// The sentry object performs various tasks,
	// such as thread synchronization and updating the stream state.

	std::istream::sentry se(is, true);
	std::streambuf *sb = is.rdbuf();

	for (;;) {
		int c = sb->sbumpc();
		switch (c) {
		case '\n':
			return is;
		case '\r':
			if (sb->sgetc() == '\n')
				sb->sbumpc();
			return is;
		case EOF:
			// Also handle the case when the last line has no line ending
			if (t.empty())
				is.setstate(std::ios::eofbit);
			return is;
		default:
			t += (char) c;
		}
	}
}

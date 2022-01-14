#pragma once

#include <glm/glm.hpp>
#include <vector>

class image {
private:
	size_t x_len;
	size_t y_len;
	std::vector<glm::vec3> pixels;

public:
	image(size_t x_len, size_t y_len);
	void setPixel(size_t x, size_t y, const glm::vec3 &pixel);
	void savePNG(const std::string &baseFilename);
	void saveHDR(const std::string &baseFilename);
};

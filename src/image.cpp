#include <string>
#include <stb_image_write.h>

#include "image.h"

image::image(size_t x_len, size_t y_len) :
	x_len(x_len),
	y_len(y_len),
	pixels(((long long) x_len) * y_len)
{}

void image::setPixel(size_t x, size_t y, const glm::vec3 &pixel)
{
	assert(x >= 0 && y >= 0 && x < x_len && y < y_len);
	pixels[y * x_len + x] = pixel;
}

void image::savePNG(const std::string &baseFilename)
{
	std::vector<unsigned char>bytes(3 * x_len * y_len);

	for (size_t y = 0; y < y_len; y++) {
		for (size_t x = 0; x < x_len; x++) {
			size_t i = y * x_len + x;
			glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
			bytes[3 * i + 0] = (unsigned char) pix.x;
			bytes[3 * i + 1] = (unsigned char) pix.y;
			bytes[3 * i + 2] = (unsigned char) pix.z;
		}
	}

	std::string filename = baseFilename + ".png";
	stbi_write_png(filename.c_str(), x_len, y_len, 3, bytes.data(), x_len * 3);
	printf("Saved %s.\n", filename.c_str());
}

void image::saveHDR(const std::string &baseFilename)
{
	std::string filename = baseFilename + ".hdr";
	stbi_write_hdr(filename.c_str(), x_len, y_len, 3, (const float *) pixels.data());
	printf("Saved %s.\n", filename.c_str());
}

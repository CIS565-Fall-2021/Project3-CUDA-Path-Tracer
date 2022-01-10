#include "main.h"
#include "preview.h"
#include <cstring>

using glm::vec3;

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static vec3 cammove;

float zoom, theta, phi;
vec3 cameraPosition;
vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char **argv)
{
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char *sceneFile = argv[1];

	// Load scene file
	scene = new Scene(sceneFile);

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera &cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	vec3 view = cam.view;
	vec3 up = cam.up;
	vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	vec3 viewXZ = vec3(view.x, 0.0f, view.z);
	vec3 viewZY = vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	// Initialize CUDA and GL components
	init();

	// GLFW main loop
	mainLoop();

	return 0;
}

void saveImage()
{
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			vec3 pix = renderState->image[index];
			img.setPixel(width - 1 - x, y, vec3(pix) / samples);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
	if (camchanged) {
		iteration = 0;
		Camera &cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		vec3 v = cam.view;
		vec3 u = vec3(0, 1, 0);//glm::normalize(cam.up);
		vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		pathtraceFree();
		pathtraceInit(scene);
	}

	if (iteration < renderState->iterations) {
		uchar4 *pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void **) &pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		pathtrace(pbo_dptr, frame, iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	} else {
		saveImage();
		pathtraceFree();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	Camera &cam = scene->state.camera;
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			saveImage();
			break;
		case GLFW_KEY_SPACE:
			camchanged = true;
			cam.lookAt = ogLookAt;
			break;
		case GLFW_KEY_UP:
			/* increase focal distance */
			camchanged = true;
			cam.focus_len = glm::max(0.0f, cam.focus_len + 0.2f);
			printf("focal length: %f\n", scene->state.camera.focus_len);
			break;
		case GLFW_KEY_DOWN:
			/* decrease focal distance */
			camchanged = true;
			cam.focus_len = glm::max(0.0f, cam.focus_len - 0.2f);
			printf("focal length: %f\n", scene->state.camera.focus_len);
			break;
		case GLFW_KEY_RIGHT:
			/* increase lens radius */
			camchanged = true;
			cam.lens_radius= glm::max(0.0f, cam.lens_radius + 0.1f);
			printf("lens radius: %f\n", scene->state.camera.lens_radius);
			break;
		case GLFW_KEY_LEFT:
			/* decrease lens radius */
			camchanged = true;
			cam.lens_radius= glm::max(0.0f, cam.lens_radius - 0.1f);
			printf("lens radius: %f\n", scene->state.camera.lens_radius);
			break;
		}

	}
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow *window, double xpos, double ypos)
{
	if (xpos == lastX || ypos == lastY)
		return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	} else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	} else if (middleMousePressed) {
		renderState = &scene->state;
		Camera &cam = renderState->camera;
		vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
		cam.lookAt += (float) (ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}

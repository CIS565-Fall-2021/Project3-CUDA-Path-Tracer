#include "main.h"
#include "preview.h"
#include <cstring>
static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

static double aperture = 0.2;
static double focus_dist = 3.0;
float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
RenderState* renderState;
int iteration;

int width;
int height;


void BuildBVH(Geom& objGeom)
{
	objGeom.Host_BVH = new float[14];
	BVH a(objGeom.triangleCount, objGeom.Host_Triangle_points_normals);
	a.GetBounds(objGeom.Host_BVH, objGeom);
}


Polygon LoadOBJ(const char* file, char* polyName, Geom& objGeom)
{
	Polygon p(polyName);
	const char* filepath = file;
	std::vector<tinyobj::shape_t> shapes; std::vector<tinyobj::material_t> materials;
	std::string errors = tinyobj::LoadObj(shapes, materials, filepath);
	std::cout << errors << std::endl;
	if (errors.size() == 0)
	{
		int min_idx = 0;
		//Read the information from the vector of shape_ts
		for (unsigned int i = 0; i < shapes.size(); i++)
		{
			std::vector<glm::vec4> pos, nor;
			std::vector<glm::vec2> uv;
			std::vector<float>& positions = shapes[i].mesh.positions;
			std::vector<float>& normals = shapes[i].mesh.normals;
			std::vector<float>& uvs = shapes[i].mesh.texcoords;
			for (unsigned int j = 0; j < positions.size() / 3; j++)
			{
				pos.push_back(glm::vec4(positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2], 1));
			}
			for (unsigned int j = 0; j < normals.size() / 3; j++)
			{
				nor.push_back(glm::vec4(normals[j * 3], normals[j * 3 + 1], normals[j * 3 + 2], 0));
			}
			for (unsigned int j = 0; j < uvs.size() / 2; j++)
			{
				uv.push_back(glm::vec2(uvs[j * 2], uvs[j * 2 + 1]));
			}
			for (unsigned int j = 0; j < pos.size(); j++)
			{
				p.AddVertex(Vertex(pos[j], glm::vec3(255, 255, 255), nor[j], uv[j]));
			}

			std::vector<unsigned int> indices = shapes[i].mesh.indices;
			for (unsigned int j = 0; j < indices.size(); j += 3)
			{
				Triangle t;
				t.m_indices[0] = indices[j] + min_idx;
				t.m_indices[1] = indices[j + 1] + min_idx;
				t.m_indices[2] = indices[j + 2] + min_idx;
				p.AddTriangle(t);
			}

			min_idx += pos.size();
		}
	}
	else
	{
		//An error loading the OBJ occurred!
		std::cout << errors << std::endl;
	}

	objGeom.triangleCount = p.m_tris.size();
	objGeom.Host_Triangle_points_normals = new glm::vec4[6 * objGeom.triangleCount];
	for (int i = 0; i < objGeom.triangleCount; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			//Geom.Triangle.vertex = Polygon.vertex[triangle[i].index[j]]
			glm::vec4 vertPos = p.m_verts[p.m_tris[i].m_indices[j]].m_pos;
			glm::vec4 vertNormal = p.m_verts[p.m_tris[i].m_indices[j]].m_normal;
			objGeom.Host_Triangle_points_normals[(6 * i) + 2 * j] = vertPos;
			objGeom.Host_Triangle_points_normals[(6 * i ) + 2 * j + 1] = vertNormal;
		}
	}
	BuildBVH(objGeom);
	return p;
}



//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];

	// Load scene file
	scene = new Scene(sceneFile);

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	cam.aperture = aperture;
	cam.focus_dist = focus_dist;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;
	cam.focus_dist = (cam.position - cam.lookAt).length();
	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);
	const char* filepathWahoo = "D:/GitHub/CIS565/Project3-CUDA-Path-Tracer/scenes/wahoo.obj";
	const char* filepathCube = "D:/GitHub/CIS565/Project3-CUDA-Path-Tracer/scenes/cube.obj";
	Polygon p;
	char* filename = "wahoo";

	for (int i = 0; i < scene->geoms.size(); i++)
	{
		if ((int)scene->geoms[i].type == 2)
		{
			p = LoadOBJ(filepathCube, filename, scene->geoms[i]);

		}
	}

	   // Initialize CUDA and GL components
	   init();

	   // GLFW main loop
	   mainLoop();

	return 0;
}


void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
			img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
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

void runCuda() {
	if (camchanged) {
		iteration = 0;
		Camera& cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;
		//cam.focus_dist = (cam.position - cam.lookAt).length();
		cam.focus_dist = focus_dist;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;
		SetCacheState(false);
	}
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		pathtraceFree();
		pathtraceInit(scene);
	}

	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		pathtrace(pbo_dptr, frame, iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	else {
		saveImage();
		pathtraceFree();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
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
			renderState = &scene->state;
			Camera& cam = renderState->camera;
			cam.lookAt = ogLookAt;
			break;
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
		cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}

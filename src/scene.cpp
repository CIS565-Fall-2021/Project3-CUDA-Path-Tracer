#include <iostream>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include "scene.h"
#include "utilities.h"


using std::string;
using std::vector;
using std::stoi;
using std::stof;

using utilities::safe_getline;
using glm::vec3;


Scene::Scene(string filename)
{
	printf("Reading scene from %s ...\n\n", filename.c_str());
	
	fp_in.open(filename);
	if (!fp_in.is_open()) {
		fprintf(stderr, "Error reading from file - aborting!\n");
		exit(EXIT_FAILURE);
	}

	while (fp_in.good()) {
		string line;
		safe_getline(fp_in, line);
		if (!line.empty()) {
			vector<string> tokens = utilities::tokenize_string(line);
			if (tokens[0] == "MATERIAL")
				loadMaterial(tokens[1]);
			if (tokens[0] == "OBJECT")
				loadGeom(tokens[1]);
			if (tokens[0] == "CAMERA")
				loadCamera();
		}
	}
}

int Scene::loadGeom(string objectid)
{
	int id = stoi(objectid);
	if (id != geoms.size()) {
		printf("ERROR: OBJECT ID does not match expected number of geoms\n");
		return -1;
	}

	printf("Loading Geom %d...\n", id);
	Geom newGeom;
	string line;

	//load object type
	safe_getline(fp_in, line);
	if (!line.empty() && fp_in.good()) {
		if (line == "sphere") {
			printf("Creating new sphere...\n");
			newGeom.type = GeomType::SPHERE;
		}
		if (line == "cube") {
			printf("Creating new cube...\n");
			newGeom.type = GeomType::CUBE;
		}
	}

	//link material
	safe_getline(fp_in, line);
	if (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilities::tokenize_string(line);
		newGeom.materialid = stoi(tokens[1]);
		printf("Connecting Geom %s to Material %d...\n", objectid.c_str(), newGeom.materialid);
	}

	//load transformations
	safe_getline(fp_in, line);
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilities::tokenize_string(line);

		//load tranformations
		if (tokens[0] == "TRANS")
			newGeom.translation = vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
		if (tokens[0] == "ROTAT")
			newGeom.rotation = vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
		if (tokens[0] == "SCALE") 
			newGeom.scale = vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));

		safe_getline(fp_in, line);
	}

	newGeom.transform = utilities::make_transform_matrix(
		newGeom.translation, newGeom.rotation, newGeom.scale);
	newGeom.inverseTransform = glm::inverse(newGeom.transform);
	newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

	geoms.push_back(newGeom);
	printf("\n");
	return 1;
}

int Scene::loadCamera()
{
	printf("Loading Camera ...\n");
	RenderState &state = this->state;
	Camera &camera = state.camera;
	float fovy;

	//load static properties
	for (int i = 0; i < 5; i++) {
		string line;
		safe_getline(fp_in, line);
		vector<string> tokens = utilities::tokenize_string(line);
		if (tokens[0] == "RES") {
			camera.resolution.x = stoi(tokens[1]);
			camera.resolution.y = stoi(tokens[2]);
		} else if (tokens[0] == "FOVY") {
			fovy = stof(tokens[1]);
		} else if (tokens[0] == "ITERATIONS") {
			state.iterations = stoi(tokens[1]);
		} else if (tokens[0] == "DEPTH") {
			state.traceDepth = stoi(tokens[1]);
		} else if (tokens[0] == "FILE") {
			state.imageName = tokens[1];
		}
	}

	string line;
	safe_getline(fp_in, line);
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilities::tokenize_string(line);
		if (tokens[0] == "EYE") {
			camera.position = vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
		} else if (tokens[0] == "LOOKAT") {
			camera.lookAt = vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
		} else if (tokens[0] == "UP") {
			camera.up = vec3(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
		}

		safe_getline(fp_in, line);
	}

	//calculate fov based on resolution
	float yscaled = tan(fovy * (PI / 180));
	float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
	float fovx = (atan(xscaled) * 180) / PI;
	camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float) camera.resolution.x,
		2 * yscaled / (float) camera.resolution.y);

	camera.view = glm::normalize(camera.lookAt - camera.position);

	//set up render camera stuff
	int arraylen = camera.resolution.x * camera.resolution.y;
	state.image.resize(arraylen);
	std::fill(state.image.begin(), state.image.end(), vec3());

	printf("Loaded camera!\n\n");
	return 1;
}

int Scene::loadMaterial(string materialid)
{
	int id = stoi(materialid);
	if (id != materials.size()) {
		printf("ERROR: MATERIAL ID does not match expected number of materials\n");
		return -1;
	}
	printf("Loading Material %d...\n\n", id);
	Material newMaterial;

	//load static properties
	for (int i = 0; i < 7; i++) {
		string line;
		safe_getline(fp_in, line);
		vector<string> tokens = utilities::tokenize_string(line);
		if (tokens[0] == "RGB") {
			vec3 color(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
			newMaterial.color = color;
		}
		if (tokens[0] == "SPECEX")
			newMaterial.specular.exponent = stof(tokens[1]);
		if (tokens[0] == "SPECRGB") {
			vec3 specColor(stof(tokens[1]), stof(tokens[2]), stof(tokens[3]));
			newMaterial.specular.color = specColor;
		}
		if (tokens[0] == "REFL")
			newMaterial.hasReflective = stof(tokens[1]);
		if (tokens[0] == "REFR")
			newMaterial.hasRefractive = stof(tokens[1]);
		if (tokens[0] == "REFRIOR")
			newMaterial.indexOfRefraction = stof(tokens[1]);
		if (tokens[0] == "EMITTANCE")
			newMaterial.emittance = stof(tokens[1]);
	}
	materials.push_back(newMaterial);
	return 1;
}

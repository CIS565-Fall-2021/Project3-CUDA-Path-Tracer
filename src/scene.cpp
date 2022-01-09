#include <iostream>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "scene.h"
#include "utilities.h"


using std::string;
using std::vector;
using std::stoi;
using std::stof;

using utilities::safe_getline;
using glm::vec3;
using glm::vec4;


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

/* returns number of triangles added */
int Scene::load_obj(string inputfile, glm::vec3 &mincoords, glm::vec3 &maxcoords)
{
	/* adapted directly from https://github.com/tinyobjloader/tinyobjloader */
	using namespace tinyobj;

	ObjReader reader;
	ObjReaderConfig reader_conf;
	reader_conf.mtl_search_path = "../scenes/meshes/";

	if (!reader.ParseFromFile(inputfile, reader_conf) || !reader.Error().empty()) {
		fprintf(stderr, "TinyObjReader: %s\n", reader.Error().c_str());
		return 0;
	}
	if (!reader.Warning().empty())
		fprintf(stderr, "TinyObjReader: %s\n", reader.Warning().c_str());

	auto &attrib = reader.GetAttrib();
	auto &shapes = reader.GetShapes();
	auto &materials = reader.GetMaterials();

	auto &vertices = attrib.vertices;

	int size = tris.size();
	
	for (int i = 0; i < vertices.size(); i++) {
		printf("vertices[%d]: %f\n", i, vertices[i]);
	}

	/* an obj file can contain multiple shapes; iterate over them: */
	for (auto &s : shapes) {
		size_t index_offset = 0;
		auto &mesh = s.mesh;
		auto &indices = mesh.indices;

		printf("shape\n");

		/* iterate over faces of the mesh (specifically the #vertices in each face */
		for (unsigned char fv : mesh.num_face_vertices) {
			printf("face %d\n", fv);
			auto idx = indices[index_offset];


			vec3 shape_first_point = { vertices[3 * idx.vertex_index + 0],
				vertices[3 * idx.vertex_index + 1],
				vertices[3 * idx.vertex_index + 2] };

			printf("first point: (%f, %f, %f)\n", shape_first_point.x, shape_first_point.y, shape_first_point.z);
			printf("idx: %d\n", idx.vertex_index);

			/* iterate over vertices of each face */
			for (unsigned char v = 1; v < fv; v += 2) {
				auto idx1 = indices[index_offset + v];
				auto idx2 = indices[index_offset + v + 1];

				 this->tris.push_back({ /* create triangle based on the two vertices */
					shape_first_point,
					vec3(vertices[3 * idx1.vertex_index + 0],
						vertices[3 * idx1.vertex_index + 1],
						vertices[3 * idx1.vertex_index + 2]),
					vec3(vertices[3 * idx2.vertex_index + 0],
						vertices[3 * idx2.vertex_index + 1],
						vertices[3 * idx2.vertex_index + 2]),
				});
				printf("idx1: %d\n", idx1.vertex_index);
				printf("idx2: %d\n", idx2.vertex_index);
				printf("added triangle\n");
			}
			index_offset += fv;
		}
	}
	for (int i = 0; i < vertices.size()-2; i += 3) { /* for bounding box */
		mincoords.x = glm::min(mincoords.x, vertices[i]);
		mincoords.y = glm::min(mincoords.y, vertices[i+1]);
		mincoords.z = glm::min(mincoords.z, vertices[i+2]);
		maxcoords.x = glm::max(maxcoords.x, vertices[i]);
		maxcoords.y = glm::max(maxcoords.y, vertices[i+1]);
		maxcoords.z = glm::max(maxcoords.z, vertices[i+2]);
	}
	return tris.size() - size;
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
		if (line == "mesh") {
			printf("Creating new mesh...\n");
			newGeom.type = GeomType::MESH;
			newGeom.triangle_start = tris.size();

			string inputfile;
			safe_getline(fp_in, inputfile);
			newGeom.triangle_n = load_obj(inputfile, newGeom.mincoords, newGeom.maxcoords);

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

	newGeom.transform = utilities::make_transform_matrix(newGeom.translation, newGeom.rotation, newGeom.scale);
	newGeom.inverseTransform = glm::inverse(newGeom.transform);
	newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

	if (newGeom.type == GeomType::MESH) {
		/* correct the coordinates of all the triangles */
		for (int i = newGeom.triangle_start; i < newGeom.triangle_start + newGeom.triangle_n; i++) {
			Triangle &t = tris[i];
			t.v[0] = vec3(newGeom.transform * vec4(t.v[0], 1.f));
			t.v[1] = vec3(newGeom.transform * vec4(t.v[1], 1.f));
			t.v[2] = vec3(newGeom.transform * vec4(t.v[2], 1.f));
		}
		newGeom.mincoords = vec3(newGeom.transform * vec4(newGeom.mincoords, 1.f));
		newGeom.maxcoords = vec3(newGeom.transform * vec4(newGeom.maxcoords, 1.f));
	}

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

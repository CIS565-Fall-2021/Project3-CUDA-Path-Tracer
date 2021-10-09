#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/vector_angle.hpp>


#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation. Requires C++11
//#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"

const int DEPTH = 3;

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int countTriangles(const std::vector<tinyobj::shape_t> shapes) {
    int numTriangles = 0;

    for (size_t s = 0; s < shapes.size(); s++) {
        numTriangles += shapes[s].mesh.num_face_vertices.size();
        //std::cout << "shape " << s << " has " << shapes[s].mesh.num_face_vertices.size() << " triangles" << std::endl;
    }

    return numTriangles;
}

void buildTriangles(std::string filename, Geom& geom) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    //if (!reader.Warning().empty()) {
    //    std::cout << "TinyObjReader: " << reader.Warning();
    //}

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    
    geom.maxDims = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    geom.minDims = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    geom.host_triangles = new std::vector<Triangle>;

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon/triangles)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            Triangle t;
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                
                glm::vec3 vertex(vx, vy, vz);
                vertex = glm::vec3(geom.transform * glm::vec4(vertex, 1.f));

                for (int i = 0; i < 3; i++) {
                    if (vertex[i] < geom.minDims[i]) {
                        geom.minDims[i] = vertex[i];
                    }

                    if (geom.maxDims[i] < vertex[i]) {
                        geom.maxDims[i] = vertex[i];
                    }
                }

                t.points[v] = vertex;

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                    // don't need to do this thrice
                    t.normal = glm::vec3(nx, ny, nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                }

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
                t.color = glm::vec3(0.0, 1.0, 0.0);
            }
            index_offset += fv;
            // per-face material
            shapes[s].mesh.material_ids[f];
            geom.host_triangles->push_back(t);
        }
    }

    geom.numTriangles = geom.host_triangles->size(); //countTriangles(shapes);

    geom.maxDims += 0.1;
    geom.minDims -= 0.1;
    std::cout << glm::to_string(geom.minDims) << " " << glm::to_string(geom.maxDims) << std::endl;
}

void Node::computeChildLocations(glm::vec3* childLocations) {
    int i = 0;

    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                float childX = this->position.x + (x ? this->dimension.x / 2.f : 0);
                float childY = this->position.y + (y ? this->dimension.y / 2.f : 0);
                float childZ = this->position.z + (z ? this->dimension.z / 2.f : 0);

                childLocations[i++] = glm::vec3(childX, childY, childZ);
            }
        }
    }
}

// https://github.com/juj/MathGeoLib/blob/master/src/Geometry/Triangle.cpp#L697
bool Node::triangleIntersectionTest(Triangle& triangle) {
    glm::vec3 a = triangle.points[0];
    glm::vec3 b = triangle.points[1];
    glm::vec3 c = triangle.points[2];

    glm::vec3 tMin = glm::min(a, glm::min(b, c));
    glm::vec3 tMax = glm::max(a, glm::max(b, c));

    glm::vec3 maxPoint = this->position + this->dimension;
    glm::vec3 minPoint = this->position;
    if (tMin.x >= maxPoint.x || tMax.x <= minPoint.x
        || tMin.y >= maxPoint.y || tMax.y <= minPoint.y
        || tMin.z >= maxPoint.z || tMax.z <= minPoint.z)
        return false;

    glm::vec3 center = (minPoint + maxPoint) * 0.5f;
    glm::vec3 h = maxPoint - center;

    const glm::vec3 t[3] = { b - a, c - a, c - b };

    glm::vec3 ac = a - center;

    glm::vec3 n = glm::cross(t[0], t[1]);
    float s = glm::dot(n, ac);
    float r = glm::abs(glm::dot(h, glm::abs(n))); // Abs(h.Dot(n.Abs()));
    if (glm::abs(s) >= r)
        return false;

    const glm::vec3 at[3] = { glm::abs(t[0]), glm::abs(t[1]), glm::abs(t[2]) };

    glm::vec3 bc = b - center;
    glm::vec3 cc = c - center;

    // SAT test all cross-axes.
    // The following is a fully unrolled loop of this code, stored here for reference:

    // eX <cross> t[0]
    float d1 = t[0].y * ac.z - t[0].z * ac.y;
    float d2 = t[0].y * cc.z - t[0].z * cc.y;
    float tc = (d1 + d2) * 0.5f;
    r = glm::abs(h.y * at[0].z + h.z * at[0].y);
    if (r + glm::abs(tc - d1) < glm::abs(tc))
        return false;

    // eX <cross> t[1]
    d1 = t[1].y * ac.z - t[1].z * ac.y;
    d2 = t[1].y * bc.z - t[1].z * bc.y;
    tc = (d1 + d2) * 0.5f;
    r = glm::abs(h.y * at[1].z + h.z * at[1].y);
    if (r + glm::abs(tc - d1) < glm::abs(tc))
        return false;

    // eX <cross> t[2]
    d1 = t[2].y * ac.z - t[2].z * ac.y;
    d2 = t[2].y * bc.z - t[2].z * bc.y;
    tc = (d1 + d2) * 0.5f;
    r = glm::abs(h.y * at[2].z + h.z * at[2].y);
    if (r + glm::abs(tc - d1) < glm::abs(tc))
        return false;

    // eY <cross> t[0]
    d1 = t[0].z * ac.x - t[0].x * ac.z;
    d2 = t[0].z * cc.x - t[0].x * cc.z;
    tc = (d1 + d2) * 0.5f;
    r = glm::abs(h.x * at[0].z + h.z * at[0].x);
    if (r + glm::abs(tc - d1) < glm::abs(tc))
        return false;

    // eY <cross> t[1]
    d1 = t[1].z * ac.x - t[1].x * ac.z;
    d2 = t[1].z * bc.x - t[1].x * bc.z;
    tc = (d1 + d2) * 0.5f;
    r = glm::abs(h.x * at[1].z + h.z * at[1].x);
    if (r + glm::abs(tc - d1) < glm::abs(tc))
        return false;

    // eY <cross> t[2]
    d1 = t[2].z * ac.x - t[2].x * ac.z;
    d2 = t[2].z * bc.x - t[2].x * bc.z;
    tc = (d1 + d2) * 0.5f;
    r = glm::abs(h.x * at[2].z + h.z * at[2].x);
    if (r + glm::abs(tc - d1) < glm::abs(tc))
        return false;

    // eZ <cross> t[0]
    d1 = t[0].x * ac.y - t[0].y * ac.x;
    d2 = t[0].x * cc.y - t[0].y * cc.x;
    tc = (d1 + d2) * 0.5f;
    r = glm::abs(h.y * at[0].x + h.x * at[0].y);
    if (r + glm::abs(tc - d1) < glm::abs(tc))
        return false;

    // eZ <cross> t[1]
    d1 = t[1].x * ac.y - t[1].y * ac.x;
    d2 = t[1].x * bc.y - t[1].y * bc.x;
    tc = (d1 + d2) * 0.5f;
    r = glm::abs(h.y * at[1].x + h.x * at[1].y);
    if (r + glm::abs(tc - d1) < glm::abs(tc))
        return false;

    // eZ <cross> t[2]
    d1 = t[2].x * ac.y - t[2].y * ac.x;
    d2 = t[2].x * bc.y - t[2].y * bc.x;
    tc = (d1 + d2) * 0.5f;
    r = glm::abs(h.y * at[2].x + h.x * at[2].y);
    if (r + glm::abs(tc - d1) < glm::abs(tc))
        return false;

    // No separating axis exists, the AABB and triangle intersect.
    return true;
}

bool Node::contains(Triangle& triangle) {
    for (int i = 0; i < 3; i++) {
        glm::vec3 p = triangle.points[i];

        bool greaterThanMin = this->position.x <= p.x && this->position.y <= p.y && this->position.z <= p.z;
        bool lessThanMax    = p.x <= this->position.x + this->dimension.x && p.y <= this->position.y + this->dimension.y && p.z <= this->position.z + this->dimension.z;

        if (greaterThanMin && lessThanMax) {
            return true;
        }
    }

    return false;
}

void initializeTree(Geom& geom) {
    std::cout << "Building octree of size " << DEPTH << std::endl;

    Node root(geom.minDims, geom.maxDims - geom.minDims, 1, DEPTH == 0);
    glm::vec3 scale = root.dimension;
    glm::vec3 translation = root.position + scale / 2.f;
    glm::vec3 rotation(0, 0, 0);
    root.transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
    root.inverseTransform = glm::inverse(root.transform);
    root.invTranspose = glm::inverseTranspose(root.transform);

    std::vector<Node>* tree = new std::vector<Node>();
    (*tree).push_back(root);

    for (int d = 0; d < DEPTH; d++) {
        int startIndex = (pow(8, d) - 1) / 7;
        int endIndex = (pow(8, d + 1) - 1) / 7;
        bool leaf = d == (DEPTH - 1);
        std::cout << "Depth " << d << " from " << startIndex << " to " << endIndex << " and leaf " << leaf << std::endl;

        glm::vec3 newDim = (*tree)[startIndex].dimension / 2.f;
        for (int i = startIndex; i < endIndex; i++) {
            glm::vec3 childLocations[8];
            (*tree)[i].computeChildLocations(childLocations);
            
            for (int j = 0; j < 8; j++) {
                int childI = leaf ? 0 : ((*tree).size() * 8 + 1);
                Node node(childLocations[j], newDim, childI, leaf);
                //std::cout << "Created node " << (*tree).size() << " with pos " << glm::to_string(node.position) << " and dim " << glm::to_string(node.dimension) << std::endl;

                glm::vec3 scale = node.dimension;
                glm::vec3 translation = node.position + scale / 2.f;

                node.transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
                node.inverseTransform = glm::inverse(node.transform);
                node.invTranspose = glm::inverseTranspose(node.transform);

                (*tree).push_back(node);
            }
        }
    }

    geom.host_tree = tree;
}

void insertTriangle(Geom& geom, int nodeIndex, int triangleIndex) {
    if ((*geom.host_tree)[nodeIndex].contains((*geom.host_triangles)[triangleIndex])) {
    //if ((*geom.host_tree)[nodeIndex].triangleIntersectionTest((*geom.host_triangles)[triangleIndex])) {
        if ((*geom.host_tree)[nodeIndex].leaf) {
            (*geom.host_tree)[nodeIndex].host_triangles->push_back((*geom.host_triangles)[triangleIndex]);
            (*geom.host_tree)[nodeIndex].numTriangles++;
        } else {
            for (int i = (*geom.host_tree)[nodeIndex].childrenStartIndex; i < (*geom.host_tree)[nodeIndex].childrenStartIndex + 8; i++) {
                insertTriangle(geom, i, triangleIndex);
            }
        }
    }
}

void insertTriangles(Geom& geom) {
    for (int i = 0; i < geom.numTriangles; i++) {
        //std::cout << "Inserting triangle " << i << std::endl;
        insertTriangle(geom, 0, i);
    }
}

void buildOctree(Geom& geom) {
    initializeTree(geom);
    insertTriangles(geom);
}

int Scene::loadGeom(string objectid) {
    std::cout << "Size of Triangle " << sizeof(Triangle) << std::endl;
    std::cout << "Size of Node " << sizeof(Node) << std::endl;
    glm::vec3 a(0, 1, 0);
    glm::vec3 b(0, 0, -1);

    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        std::string filename;
        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else {
                cout << "Creating new obj..." << endl;
                cout << line.c_str() << endl;
                newGeom.type = BB;
                filename = line.c_str();
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (newGeom.type == BB) {
            //newGeom.scale = newGeom.maxDims - newGeom.minDims;
            //newGeom.translation = newGeom.minDims + newGeom.scale / 2.f;

            buildTriangles(filename, newGeom);
            buildOctree(newGeom);
        }

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadObjFile() {
    return -1;
}
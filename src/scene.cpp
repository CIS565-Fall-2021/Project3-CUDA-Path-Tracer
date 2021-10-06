#define TINYOBJLOADER_IMPLEMENTATION // define this in only one .cc
#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"

#define USE_BB true

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;

    // initialize triangle ptr list
    this->trianglePtrs = std::vector<std::unique_ptr<std::vector<Triangle>>>();
    //this->trianglePtrs = vector<unique_ptr<vector<glm::vec3>>>();

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

bool Scene::loadObj(string filename, Geom& geom) {

    // bounding box 
    geom.boundingBox.minX = 10000000.f;
    geom.boundingBox.maxX = 0.f;
    geom.boundingBox.minY = 10000000.f;
    geom.boundingBox.maxY = 0.f;
    geom.boundingBox.minZ = 10000000.f;
    geom.boundingBox.maxZ = 0.f;

    std::vector<Triangle> triangles;

    // read in mesh and construct bounding box
    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true;
    tinyobj::ObjReader reader;

    // read from file
    if (!reader.ParseFromFile(filename, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }
    
    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    // loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {

        // loop over faces
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

            Triangle t;
            std::vector<glm::vec3> pts;
            std::vector<glm::vec3> nors;
            //loop over verts
            for (size_t v = 0; v < 3; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                    nors.push_back(glm::vec3(nx, ny, nz));
                }

                // check against bounding box bounds
                geom.boundingBox.minX = min(geom.boundingBox.minX, vx);
                geom.boundingBox.maxX = max(geom.boundingBox.maxX, vx);
                geom.boundingBox.minY = min(geom.boundingBox.minY, vy);
                geom.boundingBox.maxY = max(geom.boundingBox.maxY, vy);
                geom.boundingBox.minZ = min(geom.boundingBox.minZ, vz);
                geom.boundingBox.maxZ = max(geom.boundingBox.maxZ, vz);

                pts.push_back(glm::vec3(vx, vy, vz));
            }
            index_offset += 3;
            t.p1 = glm::vec3(pts[0].x, pts[0].y, pts[0].z);
            t.p2 = glm::vec3(pts[1].x, pts[1].y, pts[1].z);
            t.p3 = glm::vec3(pts[2].x, pts[2].y, pts[2].z);
            t.n1 = glm::vec3(nors[0].x, nors[0].y, nors[0].z);
            t.n2 = glm::vec3(nors[1].x, nors[1].y, nors[1].z);
            t.n3 = glm::vec3(nors[2].x, nors[2].y, nors[2].z);
            triangles.push_back(t);
        }
    }

    // save unique ptr to triangle vector in scene
    this->trianglePtrs.push_back(make_unique<vector<Triangle>> (triangles));
    
    // get raw ptr and save to geom
    geom.triangles = &this->trianglePtrs[this->trianglePtrs.size() - 1].get()->front();
    geom.numTriangles = triangles.size();

    return true;
}

void calcBoundingBox(Geom& geom) {
    // calc scale of bounding box in mesh's untransformed space
    glm::vec3 bbScale(geom.boundingBox.maxX - geom.boundingBox.minX,
                      geom.boundingBox.maxY - geom.boundingBox.minY,
                      geom.boundingBox.maxZ - geom.boundingBox.minZ);
    // bb scale assumes we are scaling uniformly -- translate so that it's placed correctly
    glm::vec3 unitBox(0.5, 0.5, 0.5);
    glm::vec3 bbTop(geom.boundingBox.maxX, geom.boundingBox.maxY, geom.boundingBox.maxZ);
    glm::vec3 bbTrans = bbTop - unitBox * bbScale;

    // translate/scale resulting bounding box
    bbScale *= geom.scale;
    bbTrans += geom.translation;

    geom.boundingBox.transform = utilityCore::buildTransformationMatrix(
        bbTrans, geom.rotation, bbScale);
    geom.boundingBox.inverseTransform = glm::inverse(geom.boundingBox.transform);
    geom.boundingBox.invTranspose = glm::inverseTranspose(geom.boundingBox.transform);
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strcmp(line.c_str(), "mesh") == 0) {
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //link filename for obj
        if (newGeom.type == MESH) {
            utilityCore::safeGetline(fp_in, line);
            if (!line.empty() && fp_in.good()) {
                vector<string> tokens = utilityCore::tokenizeString(line);

                if (strcmp(tokens[0].c_str(), "FILENAME") == 0) {
                    string filename = tokens[1].c_str();
                    std::cout << "Reading obj file from " << filename << " ..." << endl;

                    if (!loadObj(filename, newGeom)) return -1;
                }
            }
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        // if mesh, set bounding box transformations
        if (newGeom.type == MESH && USE_BB) {
            calcBoundingBox(newGeom);
        }
        
        //if (newGeom.type != MESH) {
            geoms.push_back(newGeom);
        //}
        
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

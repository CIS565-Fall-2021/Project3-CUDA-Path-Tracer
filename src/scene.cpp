#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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
            } 
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } 
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "SETTINGS") == 0) {
                loadSceneSettings();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadMesh(string filePath, Geom* geom) {
    // Much of the following was adapted from tinyobjloader's example code:
    // https://github.com/tinyobjloader/tinyobjloader
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filePath, reader_config)) {
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

    // set up our mesh geom struct
    long totalNumTris = 0;
    for (int s = 0; s < shapes.size(); s++) {
        totalNumTris += shapes[s].mesh.num_face_vertices.size();
    }
    geom->tris = new Tri[totalNumTris];
    geom->numTris = totalNumTris;

    // Loop over shapes
    int iTri = 0;
    glm::vec3 bboxMin = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3 bboxMax = glm::vec3(std::numeric_limits<float>::min());
    glm::vec3 newVert;
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            Tri T;

            // for each face loaded, place that vertex and vertexNormal data
            // into a tri in our new geometry
            glm::vec3* vertVecs[3] = { &T.v1,
                                    &T.v2,
                                    &T.v3 };
            glm::vec3* normVecs[3] = { &T.n1,
                                    &T.n2,
                                    &T.n3 };
            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                float vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                float vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                float vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                // create new vert with these coordinates
                newVert = glm::vec3(vx, vy, vz);
                // keep track of the min x,y, and z
                bboxMax = glm::max(bboxMax, newVert);
                bboxMin = glm::min(bboxMin, newVert);
                // add new vert
                *vertVecs[v] = newVert;

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    float nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    float ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    float nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    *normVecs[v] = glm::vec3(nx, ny, nz);
                }

            }
            geom->tris[iTri] = T;
            iTri++;
            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }

    // create a mesh bounding box
    // why a mesh instead of a boxGeom? because nested geometry and 
    // manipulating transforms does not sound at all enjoyable
    float minX = bboxMin.x;
    float minY = bboxMin.y;
    float minZ = bboxMin.z;
    float maxX = bboxMax.x;
    float maxY = bboxMax.y;
    float maxZ = bboxMax.z;

    // create vertices
    glm::vec3 bboxVerts[9];
    // ignore slot 0, it throws off my math
    bboxVerts[1] = glm::vec3(maxX, maxY, minZ);
    bboxVerts[2] = glm::vec3(maxX, minY, minZ);
    bboxVerts[3] = glm::vec3(maxX, maxY, maxZ);
    bboxVerts[4] = glm::vec3(maxX, minY, maxZ);
    bboxVerts[5] = glm::vec3(minX, maxY, minZ);
    bboxVerts[6] = glm::vec3(minX, minY, minZ);
    bboxVerts[7] = glm::vec3(minX, maxY, maxZ);
    bboxVerts[8] = glm::vec3(minX, minY, maxZ);

    // create tris from those vertices
    // pre-calculated based on the faces
    // in a cube obj I created
    int3 bboxVertIndices[12] = {make_int3(5, 3, 1),
								make_int3(3, 8, 4),
								make_int3(7, 6, 8),
								make_int3(2, 8, 6),
								make_int3(1, 4, 2),
								make_int3(5, 2, 6),
								make_int3(5, 7, 3),
								make_int3(3, 7, 8),
								make_int3(7, 5, 6),
								make_int3(2, 4, 8),
								make_int3(1, 3, 4),
								make_int3(5, 1, 2)};

    // use the indices to set verts in the tris of
    // out bounding box
    geom->boundingBox = new Tri[12];
    for (int vi = 0; vi < 12; vi++) {
        Tri tri;
        tri.v1 = bboxVerts[bboxVertIndices[vi].x];
        tri.v2 = bboxVerts[bboxVertIndices[vi].y];
        tri.v3 = bboxVerts[bboxVertIndices[vi].z];

        geom->boundingBox[vi] = tri;
    }


    return 0;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

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
            else if (line.find(".obj") != string::npos) {
                cout << "loading mesh " << line << endl;
                loadMesh(line, &newGeom);
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
    for (int i = 0; i < 8; i++) {
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
        }else if (strcmp(tokens[0].c_str(), "FOCALDIST") == 0) {
            camera.focalDist = atof(tokens[1].c_str());
        }else if (strcmp(tokens[0].c_str(), "APERTURE") == 0) {
            camera.aperture = atof(tokens[1].c_str());
        }else if (strcmp(tokens[0].c_str(), "LENSRAD") == 0) {
            camera.lensRadius = atof(tokens[1].c_str());
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
        for (int i = 0; i < 8; i++) {
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
            } else if (strcmp(tokens[0].c_str(), "FRESNELPOW") == 0) {
                newMaterial.FresnelPower = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadSceneSettings() {
    cout << "Loading Scene Settings ..." << endl;
    RenderState &state = this->state;

    //load static properties
    for (int i = 0; i < 4; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "SORT_MATERIALS") == 0) {
            state.sortMaterials = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "CACHE_BOUNCE") == 0) {
            state.cacheFirstBounce = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "USE_DOF") == 0) {
            state.useDOF = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "ANTIALIAS") == 0) {
            state.antialias = atoi(tokens[1].c_str());
        }
    }
    return 1;
}

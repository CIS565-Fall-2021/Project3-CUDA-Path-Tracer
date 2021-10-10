#define TINYOBJLOADER_IMPLEMENTATION
#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"

#define BOUNDING_BOX 0

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

//based on example tinyobj code
int Scene::loadOBJ(string filename, std::vector<Geom>& triangles, int materialID, glm::mat4 transform) {
    //tiny obj vars
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool loaded = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

    //warning + error checks
    if (!(warn.empty())) {
        std::cout << "TinyObjReader: " << warn << std::endl;
    }

    if (!(err.empty())) {
        std::cout << "TinyObjReader: " << err << std::endl;
    }

    if (!loaded) return -1;

    //shapes loop
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t offset = 0;
        //face loop
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            std::vector<glm::vec3> f_pos;
            std::vector<glm::vec3> f_nors;
            std::vector<glm::vec2> f_uvs;

            for (int v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[offset + v];
                //vertices
                tinyobj::real_t vtxX = attrib.vertices[3 * (size_t)idx.vertex_index + 0]; //cast to size_t?
                tinyobj::real_t vtxY = attrib.vertices[3 * (size_t)idx.vertex_index + 1];
                tinyobj::real_t vtxZ = attrib.vertices[3 * (size_t)idx.vertex_index + 2];
                f_pos.push_back(glm::vec3(vtxX, vtxY, vtxZ));

                //normals
                tinyobj::real_t nX = attrib.normals[3 * (size_t)idx.normal_index + 0]; 
                tinyobj::real_t nY = attrib.normals[3 * (size_t)idx.normal_index + 1];
                tinyobj::real_t nZ = attrib.normals[3 * (size_t)idx.normal_index + 2];
                f_nors.push_back(glm::vec3(nX, nY, nZ));

                //uvs (not sure if need these?)
                tinyobj::real_t tX = attrib.texcoords[3 * (size_t)idx.texcoord_index + 0];
                tinyobj::real_t tY = attrib.texcoords[3 * (size_t)idx.texcoord_index + 1];
                f_uvs.push_back(glm::vec2(tX, tY));
            }
            offset += fv;

            //form triangles
            for (int i = 1; i < f_pos.size() - 1; i++) {
                Geom triangle;
                triangle.type = TRIANGLE;
                //pos
                triangle.triangle.pt1.pos = f_pos[0];
                triangle.triangle.pt2.pos = f_pos[i];
                triangle.triangle.pt3.pos = f_pos[i + 1];

                //nor
                triangle.triangle.pt1.nor = f_nors[0];
                triangle.triangle.pt2.nor = f_nors[i];
                triangle.triangle.pt3.nor = f_nors[i + 1];

                //uv
                triangle.triangle.pt1.uv = f_uvs[0];
                triangle.triangle.pt2.uv = f_uvs[i];
                triangle.triangle.pt3.uv = f_uvs[i + 1];

                triangle.transform = transform;
                triangle.materialid = materialID;
                triangle.inverseTransform = glm::inverse(transform);
                triangle.invTranspose = glm::inverseTranspose(transform);

                //add triangle
                triangles.push_back(triangle);
            }
        }
    }

    //OLD CODE
    //geom.boundingBox.minX = 100000000.f;
    //geom.boundingBox.minY = 100000000.f;
    //geom.boundingBox.minZ = 100000000.f;
    //geom.boundingBox.maxX = 0.f;
    //geom.boundingBox.maxY = 0.f;
    //geom.boundingBox.maxZ = 0.f;

    //std::vector<Triangle> triangles;
    //tinyobj::ObjReaderConfig config;
    //config.triangulate = true;
    //tinyobj::ObjReader reader;

    //if (!reader.ParseFromFile(filename, config)) {
    //    if (!reader.Error().empty()) {
    //        std::cerr << "TinyObjReader: " << reader.Error();
    //        return -1;
    //    }
    //    exit(1);
    //}

    //if (!reader.Warning().empty()) {
    //    std::cout << "TinyObjReader: " << reader.Warning();
    //}

    //auto& shapes = reader.GetShapes();
    //auto& attrib = reader.GetAttrib();

    //for (size_t s = 0; s < shapes.size(); s++) {

    //    size_t offset = 0;
    //    for (size_t fv = 0; fv < shapes[s].mesh.num_face_vertices.size(); fv++) {

    //        Triangle t;
    //        std::vector<glm::vec3> pts;
    //        std::vector<glm::vec3> nors;

    //        for (size_t vtx = 0; vtx < 3; vtx++) {
                /*tinyobj::index_t idx = shapes[s].mesh.indices[offset + vtx];
                tinyobj::real_t vtxX = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vtxY = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vtxZ = attrib.vertices[3 * size_t(idx.vertex_index) + 2];*/

    //            // Check for normal data
    //            if (idx.normal_index >= 0) {
    //                tinyobj::real_t normX = attrib.normals[3 * size_t(idx.normal_index) + 0];
    //                tinyobj::real_t normY = attrib.normals[3 * size_t(idx.normal_index) + 1];
    //                tinyobj::real_t normZ = attrib.normals[3 * size_t(idx.normal_index) + 2];

    //                nors.push_back(glm::vec3(normX, normY, normZ));
    //            }

    //            // check with bounding box
    //            geom.boundingBox.minX = min(geom.boundingBox.minX, vtxX);
    //            geom.boundingBox.maxX = max(geom.boundingBox.maxX, vtxX);
    //            geom.boundingBox.minY = min(geom.boundingBox.minY, vtxY);
    //            geom.boundingBox.maxY = max(geom.boundingBox.maxY, vtxY);
    //            geom.boundingBox.minZ = min(geom.boundingBox.minZ, vtxZ);
    //            geom.boundingBox.maxZ = max(geom.boundingBox.maxZ, vtxZ);

    //            pts.push_back(glm::vec3(vtxX, vtxY, vtxZ));
    //        }

    //        t.p1 = glm::vec3(pts[0].x, pts[0].y, pts[0].z);
    //        t.p2 = glm::vec3(pts[1].x, pts[1].y, pts[1].z);
    //        t.p3 = glm::vec3(pts[2].x, pts[2].y, pts[2].z);
    //        t.n1 = glm::vec3(nors[0].x, nors[0].y, nors[0].z);
    //        t.n2 = glm::vec3(nors[1].x, nors[1].y, nors[1].z);
    //        t.n3 = glm::vec3(nors[2].x, nors[2].y, nors[2].z);
    //        triangles.push_back(t);
    //        offset = offset + 3;
    //    }
    //}

    //this->trianglePtrs.push_back(make_unique<vector<Triangle>>(triangles));
    //geom.triangles = &this->trianglePtrs[this->trianglePtrs.size() - 1].get()->front();
    //geom.numTriangles = triangles.size();

    return 1;
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
        
        //obj loading vars
        bool objFlag = false;
        std::string filename = "";
        int materialID = 0;
        std::vector<Geom> triangles;
        glm::mat4 transform;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            } else if (strcmp(line.c_str(), ".obj") != NULL) {
                cout << "Loading OBJ..." << endl;
                objFlag = true;
                filename = line;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            materialID = atoi(tokens[1].c_str()); //want to use this material for obj
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
        transform = newGeom.transform; //want to use this transform for obj

        if (objFlag) {
            std::cout << "OBJ LOADING" << std::endl;
            loadOBJ(filename, triangles, materialID, transform);
            for (int i = 0; i < triangles.size(); i++) {
                geoms.push_back(triangles[i]);
            }
        }
        else {
            geoms.push_back(newGeom);
        }

//#if BOUNDING_BOX
//        if (newGeom.type == MESH) {
//            glm::vec3 scale(geom.boundingBox.maxX - geom.boundingBox.minX, geom.boundingBox.maxY - geom.boundingBox.minY, geom.boundingBox.maxZ - geom.boundingBox.minZ);
//            glm::vec3 unit(0.5, 0.5, 0.5);
//            glm::vec3 top(geom.boundingBox.maxX, geom.boundingBox.maxY, geom.boundingBox.maxZ);
//            glm::vec3 transform = top - unit * scale;
//            scale *= geom.scale;
//            transform += geom.translation;
//            geom.boundingBox.transform = utilityCore::buildTransformationMatrix(transform, geom.rotation, scale);
//            geom.boundingBox.inverseTransform = glm::inverse(geom.boundingBox.transform);
//            geom.boundingBox.invTranspose = glm::inverseTranspose(geom.boundingBox.transform);
//        }
//#endif

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
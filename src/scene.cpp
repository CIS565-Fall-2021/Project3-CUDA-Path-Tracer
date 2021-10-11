#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "utilities.h"
#include <limits>

#include "options.h"

#include "tiny_gltf.h"
#include "tiny_obj_loader.h"

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;

    if(filename.substr(filename.size() - 4) == "gltf"){
      cout << " as GLTF " << endl;
      cout << " " << endl;

        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
      std::string err;
      std::string warn;

      bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

      if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
      }

      if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
      }

      if (!ret) {
        printf("Failed to parse glTF\n");
        throw;
      }

      cout << "GLTF read successfully" << endl;
      std::cout << "loaded glTF file has:\n"
                << model.accessors.size() << " accessors\n"
                << model.animations.size() << " animations\n"
                << model.buffers.size() << " buffers\n"
                << model.bufferViews.size() << " bufferViews\n"
                << model.materials.size() << " materials\n"
                << model.meshes.size() << " meshes\n"
                << model.nodes.size() << " nodes\n"
                << model.textures.size() << " textures\n"
                << model.images.size() << " images\n"
                << model.skins.size() << " skins\n"
                << model.samplers.size() << " samplers\n"
                << model.cameras.size() << " cameras\n"
                << model.scenes.size() << " scenes\n"
                << model.lights.size() << " lights\n";


      throw;
    }else if(filename.substr(filename.size() - 3) == "obj"){

      tinyobj::ObjReaderConfig reader_config;
      reader_config.mtl_search_path = utilityCore::dirnameOf(filename);

      tinyobj::ObjReader reader;

      if (!reader.ParseFromFile(filename, reader_config)) {
        if (!reader.Error().empty()) {
          std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
      }

      if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
      }

      loadGeomOBJ(reader);
      loadCameraOBJ(reader);
      loadMaterialOBJ(reader);

        //throw;


    }else {
      cout << " " << endl;


      char *fname = (char *) filename.c_str();
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

    //throw;
}



int Scene::loadGeomOBJ(tinyobj::ObjReader reader) {

  obj_maxs = glm::vec3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
  obj_mins = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  auto& shapes = reader.GetShapes();
  auto& attrib = reader.GetAttrib();
  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over faces(polygon)
    const int num_tris = shapes[s].mesh.num_face_vertices.size();
    // std::cout << "FOUND NUM TRIS: " << num_tris << '\n';
    auto geom_maxs = glm::vec3(std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min());
    auto geom_mins = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    Geom geom;
    geom.type = MESH;
    geom.num_tris = num_tris;
    std::vector<Triangle> tris;
    if (shapes[s].mesh.num_face_vertices.size() < 1) continue;
    
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

      if(fv != 3){
        std::cout << "NUM VERTS IN FACE NOT EQUAL TO THREE!!" << '\n';
        throw;
      }


      // Loop over vertices in the face.

      
      
      Triangle tri;
      

      tinyobj::index_t idx;


      // access to vertex
      idx = shapes[s].mesh.indices[index_offset + 0];
      tri.verts[0].x = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
      tri.verts[0].y = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
      tri.verts[0].z = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
      if (idx.normal_index >= 0) {
        tri.norms[0].x = attrib.normals[3 * size_t(idx.normal_index) + 0];
        tri.norms[0].y = attrib.normals[3 * size_t(idx.normal_index) + 1];
        tri.norms[0].z = attrib.normals[3 * size_t(idx.normal_index) + 2];
      }

      idx = shapes[s].mesh.indices[index_offset + 1];
      tri.verts[1].x = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
      tri.verts[1].y = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
      tri.verts[1].z = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
      if (idx.normal_index >= 0) {
        tri.norms[1].x = attrib.normals[3 * size_t(idx.normal_index) + 0];
        tri.norms[1].y = attrib.normals[3 * size_t(idx.normal_index) + 1];
        tri.norms[1].z = attrib.normals[3 * size_t(idx.normal_index) + 2];
      }

      idx = shapes[s].mesh.indices[index_offset + 2];
      tri.verts[2].x = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
      tri.verts[2].y = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
      tri.verts[2].z = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
      if (idx.normal_index >= 0) {
        tri.norms[2].x = attrib.normals[3 * size_t(idx.normal_index) + 0];
        tri.norms[2].y = attrib.normals[3 * size_t(idx.normal_index) + 1];
        tri.norms[2].z = attrib.normals[3 * size_t(idx.normal_index) + 2];
      }

      obj_maxs.x = std::max({obj_maxs.x, tri.verts[0].x, tri.verts[1].x, tri.verts[2].x});
      obj_maxs.y = std::max({obj_maxs.y, tri.verts[0].y, tri.verts[1].y, tri.verts[2].y});
      obj_maxs.z = std::max({obj_maxs.z, tri.verts[0].z, tri.verts[1].z, tri.verts[2].z});

      obj_mins.x = std::min({obj_mins.x, tri.verts[0].x, tri.verts[1].x, tri.verts[2].x});
      obj_mins.y = std::min({obj_mins.y, tri.verts[0].y, tri.verts[1].y, tri.verts[2].y});
      obj_mins.z = std::min({obj_mins.z, tri.verts[0].z, tri.verts[1].z, tri.verts[2].z});

      geom_maxs.x = std::max({geom_maxs.x, tri.verts[0].x, tri.verts[1].x, tri.verts[2].x});
      geom_maxs.y = std::max({geom_maxs.y, tri.verts[0].y, tri.verts[1].y, tri.verts[2].y});
      geom_maxs.z = std::max({geom_maxs.z, tri.verts[0].z, tri.verts[1].z, tri.verts[2].z});

      geom_mins.x = std::min({geom_mins.x, tri.verts[0].x, tri.verts[1].x, tri.verts[2].x});
      geom_mins.y = std::min({geom_mins.y, tri.verts[0].y, tri.verts[1].y, tri.verts[2].y});
      geom_mins.z = std::min({geom_mins.z, tri.verts[0].z, tri.verts[1].z, tri.verts[2].z});

      triangles.push_back(tri);
      
      

      index_offset += 3;



      }

    // use the geom itself as the bounding box
    geom.materialid = shapes[s].mesh.material_ids[0];
    // use the center of mass as the translation
    geom.translation = (geom_maxs + geom_mins) / 2.0f;
    // rotation should not be needed
    geom.rotation = glm::vec3(0, 0, 0);
    // scale should be center of mass to max value distance for x,y,z
    geom.scale = glm::vec3(geom_maxs - geom_mins) * (1.0f+EPSILON);

    //std::cout << "LOADED GEOM: maxs:" << glm::to_string(geom_maxs) << "mins: " << glm::to_string(geom_mins) << '\n';
    std::cout << "LOADED GEOM: mat:" << geom.materialid << "with num tris: " << geom.num_tris << "translation: " << glm::to_string(geom.translation) << "scale: " << glm::to_string(geom.scale) << '\n';
    geom.transform = utilityCore::buildTransformationMatrix(
            geom.translation, geom.rotation, geom.scale);
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::inverseTranspose(geom.transform);
    //triangles.push_back(tris);

    geoms.push_back(geom);



    }


  //throw;
  return 1;
}


int Scene::loadMaterialOBJ(tinyobj::ObjReader reader) {

  auto& objmaterials = reader.GetMaterials();
  cout << "Found " << objmaterials.size() << " OBJ materials \n";
  for (size_t id = 0; id < objmaterials.size(); id++) {

    Material newMaterial;

    newMaterial.color = glm::vec3(objmaterials[id].diffuse[0],
                                  objmaterials[id].diffuse[1],
                                  objmaterials[id].diffuse[2]);
    newMaterial.specular.exponent = 0;
    newMaterial.specular.color = glm::vec3(objmaterials[id].specular[0],
                                           objmaterials[id].specular[1],
                                           objmaterials[id].specular[2]);
    newMaterial.hasReflective = false;
    newMaterial.hasRefractive = false;
    newMaterial.indexOfRefraction = 0;
    newMaterial.emittance = objmaterials[id].emission[0];
    cout << "Loading Material " << id << " diffuse " << glm::to_string(newMaterial.color) << " specular " << glm::to_string(newMaterial.specular.color) << " emittance " << newMaterial.emittance << '\n';
    materials.push_back(newMaterial);
  }

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

int Scene::loadCameraOBJ(tinyobj::ObjReader reader) {

  RenderState &state = this->state;
  Camera &camera = state.camera;
  //auto loaded_cam = model.cameras[0];
  camera.resolution.x = CAMERA_RES_X;
  camera.resolution.y = CAMERA_RES_Y;
  state.iterations = CAMERA_ITERATIONS;
  state.traceDepth = MAX_BOUNCES;
  state.imageName = "testOBJ";
  /*
   * we don't have a supported way to store cameras in .OBJ
   * so, we implement a little method to choose a "good starting point" for showing the geometry.
   * for this we arbitrarily choose a starting point for the camera in the "-z" direction (facing "+z")
   * We choose a distance from the center of mass based on FOV plus a little padding
  */
  glm::vec3 com = (obj_maxs + obj_mins) / 2.0f;
  float cam_dist = (((obj_maxs.y - obj_mins.y) / 2.0f) / glm::sin(CAMERA_FOV)) + obj_maxs.z;

  camera.position = glm::vec3(com.x, com.y, com.z + cam_dist + CAMERA_ZOOM_PADDING);
  //camera.position = glm::vec3(14, 0, 0);
  camera.lookAt = com;

  std::cout << "OBJ CAM position:" << glm::to_string(camera.position) << '\n';
  std::cout << "OBJ CAM lookat:" << glm::to_string(com) << '\n';

//  camera.position = glm::vec3(1.826266, 6.070658, 5.288637);
//  camera.lookAt = glm::vec3(0.000000, 5.497307, 0.000000);
//  camera.up = glm::vec3(-0.033101, 0.989608, -0.095855);
//  state.iterations = 10;


  //calculate fov based on resolution
  float yscaled = tan(CAMERA_FOV * (PI / 180));
  float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
  float fovx = (atan(xscaled) * 180) / PI;
  camera.fov = glm::vec2(fovx, CAMERA_FOV);

  camera.right = glm::normalize(glm::cross(camera.view, camera.up));
  camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                 2 * yscaled / (float)camera.resolution.y);

  camera.view = glm::normalize(camera.lookAt - camera.position);

  //set up render camera stuff
  int arraylen = camera.resolution.x * camera.resolution.y;
  state.image.resize(arraylen);
  std::fill(state.image.begin(), state.image.end(), glm::vec3());

  return 1;

}


int Scene::loadCameraGLTF(tinygltf::Model model) {

  RenderState &state = this->state;
  Camera &camera = state.camera;
  auto loaded_cam = model.cameras[0];
  camera.resolution.x = 800;
  camera.resolution.y = 800;
  state.iterations = 5000;
  state.traceDepth = 8;
  state.imageName = "testGLTF";
  //camera.position = model.scenes[0].nodes[0]



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

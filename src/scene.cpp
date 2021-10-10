#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>  // c++11
#include <random>
#include "utilities.h"

#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "tiny_gltf.h"


static std::string GetFilePathExtension(const std::string& FileName) {
  if (FileName.find_last_of(".") != std::string::npos)
    return FileName.substr(FileName.find_last_of(".") + 1);
  return "";
}


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
            } else if (strcmp(tokens[0].c_str(), "GLTF") == 0) {
                loadGLTF(tokens[1], 3.0);
                cout << " " << endl;
            }
        }
    }

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    std::uniform_real_distribution<float> u11(-1.f, 1.f);

    for (int i = 0; i < 500; i++) {

      Material newMat;
      newMat.pbrMetallicRoughness.baseColorFactor = Color(u01(eng), u01(eng), u01(eng));
      newMat.pbrMetallicRoughness.metallicFactor = 0.f;
      materials.push_back(newMat);

    }

    for (int i = 0; i < 2000; i++) {
      Geom newGeom;
      newGeom.type = SPHERE;
      newGeom.materialid = rand() % 500 + 5;
      newGeom.translation = glm::vec3(5.f*u11(eng), 3.f*u11(eng)+3.f, 5.f*u11(eng));
      newGeom.rotation = glm::vec3(0.f);
      newGeom.scale = glm::vec3(0.1f);

      //load tranformations
      newGeom.transform = utilityCore::buildTransformationMatrix(
        newGeom.translation, newGeom.rotation, newGeom.scale);
      newGeom.inverseTransform = glm::inverse(newGeom.transform);
      newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
      geoms.push_back(newGeom);
    }
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
                newMaterial.pbrMetallicRoughness.baseColorFactor = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                float r = atof(tokens[1].c_str());
                newMaterial.hasReflective = r;
                newMaterial.pbrMetallicRoughness.metallicFactor = r;
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                float e = atof(tokens[1].c_str());
                newMaterial.emittance = e;
                newMaterial.emissiveFactor = glm::vec3(e, e, e);
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadGLTFNode(const std::vector<tinygltf::Node>& nodes, 
  const tinygltf::Node& node, const glm::mat4& pXform, bool* isLoaded) {
  
  glm::mat4 xform;
  if (node.matrix.empty()) {
    glm::vec3 t = node.translation.empty() ? glm::vec3(0.f) : glm::make_vec3(node.translation.data());
    glm::quat r = node.rotation.empty() ? glm::quat(1, 0, 0, 0) : glm::make_quat(node.rotation.data());
    glm::vec3 s = node.scale.empty() ? glm::vec3(1.f) : glm::make_vec3(node.scale.data());
    xform = utilityCore::buildTransformationMatrix(t, r, s);
  }
  else {
    xform = glm::make_mat4(node.matrix.data());
  }
  xform = xform * pXform;

  for (const int child : node.children) {
    if (!isLoaded[child]) {
      loadGLTFNode(nodes, nodes[child], xform, isLoaded);
      isLoaded[child] = true;
    }
  }

  if (node.mesh == -1)
    return 0;

  Geom newGeom;
  newGeom.type = MESH;
  newGeom.meshid = node.mesh;
  newGeom.materialid = 1;
  newGeom.transform = xform;
  newGeom.inverseTransform = glm::inverse(newGeom.transform);
  newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
  geoms.push_back(newGeom);

  return 1;
}

int Scene::loadGLTF(const std::string& filename, float scale) {
  /*
  This load function is based on tinygltf's example
  */
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;
  const std::string ext = GetFilePathExtension(filename);

  bool ret = false;
  if (ext.compare("glb") == 0) {
    // assume binary glTF.
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
  }
  else {
    // assume ascii glTF.
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());
  }

  if (!warn.empty()) {
    std::cout << "glTF parse warning: " << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << "glTF parse error: " << err << std::endl;
  }
  if (!ret) {
    std::cerr << "Failed to load glTF: " << filename << std::endl;
    return false;
  }

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

  // Store node as geometries
  bool* isLoaded = new bool[model.nodes.size()];
  memset(isLoaded, 0, model.nodes.size() * sizeof(bool));
  for (int i = 0; i < model.nodes.size(); i++) {
    if (!isLoaded[i]) {
      loadGLTFNode(model.nodes, model.nodes[i], glm::mat4(1.f), isLoaded);
      isLoaded[i] = true;
    }
  }
  delete[] isLoaded;

  //// Load all materials
  int mat_offset = materials.size();
  for (const tinygltf::Material& gltfMat : model.materials) {
    Material newMat;
    newMat.pbrMetallicRoughness.baseColorTexture = gltfMat.pbrMetallicRoughness.baseColorTexture;
    newMat.pbrMetallicRoughness.baseColorFactor = glm::make_vec3(gltfMat.pbrMetallicRoughness.baseColorFactor.data());
    newMat.pbrMetallicRoughness.metallicRoughnessTexture = gltfMat.pbrMetallicRoughness.metallicRoughnessTexture;
    newMat.pbrMetallicRoughness.metallicFactor = gltfMat.pbrMetallicRoughness.metallicFactor;
    newMat.pbrMetallicRoughness.roughnessFactor = gltfMat.pbrMetallicRoughness.roughnessFactor;
    newMat.normalTexture = gltfMat.normalTexture;
    materials.push_back(newMat);
  }

  // Iterate through all texture declaration in glTF file
  for (const tinygltf::Texture& gltfTexture : model.textures) {
    std::cout << "Found texture: " << gltfTexture.name << std::endl;
    Texture loadedTexture;
    const tinygltf::Image& image = model.images[gltfTexture.source];
    loadedTexture.components = image.component;  // number of components
    loadedTexture.width = image.width;
    loadedTexture.height = image.height;
    loadedTexture.size = image.component * image.width * image.height * sizeof(unsigned char);
    loadedTexture.image = new unsigned char[loadedTexture.size];
    memcpy(loadedTexture.image, image.image.data(), loadedTexture.size);
    textures.push_back(loadedTexture);
  }

  //glm::mat4 xform = utilityCore::buildTransformationMatrix(glm::vec3(0, 6, 0), glm::vec3(0), glm::vec3(scale, scale, scale));
  glm::mat4 xform(1.f);

  // Get all meshes
  for (const tinygltf::Mesh& gltfMesh : model.meshes) {
    std::cout << "Current mesh has " << gltfMesh.primitives.size()
      << " primitives:\n";

    // Create a mesh object
    Mesh loadedMesh;

    // For each primitive
    for (const tinygltf::Primitive& meshPrimitive : gltfMesh.primitives) {      
      const tinygltf::Accessor& indicesAccessor = model.accessors[meshPrimitive.indices];
      const tinygltf::BufferView& bufferView = model.bufferViews[indicesAccessor.bufferView];
      const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
      const uint16_t* indices = reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + indicesAccessor.byteOffset]);

      const auto byteStride = indicesAccessor.ByteStride(bufferView);
      const size_t count = indicesAccessor.count;
      loadedMesh.count = count;

      loadedMesh.mat_id = mat_offset + meshPrimitive.material;

      // Load indices
      loadedMesh.i_offset = mesh_indices.size();
      for (int i = 0; i < indicesAccessor.count; ++i) {
        mesh_indices.push_back(indices[i]);
      }
      std::cout << '\n';

      switch (meshPrimitive.mode) {
        case TINYGLTF_MODE_TRIANGLES:
        {
          std::cout << "TRIANGLES\n";

          for (const auto & attribute : meshPrimitive.attributes) {
            const tinygltf::Accessor& attribAccessor = model.accessors[attribute.second];
            const tinygltf::BufferView& bufferView = model.bufferViews[attribAccessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            const float* data = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + attribAccessor.byteOffset]);
            const int byte_stride = attribAccessor.ByteStride(bufferView);
            int offset = byte_stride / sizeof(float);

            std::cout << "current attribute has count " << count
              << " and stride " << byte_stride << " bytes\n";

            std::cout << "attribute string is : " << attribute.first << '\n';
            if (attribute.first == "POSITION") {
              std::cout << "found position attribute\n";

              // get the position min/max for computing the boundingbox
              loadedMesh.bbox_min.x = attribAccessor.minValues[0];
              loadedMesh.bbox_min.y = attribAccessor.minValues[1];
              loadedMesh.bbox_min.z = attribAccessor.minValues[2];
              loadedMesh.bbox_max.x = attribAccessor.maxValues[0];
              loadedMesh.bbox_max.y = attribAccessor.maxValues[1];
              loadedMesh.bbox_max.z = attribAccessor.maxValues[2];

              // TODO: Allow custom xform
              //utilityCore::xformVec3(loadedMesh.bbox_max, xform);
              //utilityCore::xformVec3(loadedMesh.bbox_min, xform);

              if (offset == 0)
                offset = 3;

              // Store mesh vertices
              loadedMesh.v_offset = mesh_vertices.size();
              for (int i = 0; i < attribAccessor.count; i++) {
                glm::vec3 v;
                int idx = i * offset;
                v.x = data[idx + 0];
                v.y = data[idx + 1];
                v.z = data[idx + 2];

                //utilityCore::xformVec3(v, xform);
                mesh_vertices.push_back(v);
              }
            }
            else if (attribute.first == "NORMAL") {
              std::cout << "found normal attribute\n";

              loadedMesh.n_offset = mesh_normals.size();

              if (offset == 0)
                offset = 3;

              // IMPORTANT: We need to reorder normals (and texture
              // coordinates into "facevarying" order) for each face
              // 
              // For each triangle :
              mesh_normals.resize(loadedMesh.n_offset + attribAccessor.count);
              for (int i = 0; i < loadedMesh.count; i+=3) {
                // get the i'th triange's indexes
                int f0 = indices[i + 0];
                int f1 = indices[i + 1];
                int f2 = indices[i + 2];

                int i0, i1, i2;
                i0 = f0 * offset;
                i1 = f1 * offset;
                i2 = f2 * offset;

                // get the 3 normal vectors for that face
                glm::vec3 n0, n1, n2;
                n0.x = data[i0 + 0];
                n0.y = data[i0 + 1];
                n0.z = data[i0 + 2];
                n1.x = data[i1 + 0];
                n1.y = data[i1 + 1];
                n1.z = data[i1 + 2];
                n2.x = data[i2 + 0];
                n2.y = data[i2 + 1];
                n2.z = data[i2 + 2];

                // Put them in the array in the correct order
                mesh_normals[loadedMesh.n_offset + f0] = n0;
                mesh_normals[loadedMesh.n_offset + f1] = n1;
                mesh_normals[loadedMesh.n_offset + f2] = n2;
              }
            }
            else if (attribute.first == "TEXCOORD_0") {
              std::cout << "found texture attribute\n";

              loadedMesh.uv_offset = mesh_uvs.size();
              
              if (offset == 0)
                offset = 2;

              mesh_uvs.resize(loadedMesh.uv_offset + attribAccessor.count);
              // For each triangle :
              for (int i = 0; i < loadedMesh.count; i += 3) {
                // get the i'th triange's indexes
                int f0 = indices[i + 0];
                int f1 = indices[i + 1];
                int f2 = indices[i + 2];

                int i0, i1, i2;
                i0 = f0 * offset;
                i1 = f1 * offset;
                i2 = f2 * offset;

                // get the 3 texture coordinates for each triangle
                glm::vec2 t0, t1, t2;
                t0.x = data[i0 + 0];
                t0.y = data[i0 + 1];
                t1.x = data[i1 + 0];
                t1.y = data[i1 + 1];
                t2.x = data[i2 + 0];
                t2.y = data[i2 + 1];

                // Put them in the array in the correct order
                mesh_uvs[loadedMesh.uv_offset + f0] = t0;
                mesh_uvs[loadedMesh.uv_offset + f1] = t1;
                mesh_uvs[loadedMesh.uv_offset + f2] = t2;
              }
            }
            else if (attribute.first == "TANGENT") {
              std::cout << "found tangent attribute\n";

              loadedMesh.t_offset = mesh_tangents.size();

              if (offset == 0)
                offset = 4;

              mesh_tangents.resize(loadedMesh.t_offset + attribAccessor.count);
              // For each triangle :
              for (int i = 0; i < loadedMesh.count; i += 3) {
                // get the i'th triange's indexes
                int f0 = indices[i + 0];
                int f1 = indices[i + 1];
                int f2 = indices[i + 2];

                int i0, i1, i2;
                i0 = f0 * offset;
                i1 = f1 * offset;
                i2 = f2 * offset;

                // get the 3 texture coordinates for each triangle
                glm::vec4 t0, t1, t2;
                t0.x = data[i0 + 0];
                t0.y = data[i0 + 1];
                t0.z = data[i0 + 2];
                t0.w = data[i0 + 3];
                t1.x = data[i1 + 0];
                t1.y = data[i1 + 1];
                t1.z = data[i1 + 2];
                t1.w = data[i1 + 3];
                t2.x = data[i2 + 0];
                t2.y = data[i2 + 1];
                t2.z = data[i2 + 2];
                t2.w = data[i2 + 3];

                // Put them in the array in the correct order
                mesh_tangents[loadedMesh.t_offset + f0] = t0;
                mesh_tangents[loadedMesh.t_offset + f1] = t1;
                mesh_tangents[loadedMesh.t_offset + f2] = t2;
              }
            }
          }
        }
        default:
          std::cerr << "primitive mode not implemented";
          break;
      }
    }

    meshes.push_back(loadedMesh);
    ret = true;
  }

  return ret;
}

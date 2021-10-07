#include <iostream>

#include "model_loader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader-v2.0.0/tiny_obj_loader.h"

ObjLoader::ObjLoader(const std::string& file_path) : ModelLoader() {
  is_ready_ = loadFromFile(file_path);
}

ObjLoader::ObjLoader(const std::string& file_path, const std::string& mtl_path)
    : ModelLoader() {
  is_ready_ = loadFromFile(file_path, mtl_path);
}

bool ObjLoader::loadFromFile(const std::string& file_path) {
  reader_config_.mtl_search_path = "./";
  if (!reader_.ParseFromFile(file_path, reader_config_)) {
    if (!reader_.Error().empty()) {
      std::cerr << "TinyObjReader: " << reader_.Error() << "\n";
    }
    return false;
  }
  if (!reader_.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader_.Warning();
  }
  shapes_    = reader_.GetShapes();
  materials_ = reader_.GetMaterials();
  attribute_ = reader_.GetAttrib();
  return true;
}

bool ObjLoader::loadFromFile(const std::string& file_path,
                             const std::string& mtl_path) {
  reader_config_.mtl_search_path = mtl_path;
  if (!reader_.ParseFromFile(file_path, reader_config_)) {
    if (!reader_.Error().empty()) {
      std::cerr << "TinyObjReader: " << reader_.Error() << "\n";
    }
    return false;
  }
  if (!reader_.Warning().empty()) {
    std::cout << "TinyObjReader: " << reader_.Warning();
  }
  shapes_    = reader_.GetShapes();
  materials_ = reader_.GetMaterials();
  attribute_ = reader_.GetAttrib();
  return true;
}

int ObjLoader::numShapes() const { return static_cast<int>(shapes_.size()); }

int ObjLoader::numFaces(const int shape_id) const {
  return static_cast<int>(shapes_[shape_id].mesh.num_face_vertices.size());
}

int ObjLoader::numVertices(const int shape_id, const int face_id) const {
  const auto& shape = shapes_[shape_id];
  return static_cast<int>(shape.mesh.num_face_vertices[face_id]);
}

glm::vec3 ObjLoader::getVertexPos(const int shape_id,
                                  const int vertex_id) const {
  const auto& shape    = shapes_[shape_id];
  tinyobj::index_t idx = shape.mesh.indices[vertex_id];

  tinyobj::real_t vx = attribute_.vertices[idx.vertex_index * 3 + 0];
  tinyobj::real_t vy = attribute_.vertices[idx.vertex_index * 3 + 1];
  tinyobj::real_t vz = attribute_.vertices[idx.vertex_index * 3 + 2];

  return glm::vec3(vx, vy, vz);
}

glm::vec3 ObjLoader::getNormalVec(const int shape_id,
                                  const int normal_id) const {
  const auto& shape    = shapes_[shape_id];
  tinyobj::index_t idx = shape.mesh.indices[normal_id];

  glm::vec3 normal{0.0f, 0.0f, 0.0f};

  if (idx.normal_index > 0) {
    normal.x = attribute_.normals[idx.vertex_index * 3 + 0];
    normal.y = attribute_.normals[idx.vertex_index * 3 + 1];
    normal.z = attribute_.normals[idx.vertex_index * 3 + 2];
  } else {
    std::cout << "TinyObjReader: no normal vectors found in shape " << shape_id
              << ", id " << normal_id << "!\n";
  }

  return normal;
}

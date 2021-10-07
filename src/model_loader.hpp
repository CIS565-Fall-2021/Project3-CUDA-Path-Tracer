#ifndef __CUDA_PATH_TRACING_MODEL_LOADER_HPP__
#define __CUDA_PATH_TRACING_MODEL_LOADER_HPP__

#include <string>

#include "glm/glm.hpp"

class ModelLoader {
public:
  ModelLoader() : is_ready_(false) {}

  bool isReady() const { return is_ready_; }

  // load model from file
  virtual bool loadFromFile(const std::string& file_path) = 0;

protected:
  bool is_ready_;
};

#include "tinyobjloader-v2.0.0/tiny_obj_loader.h"
class ObjLoader : public ModelLoader {
public:
  ObjLoader() : ModelLoader() {}
  ObjLoader(const std::string& file_path);
  ObjLoader(const std::string& file_path, const std::string& mtl_path);

  /* manual loaders */
  bool loadFromFile(const std::string& file_path) override;
  bool loadFromFile(const std::string& file_path, const std::string& mtl_path);

  /* get sizes */
  int numShapes() const;
  int numFaces(const int shape_id) const;
  int numVertices(const int shape_id, const int face_id) const;

  /* get attributes */
  /**
   * @brief Get vertex 3D position in mesh. Vertex is ordered from 0 per shape
   * in face order, i.e., for 1st vertex in face 2 in any shape,
   *
   * vertex_id = numVertices(face_0) + numVertices(face_1) + 1.
   *
   * @param shape_id
   * @param vertex_id
   * @return glm::vec3
   */
  glm::vec3 getVertexPos(const int shape_id, const int vertex_id) const;
  /**
   * @brief Get normal vector per vertex in mesh. Normal vector is ordered from
   * 0 per shape in face order, i.e., for 1st normal vector in face 2 in any
   * shape,
   *
   * normal_id = numVertices(face_0) + numVertices(face_1) + 1.
   *
   * @param shape_id
   * @param normal_id
   * @return glm::vec3
   */
  glm::vec3 getNormalVec(const int shape_id, const int normal_id) const;

private:
  tinyobj::ObjReader reader_;
  tinyobj::ObjReaderConfig reader_config_;

  tinyobj::attrib_t attribute_;
  std::vector<tinyobj::shape_t> shapes_;
  std::vector<tinyobj::material_t> materials_;
};

#endif /* __CUDA_PATH_TRACING_MODEL_LOADER_HPP__ */

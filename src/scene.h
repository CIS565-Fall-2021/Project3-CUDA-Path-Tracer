#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#define MAX_LEVEL 2

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid, string objFilename);
    int loadCamera();
    int loadMesh(std::string inputFile);
public:
    Scene(string filename);
    Scene(string filename, string objFilename);

    void makeOctree();
    void makeOctreeNode(OctreeNode parent, int level);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<OctreeNode> octree;
    std::vector<Triangle> octTriangles;
    Mesh mesh;
    RenderState state;
};

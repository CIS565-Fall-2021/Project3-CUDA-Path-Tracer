#include "sceneStructs.h"
#include "utilities.h"
#include <glm/gtc/matrix_inverse.hpp>

glm::vec3 getCenter(std::array<glm::vec3, 3>& triangle, int axis)
{
    glm::vec3 v = glm::vec3(0);

    if (axis == 0)
    {
        for (auto& vertex : triangle)
        {
            v += glm::vec3(vertex.x, 0, 0);
        }
        return v / 3.f;
    }
    else if (axis == 1)
    {
        for (auto& vertex : triangle)
        {
            v += glm::vec3(0,vertex.y, 0);
        }
        return v / 3.f;
    }
    else if (axis == 2)
    {
        for (auto& vertex : triangle)
        {
            v += glm::vec3(0,0,vertex.z);
        }
        return v / 3.f;
    }
    return v;
}

// TODO: use getMinVertex to clean up code
bool xSort(std::array<glm::vec3, 3> a, std::array<glm::vec3, 3> b) {
    glm::vec3 centerA = getCenter(a, 0);
    glm::vec3 centerB = getCenter(b, 0);

    return centerA.x < centerB.x;
}
bool ySort(std::array<glm::vec3, 3> a, std::array<glm::vec3, 3> b) {
    glm::vec3 centerA = getCenter(a, 1);
    glm::vec3 centerB = getCenter(b, 1);

    return centerA.x < centerB.x;
}
bool zSort(std::array<glm::vec3, 3> a, std::array<glm::vec3, 3> b) {
    glm::vec3 centerA = getCenter(a, 2);
    glm::vec3 centerB = getCenter(b, 2);

    return centerA.x < centerB.x;
}

void buildTree(
    int node,
    std::vector<std::array<glm::vec3, 3>>& triangles,
    std::vector<KDNode>* kdNodes)
{
    // update the min and max corner of parent node
    // update the min and max corners
    float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
    float maxX = FLT_MIN, maxY = FLT_MIN, maxZ = FLT_MIN;
    for (auto& triangle : triangles)
    {
        for (const glm::vec3& vertex : triangle)
        {
            if (vertex.x < minX)
                minX = vertex.x;
            if (vertex.x > maxX)
                maxX = vertex.x;
            if (vertex.y < minY)
                minY = vertex.y;
            if (vertex.y > maxY)
                maxY = vertex.y;
            if (vertex.z < minZ)
                minZ = vertex.z;
            if (vertex.z > maxZ)
                maxZ = vertex.z;
        }

    }
    glm::vec3 minCorner = glm::vec3(minX, minY, minZ);
    glm::vec3 maxCorner = glm::vec3(maxX, maxY, maxZ);

    // create box geom
    kdNodes->at(node).boundingBox.type = GeomType::CUBE;
    kdNodes->at(node).boundingBox.translation = (maxCorner - minCorner) / 2.f;
    kdNodes->at(node).boundingBox.scale = maxCorner / 0.5f;
    kdNodes->at(node).boundingBox.rotation = glm::vec3(0);
    kdNodes->at(node).boundingBox.transform = utilityCore::buildTransformationMatrix(
        kdNodes->at(node).boundingBox.translation, kdNodes->at(node).boundingBox.rotation, kdNodes->at(node).boundingBox.scale);
    kdNodes->at(node).boundingBox.inverseTransform = glm::inverse(kdNodes->at(node).boundingBox.transform);
    kdNodes->at(node).boundingBox.invTranspose = glm::inverseTranspose(kdNodes->at(node).boundingBox.transform);

    // sort triangles
    std::sort(triangles.begin(), triangles.end(), kdNodes->at(node).axis == 0 ? xSort : kdNodes->at(node).axis == 1 ? ySort : zSort);

    // get the median value
    unsigned long long arraySize = triangles.size();
    float median = FLT_MAX;
    if (kdNodes->at(node).axis == 0)
    {
        median = (arraySize % 2 == 1) ? getCenter(triangles.at(arraySize / 2), kdNodes->at(node).axis).x :
            (getCenter(triangles.at(arraySize / 2), kdNodes->at(node).axis).x + getCenter(triangles.at(arraySize / 2 - 1), kdNodes->at(node).axis).x) / 2;
    }
    else if (kdNodes->at(node).axis == 1)
    {
        median = (arraySize % 2 == 1) ? getCenter(triangles.at(arraySize / 2), kdNodes->at(node).axis).y :
            (getCenter(triangles.at(arraySize / 2), kdNodes->at(node).axis).y + getCenter(triangles.at(arraySize / 2 - 1), kdNodes->at(node).axis).y) / 2;
    }
    else
    {
        median = (arraySize % 2 == 1) ? getCenter(triangles.at(arraySize / 2), kdNodes->at(node).axis).z :
            (getCenter(triangles.at(arraySize / 2), kdNodes->at(node).axis).z + getCenter(triangles.at(arraySize / 2 - 1), kdNodes->at(node).axis).z) / 2;
    }

    // split the triangles by the axis-median
    std::vector<std::array<glm::vec3, 3>> ltriangles;
    std::vector<std::array<glm::vec3, 3>> rtriangles;
    for (auto& triangle : triangles)
    {
        float comparison = FLT_MAX;
        if (kdNodes->at(node).axis == 0)
            comparison = getCenter(triangle, 0).x;
        if (kdNodes->at(node).axis == 1)
            comparison = getCenter(triangle, 1).y;
        if (kdNodes->at(node).axis == 2)
            comparison = getCenter(triangle, 2).z;

        if (comparison < median)
            ltriangles.push_back(triangle);
        else
            rtriangles.push_back(triangle);
    }

    // create the two children nodes
    kdNodes->push_back(KDNode());
    kdNodes->at(node).leftChild = kdNodes->size() - 1;
    kdNodes->push_back(KDNode());
    kdNodes->at(node).rightChild = kdNodes->size() - 1;

    // update children node axis
    if (kdNodes->at(node).axis < 2)
    {
        kdNodes->at(kdNodes->at(node).leftChild).axis = kdNodes->at(node).axis + 1;
        kdNodes->at(kdNodes->at(node).rightChild).axis = kdNodes->at(node).axis + 1;
    }
    else
    {
        kdNodes->at(kdNodes->at(node).leftChild).axis = 0;
        kdNodes->at(kdNodes->at(node).rightChild).axis = 0;
    }

    // recursively call function
    unsigned long long lsize = ltriangles.size(), rsize = rtriangles.size();
    if (lsize <= 1)
    {
        if (lsize == 1)
        {
            // leaf node 
            kdNodes->at(kdNodes->at(node).leftChild).particles.
                insert(kdNodes->at(kdNodes->at(node).leftChild).particles.end(), ltriangles.begin(), ltriangles.end());
        }

    }
    else
        buildTree(kdNodes->at(node).leftChild, ltriangles, kdNodes);

    if (rsize <= 1)
    {
        if (rsize == 1)
        {
            // leaf node 
            kdNodes->at(kdNodes->at(node).rightChild).particles.
                insert(kdNodes->at(kdNodes->at(node).rightChild).particles.end(), rtriangles.begin(), rtriangles.end());
        }

    }
    else
        buildTree(kdNodes->at(node).rightChild, rtriangles, kdNodes);
}
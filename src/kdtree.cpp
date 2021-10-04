#include "sceneStructs.h"

glm::vec3* getMinVertex(std::array<glm::vec3, 3>* triangle, int axis)
{
    glm::vec3* v = &glm::vec3(FLT_MAX);

    if (axis == 0)
    {
        for (auto& vertex : *triangle)
        {
            if (vertex.x < v->x)
                v = &vertex;
        }
        return v;
    }
    else if (axis == 1)
    {
        for (auto& vertex : *triangle)
        {
            if (vertex.y < v->y)
                v = &vertex;
        }
        return v;
    }
    else if (axis == 2)
    {
        for (auto& vertex : *triangle)
        {
            if (vertex.z < v->z)
                v = &vertex;
        }
        return v;
    }
    return v;
}

// TODO: use getMinVertex to clean up code
bool xSort(std::array<glm::vec3, 3>* a, std::array<glm::vec3, 3>* b) {
    // get the minimum A
    glm::vec3 minA = glm::vec3(FLT_MAX);
    for (auto& vertex : *a)
    {
        if (vertex.x < minA.x)
            minA = vertex;
    }
    // get the minimum B
    glm::vec3 minB = glm::vec3(FLT_MAX);
    for (auto& vertex : *b)
    {
        if (vertex.x < minB.x)
            minB = vertex;
    }
    return minA.x < minB.x;
}
bool ySort(std::array<glm::vec3, 3>* a, std::array<glm::vec3, 3>* b) {
    // get the minimum A
    glm::vec3 minA = glm::vec3(FLT_MAX);
    for (auto& vertex : *a)
    {
        if (vertex.y < minA.y)
            minA = vertex;
    }
    // get the minimum B
    glm::vec3 minB = glm::vec3(FLT_MAX);
    for (auto& vertex : *b)
    {
        if (vertex.y < minB.y)
            minB = vertex;
    }
    return minA.y < minB.y;
}
bool zSort(std::array<glm::vec3, 3>* a, std::array<glm::vec3, 3>* b) {
    // get the minimum A
    glm::vec3 minA = glm::vec3(FLT_MAX);
    for (auto& vertex : *a)
    {
        if (vertex.z < minA.z)
            minA = vertex;
    }
    // get the minimum B
    glm::vec3 minB = glm::vec3(FLT_MAX);
    for (auto& vertex : *b)
    {
        if (vertex.z < minB.z)
            minB = vertex;
    }
    return minA.z < minB.z;
}

void buildTree(KDNode* node, std::vector<std::array<glm::vec3, 3>*>& triangles)
{
    // update the min and max corner of parent node
    // update the min and max corners
    float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
    float maxX = FLT_MIN, maxY = FLT_MIN, maxZ = FLT_MIN;
    for (auto* triangle : triangles)
    {
        for (const glm::vec3& vertex : *triangle)
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
    node->minCorner = glm::vec3(minX, minY, minZ);
    node->maxCorner = glm::vec3(maxX, maxY, maxZ);

    // sort triangles
    std::sort(triangles.begin(), triangles.end(), node->axis == 0 ? xSort : node->axis == 1 ? ySort : zSort);

    // get the median value
    unsigned long long arraySize = triangles.size();
    float median = FLT_MAX;
    if (node->axis == 0)
    {
        // TODO: check if you need to handle arraySize being odd
        glm::vec3* vertex = getMinVertex(triangles.at(arraySize / 2), 0);
        median = vertex->x;
    }
    else if (node->axis == 1)
    {
        glm::vec3* vertex = getMinVertex(triangles.at(arraySize / 2), 1);
        median = vertex->y;
    }
    else
    {
        glm::vec3* vertex = getMinVertex(triangles.at(arraySize / 2), 2);
        median = vertex->z;
    }

    // split the triangles by the axis-median
    std::vector<std::array<glm::vec3, 3>*> ltriangles;
    std::vector<std::array<glm::vec3, 3>*> rtriangles;
    for (auto* triangle : triangles)
    {
        float comparison = FLT_MAX;
        if (node->axis == 0)
            comparison = getMinVertex(triangle, 0)->x;
        if (node->axis == 1)
            comparison = getMinVertex(triangle, 1)->y;
        if (node->axis == 2)
            comparison = getMinVertex(triangle, 2)->z;

        if (comparison < median)
            ltriangles.push_back(triangle);
        else
            rtriangles.push_back(triangle);
    }

    // create the two children nodes
    node->leftChild = new KDNode();
    node->rightChild = new KDNode();

    // update children node axis
    if (node->axis < 2)
    {
        node->leftChild->axis = node->axis + 1;
        node->rightChild->axis = node->axis + 1;
    }
    else
    {
        node->leftChild->axis = 0;
        node->rightChild->axis = 0;
    }

    // recursively call function
    unsigned long long lsize = ltriangles.size(), rsize = rtriangles.size();
    if (lsize <= 1)
    {
        if (lsize == 1)
            node->leftChild->particles.
            insert(node->leftChild->particles.end(), ltriangles.begin(), ltriangles.end());
    }
    else
        buildTree(node->leftChild, ltriangles);

    if (rsize <= 1)
    {
        if (rsize == 1)
            node->rightChild->particles.
            insert(node->rightChild->particles.end(), rtriangles.begin(), rtriangles.end());
    }
    else
        buildTree(node->rightChild, rtriangles);
}

// TODO: make accessable to both device and host
bool withinBounds(KDNode* node, glm::vec3 boundingMin, glm::vec3 boundingMax)
{
    return node->minCorner.x < boundingMin.x&&
        node->minCorner.y < boundingMin.y&&
        node->minCorner.z < boundingMin.z&&
        node->maxCorner.x > boundingMax.x&&
        node->maxCorner.y > boundingMax.y&&
        node->maxCorner.z > boundingMax.z;
}

KDNode* depthSearch(KDNode* node, glm::vec3 boundingMin, glm::vec3 boundingMax)
{
    if (withinBounds(node, boundingMin, boundingMax))
    {
        if (node->leftChild != nullptr && withinBounds(node->leftChild, boundingMin, boundingMax))
            return depthSearch(node->leftChild, boundingMin, boundingMax);
        if (node->rightChild != nullptr && withinBounds(node->rightChild, boundingMin, boundingMax))
            return depthSearch(node->rightChild, boundingMin, boundingMax);
        return node;
    }
    return nullptr;
}

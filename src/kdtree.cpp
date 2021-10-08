#include <glm/gtc/matrix_inverse.hpp>

#include "sceneStructs.h"
#include "utilities.h"

struct TriangleStruct
{
    Triangle arr;
    int index;
    TriangleStruct(Triangle arr, int index) : arr(arr), index(index)
    {}
};

//glm::vec3 getCenter(Triangle& triangle, int axis)
//{
//    glm::vec3 v = glm::vec3(0);
//
//    if (axis == 0)
//    {
//        for (auto& vertex : triangle)
//        {
//            v += glm::vec3(vertex.x, 0, 0);
//        }
//        return v / 3.f;
//    }
//    else if (axis == 1)
//    {
//        for (auto& vertex : triangle)
//        {
//            v += glm::vec3(0,vertex.y, 0);
//        }
//        return v / 3.f;
//    }
//    else if (axis == 2)
//    {
//        for (auto& vertex : triangle)
//        {
//            v += glm::vec3(0,0,vertex.z);
//        }
//        return v / 3.f;
//    }
//    return v;
//}

float getMin(Triangle& triangle, int axis)
{
    return std::min(triangle[0][axis], std::min(triangle[1][axis], triangle[2][axis]));
}

bool xSort(TriangleStruct a, TriangleStruct b) {
    return getMin(a.arr, 0) < getMin(b.arr, 0);
}
bool ySort(TriangleStruct a, TriangleStruct b) {
    return getMin(a.arr, 1) < getMin(b.arr, 1);
}
bool zSort(TriangleStruct a, TriangleStruct b) {
    return getMin(a.arr, 2) < getMin(b.arr, 2);
}

void buildTree(
    int node,
    std::vector<KDNode>* kdNodes,
    std::vector<Triangle>* primitives,
    int startInx, int endInx, // TODO: change name to be clearer
    Transform& transform)
{
    // update the min and max corner of parent node
    // update the min and max corners
    float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
    float maxX = FLT_MIN, maxY = FLT_MIN, maxZ = FLT_MIN;
    for (int i = startInx; i <= endInx; i++)
    {
        Triangle& triangle = primitives->at(i);
        // transform triangle
        glm::mat4 tm = utilityCore::buildTransformationMatrix(transform.translate, transform.rotate, transform.scale);
        triangle[0] = glm::vec3(tm * glm::vec4(triangle[0], 1.f));
        triangle[1] = glm::vec3(tm * glm::vec4(triangle[1], 1.f));
        triangle[2] = glm::vec3(tm * glm::vec4(triangle[2], 1.f));

        for (int i = 0; i < 3; i++)
        {
            glm::vec3& vertex = triangle[i];

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
    glm::vec3 minCorner{ minX, minY, minZ };
    glm::vec3 maxCorner{ maxX, maxY, maxZ };

    // create bounding box geom
    kdNodes->at(node).boundingBox.type = GeomType::CUBE;
    kdNodes->at(node).boundingBox.translation = (maxCorner + minCorner) / 2.f;
    kdNodes->at(node).boundingBox.scale = (maxCorner- kdNodes->at(node).boundingBox.translation) / 0.5f;
    kdNodes->at(node).boundingBox.scale += 0.01f; // epsilon addition
    kdNodes->at(node).boundingBox.rotation = glm::vec3(0);
    kdNodes->at(node).boundingBox.transform = utilityCore::buildTransformationMatrix(
        kdNodes->at(node).boundingBox.translation, kdNodes->at(node).boundingBox.rotation, kdNodes->at(node).boundingBox.scale);
    kdNodes->at(node).boundingBox.inverseTransform = glm::inverse(kdNodes->at(node).boundingBox.transform);
    kdNodes->at(node).boundingBox.invTranspose = glm::inverseTranspose(kdNodes->at(node).boundingBox.transform);

    if ((endInx - startInx + 1) > 1)
    {
        // sort triangles
        std::vector<TriangleStruct> mTriangleStructs;
        for (int i = startInx; i <= endInx; i++)
        {
            mTriangleStructs.push_back(TriangleStruct(primitives->at(i), i));
        }
        std::sort(mTriangleStructs.begin(), mTriangleStructs.end(), kdNodes->at(node).axis == 0 ? xSort : kdNodes->at(node).axis == 1 ? ySort : zSort);
        std::vector<int> triangles;
        for (int i = startInx, j = 0; i <= endInx; i++, j++)
        {
            primitives->at(i) = mTriangleStructs.at(j).arr;
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

        // split triangles
        int midpoint = startInx + (endInx - startInx) / 2;

        buildTree(kdNodes->at(node).leftChild, kdNodes, primitives, startInx, midpoint, transform);
        buildTree(kdNodes->at(node).rightChild, kdNodes, primitives, midpoint + 1, endInx, transform);
    }
}
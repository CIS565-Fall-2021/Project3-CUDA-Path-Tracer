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

float getMin(Triangle& triangle, int axis)
{
    return std::min(triangle[0][axis], std::min(triangle[1][axis], triangle[2][axis]));
}

float getMid(Triangle& triangle, int axis)
{
    return (triangle[0][axis] + triangle[1][axis] + triangle[2][axis]) / 3.f;
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
    std::vector<BVHNode>* bvhNodes,
    std::vector<Triangle>* primitives,
    int node, int primStart, int primEnd)
{
    // update the min and max corner of parent node
    // update the min and max corners
    float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
    float maxX = FLT_MIN, maxY = FLT_MIN, maxZ = FLT_MIN;

    for (int i = primStart; i <= primEnd; i++)
    {
        Triangle& triangle = primitives->at(i);

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
    bvhNodes->at(node).boundingBox.type = GeomType::CUBE;
    bvhNodes->at(node).boundingBox.translation = (maxCorner + minCorner) / 2.f;
    bvhNodes->at(node).boundingBox.scale = (maxCorner - bvhNodes->at(node).boundingBox.translation) / 0.5f;
    bvhNodes->at(node).boundingBox.scale += 0.001f; // epsilon addition
    bvhNodes->at(node).boundingBox.rotation = glm::vec3(0);
    bvhNodes->at(node).boundingBox.transform = utilityCore::buildTransformationMatrix(
        bvhNodes->at(node).boundingBox.translation, bvhNodes->at(node).boundingBox.rotation, bvhNodes->at(node).boundingBox.scale);
    bvhNodes->at(node).boundingBox.inverseTransform = glm::inverse(bvhNodes->at(node).boundingBox.transform);
    bvhNodes->at(node).boundingBox.invTranspose = glm::inverseTranspose(bvhNodes->at(node).boundingBox.transform);

    if ((primEnd - primStart + 1) > 1)
    {
        // sort triangles
        std::vector<TriangleStruct> mTriangleStructs; // needed to swap triangles
        for (int i = primStart; i <= primEnd; i++)
        {
            mTriangleStructs.push_back(TriangleStruct(primitives->at(i), i));
        }
        std::sort(mTriangleStructs.begin(), mTriangleStructs.end(), bvhNodes->at(node).axis == 0 ? xSort : bvhNodes->at(node).axis == 1 ? ySort : zSort);
        for (int i = primStart, j = 0; i <= primEnd; i++, j++)
        {
            primitives->at(i) = mTriangleStructs.at(j).arr; // very important
        }

        // create the two children nodes
        bvhNodes->push_back(BVHNode());
        bvhNodes->at(node).leftChild = bvhNodes->size() - 1;
        bvhNodes->push_back(BVHNode());
        bvhNodes->at(node).rightChild = bvhNodes->size() - 1;

        // update children node axis
        if (bvhNodes->at(node).axis < 2)
        {
            bvhNodes->at(bvhNodes->at(node).leftChild).axis = bvhNodes->at(node).axis + 1;
            bvhNodes->at(bvhNodes->at(node).rightChild).axis = bvhNodes->at(node).axis + 1;
        }
        else
        {
            bvhNodes->at(bvhNodes->at(node).leftChild).axis = 0;
            bvhNodes->at(bvhNodes->at(node).rightChild).axis = 0;
        }

        // split triangles
        int midpoint = primStart + (primEnd - primStart) / 2;

        buildTree(bvhNodes, primitives, bvhNodes->at(node).leftChild, primStart, midpoint);
        buildTree(bvhNodes, primitives, bvhNodes->at(node).rightChild, midpoint + 1, primEnd);
    }
    else
    {
        // leaf node
        Geom geom;
        geom.type = GeomType::TRIANGLE;

        geom.pos1 = primitives->at(primStart)[0];
        geom.pos2 = primitives->at(primStart)[1];
        geom.pos3 = primitives->at(primStart)[2];

        bvhNodes->at(node).triangle = geom;
    }
}
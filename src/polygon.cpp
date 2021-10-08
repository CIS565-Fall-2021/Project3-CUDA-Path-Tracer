#include "polygon.h"
#include <glm/gtx/transform.hpp>

void Polygon::Triangulate()
{
    // Populate list of triangles
    for (unsigned int i = 0; i < this->m_verts.size() - 2; i++)
    {
        Triangle tri({0, i+1, i+2});
        m_tris.push_back(tri);
    }
}

// Creates a polygon from the input list of vertex positions and colors
Polygon::Polygon(const std::vector<glm::vec4>& pos, const std::vector<glm::vec3>& col)
    : m_tris(), m_verts()
{
    for(unsigned int i = 0; i < pos.size(); i++)
    {
        m_verts.push_back(Vertex(pos[i], col[i], glm::vec4()));
    }
    Triangulate();
}

// Creates a regular polygon with a number of sides indicated by the "sides" input integer.
// All of its vertices are of color "color", and the polygon is centered at "pos".
// It is rotated about its center by "rot" degrees, and is scaled from its center by "scale" units
Polygon::Polygon(int sides, glm::vec3 color, glm::vec4 pos, float rot, glm::vec4 scale)
    : m_tris(), m_verts()
{
    glm::vec4 v(0.f, 1.f, 0.f, 1.f);
    float angle = 360.f / sides;
    for(int i = 0; i < sides; i++)
    {
        glm::vec4 vert_pos = glm::translate(glm::vec3(pos.x, pos.y, pos.z))
                           * glm::rotate(rot, glm::vec3(0.f, 0.f, 1.f))
                           * glm::scale(glm::vec3(scale.x, scale.y, scale.z))
                           * glm::rotate(i * angle, glm::vec3(0.f, 0.f, 1.f))
                           * v;
        m_verts.push_back(Vertex(vert_pos, color, glm::vec4()));
    }

    Triangulate();
}

Polygon::Polygon()
    : m_tris(), m_verts()
{}

Polygon::Polygon(const Polygon& p)
    : m_tris(p.m_tris), m_verts(p.m_verts)
{

}

Polygon::~Polygon()
{
    
}

void Polygon::AddTriangle(const Triangle& t)
{
    m_tris.push_back(t);
}

void Polygon::AddVertex(const Vertex& v)
{
    m_verts.push_back(v);
}

void Polygon::ClearTriangles()
{
    m_tris.clear();
}

Triangle& Polygon::TriAt(unsigned int i)
{
    return m_tris[i];
}

Triangle Polygon::TriAt(unsigned int i) const
{
    return m_tris[i];
}

Vertex &Polygon::VertAt(unsigned int i)
{
    return m_verts[i];
}

Vertex Polygon::VertAt(unsigned int i) const
{
    return m_verts[i];
}

std::vector<glm::vec2> Polygon::findBoundingBox(Triangle tri) const {
    std::vector<glm::vec2> coords;
    if (m_verts.size() < 3) {
        std::cout << "Warning: invalid number of vertices" << "\n";
        return coords;
    }
    float minX, maxX, minY, maxY;
    minX = std::min(m_verts[tri.m_indices[0]].m_pos.x,
            std::min(m_verts[tri.m_indices[1]].m_pos.x,
                     m_verts[tri.m_indices[2]].m_pos.x));
    minX = std::max(0.f, minX);

    maxX = std::max(m_verts[tri.m_indices[0]].m_pos.x,
            std::max(m_verts[tri.m_indices[1]].m_pos.x,
                     m_verts[tri.m_indices[2]].m_pos.x));
    maxX = std::min(512.f, maxX);

    minY = std::min(m_verts[tri.m_indices[0]].m_pos.y,
            std::min(m_verts[tri.m_indices[1]].m_pos.y,
                     m_verts[tri.m_indices[2]].m_pos.y));
    minY = std::max(0.f, minY);

    maxY = std::max(m_verts[tri.m_indices[0]].m_pos.y,
            std::max(m_verts[tri.m_indices[1]].m_pos.y,
                     m_verts[tri.m_indices[2]].m_pos.y));
    maxY = std::min(512.f, maxY);

    glm::vec2 min(minX, minY);
    glm::vec2 max(maxX, maxY);
    coords.push_back(min);
    coords.push_back(max);

    return coords;
}

glm::vec3 Polygon::interpolate(Triangle tri, glm::vec2 coord) const {
    // create a copy of each vertex with z axis zeroed
    glm::vec3 v0(m_verts[tri.m_indices[0]].m_pos.x, m_verts[tri.m_indices[0]].m_pos.y, 0);
    glm::vec3 v1(m_verts[tri.m_indices[1]].m_pos.x, m_verts[tri.m_indices[1]].m_pos.y, 0);
    glm::vec3 v2(m_verts[tri.m_indices[2]].m_pos.x, m_verts[tri.m_indices[2]].m_pos.y, 0);
    // expand point p into vec3
    glm::vec3 p(coord, 0);
    // calculate areas
    float area = 0.5 * glm::length(glm::cross(v0-v1, v2-v1));
    float area0 = 0.5 * glm::length(glm::cross(p-v1, v2-v1));
    float area1 = 0.5 * glm::length(glm::cross(v0-p, v2-p));
    float area2 = 0.5 * glm::length(glm::cross(v0-v1, p-v1));
//    std::cout << "Total area: " << area << "\n\tPartition0: " << area0 <<
//                 "\n\tPartition1: " << area1 << "\n\tPartition2: " << area2 <<
//                 "\n\t\tSum of partitions: " << (area0 + area1 + area2) << "\n";

//    if (area - (area0 + area1 + area2) > EPSILON) {
//        std::cout << "Warning: barycentric interpolation on invalid vertex" << "\n";
//        return glm::vec3(std::numeric_limits<double>::quiet_NaN());
//    }
    return glm::vec3(area0/area, area1/area, area2/area);
}

glm::vec4 Polygon::interpolatePC(Vertex v0, Vertex v1, Vertex v2, glm::vec4 coord) const {
    glm::vec3 v0Flat(v0.m_pos.x, v0.m_pos.y, 0);
    glm::vec3 v1Flat(v1.m_pos.x, v1.m_pos.y, 0);
    glm::vec3 v2Flat(v2.m_pos.x, v2.m_pos.y, 0);
    glm::vec3 nCoord(coord.x, coord.y, 0);

    float area =  0.5 * glm::length(glm::cross(v0Flat - v1Flat, v2Flat - v1Flat));
    float area0 = 0.5 * glm::length(glm::cross(nCoord - v1Flat, v2Flat - v1Flat));
    float area1 = 0.5 * glm::length(glm::cross(v0Flat - nCoord, v2Flat - nCoord));
    float area2 = 0.5 * glm::length(glm::cross(v0Flat - v1Flat, nCoord - v1Flat));

    float s0 = area0/area;
    float s1 = area1/area;
    float s2 = area2/area;

//    std::cout << area << " /0/ " << area0 << " /1/ " << area1 << " /2/ " << area2 << " /3/ "
//        << s0 << " /4/ " << s1 << " /5/ " << s2 << " /6/ " << v0.m_pos.z << " /7/ "
//        << v1.m_pos.z << " /8/ " << v2.m_pos.z << "\n";

    float z = 1.f / ((s0 / v0.m_pos.z) + (s1 / v1.m_pos.z) + (s2 / v2.m_pos.z));

    return glm::vec4(coord.x, coord.y, z, coord.w);
}

glm::vec4 Polygon::interpolateNorm(Vertex v0, Vertex v1, Vertex v2, glm::vec4 coord) const {
    glm::vec3 v0Flat(v0.m_pos.x, v0.m_pos.y, 0);
    glm::vec3 v1Flat(v0.m_pos.x, v0.m_pos.y, 0);
    glm::vec3 v2Flat(v0.m_pos.x, v0.m_pos.y, 0);
    glm::vec3 nCoord(coord.x, coord.y, 0);

    float area =  0.5 * glm::length(glm::cross(v0Flat - v1Flat, v2Flat - v1Flat));
    float area0 = 0.5 * glm::length(glm::cross(nCoord - v1Flat, v2Flat - v1Flat));
    float area1 = 0.5 * glm::length(glm::cross(v0Flat - nCoord, v2Flat - nCoord));
    float area2 = 0.5 * glm::length(glm::cross(v0Flat - v1Flat, nCoord - v1Flat));

    float s0 = area0/area;
    float s1 = area1/area;
    float s2 = area2/area;

    return glm::vec4(coord.z * ((v0.m_normal * s0 / v0.m_pos.z) +
                                (v1.m_normal * s1 / v1.m_pos.z) +
                                (v2.m_normal * s2 / v2.m_pos.z)));
}














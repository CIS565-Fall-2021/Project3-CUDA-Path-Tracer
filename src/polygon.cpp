#include "polygon.h"
#include <glm/gtx/transform.hpp>

void Polygon::Triangulate()
{

    Triangle temp_triangle;
    int num_vertices = this->m_verts.size();
    for (int i = 0; i < num_vertices - 2; i++)
    {
        temp_triangle.m_indices[0] = 0;
        temp_triangle.m_indices[1] = i + 1;
        temp_triangle.m_indices[2] = i + 2;
        this->m_tris.push_back(temp_triangle);
    }
}


//Use the one above if using textures
// Creates a polygon from the input list of vertex positions and colors
Polygon::Polygon(char* name, const std::vector<glm::vec4>& pos, const std::vector<glm::vec3>& col)
    : m_tris(), m_verts(), m_name(name)
{
    for (unsigned int i = 0; i < pos.size(); i++)
    {
        m_verts.push_back(Vertex(pos[i], col[i], glm::vec4(), glm::vec2()));
    }
    Triangulate();
}

Polygon::Polygon()
    : m_tris(), m_verts(), m_name("Polygon")
{}

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

Vertex& Polygon::VertAt(unsigned int i)
{
    return m_verts[i];
}

Vertex Polygon::VertAt(unsigned int i) const
{
    return m_verts[i];
}
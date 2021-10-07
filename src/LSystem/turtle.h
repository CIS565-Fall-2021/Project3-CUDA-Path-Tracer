#pragma once
#ifndef TURTLE_H
#define TURTLE_H
#include<vector>
#include <glm/glm.hpp>
#include<glm/gtx/transform.hpp>
class Turtle
{
private:

public:
    Turtle();
    Turtle(glm::vec3 a_pos, glm::vec3 a_right, glm::vec3 a_up, glm::vec3 a_foward, float a_lambda = 20, float a_radius = 10);
    Turtle(const Turtle&);
    glm::vec3 m_Position;
    glm::vec3 m_forward;
    glm::vec3 m_Right;
    glm::vec3 m_Up;
    std::vector<Turtle> TurtleStates;
    float lambda; //Storing Length for Carving Terrain
    float radius; //Storing radius for Carving Terrain

    bool m_takeRandomRotationsForward;
    float m_rotateAngle;
    void TurnLeft();
    void TurnRight();
    void Move();
//    void drawThinerLines();
    void LoadState();
    void SaveState();
};

#endif // TURTLE_H

#include "turtle.h"

Turtle::Turtle() : m_takeRandomRotationsForward(true), m_rotateAngle(30.f)
{
    m_Position = glm::vec3(200.0f, 133.0f, 200.0f);
    m_forward = glm::vec3(1.0, 0.0, 0.0);
    m_Right = glm::vec3(0.0, 0.0, 1.0);
    m_Up = glm::vec3(0.0, 1.0, 0.0);
    lambda = 20.0f;
    radius = 10.0f;
}

Turtle::Turtle(glm::vec3 a_pos, glm::vec3 a_right, glm::vec3 a_up, glm::vec3 a_foward, float a_lambda, float a_radius) : m_Position(a_pos),
    m_forward(a_foward), m_Right(a_right), m_Up(a_up), lambda(a_lambda), radius(a_radius), m_takeRandomRotationsForward(true), m_rotateAngle(30.f)
{}

Turtle::Turtle(const Turtle &a_turtle)
{
    m_Position = a_turtle.m_Position;
    m_forward = a_turtle.m_forward;
    m_Right = a_turtle.m_Right;
    m_Up = a_turtle.m_Up;
    lambda = a_turtle.lambda;
    radius = a_turtle.radius;
    m_takeRandomRotationsForward = a_turtle.m_takeRandomRotationsForward;
    m_rotateAngle = a_turtle.m_rotateAngle;
    TurtleStates = a_turtle.TurtleStates;


}

// Generate a random degree and rotate the direction to the left
void Turtle::TurnLeft(){
    glm::mat4 rotMax = glm::rotate(glm::mat4(), glm::radians( -1 * m_rotateAngle),m_Up);
    m_forward = glm::vec3(glm::vec4(m_forward, 0) * rotMax);
    m_Right = glm::vec3(glm::vec4(m_Right , 0) * rotMax);
}

// Generate a random degree and rotate the direction to the right
void Turtle::TurnRight(){
    glm::mat4 rotMax = glm::rotate(glm::mat4(), glm::radians(m_rotateAngle),m_Up);
    m_forward = glm::vec3(glm::vec4(m_forward, 0) * rotMax);
    m_Right = glm::vec3(glm::vec4(m_Right , 0) * rotMax);
}

// Draw a line along the direction in certain distance and carve out the terrain nearby
void Turtle::Move(){
    if(m_takeRandomRotationsForward)
    {
        float randomDegree = rand() % 19 + (-9);
        glm::mat4 rotMax = glm::rotate(glm::mat4(), glm::radians(randomDegree),m_Up);
        m_forward = glm::vec3(glm::vec4(m_forward, 0) * rotMax);
        m_Right = glm::vec3(glm::vec4(m_Right , 0) * rotMax);
    }
    float amount = lambda;
    m_Position += amount * m_forward;
}

// Pop out the position and direction out of the stack
void Turtle::LoadState(){
    Turtle State =  TurtleStates.back();
    TurtleStates.pop_back();
    this->m_Position = State.m_Position;
    this->m_forward = State.m_forward;
    this->m_Right = State.m_Right;
    this->m_Up = State.m_Up;
    this->lambda = State.lambda;
    this->radius = State.radius ;
}

// Push the position and direction into the stack
void Turtle::SaveState(){
    TurtleStates.push_back(*this);
}

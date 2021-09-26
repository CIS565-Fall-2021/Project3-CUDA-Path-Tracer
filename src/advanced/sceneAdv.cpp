#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "sceneAdv.h"

bool SceneAdvance::readFromToken(const std::vector<std::string>& tokens)
{
    if (Scene::readFromToken(tokens))
    {
        return true;
    }
    //TODO
    return false;
}

SceneAdvance::SceneAdvance(std::string filename)
    : Scene(filename) {}

SceneAdvance::~SceneAdvance() {}

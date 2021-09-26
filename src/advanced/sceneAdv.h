#pragma once

#include "materialAdv.h"
#include "../scene.h"

class SceneAdvance : public Scene {
protected:
    virtual bool readFromToken(const std::vector<std::string>& tokens) override;
public:
    SceneAdvance(std::string filename);
    virtual ~SceneAdvance();

    std::vector<MaterialAdvance> advanceMaterials;
};

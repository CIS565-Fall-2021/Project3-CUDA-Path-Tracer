#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define DEBUG_TOKENS 0//1

int Scene::loadMaterial(std::string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        std::cout << "ERROR: MATERIAL ID does not match expected number of materials" << std::endl;
        return -1;
    } else {
        std::cout << "Loading Material " << id << "..." << std::endl;
        Material newMaterial;

        bool customMaterialType = false;

        ////load static properties
        //for (int i = 0; i < 7; i++) {
        //    std::string line;
        //    utilityCore::safeGetline(fp_in, line);
        std::string line;
        while (fp_in.good()) {
            utilityCore::safeGetline(fp_in, line);
            if (line.empty()) {
                break;
            }
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
#if DEBUG_TOKENS
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::cout << "token[" << i << "] = <" << tokens[i] << '>' << std::endl;
            }
            std::cout << std::endl;
#endif // DEBUG_TOKENS
            if (tokens.empty() || tokens[0].empty()) {
                break;
            }

            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } 
            else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } 
            else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } 
            else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } 
            else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } 
            else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } 
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            } 
            else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
                newMaterial.metallic = glm::clamp(1.f - static_cast<float>(atof(tokens[1].c_str())), 0.f, 1.f);
            } 
            else if (strcmp(tokens[0].c_str(), "MATERIAL_TYPE") == 0) {
                if (strcmp(tokens[1].c_str(), "PHONG") == 0) {
                    std::cout << "Creating new Phong material..." << std::endl;
                    newMaterial.materialType = MaterialType::PHONG;
                    customMaterialType = true;
                } 
                else if (strcmp(tokens[1].c_str(), "DIELECTRIC") == 0) { // Dieletric is not correct...
                    std::cout << "Creating new Dielectric material..." << std::endl;
                    newMaterial.materialType = MaterialType::DIELECTRIC;
                    customMaterialType = true;
                }
                else if (strcmp(tokens[1].c_str(), "MICROFACET_GGX") == 0) {
                    std::cout << "Creating new Microfacet-GGX material..." << std::endl;
                    newMaterial.materialType = MaterialType::MICROFACET_GGX;
                    customMaterialType = true;
                }
            }
            else if (strcmp(tokens[0].c_str(), "DIFFUSE_TEXTURE") == 0) {
                addTextureToLoad(materials.size(), utilityCore::getAddrOffsetInStruct(&newMaterial, &newMaterial.diffuseTexture), basePath + tokens[1]);
                //newMaterial.diffuseTexture = loadTexture((basePath + tokens[1]).c_str());
            }
            else if (strcmp(tokens[0].c_str(), "SPECULAR_TEXTURE") == 0) {
                addTextureToLoad(materials.size(), utilityCore::getAddrOffsetInStruct(&newMaterial, &newMaterial.specularTexture), basePath + tokens[1]);
                //newMaterial.specularTexture = loadTexture((basePath + tokens[1]).c_str());
            }
            else if (strcmp(tokens[0].c_str(), "NORMAL_TEXTURE") == 0) {
                addTextureToLoad(materials.size(), utilityCore::getAddrOffsetInStruct(&newMaterial, &newMaterial.normalTexture), basePath + tokens[1]);
                //newMaterial.normalTexture = loadTexture((basePath + tokens[1]).c_str());
            }
        }
        if (!customMaterialType) {
            std::cout << "Creating new Phong material..." << std::endl;
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadGeom(std::string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        std::cout << "ERROR: OBJECT ID does not match expected number of geoms" << std::endl;
        return -1;
    } else {
        std::cout << "Loading Geom " << id << "..." << std::endl;
        Geom newGeom;
        newGeom.geometryid = id;
        std::string line;

        ////load object type
        //utilityCore::safeGetline(fp_in, line);
        //if (!line.empty() && fp_in.good()) {
        //    if (strcmp(line.c_str(), "sphere") == 0) {
        //        std::cout << "Creating new sphere..." << std::endl;
        //        newGeom.type = GeomType::SPHERE;
        //    } 
        //    else if (strcmp(line.c_str(), "cube") == 0) {
        //        std::cout << "Creating new cube..." << std::endl;
        //        newGeom.type = GeomType::CUBE;
        //    }
        //    else if (strcmp(line.c_str(), "trimesh") == 0) {
        //        std::cout << "Creating new trimesh..." << std::endl;
        //        newGeom.type = GeomType::TRI_MESH;
        //    }
        //}

        ////link material
        //utilityCore::safeGetline(fp_in, line);
        //if (!line.empty() && fp_in.good()) {
        //    std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        //    newGeom.materialid = atoi(tokens[1].c_str());
        //    std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << std::endl;
        //}

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);

#if DEBUG_TOKENS
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::cout << "token[" << i << "] = <" << tokens[i] << '>' << std::endl;
            }
            std::cout << std::endl;
#endif // DEBUG_TOKENS
            //load geom type
            if (strcmp(tokens[0].c_str(), "sphere") == 0) {
                std::cout << "Creating new sphere..." << std::endl;
                newGeom.type = GeomType::SPHERE;
            } 
            else if (strcmp(tokens[0].c_str(), "cube") == 0) {
                std::cout << "Creating new cube..." << std::endl;
                newGeom.type = GeomType::CUBE;
            }
            else if (strcmp(tokens[0].c_str(), "trimesh") == 0) {
                std::cout << "Creating new trimesh..." << std::endl;
                newGeom.type = GeomType::TRI_MESH;
            }

            //link material
            if (strcmp(tokens[0].c_str(), "material") == 0) {
                std::vector<std::string> tokens = utilityCore::tokenizeString(line);
                newGeom.materialid = atoi(tokens[1].c_str());
                std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << std::endl;
            }

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            //load models
            if (strcmp(tokens[0].c_str(), "MODEL") == 0) {
                addModelToLoad(geoms.size(), utilityCore::getAddrOffsetInStruct(&newGeom, &newGeom.trimeshRes), basePath + tokens[1]);
            }
            else if (strcmp(tokens[0].c_str(), "STENCIL") == 0) {
                newGeom.stencilid = atoi(tokens[1].c_str());
            }


            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    std::cout << "Loading Camera ..." << std::endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    state.imageName = "../img/output/";

    ////load static properties
    //for (int i = 0; i < 5; i++) {
    //    std::string line;
    //    utilityCore::safeGetline(fp_in, line);
    std::string line;
    int ppcount = 0;
    while (fp_in.good()) {
        utilityCore::safeGetline(fp_in, line);
        if (line.empty()) {
            break;
        }

        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
#if DEBUG_TOKENS
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << "token[" << i << "] = <" << tokens[i] << '>' << std::endl;
        }
        std::cout << std::endl;
#endif // DEBUG_TOKENS
        if (tokens.empty() || tokens[0].empty()) {
            break;
        }

        if (ppcount > 0) {
            if (strcmp(tokens[0].c_str(), "COLOR_RAMP") == 0) {
                addTextureToLoad(Background::COLOR_RAMP_MATERIAL_INDEX, utilityCore::getAddrOffsetInStruct(&rampMap, &rampMap), basePath + tokens[1]);
                postprocesses.push_back(std::make_pair(PostProcessType::COLOR_RAMP, true));
                --ppcount;
                std::cout << "--Add COLOR_RAMP post process..." << std::endl;
                continue;
            }
            else if (strcmp(tokens[0].c_str(), "OUTLINE_BY_STENCIL") == 0) { // e.g. OUTLINE_BY_STENCIL 1 0. 1. 0. 3
                int stencilid = atoi(tokens[1].c_str());
                if (stencilid > 0) {
                    ppToStencilMap[postprocesses.size()] = stencilid;
                    stencilOutlineColorWidths[stencilid] = std::make_pair(glm::vec3(atof(tokens[2].c_str()), atof(tokens[3].c_str()), atof(tokens[4].c_str())), glm::max(1, atoi(tokens[5].c_str())));
                    postprocesses.push_back(std::make_pair(PostProcessType::OUTLINE_BY_STENCIL, true));
                    --ppcount;
                    std::cout << "--Add OUTLINE_BY_STENCIL post process..." << std::endl;
                }
                continue;
            }
        }
        if (strcmp(tokens[0].c_str(), "POSTPROCESS") == 0) {
            ppcount = atoi(tokens[1].c_str());
            postprocesses.reserve(ppcount);
            std::cout << "Set " << ppcount << " post process..." << std::endl;
        }

        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName += tokens[1];
        }


        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
    }

    // Start image name
#if ENABLE_JITTER_ANTI_ALIASING
    state.imageName += "_JAA";
#endif // JITTER_ANTI_ALIASING
#if ENABLE_ADVANCED_PIPELINE
    state.imageName += "_ADVPIPE";
#endif // ENABLE_ADVANCED_PIPELINE
    state.imageName += "_depth" + std::to_string(state.traceDepth);

    std::string postprocessStr = "_PP";
    size_t activePP = 0;
    for (size_t i = 0; i < postprocesses.size(); ++i) {
        if (postprocesses[i].second) {
            postprocessStr += std::to_string(i);
            ++activePP;
        }
    }
    if (activePP > 0) {
        state.imageName += postprocessStr;
    }
    ////
#if ENABLE_PARTITION
    state.imageName += "--PARTITION";
#endif // ENABLE_PARTITION
#if ENABLE_SORTING
    state.imageName += "--SORTING";
#endif // ENABLE_SORTING
#if ENABLE_CACHE_FIRST_INTERSECTION
    state.imageName += "--CACHE1st";
#endif // ENABLE_CACHE_FIRST_INTERSECTION
#if ENABLE_BVH
    state.imageName += "--BVH";
#if SORT_BEFORE_BUILD_BVH
    state.imageName += "-PRESORT";
#endif // SORT_BEFORE_BUILD_BVH
#endif // ENABLE_BVH
    ////

    // End image name

    //utilityCore::safeGetline(fp_in, line);
    //while (!line.empty() && fp_in.good()) {
    //    std::vector<std::string> tokens = utilityCore::tokenizeString(line);
    //    if (strcmp(tokens[0].c_str(), "EYE") == 0) {
    //        camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
    //    } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
    //        camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
    //    } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
    //        camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
    //    }

    //    utilityCore::safeGetline(fp_in, line);
    //}

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), background.backgroundColor);

    std::cout << "Loaded camera!" << std::endl;
    return 1;
}

int Scene::loadBackground() {
    std::cout << "Loading Background ..." << std::endl;
    std::string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
#if DEBUG_TOKENS
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << "token[" << i << "] = <" << tokens[i] << '>' << std::endl;
        }
        std::cout << std::endl;
#endif // DEBUG_TOKENS
        if (strcmp(tokens[0].c_str(), "RGB") == 0) {
            background.backgroundColor = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SPHERE_MAP") == 0) {
            addTextureToLoad(Background::BACKGROUND_MATERIAL_INDEX, utilityCore::getAddrOffsetInStruct(&background, &background.sphereMap), basePath + tokens[1]);
        }

        utilityCore::safeGetline(fp_in, line);
    }
    std::fill(state.image.begin(), state.image.end(), background.backgroundColor);

    std::cout << "Loaded background!" << std::endl;
    return 1;
}

bool Scene::readFromToken(const std::vector<std::string>& tokens) {
    if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
        loadMaterial(tokens[1]);
        std::cout << " " << std::endl;
        return true;
    } 
    else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
        loadGeom(tokens[1]);
        std::cout << " " << std::endl;
        return true;
    } 
    else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
        loadCamera();
        std::cout << " " << std::endl;
        return true;
    }
    else if (strcmp(tokens[0].c_str(), "BACKGROUND") == 0) {
        loadBackground();
        std::cout << " " << std::endl;
        return true;
    }
    return false;
}

Scene::Scene(std::string filename) {
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        std::cout << "Error reading from file - aborting!" << std::endl;
        throw;
    }

    basePath = utilityCore::getBaseDirectory(filename);
    std::cout << "Base path: " << basePath << std::endl;
    
    while (fp_in.good()) {
        std::string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            std::vector<std::string> tokens = utilityCore::tokenizeString(line);
            readFromToken(tokens);
        }
    }

    //initCallbacks.push([this]() {
    //    initTextures();
    //});    
    //initCallbacks.push([this]() {
    //    initModels();
    //});
}

Scene::~Scene() {
    freeGBuffer();
    freeModels();
    freeTextures();
}

void Scene::execInitCallbacks() {
    initTextures();
    initModels();
    initGBuffer();
    //while (initCallbacks.size()) {
    //    initCallbacks.front()();
    //    initCallbacks.pop();
    //}
}

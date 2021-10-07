#include <iostream>
#include "scene.h"
#include <cstring>
#include "glm/glm.hpp"
#include "glm/gtx/intersect.hpp"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_gltf.h"

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char *fname = (char *)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open())
    {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good())
    {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty())
        {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0)
            {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0)
            {
                loadGeom(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0)
            {
                loadCamera();
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "GLTF") == 0)
            {
                loadGLTF();
            }
        }
    }
}

int Scene::loadGLTF()
{
    string line;

    //load transformations
    glm::vec3 translation, rotation, scale;

    // load translation
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good())
    {
        vector<string> tokens = utilityCore::tokenizeString(line);
        translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
    }

    // load rotation
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good())
    {
        vector<string> tokens = utilityCore::tokenizeString(line);
        rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
    }

    // load scale
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good())
    {
        vector<string> tokens = utilityCore::tokenizeString(line);
        scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
    }

    // load gltf
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good())
    {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        string err;
        string warn;
        bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, line.c_str());
        if (!warn.empty())
        {
            printf("Warn: %s\n", warn.c_str());
        }

        if (!err.empty())
        {
            printf("Err: %s\n", err.c_str());
        }

        if (!ret)
        {
            printf("Failed to parse glTF\n");
            return -1;
        }
        std::cout << line.c_str() << "\n";
        for (const tinygltf::Mesh &mesh : model.meshes)
        {
            for (const tinygltf::Primitive &primitive : mesh.primitives)
            {

                const tinygltf::Accessor &idxAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView &idxBufferView = model.bufferViews[idxAccessor.bufferView];
                const tinygltf::Buffer &idxBuffer = model.buffers[idxBufferView.buffer];
                const unsigned short *indices = reinterpret_cast<const unsigned short *>(&idxBuffer.data[idxBufferView.byteOffset + idxAccessor.byteOffset]);

                const tinygltf::Accessor &posAccessor = model.accessors[primitive.attributes.at("POSITION")];
                const tinygltf::BufferView &posBufferView = model.bufferViews[posAccessor.bufferView];
                const tinygltf::Buffer &posBuffer = model.buffers[posBufferView.buffer];
                const float *positions = reinterpret_cast<const float *>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);
                //std::cout << primitive.attributes.at("NORMAL") << "\n";

                const float *normals = nullptr;
                if (primitive.attributes.count("NORMAL"))
                {
                    std::cout << "hi"
                              << "\n";
                    const tinygltf::Accessor &norAccessor = model.accessors[primitive.attributes.at("NORMAL")];
                    const tinygltf::BufferView &norBufferView = model.bufferViews[norAccessor.bufferView];
                    const tinygltf::Buffer &norBuffer = model.buffers[norBufferView.buffer];
                    normals = reinterpret_cast<const float *>(&norBuffer.data[norBufferView.byteOffset + norAccessor.byteOffset]);
                }

                std::cout << idxAccessor.count << " : " << posAccessor.count << "\n";
                for (size_t i = 0; i < idxAccessor.count; i += 3)
                {

                    Geom newGeom;
                    newGeom.type = TRIANGLE;
                    newGeom.materialid = 0;
                    for (int j = 0; j < 3; j++)
                    {
                        int idx = indices[j + i]; // indices[j + i];
                                                  //  std::cout << idx << "\n";
                        newGeom.vertices[j] = glm::vec3(positions[idx * 3 + 0], positions[idx * 3 + 1], positions[idx * 3 + 2]);
                        //std::cout << glm::to_string(newGeom.vertices[j]) << '\n';
                        if (normals) {
                            newGeom.n1 = glm::vec3(normals[idx * 3 + 0], normals[idx * 3 + 1], normals[idx * 3 + 2]);
                            newGeom.n2 = glm::vec3(normals[idx * 3 + 3], normals[idx * 3 + 4], normals[idx * 3 + 5]);
                            newGeom.n3 = glm::vec3(normals[idx * 3 + 6], normals[idx * 3 + 7], normals[idx * 3 + 8]);
                            }
                    }

                    
                    newGeom.transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
                    newGeom.inverseTransform = glm::inverse(newGeom.transform);
                    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
                    geoms.push_back(newGeom);
                }
            }
        }
    }

    std::cout << "geoms size: " << geoms.size() << "\n";
}

int Scene::loadGeom(string objectid)
{
    int id = atoi(objectid.c_str());
    if (id != geoms.size())
    {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else
    {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good())
        {
            if (strcmp(line.c_str(), "sphere") == 0)
            {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0)
            {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good())
        {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good())
        {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0)
            {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0)
            {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0)
            {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
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

int Scene::loadCamera()
{
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++)
    {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0)
        {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FOVY") == 0)
        {
            fovy = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0)
        {
            state.iterations = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "DEPTH") == 0)
        {
            state.traceDepth = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FILE") == 0)
        {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good())
    {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0)
        {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0)
        {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "UP") == 0)
        {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

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
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid)
{
    int id = atoi(materialid.c_str());
    if (id != materials.size())
    {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    }
    else
    {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++)
        {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0)
            {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.color = color;
            }
            else if (strcmp(tokens[0].c_str(), "SPECEX") == 0)
            {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0)
            {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            }
            else if (strcmp(tokens[0].c_str(), "REFL") == 0)
            {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFR") == 0)
            {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0)
            {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0)
            {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

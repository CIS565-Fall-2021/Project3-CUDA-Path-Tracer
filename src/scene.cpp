#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "../external/include/json.hpp"
#include "../external/include/stb_image.h"
#include "../external/include/stb_image_write.h"
#include "../external/include/tiny_gltf.h"
#include "tiny_obj_loader.h"

#include <fstream>
#include <iterator>
#include <vector>

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
            else if (strcmp(tokens[0].c_str(), "BACK") == 0)
            {
                loadBackground();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadBackground()
{
    cout << "Loading Background "
         << "..." << endl;
    string line;
    //load object type
    utilityCore::safeGetline(fp_in, line);
    vector<string> tokens = utilityCore::tokenizeString(line);
    if (!line.empty() && fp_in.good())
    {
        if (strcmp(tokens[0].c_str(), "IMG") == 0)
        {
            std::string inputfile = tokens[1];
            if (tokens.size() > 1)
            {
                // load color texture
                std::string background = tokens[1];
                int width, height, byteStride;
                unsigned char *imgBuff = stbi_load(background.c_str(), &width, &height, &byteStride, 0);
                if (imgBuff == nullptr)
                {
                    cout << stbi_failure_reason() << endl;
                }
                backHeight = height;
                backWidth = width;
                backTex.reserve((size_t)width * (size_t)height);
                for (int idx = 0; idx < width * height; idx++)
                {
                    int tmpIdx = idx * byteStride;
                    glm::vec3 t(imgBuff[tmpIdx] / 255.f, imgBuff[tmpIdx + 1] / 255.f, imgBuff[tmpIdx + 2] / 255.f);
                    // t.bCol[0] = imgBuff[tmpIdx];
                    // t.bCol[1] = imgBuff[tmpIdx + 1];
                    // t.bCol[2] = imgBuff[tmpIdx + 2];
                    // backTex.push_back(t);
                    t = glm::mix(glm::vec3(0.f), t, glm::length(t));                // interp again for darker
                    backTex.push_back(glm::mix(glm::vec3(0.f), t, glm::length(t))); // my specific background is too bright
                }
                cout << "vector size " << backTex.size() << endl;
                stbi_image_free(imgBuff);
            }
        }
    }
    return 1;
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
        vector<Geom> triangs;
        string line;

        vector<Geom> meshs;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
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
            else if (strcmp(tokens[0].c_str(), "mesh") == 0)
            {
                if (tokens.size() < 2)
                {
                    return -1;
                }
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;

                std::string inputfile = tokens[1];
                tinyobj::ObjReaderConfig reader_config;
                reader_config.mtl_search_path = "../scenes/"; // Path to material files

                tinyobj::ObjReader reader;

                if (!reader.ParseFromFile(inputfile, reader_config))
                {
                    if (!reader.Error().empty())
                    {
                        std::cerr << "TinyObjReader: " << reader.Error();
                    }
                    exit(1);
                }

                if (!reader.Warning().empty())
                {
                    std::cout << "TinyObjReader: " << reader.Warning();
                }

                auto &attrib = reader.GetAttrib();
                auto &shapes = reader.GetShapes();
                auto &materials = reader.GetMaterials();

                cout << "sizeof shapes " << shapes.size() << endl;
                cout << "sizeof materials " << materials.size() << endl;

                vector<Geom> meshVec(shapes.size());

                // foreach mesh
                for (size_t s = 0; s < shapes.size(); s++)
                {
                    vector<struct Triangle> meshTris;
                    // foreach poly in mesh
                    size_t index_offset = 0;
                    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
                    {
                        size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
                        struct Triangle tri;
                        // foreach vert in poly
                        for (size_t v = 0; v < fv; v++)
                        {
                            // access to vertex
                            tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                            tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                            tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                            tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                            tri.pos[v] = glm::vec3(vx, vy, vz);

                            // Check if `normal_index` is zero or positive. negative = no normal data
                            if (idx.normal_index >= 0)
                            {
                                tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                                tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                                tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                                tri.norm[v] = glm::vec3(nx, ny, nz);
                            }

                            // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                            if (idx.texcoord_index >= 0)
                            {
                                tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                                tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                                tx = tx < 0 ? -tx : tx;
                                ty = ty < 0 ? -ty : ty;
                                tri.uv[v] = glm::vec2(tx, ty);
                                // if (f % 1024 == 0)
                                // {
                                //     cout << "sample uvs " << glm::to_string(tri.uv[v]) << endl;
                                // }
                            }
                        }
                        index_offset += fv;
                        // precalc tan, bitan, plannorm
                        // ABC, HKL uvs
                        // D = B-A
                        // E = C-A
                        // F = K-H
                        // G = L-H
                        // D = F.s * T + F.t * U
                        // E = G.s * T + G.t * U
                        // | D.x D.y D.z |   | F.s F.t | | T.x T.y T.z |
                        // |             | = |         | |             |
                        // | E.x E.y E.z |   | G.s G.t | | U.x U.y U.z |
                        glm::vec3 d = tri.pos[1] - tri.pos[0];
                        glm::vec3 e = tri.pos[2] - tri.pos[0];
                        glm::vec2 uvF = tri.uv[1] - tri.uv[0];
                        glm::vec2 uvG = tri.uv[2] - tri.uv[0];
                        auto tmpMatSquare = glm::inverse(glm::transpose(glm::mat2(uvF, uvG)));
                        auto tmpMatRectan = glm::transpose(glm::mat2x3(d, e));
                        auto prodMat = glm::transpose(tmpMatSquare * tmpMatRectan);
                        tri.planarNorm = glm::cross(d, e);
                        tri.tangent = prodMat[0];
                        tri.bitangent = prodMat[1];
                        // triangs.push_back(tri);
                        meshTris.push_back(tri);

                        // per-face material
                        //shapes[s].mesh.material_ids[f];
                    }
                    // meshVec[s].ts = meshTris;
                    // meshVec[s].useTexture = meshVec[s].numTris > 20000 ? s != 0 : 1;
                    // meshVec[s].useTexture = true;   // normal
                    meshVec[s].useTexture = s != 0; // ebon hawk windshield
                    meshVec[s].type = MESH;
                    // meshVec[s].meshIdx = s;
                    //TODO: Calculate AABBs
                    meshVec[s].min = meshTris.at(0).pos[0];
                    meshVec[s].max = meshTris.at(0).pos[0];
                    for (auto const &t : meshTris)
                    {
                        auto tmpmin = glm::min(glm::min(t.pos[0], t.pos[1]), t.pos[2]);
                        meshVec[s].min = glm::min(meshVec[s].min, tmpmin);
                        auto tmpmax = glm::max(glm::max(t.pos[0], t.pos[1]), t.pos[2]);
                        meshVec[s].max = glm::max(meshVec[s].max, tmpmax);
                    }
                    meshVec[s].numTris = meshTris.size();
                    meshVec[s].triIdx = triangles.size();
                    triangles.insert(triangles.end(), meshTris.begin(), meshTris.end());
                    cout << "min " << glm::to_string(meshVec[s].min) << " max " << glm::to_string(meshVec[s].max) << endl;
                }
                meshs = meshVec;
                numMeshs = meshs.size();
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

        if (meshs.size() > 0)
        {
            for (auto &m : meshs)
            {
                m.translation = newGeom.translation;
                m.rotation = newGeom.rotation;
                m.scale = newGeom.scale;
                m.transform = newGeom.transform;
                m.inverseTransform = newGeom.inverseTransform;
                m.invTranspose = newGeom.invTranspose;
                m.materialid = newGeom.materialid;

                // m.type = TRIANGLE;
                // cout << "norm " << glm::to_string(tri.norm[0]) << " "
                //      << glm::to_string(tri.norm[1]) << " "
                //      << glm::to_string(tri.norm[2]) << endl;
                // for (int trivert = 0; trivert < 3; trivert++)
                // {
                //     tri.pos[trivert] = glm::vec3(tri.transform * glm::vec4(tri.pos[trivert], 1.f));
                //     tri.norm[trivert] = glm::vec3(tri.invTranspose * glm::vec4(tri.norm[trivert], 0.f));
                // }
            }
            geoms.insert(geoms.end(), meshs.begin(), meshs.end());
            cout << "geoms size: " << geoms.size() << endl;
        }
        else
        {
            geoms.push_back(newGeom);
        }
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
        for (int i = 0; i < 11; i++)
        {
            string line;
            utilityCore::safeGetline(fp_in, line);
            if (line.length() < 1)
                continue;
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
            else if (strcmp(tokens[0].c_str(), "COLORMAP") == 0)
            {
                // NB: colormap goes first so this will reserve the vector and push back the elems
                if (tokens.size() > 1)
                {
                    // load color texture
                    std::string bcolorTexFile = tokens[1];
                    int width, height, byteStride;
                    unsigned char *imgBuff = stbi_load(bcolorTexFile.c_str(), &width, &height, &byteStride, 0);
                    if (imgBuff == nullptr)
                    {
                        cout << stbi_failure_reason() << endl;
                    }
                    newMaterial.texHeight = height;
                    newMaterial.texWidth = width;
                    texData.reserve((size_t)width * (size_t)height);
                    for (int idx = 0; idx < width * height; idx++)
                    {
                        int tmpIdx = idx * byteStride;
                        struct TexData t;
                        t.bCol[0] = imgBuff[tmpIdx];
                        t.bCol[1] = imgBuff[tmpIdx + 1];
                        t.bCol[2] = imgBuff[tmpIdx + 2];
                        texData.push_back(t);
                    }
                    cout << "vector size " << texData.size() << endl;
                    stbi_image_free(imgBuff);
                }
                newMaterial.useTex = true;
            }
            else if (strcmp(tokens[0].c_str(), "EMITMAP") == 0)
            {
                if (tokens.size() > 1)
                {
                    // load color texture
                    std::string emitTexFile = tokens[1];
                    int width, height, byteStride;
                    unsigned char *imgBuff = stbi_load(emitTexFile.c_str(), &width, &height, &byteStride, 0);
                    if (imgBuff == nullptr)
                    {
                        cout << stbi_failure_reason() << endl;
                    }
                    for (int idx = 0; idx < width * height; idx++)
                    {
                        int tmpIdx = idx * byteStride;
                        texData[idx].emit = imgBuff[tmpIdx]; // assumes emitmap is 1 byte per pixel
                    }
                    stbi_image_free(imgBuff);
                }
            }
            else if (strcmp(tokens[0].c_str(), "ROUGHMAP") == 0)
            {
                if (tokens.size() > 1)
                {
                    // load ao_rough_metal texture
                    std::string ao_rough_metal = tokens[1];
                    int width, height, byteStride;
                    unsigned char *imgBuff = stbi_load(ao_rough_metal.c_str(), &width, &height, &byteStride, 0);
                    if (imgBuff == nullptr)
                    {
                        cout << stbi_failure_reason() << endl;
                    }
                    for (int idx = 0; idx < width * height; idx++)
                    {
                        int tmpIdx = idx * byteStride;
                        texData[idx].amOc = imgBuff[tmpIdx];
                        texData[idx].rogh = imgBuff[tmpIdx + 1];
                        texData[idx].metl = imgBuff[tmpIdx + 2];
                    }
                    stbi_image_free(imgBuff);
                }
                // else if (newMaterial.useTex)
                // {
                //     for (int idx = 0; idx < newMaterial.texWidth * newMaterial.texHeight; idx++)
                //     {
                //         int tmpIdx = idx * 3;
                //         texData[idx].amOc = imgBuff[tmpIdx];
                //         texData[idx].rogh = imgBuff[tmpIdx + 1];
                //         texData[idx].metl = imgBuff[tmpIdx + 2];
                //     }
                // }
            }
            else if (strcmp(tokens[0].c_str(), "NORMALMAP") == 0)
            {
                if (tokens.size() > 1)
                {
                    // load normal/bump texture
                    std::string normtex = tokens[1];
                    int width, height, byteStride;
                    unsigned char *imgBuff = stbi_load(normtex.c_str(), &width, &height, &byteStride, 0);
                    if (imgBuff == nullptr)
                    {
                        cout << stbi_failure_reason() << endl;
                    }
                    for (int idx = 0; idx < width * height; idx++)
                    {
                        int tmpIdx = idx * 3;
                        texData[idx].bump[0] = imgBuff[tmpIdx];
                        texData[idx].bump[1] = imgBuff[tmpIdx + 1];
                        texData[idx].bump[2] = imgBuff[tmpIdx + 2];
                    }
                    stbi_image_free(imgBuff);
                }
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

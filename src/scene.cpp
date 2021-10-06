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
        }
    }
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

                // Loop over shapes
                for (size_t s = 0; s < shapes.size(); s++)
                {
                    // Loop over faces(polygon)
                    size_t index_offset = 0;
                    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
                    {
                        size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                        //! with ebon hawk model, all triangles so ignoring other polygons
                        Geom tri;
                        // Loop over vertices in the face.
                        for (size_t v = 0; v < fv; v++)
                        {
                            // access to vertex
                            tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                            tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                            tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                            tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                            tri.t.pos[v] = glm::vec3(vx, vy, vz);

                            // Check if `normal_index` is zero or positive. negative = no normal data
                            if (idx.normal_index >= 0)
                            {
                                tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                                tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                                tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                                tri.t.norm[v] = glm::vec3(nx, ny, nz);
                                // tri.t.norm[v] = glm::normalize(tri.t.pos[v]); // TMP for test norm interp
                            }

                            // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                            if (idx.texcoord_index >= 0)
                            {
                                tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                                tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                                tx = tx < 0 ? -tx : tx;
                                ty = ty < 0 ? -ty : ty;
                                // tx = tx < 0 ? 1 + tx : tx;
                                // ty = ty < 0 ? 1 + ty : ty;
                                tri.t.uv[v] = glm::vec2(tx, ty);
                                // if (f % 1024 == 0)
                                // {
                                //     cout << "sample uvs " << glm::to_string(tri.t.uv[v]) << endl;
                                // }
                            }

                            // Optional: vertex colors
                            // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                            // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                            // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
                        }
                        index_offset += fv;
                        triangs.push_back(tri);

                        // per-face material
                        shapes[s].mesh.material_ids[f];
                    }
                }
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

        //TODO: if is mesh, map these to every triangle in the mesh
        //TODO push all triangles in container to geoms
        //TODO else to the line below
        if (triangs.size() > 0)
        {
            for (auto &tri : triangs)
            {
                tri.translation = newGeom.translation;
                tri.rotation = newGeom.rotation;
                tri.scale = newGeom.scale;
                tri.transform = newGeom.transform;
                tri.inverseTransform = newGeom.inverseTransform;
                tri.invTranspose = newGeom.invTranspose;
                tri.materialid = newGeom.materialid;
                tri.type = TRIANGLE;
                // cout << "norm " << glm::to_string(tri.t.norm[0]) << " "
                //      << glm::to_string(tri.t.norm[1]) << " "
                //      << glm::to_string(tri.t.norm[2]) << endl;
                // for (int trivert = 0; trivert < 3; trivert++)
                // {
                //     tri.t.pos[trivert] = glm::vec3(tri.transform * glm::vec4(tri.t.pos[trivert], 1.f));
                //     tri.t.norm[trivert] = glm::vec3(tri.invTranspose * glm::vec4(tri.t.norm[trivert], 0.f));
                // }
            }
            geoms.insert(geoms.end(), triangs.begin(), triangs.end());
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
                if (tokens.size() > 1)
                {
                    // load color texture
                    std::string bcolorTexFile = tokens[1];
#define STBI_NO_FAILURE_STRINGS
#define STBI_FAILURE_USERMSG
                    int width, height, byteStride;
                    unsigned char *imgBuff = stbi_load(bcolorTexFile.c_str(), &width, &height, &byteStride, 0);
                    if (imgBuff == nullptr)
                    {
                        cout << stbi_failure_reason() << endl;
                    }
                    // std::fill(baseColorVec.begin(), baseColorVec.end(), glm::vec3(0.f));
                    baseColorVec.reserve(width * height);
                    // baseColorVec.reserve(width * height);
                    // ... process data if not NULL ...
                    // ... x = width, y = height, n = # 8-bit components per pixel ...
                    // ... replace '0' with '1'..'4' to force that many components per pixel
                    // ... but 'n' will always be the number that it would have been if you said 0

                    // The return value from an image loader is an 'unsigned char *' which points
                    // to the pixel data, or NULL on an allocation failure or if the image is
                    // corrupt or invalid. The pixel data consists of *y scanlines of *x pixels,
                    // with each pixel consisting of N interleaved 8-bit components; the first
                    // pixel pointed to is top-left-most in the image. There is no padding between
                    // image scanlines or between pixels, regardless of format. The number of
                    // components N is 'desired_channels' if desired_channels is non-zero, or
                    // *channels_in_file otherwise. If desired_channels is non-zero,
                    // *channels_in_file has the number of components that _would_ have been
                    // output otherwise. E.g. if you set desired_channels to 4, you will always
                    // get RGBA output, but you can check *channels_in_file to see if it's trivially
                    // opaque because e.g. there were only 3 channels in the source image.
                    // height lines of width pixels ->
                    //    u * width, v * height, floor both
                    //    nU, nV -> nU + width * nV
                    cout << "image read size " << width * height * byteStride << endl;
                    cout << "bytes per pix " << byteStride << endl;
                    for (int idx = 0; idx < width * height; idx++)
                    {
                        int tmpIdx = idx * 3;
                        baseColorVec.push_back(glm::vec3(imgBuff[tmpIdx], imgBuff[tmpIdx + 1], imgBuff[tmpIdx + 2]));
                    }
                    cout << "vector size " << baseColorVec.size() << endl;
                    stbi_image_free(imgBuff);
                }
            }
            else if (strcmp(tokens[0].c_str(), "EMITMAP") == 0)
            {
                if (tokens.size() > 1)
                {
                    // load color texture
                    // newMaterial.emissiveTexID;
                }
            }
            else if (strcmp(tokens[0].c_str(), "ROUGHMAP") == 0)
            {
                if (tokens.size() > 1)
                {
                    // load color texture
                    // newMaterial.roughTexID;
                }
            }
            else if (strcmp(tokens[0].c_str(), "NORMALMAP") == 0)
            {
                if (tokens.size() > 1)
                {
                    // load color texture
                    // newMaterial.normalTexID;
                }
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

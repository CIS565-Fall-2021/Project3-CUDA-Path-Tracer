#define TINYOBJLOADER_IMPLEMENTATION

#include "main.h"
#include "preview.h"
#include <cstring>
#include "tiny_obj_loader.h"
#include "polygon.h"

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
int iteration;

int width;
int height;

void LoadOBJ(const char* file, Geom& geom)
{
    Polygon p;
    std::vector<tinyobj::shape_t> shapes; std::vector<tinyobj::material_t> materials;
    std::string errors = tinyobj::LoadObj(shapes, materials, file);
    std::cout << errors << std::endl;
    if (errors.size() == 0)
    {
        int min_idx = 0;
        //Read the information from the vectxor of shape_ts
        for (int i = 0; i < shapes.size(); i++)
        {
            std::vector<glm::vec4> pos, nor;
            std::vector<glm::vec2> uv;
            std::vector<float>& positions = shapes[i].mesh.positions;
            std::vector<float>& normals   = shapes[i].mesh.normals;
            for (int j = 0; j < positions.size() / 3; j++)
            {
                pos.push_back(glm::vec4(positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2], 1));
            }
            for (int j = 0; j < normals.size() / 3; j++)
            {
                nor.push_back(glm::vec4(normals[j * 3], normals[j * 3 + 1], normals[j * 3 + 2], 0));
            }
            for (int j = 0; j < pos.size(); j++)
            {
                p.AddVertex(Vertex(pos[j], glm::vec3(255, 255, 255), nor[j], glm::vec2(0.f, 0.f)));
            }

            std::vector<unsigned int> indices = shapes[i].mesh.indices;
            for (unsigned int j = 0; j < indices.size(); j += 3)
            {
                Triangle t;
                t.m_indices[0] = indices[j] + min_idx;
                t.m_indices[1] = indices[j + 1] + min_idx;
                t.m_indices[2] = indices[j + 2] + min_idx;
                p.AddTriangle(t);
            }

            min_idx += pos.size();
        }
    }
    else
    {
        //An error loading the OBJ occurred!
        std::cout << errors << std::endl;
        return;
    }
    
    geom.triCount = p.m_tris.size(); 
    geom.host_VecNorArr = (glm::vec4*) malloc(p.m_tris.size() * 6 * sizeof(glm::vec4));

    for (int i = 0; i < p.m_tris.size(); i++) {
        for (int j = 0; j < 3; j++) {
            geom.host_VecNorArr[6 * i + 2 * j]     = p.m_verts[p.m_tris[i].m_indices[j]].m_pos;
            geom.host_VecNorArr[6 * i + 2 * j + 1] = p.m_verts[p.m_tris[i].m_indices[j]].m_normal;
        }
    }
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];
    
#if USE_MESH_LOADING
    const char* objFile = NULL;
    if (argc > 2 && USE_MESH_LOADING) {
        objFile = argv[2];
    }
#endif // USE_MESH_LOADING

    // Load scene file
    scene = new Scene(sceneFile);

#if USE_MESH_LOADING
    const char* objPath = "C:/Users/yangr/OneDrive/Desktop/wahoo.obj";
    for (int i = 0; i < scene->geoms.size(); i++) {
        if (scene->geoms[i].type == GeomType::OBJ) {
            LoadOBJ(objPath, scene->geoms[i]);

            float minX = std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float minZ = std::numeric_limits<float>::max();

            float maxX = std::numeric_limits<float>::min();
            float maxY = std::numeric_limits<float>::min();
            float maxZ = std::numeric_limits<float>::min();

            for (int i = 0; i < scene->geoms[i].triCount * 6; i += 2) {
                minX = (scene->geoms[i].host_VecNorArr[i].x < minX) ? scene->geoms[i].host_VecNorArr[i].x : minX;
                minY = (scene->geoms[i].host_VecNorArr[i].y < minY) ? scene->geoms[i].host_VecNorArr[i].y : minY;
                minZ = (scene->geoms[i].host_VecNorArr[i].z < minZ) ? scene->geoms[i].host_VecNorArr[i].z : minZ;

                maxX = (scene->geoms[i].host_VecNorArr[i].x > maxX) ? scene->geoms[i].host_VecNorArr[i].x : maxX;
                maxY = (scene->geoms[i].host_VecNorArr[i].y > maxY) ? scene->geoms[i].host_VecNorArr[i].y : maxY;
                maxZ = (scene->geoms[i].host_VecNorArr[i].z > maxZ) ? scene->geoms[i].host_VecNorArr[i].z : maxZ;
            }

            scene->geoms[i].bbScale = glm::vec3(glm::abs((maxX - minX) / 2.0f), glm::abs((maxY - minY) / 2.0f), glm::abs((maxZ - minZ) / 2.0f));
            scene->geoms[i].bbInverseTransform = glm::inverse(utilityCore::buildTransformationMatrix(
                scene->geoms[i].translation, scene->geoms[i].rotation, scene->geoms[i].bbScale));
        } 

    }
#endif // USE_MESH_LOADING

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

#if USE_DOF
    cam.aperture  = DOF_APERATURE;
    cam.focusDist = DOF_FOCUSDIST; 
#endif // USE_DOF


    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage() {
    float samples = iteration;
    // output image file
    image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
    if (camchanged) {
        iteration = 0;
        Camera &cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
      }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations) {
        uchar4 *pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    } else {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
#if USE_MESH_LOADING
        for (int i = 0; i < scene->geoms.size(); i++) {
            if (scene->geoms[i].type == GeomType::OBJ) {
                free(scene->geoms[i].host_VecNorArr);
            }
        }
#endif // USE_MESH_LOADING
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        saveImage();
        break;
      case GLFW_KEY_SPACE:
        camchanged = true;
        renderState = &scene->state;
        Camera &cam = renderState->camera;
        cam.lookAt = ogLookAt;
        break;
      }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  }
  else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}

#include "main.h"
#include "preview.h"
#include <cstring>
#include <glm/gtx/string_cast.hpp>
// #include "intersections.h"
#include "sceneStructs.h"

// #include "intersections.h"

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

//-------------------------------
//-------------MAIN--------------
//-------------------------------
// glm::vec3 multiplyMV2(glm::mat4 m, glm::vec4 v)
// {
//   return glm::vec3(m * v);
// }
// glm::vec3 triIntersect2(Ray const &r, Triangle const &tri)
// {
//   using namespace glm;
//   vec3 ro = r.origin;
//   vec3 rd = normalize(r.direction);
//   vec3 vertA = tri.pos[0];
//   vec3 vertB = tri.pos[1];
//   vec3 vertC = tri.pos[2];
//   vec3 aToB = vertB - vertA;
//   vec3 aToC = vertC - vertA;
//   vec3 aToRo = ro - vertA;

//   // q, intersect point in triangle, = v0 + alpha(v1-v0) + beta(v2-v0)
//   // ray is p + td where p is point and d is direction
//   // Mx = s -> M = {-d, v1-v0, v2-v0},
//   //           x = {p-a},
//   //           s = {t, alpha, beta}Transpose

//   vec3 triNorm = cross(aToB, aToC); // Triangle normal from points
//   vec3 q = cross(aToRo, rd);        // used for scalar triple product
//   float negDet = dot(rd, triNorm);
//   float negReciporicalDet = 1.f / negDet;
//   float t = negReciporicalDet * -1.f * dot(triNorm, aToRo);
//   float u = negReciporicalDet * -1.f * dot(q, aToC);
//   float v = negReciporicalDet * dot(q, aToB);
//   if (u < 0.f || u > 1.f || v < 0.f || (u + v) > 1.f)
//   {
//     t = -1.f;
//   }
//   return vec3(t, u, v);
// }
// float triangleIntersectionTest2(Geom const &tri, Ray r, glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside)
// {
//   Ray q; // in triangle space
//   q.origin = multiplyMV2(tri.inverseTransform, glm::vec4(r.origin, 1.0f));
//   q.direction = glm::normalize(multiplyMV2(tri.inverseTransform, glm::vec4(r.direction, 0.0f)));
//   glm::vec3 tuv = triIntersect2(q, tri.t);
//   // glm::vec3 tuv(0.f);
//   if (tuv.x == -1.f)
//   {
//     return -1.f;
//   }
//   glm::vec3 normalTri(
//       (1.f - tuv.y - tuv.z) * tri.t.norm[0] +
//       tuv.y * tri.t.norm[1] +
//       tuv.z * tri.t.norm[2]);
//   glm::vec3 intersectionPointTri(
//       q.origin + tuv.x * q.direction);
//   float tmp = glm::dot(normalTri, q.direction);
//   cout << "tmp " << tmp << endl;
//   outside = tmp < 0.f;
//   // back from triangle space
//   intersectionPoint = multiplyMV2(tri.transform, glm::vec4(intersectionPointTri, 1.f));
//   normal = glm::normalize(multiplyMV2(tri.invTranspose, glm::vec4(normalTri, 0.f)));
//   return glm::length(r.origin - intersectionPoint);
// }
int main(int argc, char **argv)
{
  // // Testing
  // // triangle 000 300 030, ray from 1,1,-1 in the 0,0,1 direction should be t=1
  // bool outside;
  // glm::vec3 isecpt, norm;
  // struct Ray ray;
  // ray.direction = glm::vec3(0.f, 0.f, -1.f);
  // ray.origin = glm::vec3(1.f, 1.f, 1.f);
  // struct Geom trigeom;
  // trigeom.inverseTransform = glm::mat4(1.f);
  // trigeom.invTranspose = glm::mat4(1.f);
  // trigeom.transform = glm::mat4(1.f);
  // trigeom.type = TRIANGLE;
  // trigeom.t.norm[0] = glm::vec3(0, 0, 1);
  // trigeom.t.norm[1] = glm::vec3(0, 0, 1);
  // trigeom.t.norm[2] = glm::vec3(0, 0, 1);
  // // trigeom.t.norm = {glm::vec3(1, 0, 0), glm::vec3(1, 0, 0), glm::vec3(1, 0, 0)};
  // trigeom.t.pos[0] = glm::vec3(0);
  // trigeom.t.pos[1] = glm::vec3(3, 0, 0);
  // trigeom.t.pos[2] = glm::vec3(0, 3, 0);
  // // trigeom.t.pos = {glm::vec3(0), glm::vec3(3, 0, 0), glm::vec3(0, 3, 0)};
  // auto t_parametric = triangleIntersectionTest2(trigeom, ray, isecpt, norm, outside);
  // cout << "collision t is " << t_parametric << endl;
  // cout << "intersection point is " << glm::to_string(isecpt) << endl;
  // cout << "tri normal is " << glm::to_string(norm) << endl;
  // cout << "isoutside = " << outside << endl;
  // // End Testing

  startTimeString = currentTimeString();

  if (argc < 2)
  {
    printf("Usage: %s SCENEFILE.txt\n", argv[0]);
    return 1;
  }

  const char *sceneFile = argv[1];

  // Load scene file
  scene = new Scene(sceneFile);

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

void saveImage()
{
  float samples = iteration;
  // output image file
  image img(width, height);

  for (int x = 0; x < width; x++)
  {
    for (int y = 0; y < height; y++)
    {
      int index = x + (y * width);
      glm::vec3 pix = renderState->image[index];
      img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
    }
  }

  std::string filename = renderState->imageName;
  std::ostringstream ss;
  ss << "../renders/" << filename << "." << startTimeString << "." << samples << "samp";
  // ss << filename << "." << startTimeString << "." << samples << "samp";
  filename = ss.str();

  // CHECKITOUT
  img.savePNG(filename);
  //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
  if (camchanged)
  {
    iteration = 0;
    Camera &cam = renderState->camera;
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.y = zoom * cos(theta);
    cameraPosition.z = zoom * cos(phi) * sin(theta);

    cam.view = -glm::normalize(cameraPosition);
    glm::vec3 v = cam.view;
    glm::vec3 u = glm::vec3(0, 1, 0); //glm::normalize(cam.up);
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

  if (iteration == 0)
  {
    pathtraceFree();
    pathtraceInit(scene);
  }

  if (iteration < renderState->iterations)
  {
    uchar4 *pbo_dptr = NULL;
    iteration++;
    cudaGLMapBufferObject((void **)&pbo_dptr, pbo);

    // execute the kernel
    int frame = 0;
    pathtrace(pbo_dptr, frame, iteration);

    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);
  }
  else
  {
    saveImage();
    pathtraceFree();
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
  }
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (action == GLFW_PRESS)
  {
    switch (key)
    {
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

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow *window, double xpos, double ypos)
{
  if (xpos == lastX || ypos == lastY)
    return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed)
  {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (rightMousePressed)
  {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  }
  else if (middleMousePressed)
  {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
    cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}

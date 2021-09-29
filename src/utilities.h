#pragma once

#pragma warning(push)
#pragma warning(disable:4244)
#pragma warning(disable:4996)

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define INV_PI            0.31830988618f
#define HALF_INV_PI       0.15915494309f
#define EPSILON           0.00001f

#define ENABLE_COMPACTION 0//1
#define ENABLE_SORTING 0//1
#define JITTER_ANTI_ALIASING 1//1
#define ENABLE_BVH 0//1
#define ENABLE_ADVANCED_PIPELINE 1//0

using ui8 = unsigned char;
using ui16 = unsigned short;
using ui32 = unsigned int;
using ui64 = unsigned long long;
using i8 = char;
using i16 = short;
using i32 = int;
using i64 = long long;
using f32 = float;
using f64 = double;

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
    extern std::string getBaseDirectory(const std::string& filename);

    template<typename T>
    T* getPtrInStruct(const void* struct_ptr,i64 offset);
    extern i64 getAddrOffsetInStruct(const void* struct_ptr, const void* var_ptr);
    extern std::string getFileExtension(const std::string& filename);
}

template<typename T>
T* utilityCore::getPtrInStruct(const void* struct_ptr, i64 offset) {
    return reinterpret_cast<T*>(reinterpret_cast<i64>(struct_ptr) + offset);
}
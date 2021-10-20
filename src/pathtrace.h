#pragma once

#include <vector>

#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBufferNormal(uchar4 *pbo);
void showGBufferPosition(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);

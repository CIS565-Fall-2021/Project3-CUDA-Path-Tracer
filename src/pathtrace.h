#pragma once

#include <vector>

#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBufferNormal(uchar4 *pbo);
void showGBufferPosition(uchar4 *pbo);
void showGBufferWeights(uchar4 *pbo);
void showGBufferPositionWeights(uchar4 *pbo);
void showGBufferNormalWeights(uchar4 *pbo);
void showGBufferColorWeights(uchar4 *pbo);
void showImage(uchar4 *pbo, int iter);
void showDenoisedImage(uchar4 *pbo, int iter);
void denoiseImage(int filter_width, float c_phi, float n_phi, float p_phi);

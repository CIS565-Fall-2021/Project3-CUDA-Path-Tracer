#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, bool sortByMaterial, bool cachFirstBounce, bool stochasticAA, bool depthOfField, 
					bool boundingVolumeCulling);
performanceAnalysis::PerformanceTimer& timer();

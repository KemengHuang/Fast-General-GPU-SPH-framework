#ifndef _PARAMETERS_H
#define _PARAMETERS_H

#include <cuda_runtime.h>

// common
extern const bool DEBUG;
extern const float TIME_STEP;
extern const float3 WORLD_SIZE;
extern const float3 GRAVITY;

// SPH simulation
extern const float KERNAL_RADIUS;
extern const float MASS;
extern const float VICOSITY_COEFFICIENT;
extern const float REST_DENSITY;
extern const float WALL_DAMPING;
extern const float GAS_CONSTANT;
extern const int pcisph_min_loops;
extern const int pcisph_max_loops;
extern const float pcisph_max_density_error_allowed;

// Eulerian simulation
extern const int eulerDim[3];

#endif /*_PARAMETERS_H*/
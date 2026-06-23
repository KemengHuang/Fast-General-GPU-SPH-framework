//
// camera_state.cpp
// Definitions for the camera / GUI global state.
//

#include "render/camera_state.h"

float window_width = 1000.0f;
float window_height = 750.0f;

float xRot = 0.0f;
float yRot = 0.0f;
float xTrans = 0.0f;
float yTrans = 0.0f;
float zTrans = -175.0f;

int psize = 12;
int ox = 0;
int oy = 0;
int buttonState = 0;
float xRotLength = 0.0f;
float yRotLength = 0.0f;

float3 real_world_origin;
float3 real_world_side;
float3 sim_ratio;

float world_width = 0.0f;
float world_height = 0.0f;
float world_length = 0.0f;

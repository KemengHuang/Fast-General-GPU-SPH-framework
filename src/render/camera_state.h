//
// camera_state.h
// Camera / GUI global state shared between the application and the renderer.
//

#ifndef _CAMERA_STATE_H
#define _CAMERA_STATE_H

#include "core/sph_header.h"

extern float window_width;
extern float window_height;

extern float xRot;
extern float yRot;
extern float xTrans;
extern float yTrans;
extern float zTrans;

extern int psize;
extern int ox;
extern int oy;
extern int buttonState;
extern float xRotLength;
extern float yRotLength;

extern float3 real_world_origin;
extern float3 real_world_side;
extern float3 sim_ratio;

extern float world_width;
extern float world_height;
extern float world_length;

#endif/*_CAMERA_STATE_H*/

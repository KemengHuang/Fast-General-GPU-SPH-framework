#ifndef __SPHDATA_H__
#define __SPHDATA_H__

#include "sph_header.h"

float window_width = 1000;
float window_height = 750;

float xRot = 0.0f;
float yRot = 0.0f;
float xTrans = 0;
float yTrans = 0;
float zTrans = -175.0;

int psize = 12;
int ox;
int oy;
int buttonState;
float xRotLength = 0.0f;
float yRotLength = 0.0f;

float3 real_world_origin;
float3 real_world_side;
float3 sim_ratio;

float world_width;
float world_height;
float world_length;

#endif
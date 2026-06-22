//
// sph_marching_cube.h
// Hybrid_Parallel_SPH 
//
// created by ruanjm on 22/04/16
// Copyright (c) 2016 ruanjm. All rights reserved.
//

#ifndef _SPH_MARCHING_CUBE_H
#define _SPH_MARCHING_CUBE_H

#include <vector_functions.h>
#include "sph_parameter.h"

namespace sph
{

bool generateMesh(float3 *pos, unsigned int nump, SystemParameter *sys_para, unsigned int loop_times);

void outputMesh(SystemParameter *sys_para, unsigned int loop_times);

}

#endif/*_SPH_MARCHING_CUBE_H*/

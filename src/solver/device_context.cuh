//
// device_context.cuh
// CUDA context symbols and kernel management helpers.
//

#ifndef _SOLVER_DEVICE_CONTEXT_CUH_
#define _SOLVER_DEVICE_CONTEXT_CUH_

#include "solver/kernel_common.cuh"
#include "core/cuda_call_check.h"

namespace sph {

extern cudaStream_t sms_stream;
extern cudaEvent_t sms_density_event;
extern cudaEvent_t sms_force_event;

void transSysParaToDevice(const SystemParameter *host_para);
void initializeKernel();
void releaseKernel();

}

#endif

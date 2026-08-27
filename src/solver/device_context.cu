//
// device_context.cu
// CUDA context symbols and kernel management helpers.
//

#include "solver/device_context.cuh"

namespace sph {

__constant__ SystemParameter kDevSysPara;

cudaStream_t sms_stream;
cudaEvent_t sms_density_event;
cudaEvent_t sms_force_event;

void transSysParaToDevice(const SystemParameter *host_para)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(kDevSysPara, host_para, sizeof(SystemParameter)));
}

void initializeKernel()
{
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&sms_stream, cudaStreamDefault));
    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&sms_density_event, cudaEventDisableTiming));
    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&sms_force_event, cudaEventDisableTiming));
}

void releaseKernel()
{
    CUDA_SAFE_CALL(cudaStreamDestroy(sms_stream));
    CUDA_SAFE_CALL(cudaEventDestroy(sms_density_event));
    CUDA_SAFE_CALL(cudaEventDestroy(sms_force_event));
}

}

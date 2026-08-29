#include "render/particle_vbo_copy.cuh"

#include "core/sph_utils.cuh"

namespace sph
{
namespace
{

__global__ void copyParticleDataToVBOsKernel(ParticleBufferList buffer_list,
                                             unsigned int particle_count,
                                             float3 *device_positions,
                                             uint *device_colors)
{
    const unsigned int index = threadIdx.x +
                               __umul24(blockIdx.x, blockDim.x);
    if (index >= particle_count) return;

    device_positions[index] = buffer_list.final_position[index];
    device_colors[index] = buffer_list.color[index];
}

} // namespace

void copyParticleDataToVBOs(ParticleBufferList buffer_list,
                            unsigned int particle_count,
                            float3 *device_positions,
                            uint *device_colors)
{
    if (particle_count == 0 || !device_positions || !device_colors) return;

    constexpr int threads_per_block = 128;
    const int block_count = ceil_int(static_cast<int>(particle_count),
                                     threads_per_block);
    copyParticleDataToVBOsKernel<<<block_count, threads_per_block>>>(
        buffer_list, particle_count, device_positions, device_colors);
}

} // namespace sph

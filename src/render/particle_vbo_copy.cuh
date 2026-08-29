#ifndef GSPH_PARTICLE_VBO_COPY_CUH_
#define GSPH_PARTICLE_VBO_COPY_CUH_

#include "particle/particle_buffer.h"

namespace sph
{

void copyParticleDataToVBOs(ParticleBufferList buffer_list,
                            unsigned int particle_count,
                            float3 *device_positions,
                            uint *device_colors);

} // namespace sph

#endif // GSPH_PARTICLE_VBO_COPY_CUH_

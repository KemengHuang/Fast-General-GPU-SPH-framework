//
// particle_buffer.h
// Hybrid_Parallel_SPH
//
// Particle buffer layout and buffer object management.
//

#ifndef _PARTICLE_BUFFER_H
#define _PARTICLE_BUFFER_H

#include <windows.h>
#include "core/cuda_call_check.h"

typedef unsigned int uint;

namespace sph
{

struct Vlmfraction {
    float a1;
    float a2;
};

struct mixVelocity {
    float3 v1;
    float3 v2;
    float3 mixv;
};

struct DriftVelocity {
    float3 Vm1;
    float3 Vm2;
};

struct mixMass {
    float mass1;
    float mass2;
    float massMix;
};

struct mixDensity {
    float den1;
    float den2;
    float mixDen;
};

struct mixPressure {
    float pres1;
    float pres2;
    float mixPres;
};

enum condition { FLUID, SOLID };

// Bit-packed block task used by the grid scheduler.
// xxi/xxx, yyi/yyy, zzi/zzz bound the task's local micro-cell coordinates.
// isSame is assigned by the historical judgeTask same-cell pairing policy.
// For a paired block, the first descriptor stores the combined search bounds
// in paired_bounds while its hot x/y/z fields retain the original task bounds.
struct BlockTask {
    char isSame;
    // Six 2-bit paired bounds fit in padding that precedes cellid. The hot
    // x/y/z fields stay original; cached cell metadata brings the task to 32 B.
    unsigned short paired_bounds;
    int cellid;
    ushort3 cell_pos;
    unsigned short p_offset;
    char xxi;
    char xxx;
    char yyi;
    char yyy;
    char zzi;
    char zzz;
    int cell_begin;
    int cell_particle_count;
};

struct SameCellForceAccum
{
    float3 pressure;
    float3 viscosity;
    float3 gradient;
    float laplacian;
};

static_assert(sizeof(BlockTask) == 32,
              "Unexpected BlockTask layout");

__host__ __device__ inline unsigned short packSmsTaskBounds(
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z)
{
    return static_cast<unsigned short>(
        min_x | (max_x << 2) | (min_y << 4) |
        (max_y << 6) | (min_z << 8) | (max_z << 10));
}

enum BufferType {
    kBuffTypeNone,
    kBuffTypeDevice,
    kBuffTypeHostPageable,
    kBuffTypeHostPinned
};

struct sumGrad {
    float3 sumGradW = make_float3(0.0f, 0.0f, 0.0f);
    float sumGradWDot = 0.0f;
    uint num_neigh = 0;
};

struct ParticleIdxRange {
    __host__ __device__ ParticleIdxRange() {}
    __host__ __device__ ParticleIdxRange(int b, int e) : begin(b), end(e) {}
    int begin, end;
};

struct ParticleBufferList {
    float4* __restrict__ position_d;
    float4* __restrict__ evaluated_velocity;
    float3* __restrict__ velocity;
    float3* __restrict__ acceleration;

    float3* __restrict__ final_position;

    // PCI-SPH fields
    float3* predicted_pos;
    float3* correction_pressure_force;
    float* pressure;
    float* predicted_density;
    float* densityError;
    float* correction_pressure;

    unsigned int* __restrict__ color;
    condition* phase;   // particle phase/type

    Vlmfraction* vlfrt;
    mixVelocity* mixV;
    DriftVelocity* Vm;
    mixPressure* mixP;
    mixMass* mixM;
    mixDensity* Mixden;

    __device__ condition & getPhase(unsigned int idx) {
        return phase[idx];
    }
};

class ParticleBufferObject {
public:
    explicit ParticleBufferObject();
    ParticleBufferObject(unsigned int cap, BufferType type);
    ParticleBufferObject(const ParticleBufferObject&) = delete;
    ParticleBufferObject& operator=(const ParticleBufferObject&) = delete;
    ~ParticleBufferObject();

    void allocate(unsigned int nump, BufferType type);
    void allocateSubBuffer(const ParticleBufferObject* base_buffer);
    void reallocate(unsigned int new_nump);
    void free();
    void transfer(ParticleBufferObject &dst_buff_obj, unsigned int offset, unsigned int nump, cudaMemcpyKind kind);

    inline BufferType get_type() { return type_; }
    inline unsigned int get_capacity() { return capacity_; }
    inline ParticleBufferList& get_buff_list() { return buff_list_; }

    void swapObj(ParticleBufferObject &obj);

private:
    ParticleBufferList buff_list_;
    BufferType type_ = kBuffTypeNone;
    unsigned int capacity_ = 0U;
    const ParticleBufferObject *base_buffer_ = nullptr;
};

} // namespace sph

#endif/*_PARTICLE_BUFFER_H*/

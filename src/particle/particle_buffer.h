//
// particle_buffer.h
// Hybrid_Parallel_SPH
//
// Particle buffer layout and buffer object management.
//

#ifndef _PARTICLE_BUFFER_H
#define _PARTICLE_BUFFER_H

#include <windows.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
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
// xxi/xxx, yyi/yyy, zzi/zzz encode whether the neighbor cell offset is -1, 0 or +1
// along each axis and whether the current block is the same cell as the neighbor.
struct BlockTask {
    char isSame;
    int cellid;
    unsigned short p_offset;
    char xxi;
    char xxx;
    char yyi;
    char yyy;
    char zzi;
    char zzz;
};

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
    float4* position_d;
    float4* evaluated_velocity;
    float3* velocity;
    float3* acceleration;

    float3* final_position;

    // PCI-SPH fields
    float3* predicted_pos;
    float3* correction_pressure_force;
    float* pressure;
    float* predicted_density;
    float* densityError;
    float* correction_pressure;

    unsigned int* color;
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
    ParticleBufferObject& operator=(const ParticleBufferList&) = delete;
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

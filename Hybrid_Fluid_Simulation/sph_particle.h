//
// particle.h
// SPH_SMS_Benchmark
//
// created by ruanjm on 2016/03/26
// Copyright (c) 2016 ruanjm. All rights reserved.
//

#ifndef _PARTICLE_H
#define _PARTICLE_H

#include <windows.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "cuda_call_check.h"
//#include "sph_hybrid_system.h"

typedef unsigned int uint;

namespace sph
{
struct Vlmfraction{
    float a1;
    float a2;
};

struct mixVelocity{
    float3 v1;
    float3 v2;
    float3 mixv;
};

struct DriftVelocity{
    float3 Vm1;
    float3 Vm2;
};

struct mixMass{
    float mass1;
    float mass2;
    float massMix;
};


struct mixDensity{
    float den1;
    float den2;
    float mixDen;
};

struct mixPressure{
    float pres1;
    float pres2;
    float mixPres;
};










//sf add 粒子状态
enum condition { FLUID, SOLID,  };



struct BlockTask
{
	
	char isSame;
	int cellid;
//	ushort3 cell_pos;
	unsigned short p_offset;
	char xxi;
	char xxx;
	char yyi;
	char yyy;
	char zzi;
	char zzz;


//	char zzz1;
//	char zzz2; 
//	char zzz3; 
//	char zzz41;
//	char zzz5;
//	char zzz6;
//	char zzz7;

    
	
	
};

enum BufferType
{
    kBuffTypeNone,
    kBuffTypeDevice,
    kBuffTypeHostPageable,
    kBuffTypeHostPinned
};

struct sumGrad{
	//public:
	//sumGrad(){ sumGradW = make_float3(0.0, 0.0, 0.0); sumGradWDot = 0; num_neigh = 0; }
	float3 sumGradW = make_float3(0.0, 0.0, 0.0);
	float sumGradWDot = 0.0f;
	uint num_neigh = 0;
	//condition ph;
};

struct ParticleBufferList
{
    float4* position_d;
    float4* evaluated_velocity;
    float3* velocity;
    float3* acceleration;
    
    float3* final_position;

    float3*		predicted_pos;  //sf
    float3*		correction_pressure_force;  //sf
  //  float*  density;
    float*  pressure;
    //float*  surface_normal_vector;
    
    //unsigned int* particle_type;

	//sf pcisph--------------------------------
	
	float*		predicted_density; //sf
	float*		densityError;//sf
	float*		correction_pressure;  //sf

    unsigned int* color;
	//sf others-------------------------------
	condition*	phase;             //sf 粒子种类




    Vlmfraction* vlfrt;
    mixVelocity* mixV;
    DriftVelocity* Vm;
    mixPressure* mixP;
    mixMass* mixM;
    mixDensity* Mixden;


    __device__
    condition & getPhase(unsigned int idx) {
        return phase[idx];
    }
};

class ParticleBufferObject
{
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

    //ParticleBufferList& mapResources();
    //void unmapResources();

    //GLuint get_final_position_vbo() { return final_position_vbo_; }
    //GLuint get_color_vbo() { return color_vbo_; }

    inline BufferType get_type() { return type_; }
    inline unsigned int get_capacity() { return capacity_; }
    inline ParticleBufferList& get_buff_list() { return buff_list_; }

    void swapObj(ParticleBufferObject &obj);

private:
    ParticleBufferList buff_list_;
    BufferType type_ = kBuffTypeNone;
    unsigned int capacity_ = 0U;
    const ParticleBufferObject *base_buffer_ = nullptr;
    //GLuint final_position_vbo_, color_vbo_;
    //cudaGraphicsResource *final_position_vbo_cuda_, *color_vbo_cuda_;
};


}

#endif/*_PARTICLE_H*/
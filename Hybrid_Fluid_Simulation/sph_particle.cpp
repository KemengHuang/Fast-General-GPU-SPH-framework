//
// sph_particle.cpp
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#include "sph_particle.h"

namespace sph
{

/****************************** Tools ******************************/

template<typename T>
inline void swap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}

template<typename T>
inline void reallocDeviceBuffer(T *&p, unsigned int old_nump, unsigned int new_nump)
{
    T *temp;
    CUDA_SAFE_CALL(cudaMalloc(&temp, new_nump * sizeof(T)));
    CUDA_SAFE_CALL(cudaMemcpy(temp, p, old_nump * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaFree(p));
    p = temp;
}

//inline void reallocGLBuffer(GLuint &vbo, cudaGraphicsResource *&vbo_cuda, cudaGraphicsMapFlags map_flag, unsigned int new_size/* in bytes */)
//{
//    GLuint temp_vbo;
//    cudaGraphicsResource *temp_vbo_cuda;
//
//    glGenBuffers(1, &temp_vbo);
//    glBindBuffer(GL_ARRAY_BUFFER, temp_vbo);
//    glBufferData(GL_ARRAY_BUFFER, new_size, 0, GL_DYNAMIC_DRAW);
//    glBindBuffer(GL_ARRAY_BUFFER, 0);
//    CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&temp_vbo_cuda, temp_vbo, map_flag));
//
//    // copy data
//    if (cudaGraphicsMapFlagsNone == map_flag || cudaGraphicsMapFlagsReadOnly == map_flag)
//    {
//        GLint data_size;
//        glBindBuffer(GL_COPY_READ_BUFFER, vbo);
//        glGetBufferParameteriv(GL_COPY_READ_BUFFER, GL_BUFFER_SIZE, &data_size);
//        glBindBuffer(GL_COPY_WRITE_BUFFER, temp_vbo);
//        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, data_size);
//        glBindBuffer(GL_COPY_READ_BUFFER, 0);
//        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
//    }
//
//    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(vbo_cuda));
//    glDeleteBuffers(1, &vbo);
//
//    vbo = temp_vbo;
//    vbo_cuda = temp_vbo_cuda;
//}

template<typename T>
inline void reallocHostPinnedBuffer(T *&p, unsigned int old_nump, unsigned int new_nump)
{
    T *temp;
    CUDA_SAFE_CALL(cudaMallocHost(&temp, new_nump * sizeof(T)));
    std::memcpy(temp, p, old_nump * sizeof(T));
    CUDA_SAFE_CALL(cudaFreeHost(p));
    p = temp;
}

template<typename T>
inline void reallocHostPageableBuffer(T *&p, unsigned int old_nump, unsigned int new_nump)
{
    T *temp;
    temp = new T[new_nump];
    std::memcpy(temp, p, old_nump * sizeof(T));
    delete[]p;
    p = temp;
}

/****************************** Interface ******************************/

/****************************** ParticleBufferObject ******************************/
ParticleBufferObject::ParticleBufferObject()
{

}

ParticleBufferObject::ParticleBufferObject(unsigned int cap, BufferType type)
{
    allocate(cap, type);
}

ParticleBufferObject::~ParticleBufferObject()
{
    this->free();
}

void ParticleBufferObject::allocate(unsigned int nump, BufferType type)
{
    if (kBuffTypeNone != type_) return;

    capacity_ = nump;
    type_ = type;

    if (kBuffTypeDevice == type)
    {
        CUDA_SAFE_CALL(cudaMalloc(&buff_list_.position_d, nump * sizeof(float4)));
        CUDA_SAFE_CALL(cudaMalloc(&buff_list_.velocity, nump * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMalloc(&buff_list_.acceleration, nump * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMalloc(&buff_list_.evaluated_velocity, nump * sizeof(float4)));
        CUDA_SAFE_CALL(cudaMalloc(&buff_list_.final_position, nump * sizeof(float3)));
  //      CUDA_SAFE_CALL(cudaMalloc(&buff_list_.density, nump * sizeof(float)));
 //       CUDA_SAFE_CALL(cudaMalloc(&buff_list_.pressure, nump * sizeof(float)));
        //CUDA_SAFE_CALL(cudaMalloc(&buff_list_.surface_normal_vector, nump * sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc(&buff_list_.color, nump * sizeof(unsigned int)));
        //CUDA_SAFE_CALL(cudaMalloc(&buff_list_.particle_type, nump * sizeof(unsigned int)));
		
	

//		CUDA_SAFE_CALL(cudaMalloc(&buff_list_.phase, nump * sizeof(condition)));



    }
    if (kBuffTypeHostPinned == type)
    {
        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.position_d, nump * sizeof(float4)));
        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.velocity, nump * sizeof(float3)));
//        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.acceleration, nump * sizeof(float3)));
//        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.evaluated_velocity, nump * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.final_position, nump * sizeof(float3)));
//        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.density, nump * sizeof(float)));
//        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.pressure, nump * sizeof(float)));
        //CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.surface_normal_vector, nump * sizeof(float)));
        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.color, nump * sizeof(unsigned int)));
        //CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.particle_type, nump * sizeof(unsigned int)));

	

	//	CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.phase, nump * sizeof(condition)));




    }
    if (kBuffTypeHostPageable == type)
    {
        buff_list_.position_d = new float4[nump];
        buff_list_.velocity = new float3[nump];
        buff_list_.acceleration = new float3[nump];
        buff_list_.evaluated_velocity = new float4[nump];
        buff_list_.final_position = new float3[nump];
//        buff_list_.density = new float[nump];
 //       buff_list_.pressure = new float[nump];
        //buff_list_.surface_normal_vector = new float[nump];
        buff_list_.color = new unsigned int[nump];
        //buff_list_.particle_type = new unsigned int[nump];

	
    }
}

void ParticleBufferObject::allocateSubBuffer(const ParticleBufferObject* base_buffer)
{
    if (kBuffTypeNone != type_) return;

    capacity_ = base_buffer->capacity_;
    type_ = base_buffer->type_;
    base_buffer_ = base_buffer;

    if (kBuffTypeDevice == type_)
    {
        CUDA_SAFE_CALL(cudaMalloc(&buff_list_.position_d, capacity_ * sizeof(float4)));
        CUDA_SAFE_CALL(cudaMalloc(&buff_list_.velocity, capacity_ * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMalloc(&buff_list_.evaluated_velocity, capacity_ * sizeof(float4)));
        //CUDA_SAFE_CALL(cudaMalloc(&buff_list_.particle_type, capacity_ * sizeof(unsigned int)));
		CUDA_SAFE_CALL(cudaMalloc(&buff_list_.color, capacity_ * sizeof(unsigned int)));

		//sf add
//		CUDA_SAFE_CALL(cudaMalloc(&buff_list_.phase, capacity_ * sizeof(condition)));

        buff_list_.acceleration = base_buffer->buff_list_.acceleration;
 //       buff_list_.density = base_buffer->buff_list_.density;
 //       buff_list_.pressure = base_buffer->buff_list_.pressure;
        //buff_list_.surface_normal_vector = base_buffer->buff_list_.surface_normal_vector;
        buff_list_.final_position = base_buffer->buff_list_.final_position;
		
	
    }
    if (kBuffTypeHostPinned == type_)
    {
        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.position_d, capacity_ * sizeof(float4)));
        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.velocity, capacity_ * sizeof(float3)));
//        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.evaluated_velocity, capacity_ * sizeof(float3)));
        CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.color, capacity_ * sizeof(unsigned int)));
        //CUDA_SAFE_CALL(cudaMallocHost(&buff_list_.particle_type, capacity_ * sizeof(unsigned int)));

	

//        buff_list_.acceleration = base_buffer->buff_list_.acceleration;
        buff_list_.final_position = base_buffer->buff_list_.final_position;
//        buff_list_.density = base_buffer->buff_list_.density;
//        buff_list_.pressure = base_buffer->buff_list_.pressure;
        //buff_list_.surface_normal_vector = base_buffer->buff_list_.surface_normal_vector;

	
    }
    if (kBuffTypeHostPageable == type_)
    {
        buff_list_.position_d = new float4[capacity_];
        buff_list_.velocity = new float3[capacity_];
        buff_list_.evaluated_velocity = new float4[capacity_];
        buff_list_.color = new unsigned int[capacity_];
        //buff_list_.particle_type = new unsigned int[capacity_];

	

        buff_list_.acceleration = base_buffer->buff_list_.acceleration;
        buff_list_.final_position = base_buffer->buff_list_.final_position;
//        buff_list_.density = base_buffer->buff_list_.density;
//        buff_list_.pressure = base_buffer->buff_list_.pressure;
        //buff_list_.surface_normal_vector = base_buffer->buff_list_.surface_normal_vector;

	
    }
}

void ParticleBufferObject::reallocate(unsigned int new_nump)
{
    if (new_nump <= capacity_) return;

    if (kBuffTypeDevice == type_)
    {
        reallocDeviceBuffer(buff_list_.position_d, capacity_, new_nump);
        reallocDeviceBuffer(buff_list_.velocity, capacity_, new_nump);
        reallocDeviceBuffer(buff_list_.evaluated_velocity, capacity_, new_nump);
        //reallocDeviceBuffer(buff_list_.particle_type, capacity_, new_nump);
        reallocDeviceBuffer(buff_list_.color, capacity_, new_nump);

	
        if (base_buffer_)
        {
            buff_list_.acceleration = base_buffer_->buff_list_.acceleration;
  //          buff_list_.density = base_buffer_->buff_list_.density;
  //          buff_list_.pressure = base_buffer_->buff_list_.pressure;
            //buff_list_.surface_normal_vector = base_buffer_->buff_list_.surface_normal_vector;
            buff_list_.final_position = base_buffer_->buff_list_.final_position;

		
        }
        else
        {
            reallocDeviceBuffer(buff_list_.acceleration, capacity_, new_nump);
 //           reallocDeviceBuffer(buff_list_.density, capacity_, new_nump);
  //          reallocDeviceBuffer(buff_list_.pressure, capacity_, new_nump);
            //reallocDeviceBuffer(buff_list_.surface_normal_vector, capacity_, new_nump);
            reallocDeviceBuffer(buff_list_.final_position, capacity_, new_nump);

		
        }
    }
    if (kBuffTypeHostPinned == type_)
    {
        reallocHostPinnedBuffer(buff_list_.position_d, capacity_, new_nump);
        reallocHostPinnedBuffer(buff_list_.velocity, capacity_, new_nump);
//        reallocHostPinnedBuffer(buff_list_.evaluated_velocity, capacity_, new_nump);
        reallocHostPinnedBuffer(buff_list_.color, capacity_, new_nump);
        //reallocHostPinnedBuffer(buff_list_.particle_type, capacity_, new_nump);

		

        if (base_buffer_)
        {
//            buff_list_.acceleration = base_buffer_->buff_list_.acceleration;
            buff_list_.final_position = base_buffer_->buff_list_.final_position;
//            buff_list_.density = base_buffer_->buff_list_.density;
//            buff_list_.pressure = base_buffer_->buff_list_.pressure;
            //buff_list_.surface_normal_vector = base_buffer_->buff_list_.surface_normal_vector;

			
        }
        else
        {
//            reallocHostPinnedBuffer(buff_list_.acceleration, capacity_, new_nump);
            reallocHostPinnedBuffer(buff_list_.final_position, capacity_, new_nump);
//            reallocHostPinnedBuffer(buff_list_.density, capacity_, new_nump);
//            reallocHostPinnedBuffer(buff_list_.pressure, capacity_, new_nump);
            //reallocHostPinnedBuffer(buff_list_.surface_normal_vector, capacity_, new_nump);

		
        }
    }
    if (kBuffTypeHostPageable == type_)
    {
        reallocHostPageableBuffer(buff_list_.position_d, capacity_, new_nump);
        reallocHostPageableBuffer(buff_list_.velocity, capacity_, new_nump);
        reallocHostPageableBuffer(buff_list_.evaluated_velocity, capacity_, new_nump);
        reallocHostPageableBuffer(buff_list_.color, capacity_, new_nump);
        //reallocHostPageableBuffer(buff_list_.particle_type, capacity_, new_nump);

	

        if (base_buffer_)
        {
            buff_list_.acceleration = base_buffer_->buff_list_.acceleration;
            buff_list_.final_position = base_buffer_->buff_list_.final_position;
 //           buff_list_.density = base_buffer_->buff_list_.density;
 //           buff_list_.pressure = base_buffer_->buff_list_.pressure;
            //buff_list_.surface_normal_vector = base_buffer_->buff_list_.surface_normal_vector;

		
        }
        else
        {
            reallocHostPageableBuffer(buff_list_.acceleration, capacity_, new_nump);
            reallocHostPageableBuffer(buff_list_.final_position, capacity_, new_nump);
 //           reallocHostPageableBuffer(buff_list_.density, capacity_, new_nump);
  //          reallocHostPageableBuffer(buff_list_.pressure, capacity_, new_nump);
            //reallocHostPageableBuffer(buff_list_.surface_normal_vector, capacity_, new_nump);

		

        }
    }

    capacity_ = new_nump;
}

void ParticleBufferObject::free()
{
    if (kBuffTypeDevice == type_)
    {
        CUDA_SAFE_CALL(cudaFree(buff_list_.position_d));
        CUDA_SAFE_CALL(cudaFree(buff_list_.velocity));
        CUDA_SAFE_CALL(cudaFree(buff_list_.evaluated_velocity));
        //CUDA_SAFE_CALL(cudaFree(buff_list_.particle_type));
        CUDA_SAFE_CALL(cudaFree(buff_list_.color));

        if (!base_buffer_)
        {
            CUDA_SAFE_CALL(cudaFree(buff_list_.acceleration));
 //           CUDA_SAFE_CALL(cudaFree(buff_list_.density));
    //        CUDA_SAFE_CALL(cudaFree(buff_list_.pressure));
            //CUDA_SAFE_CALL(cudaFree(buff_list_.surface_normal_vector));
            CUDA_SAFE_CALL(cudaFree(buff_list_.final_position));

		
        }
    }
    if (kBuffTypeHostPageable == type_)
    {
        delete[]buff_list_.position_d;
        delete[]buff_list_.velocity;
        delete[]buff_list_.evaluated_velocity;
        delete[]buff_list_.color;
        //delete[]buff_list_.particle_type;

	

        if (!base_buffer_)
        {
            delete[]buff_list_.acceleration;
            delete[]buff_list_.final_position;
  //          delete[]buff_list_.density;
  //          delete[]buff_list_.pressure;
            //delete[]buff_list_.surface_normal_vector;

		
        }
    }
    if (kBuffTypeHostPinned == type_)
    {
        CUDA_SAFE_CALL(cudaFreeHost(buff_list_.position_d));
        CUDA_SAFE_CALL(cudaFreeHost(buff_list_.velocity));
//        CUDA_SAFE_CALL(cudaFreeHost(buff_list_.evaluated_velocity));
        CUDA_SAFE_CALL(cudaFreeHost(buff_list_.color));
        //CUDA_SAFE_CALL(cudaFreeHost(buff_list_.particle_type));

        if (!base_buffer_)
        {
//            CUDA_SAFE_CALL(cudaFreeHost(buff_list_.acceleration));
            CUDA_SAFE_CALL(cudaFreeHost(buff_list_.final_position));
//            CUDA_SAFE_CALL(cudaFreeHost(buff_list_.density));
//            CUDA_SAFE_CALL(cudaFreeHost(buff_list_.pressure));
            //CUDA_SAFE_CALL(cudaFreeHost(buff_list_.surface_normal_vector));

        }
    }

    type_ = kBuffTypeNone;
    capacity_ = 0U;
    base_buffer_ = nullptr;
}

void ParticleBufferObject::transfer(ParticleBufferObject &dst_buff_obj, unsigned int offset, unsigned int nump, cudaMemcpyKind kind)
{
    CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.position_d + offset, buff_list_.position_d + offset, nump * sizeof(float4), kind));
    CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.velocity + offset, buff_list_.velocity + offset, nump * sizeof(float3), kind));
//    CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.acceleration + offset, buff_list_.acceleration + offset, nump * sizeof(float3), kind));
//    CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.evaluated_velocity + offset, buff_list_.evaluated_velocity + offset, nump * sizeof(float3), kind));
    CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.final_position + offset, buff_list_.final_position + offset, nump * sizeof(float3), kind));
//    CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.density + offset, buff_list_.density + offset, nump * sizeof(float), kind));
//    CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.pressure + offset, buff_list_.pressure + offset, nump * sizeof(float), kind));
    //CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.surface_normal_vector + offset, buff_list_.surface_normal_vector + offset, nump * sizeof(float), kind));
    CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.color + offset, buff_list_.color + offset, nump * sizeof(unsigned int), kind));
    //CUDA_SAFE_CALL(cudaMemcpy(dst_buff_obj.buff_list_.particle_type + offset, buff_list_.particle_type + offset, nump * sizeof(unsigned int), kind));
}

void ParticleBufferObject::swapObj(ParticleBufferObject &obj)
{
    swap(this->buff_list_, obj.buff_list_);
    swap(this->type_, obj.type_);
    swap(this->capacity_, obj.capacity_);
    //swap(this->base_buffer_, obj.base_buffer_);
}

}

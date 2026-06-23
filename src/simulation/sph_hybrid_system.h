//
// sph_hybrid_system.h
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#ifndef _SPH_HYBRID_SYSTEM_H
#define _SPH_HYBRID_SYSTEM_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include "render/gl_texture.h"
#include "core/high_resolution_timer.h"
#include "grid/sph_arrangement.h"
#include "core/sph_parameter.h"
#include "particle/particle_buffer.h"

typedef unsigned int uint;

namespace sph
{

const int kDefaultNumParticles = 65536;

struct Scene
{
    std::vector<std::pair<float3, float3>> fluid_blocks;
    float interval = 0.5f;
    float mass = 0.02f;
    uint recomm_nump = kDefaultNumParticles;
	float x, y, z;
};

class HybridSystem
{
public:
    HybridSystem(const float3 &real_world_side, const float3 &sim_origin, bool headless = false);
    ~HybridSystem();

    void tick();
    void runBenchmark(int frames);
    void setPause();
    bool isRunning();
    uint getNumParticles();
    float3 getPosition(uint idx);

	void insertParticles(unsigned int type);

    void drawParticles(float rad, int size);
    void drawInfo(GLdouble w, GLdouble h);
	int loop;

private:
    void initializeScene(const std::string &file_name, Scene scene);
    void initializeScene2(const std::string &file_name);
    void resetBuffer(uint nump);
	void addParticle2(float3 position, float3 velocity, condition phase, float temperature);
    void addParticle(float3 position, float3 velocity = make_float3(0.0f, 0.0f, 0.0f), int color_type = 1);

    void createPersistentCudaResources();
    void destroyPersistentCudaResources();
    void waitForGraphicsCopy();

    bool is_running_ = false;
    uint nump_ = 0U;
    uint buff_capacity_ = 0U;
    ParticleBufferObject host_buff_;
    ParticleBufferObject device_buff_;
    ParticleBufferObject device_buff_temp_;

    ParticleBufferObject device_buff_data_;

    SystemParameter sys_para_;
    //std::unique_ptr<Arrangement> arrangement_;
	Arrangement *arrangement_;
	float particle_interval = 0.5f;
    HighResolutionTimerForWin frame_timer_;
    bool headless_mode_ = false;
    bool get_detailed_time_ = false;
    float total_time_ = 0.0f;
    float pre_time_, density_time_, force_time_;
    bool generate_mesh_;
	bool add_smoke_;

    // render
    PNGTexture particle_texture_;
    GLuint position_vbo_;
    GLuint color_vbo_;

    void registerGraphicsResources();
    void unregisterGraphicsResources();

    cudaGraphicsResource_t position_vbo_res_ = nullptr;
    cudaGraphicsResource_t color_vbo_res_ = nullptr;
    bool vbo_resources_registered_ = false;

    // asynchronous device->host copy resources
    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t compute_done_event_ = nullptr;
    cudaEvent_t copy_done_event_ = nullptr;
    cudaEvent_t tick_events_[5] = {};
    bool tick_events_created_ = false;

	// action
	void action1();
	bool action1_ = false;

	//sf add
	float pcisph_density_factor;
};

}

#endif/*_SPH_HYBRID_SYSTEM_H*/

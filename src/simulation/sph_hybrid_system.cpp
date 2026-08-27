//
// sph_hybrid_system.cpp
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#define _USE_MATH_DEFINES

#include "simulation/sph_hybrid_system.h"
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include "json/json.h"
#include "json/reader.h"
#include "core/cuda_math.cuh"
#include "core/cuda_call_check.h"
#include "solver/kernel_dispatch.cuh"
#include "simulation/sph_marching_cube.h"
#include "simulation/pcisph_factor.h"


namespace sph
{

const char *kDefaultSceneFileName = "assets/scene_default.json";
const uint kDefaultBufferCapacity = 65536U;

/****************************** utilities ******************************/

#define COLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )
bool readSceneFromJsonFile(Scene *out, const std::string &file_name)
{
	std::ifstream input(file_name, std::ios::binary);
	if (!input.is_open())
	{
		std::cout << "can not open " << file_name << std::endl;
		return false;
	}

	Json::Reader js_reader;
	Json::Value root;

	if (js_reader.parse(input, root))
	{
		if (root.isMember("xx")){
			out->x = root["xx"].asFloat();
		}
		if (root.isMember("yy")){
			out->y = root["yy"].asFloat();
		}
		if (root.isMember("zz")){
			out->z = root["zz"].asFloat();
		}
		if (root.isMember("mass"))
		{
			out->mass = root["mass"].asFloat();
		}
		if (root.isMember("interval"))
		{
			out->interval = root["interval"].asFloat();
		}
		if (root.isMember("recomm_nump"))
		{
			out->recomm_nump = root["recomm_nump"].asUInt();
		}
		if (root.isMember("fluid_block"))
		{
			Json::Value fluid_blocks = root["fluid_block"];
			for (int i = 0; i < fluid_blocks.size(); ++i)
			{
				float3 begin = make_float3(fluid_blocks[i]["begin_x"].asFloat(),
					fluid_blocks[i]["begin_y"].asFloat(),
					fluid_blocks[i]["begin_z"].asFloat());
				float3 end = make_float3(fluid_blocks[i]["end_x"].asFloat(),
					fluid_blocks[i]["end_y"].asFloat(),
					fluid_blocks[i]["end_z"].asFloat());
				out->fluid_blocks.push_back(std::make_pair(begin, end));
			}
		}
	}
	else
	{
		std::cout << "can not parse scene file " << file_name << std::endl;
		return false;
	}

	return true;
}
inline void defaultInitializeSPHSysPara(SystemParameter &sys_para, Scene *scene)
{
	
	if (!readSceneFromJsonFile(scene, kDefaultSceneFileName))
	{
		return;
	}


    sys_para.to = 0.0000005;
    sys_para.limita = 0.0005;

    sys_para.viscosity1 = 6.5f;
    sys_para.viscosity2 = 6.5f;


    sys_para.kernel = 0.03f;
    //sys_para.kernel = 0.018f;
	sys_para.mass = scene->mass;
    sys_para.kernel_2 = sys_para.kernel * sys_para.kernel;

	sys_para.world_size = make_float3(scene->x, scene->y, scene->z);
    sys_para.cell_size = sys_para.kernel;
    sys_para.inv_cell_size = 1.0f / sys_para.cell_size;
    sys_para.grid_size = make_ushort3((int)ceil(sys_para.world_size.x / sys_para.cell_size),
                                   (int)ceil(sys_para.world_size.y / sys_para.cell_size),
                                   (int)ceil(sys_para.world_size.z / sys_para.cell_size));

    sys_para.gravity = make_float3(0.0f, -9.8f, 0.0f);
    sys_para.wall_damping = -0.5f;


    sys_para.rest_density = 1000.0f;
    sys_para.rest_density1 = 1000.0f;
    sys_para.rest_density2 = 1000.0f;

    sys_para.gas_constant = 1.0f;
    sys_para.viscosity = 6.5f;
    sys_para.time_step = 0.003f;
    sys_para.surface_normal = 0.1f;
    sys_para.surface_coe = 0.2f;

    sys_para.poly6_value = 315.0f / (64.0f * M_PI * pow(sys_para.kernel, 9));
    sys_para.spiky_value = -45.0f / (M_PI * pow(sys_para.kernel, 6));
    sys_para.visco_value = 45.0f / (M_PI * pow(sys_para.kernel, 6));

    sys_para.grad_poly6 = -945 / (32 * M_PI * pow(sys_para.kernel, 9));
    sys_para.lplc_poly6 = 945 / (8 * M_PI * pow(sys_para.kernel, 9));
    sys_para.self_density = sys_para.mass * sys_para.poly6_value * pow(sys_para.kernel, 6);
    // NOTE: the "3 / 4" below is integer division (evaluates to 0), so self_lplc_color
    // is always 0. The field is currently not consumed anywhere, so the formula is left
    // as-is to preserve behavior; fix to 3.0f / 4.0f if it ever gets used.
    sys_para.self_lplc_color = sys_para.lplc_poly6 * sys_para.mass * sys_para.kernel_2 * (0 - 3 / 4 * sys_para.kernel_2);

    sys_para.bound_interval = sys_para.kernel;
    sys_para.bound_min = make_float3(sys_para.bound_interval, sys_para.bound_interval, sys_para.bound_interval);
    sys_para.bound_max = make_float3(sys_para.world_size.x - sys_para.bound_interval,
                                     sys_para.world_size.y - sys_para.bound_interval,
                                     sys_para.world_size.z - sys_para.bound_interval);

    //sf add
//    sys_para.spacing_fluid = pow(sys_para.mass / sys_para.rest_density, 1 / 3.0f);

    //sf pcisph
//    sys_para.pcisph_min_loops = pcisph_min_loops;
//    sys_para.pcisph_max_loops = pcisph_max_loops;
//    sys_para.pcisph_max_density_error_allowed = pcisph_max_density_error_allowed;
}



void setOrthographicProjection(GLdouble w, GLdouble h)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w, h, 0);
    glMatrixMode(GL_MODELVIEW);
}

void restorePerspectiveProjection()
{
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void renderBitmapString(float x, float y, float z, void *font, const std::stringstream &ss)
{
    std::string str = ss.str();
    const char *c;
    glRasterPos3f(x, y, z);
    for (c = str.c_str(); *c != '\0'; c++) {
        glutBitmapCharacter(font, *c);
    }
}

/****************************** HybridSystem ******************************/

HybridSystem::HybridSystem(const float3 &real_world_side, const float3 &sim_origin, bool headless)
    : headless_mode_(headless)
{
	Scene scene;
    defaultInitializeSPHSysPara(sys_para_, &scene);
    sys_para_.sim_ratio = make_float3(real_world_side.x / sys_para_.world_size.x,
                                      real_world_side.y / sys_para_.world_size.y,
                                      real_world_side.z / sys_para_.world_size.z);
    sys_para_.sim_origin = sim_origin;

    // Headless benchmark mode needs timing events, so decide before creating them.
    get_detailed_time_ = headless_mode_;

    initializeKernel();
    createPersistentCudaResources();

	initializeScene(kDefaultSceneFileName, scene);

    if (!headless_mode_)
    {
        // render
        particle_texture_.loadPNG("assets/ball32.png");
        glGenBuffers(1, &position_vbo_);
        glGenBuffers(1, &color_vbo_);

        // Allocate GPU storage once, then register the VBOs with CUDA so the
        // particle data can be copied directly from device memory each frame.
        glBindBuffer(GL_ARRAY_BUFFER, position_vbo_);
        glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);
        glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(uint), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        registerGraphicsResources();
    }

    generate_mesh_ = false;
    add_smoke_ = false;
}

HybridSystem::~HybridSystem()
{
    waitForGraphicsCopy();
    unregisterGraphicsResources();
    destroyPersistentCudaResources();
    delete arrangement_;
    arrangement_ = nullptr;
    resetBuffer(0);
    releaseKernel();
}
int tt = 0;
float time = 0;
void HybridSystem::tick()
{
    if (!is_running_) return;
	tt++;
    tick_timer_.set_start();
    static int step = 0;
    if (get_detailed_time_ && !tick_events_created_)
    {
        createPersistentCudaResources();
    }
    int *d_index = arrangement_->getDevCellIndex();
    int *cell_offset = arrangement_->getDevCellOffset();

	int *cell_offsetM = arrangement_->getDevCellOffsetM();

    int *cell_nump = arrangement_->getDevCellNumP();
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[0]));

    arrangement_->arrangeHybridMode9M();
//    arrangement_->CountingSortCUDA();
//    arrangement_->assignTasksFixedCTA();

#if HYBRID_DEVICE_GRID_SIZING
    // TRA particles occupy [0, middle) of the compaction index. The physics kernels
    // read the actual split point from device memory, so the upper bound suffices
    // here and no host readback is needed.
    ParticleIdxRange tra_range(0, nump_);
    // Safe upper bound on the SMS task count: one task per 32 particles plus at
    // most one partial task per cell.
    int sms_task_bound = (nump_ + 31) / 32 + arrangement_->getNumC();
#else
    // Host-synced sizing: middle_value_ is valid after the arrange sync.
    int middle_host = arrangement_->getMiddleValue();
    if (middle_host < 0 || middle_host > (int)nump_) middle_host = nump_;
    ParticleIdxRange tra_range(0, middle_host);
    int sms_task_bound = 0;  // unused on the host-synced path
#endif
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[1]));

	computeDensityHybrid128n(cell_offsetM, tra_range, device_buff_.get_buff_list(), d_index, cell_offset, cell_nump, arrangement_->getBlockTasks(), arrangement_->getNumBlockSMSMode(), arrangement_->getDevNumCTA(), arrangement_->getDevMiddleValue(), sms_task_bound);
//    computeDensitySMS64(device_buff_.get_buff_list(), cell_offset, cell_nump, arrangement_->getBlockTasks(), arrangement_->getNumBlockSMSMode());
//    computeDensityTRA(device_buff_.get_buff_list(), ParticleIdxRange(0, nump_), cell_offset, cell_nump);
    //   std::cout << step << std::endl;
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[2]));

	computeForceHybrid128n(cell_offsetM, tra_range, device_buff_.get_buff_list(), d_index, cell_offset, cell_nump, arrangement_->getBlockTasks(), arrangement_->getNumBlockSMSMode(), arrangement_->getDevNumCTA(), arrangement_->getDevMiddleValue(), sms_task_bound);
//    computeForceSMS64(device_buff_.get_buff_list(), cell_offset, cell_nump, arrangement_->getBlockTasks(), arrangement_->getNumBlockSMSMode());
//    computeForceTRA(device_buff_.get_buff_list(), ParticleIdxRange(0, nump_), cell_offset, cell_nump);
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[3]));

    advance(device_buff_.get_buff_list(), nump_);
	//advanceWave(device_buff_.get_buff_list(), nump_,time);
	time += sys_para_.time_step;
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[4]));

    // Record the point where all simulation kernels for this frame finish.
    // If CUDA-GL interop is unavailable we fall back to asynchronous D2H
    // copies that the draw path will synchronize on.
    CUDA_SAFE_CALL(cudaEventRecord(compute_done_event_, 0));
    if (!headless_mode_ && !vbo_resources_registered_)
    {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(copy_stream_, compute_done_event_, 0));
        CUDA_SAFE_CALL(cudaMemcpyAsync(host_buff_.get_buff_list().final_position,
                                       device_buff_.get_buff_list().final_position,
                                       nump_ * sizeof(float3), cudaMemcpyDeviceToHost, copy_stream_));
        CUDA_SAFE_CALL(cudaMemcpyAsync(host_buff_.get_buff_list().color,
                                       device_buff_.get_buff_list().color,
                                       nump_ * sizeof(uint), cudaMemcpyDeviceToHost, copy_stream_));
        CUDA_SAFE_CALL(cudaEventRecord(copy_done_event_, copy_stream_));
    }

    tick_timer_.set_end();
    total_time_ = static_cast<float>(tick_timer_.get_millisecond());

    if (get_detailed_time_)
    {
        // Events are recorded asynchronously; wait for the last one before
        // reading elapsed times so the measurements are accurate.
        CUDA_SAFE_CALL(cudaEventSynchronize(tick_events_[4]));

        CUDA_SAFE_CALL(cudaEventElapsedTime(&pre_time_, tick_events_[0], tick_events_[1]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&density_time_, tick_events_[1], tick_events_[2]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&force_time_, tick_events_[2], tick_events_[3]));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&total_time_, tick_events_[0], tick_events_[4]));
    }
    ++step;
    loop = step;
    if (loop > 640) {
  //      exit(0);
    }
}

void HybridSystem::runBenchmark(int frames)
{
    if (frames <= 0) return;

    // Make sure the simulation is running and detailed timing is enabled.
    if (!is_running_) setPause();
    get_detailed_time_ = true;
    if (!tick_events_created_)
    {
        destroyPersistentCudaResources();
        createPersistentCudaResources();
    }

    // Warm-up with detailed per-frame timing. The event synchronize in tick()
    // serializes CPU and GPU, so it is only used here to collect per-stage averages.
    const int warmup_frames = frames < 5 ? frames : 5;
    double total_arrange = 0.0;
    double total_density = 0.0;
    double total_force = 0.0;
    double total_kernel = 0.0;
    for (int i = 0; i < warmup_frames; ++i)
    {
        tick();
        total_arrange += pre_time_;
        total_density += density_time_;
        total_force += force_time_;
        total_kernel += total_time_;
    }

    // Measured loop: detailed timing off, so no per-frame CPU-GPU synchronization
    // distorts the wall-clock throughput. The pipeline is drained once at the end.
    const int measured_frames = frames - warmup_frames;
    get_detailed_time_ = false;
    HighResolutionTimerForWin bench_timer;
    bench_timer.set_start();
    for (int i = 0; i < measured_frames; ++i)
    {
        tick();
    }
    if (measured_frames > 0) CUDA_SAFE_CALL(cudaDeviceSynchronize());
    bench_timer.set_end();
    get_detailed_time_ = true;

    double elapsed_ms = bench_timer.get_millisecond();
    double avg_frame = measured_frames > 0 ? elapsed_ms / measured_frames
                                           : total_kernel / warmup_frames;
    double fps = 1000.0 / avg_frame;

    std::cout << "\n========== Headless benchmark (" << measured_frames << " measured + " << warmup_frames << " warm-up frames) ==========\n";
    std::cout << "Total wall time: " << elapsed_ms << " ms\n";
    std::cout << "Average frame time: " << avg_frame << " ms\n";
    std::cout << "FPS: " << fps << "\n";
    std::cout << "Per-stage averages (CUDA events, warm-up frames):\n";
    std::cout << "  arrange/grid : " << (total_arrange / warmup_frames) << " ms\n";
    std::cout << "  density      : " << (total_density / warmup_frames) << " ms\n";
    std::cout << "  force        : " << (total_force / warmup_frames) << " ms\n";
    std::cout << "  total kernel : " << (total_kernel / warmup_frames) << " ms\n";
    std::cout << "==================================================\n" << std::endl;
}

void HybridSystem::initializeScene(const std::string &file_name, Scene scene)
{
   
    sys_para_.mass = scene.mass;
    transSysParaToDevice(&sys_para_);

    particle_interval = scene.interval;
    // Count the particles exactly before allocating, so the buffers are not
    // sized by the (potentially much larger) recomm_nump hint.
    uint exact_count = 0;
    for (const auto &range : scene.fluid_blocks)
    {
        for (float x = range.first.x; x < range.second.x; x += sys_para_.kernel * particle_interval)
            for (float y = range.first.y; y < range.second.y; y += sys_para_.kernel * particle_interval)
                for (float z = range.first.z; z < range.second.z; z += sys_para_.kernel * particle_interval)
                    ++exact_count;
    }
    resetBuffer(exact_count > 0 ? exact_count : scene.recomm_nump);
    int t = 0;
    for (const auto &range : scene.fluid_blocks)
    {
        t++;
        for (float x = range.first.x; x < range.second.x; x += sys_para_.kernel * particle_interval)
        {
            for (float y = range.first.y; y < range.second.y; y += sys_para_.kernel * particle_interval)
            {
                for (float z = range.first.z; z < range.second.z; z += sys_para_.kernel * particle_interval)
                {
                    addParticle(make_float3(x, y, z), make_float3(0.0f, 0.0f, 0.0f), t);
                }
            }
        }
    }
    BuffInit(device_buff_.get_buff_list(), nump_);
    std::cout << "Number of particles: " << nump_ << std::endl;

    host_buff_.transfer(device_buff_, 0, nump_, cudaMemcpyHostToDevice);

    //arrangement_.reset(new Arrangement(device_buff_, device_buff_temp_, nump_, sys_para_.inv_cell_size, sys_para_.grid_size));
    arrangement_ = //new Arrangement(device_buff_, device_buff_temp_, nump_, buff_capacity_, sys_para_.inv_cell_size, sys_para_.grid_size);
        new Arrangement(device_buff_, device_buff_temp_,  nump_, buff_capacity_, sys_para_.inv_cell_size, sys_para_.grid_size);
}




void HybridSystem::initializeScene2(const std::string &file_name)
{
    Scene scene;
    //if (!readSceneFromJsonFile(scene, kDefaultSceneFileName))
    //{
    //    return;
    //}

    //sys_para_.mass = scene.mass;
    transSysParaToDevice(&sys_para_);

    resetBuffer(scene.recomm_nump);
    particle_interval = scene.interval;

    printf("spacing_lava %f \n", sys_para_.spacing_fluid);
    printf("self_density %f \n", sys_para_.self_density);

    float3 vel = make_float3(0, 0, 0);
    float tempera_1 = 50;
    float tempera_2 = 100;
    condition pha = FLUID;




    float low = 0.01;
    float hig = 2.99;
    for (float x = low; x < hig; x += sys_para_.spacing_fluid) //sf float x = range.first.x; x < range.second.x / 4; x += sys_para_.kernel * scene.interval
    {
        for (float y = low; y < hig / 4; y += sys_para_.spacing_fluid)
        {
            for (float z = hig*3 / 4; z < hig; z += sys_para_.spacing_fluid)
            {
                addParticle2(make_float3(x, y, z), vel, pha, tempera_1);
            }
        }
    }




    low = 0.01;
    hig = 2.99;
    for (float x = low; x < hig; x += sys_para_.spacing_fluid) //sf float x = range.first.x; x < range.second.x / 4; x += sys_para_.kernel * scene.interval
    {
        for (float y = low; y < hig/4; y += sys_para_.spacing_fluid)
        {
            for (float z = low; z < hig/4; z += sys_para_.spacing_fluid)
            {
                addParticle2(make_float3(x, y, z), vel, pha, tempera_1);
            }
        }
    }
    //pha = SOLID;
    low = 1.25;
    hig = 1.75;
    for (float x = low; x < hig; x += sys_para_.spacing_fluid) //sf float x = range.first.x; x < range.second.x / 4; x += sys_para_.kernel * scene.interval
    {
        for (float y = low; y < hig; y += sys_para_.spacing_fluid)
        {
            for (float z = low; z < hig; z += sys_para_.spacing_fluid)
            {
                if (pow(x - 1.5, 2) + pow(y - 1.5, 2) + pow(z - 1.5, 2) < 0.0625f){

                    addParticle2(make_float3(x, y, z), vel, pha, tempera_1);
                }
            }
        }
    }

    std::cout << "Number of particles: " << nump_ << std::endl;

    host_buff_.transfer(device_buff_, 0, nump_, cudaMemcpyHostToDevice);
    //arrangement_.reset(new Arrangement(device_buff_, device_buff_temp_, nump_, sys_para_.inv_cell_size, sys_para_.grid_size));
    arrangement_ = new Arrangement(device_buff_, device_buff_temp_, nump_, buff_capacity_, sys_para_.inv_cell_size, sys_para_.grid_size);
}

void HybridSystem::setPause()
{
    is_running_ = !is_running_;
}

bool HybridSystem::isRunning()
{
    return is_running_;
}

uint HybridSystem::getNumParticles()
{
    return nump_;
}

float3 HybridSystem::getPosition(unsigned int idx)
{
    //return host_buff_.position[idx] * sys_para_.sim_ratio + sys_para_.sim_origin;
    return host_buff_.get_buff_list().final_position[idx];
}

void HybridSystem::insertParticles(unsigned int type)
{
    if (1 == type)
    {
        action1_ = !action1_;
    }
}

void HybridSystem::drawParticles(float rad, int size)
{
    glEnable(GL_BLEND);
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.5);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glEnable(GL_POINT_SPRITE_ARB);
    float quadratic[] = { 1.0f, 0.01f, 0.001f };
    glEnable(GL_POINT_DISTANCE_ATTENUATION);
    glPointParameterfvARB(GL_POINT_DISTANCE_ATTENUATION, quadratic);
    glPointSize(size);
    glPointParameterfARB(GL_POINT_SIZE_MAX, 32);
    glPointParameterfARB(GL_POINT_SIZE_MIN, 1.0f);

    // Texture and blending mode
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, particle_texture_.get_texture());
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Upload particle data.  Prefer CUDA-GL interop (device->device copy)
    // over a host round-trip.
    if (vbo_resources_registered_)
    {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(0, compute_done_event_, 0));

        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &position_vbo_res_, 0));
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &color_vbo_res_, 0));

        size_t num_bytes_pos = 0, num_bytes_col = 0;
        float3 *d_position_vbo = nullptr;
        uint *d_color_vbo = nullptr;
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_position_vbo, &num_bytes_pos, position_vbo_res_));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_color_vbo, &num_bytes_col, color_vbo_res_));

        copyParticleDataToVBOs(device_buff_.get_buff_list(), nump_, d_position_vbo, d_color_vbo);

        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &position_vbo_res_, 0));
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &color_vbo_res_, 0));
    }
    else
    {
        waitForGraphicsCopy();
    }

    // Point buffers
	//GLint gsize = size;
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo_);
    if (!vbo_resources_registered_)
    {
        glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(float3), host_buff_.get_buff_list().final_position, GL_DYNAMIC_DRAW);
    }
	glVertexPointer(3, GL_FLOAT, 0, 0x0);
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);
    if (!vbo_resources_registered_)
    {
        glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(uint), host_buff_.get_buff_list().color, GL_DYNAMIC_DRAW);
    }
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0x0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    //for (size_t i = 1000; i < 1020; ++i)
    //{
    //    printf("color: %u\n", host_buff_.color[i]);
    //}

    // Render - Point Sprites
    glNormal3f(0, 1, 0.001);
    glColor4f(1, 1, 1, 1);
    glDrawArrays(GL_POINTS, 0, nump_);

    // Restore state
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisable(GL_POINT_SPRITE_ARB);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_TEXTURE_2D);
    glDepthMask(GL_TRUE);
}
void HybridSystem::drawInfo(GLdouble w, GLdouble h)
{
    float x = 20, y = 20, delta_y = 20;
    std::stringstream ss;
    static unsigned int frame = 0;
    static float time = 0.0f, acc_time = 0.0f;

    setOrthographicProjection(w, h);
	//tt++;
    glPushMatrix();
    glLoadIdentity();
    glColor3f(1.0f, 1.0f, 1.0f);
    glDisable(GL_LIGHTING);

    // output particles
    frame_timer_.set_end();
    acc_time += frame_timer_.get_millisecond();
    frame_timer_.set_start();
    ++frame;
    if (acc_time > 100.0f) {
        time = 1000 * frame / acc_time; frame = 0; acc_time = 0.0f;
    }

    ss << "FPS: " << time;  // smoothed wall-clock FPS


    frame_timer_.set_start();
    renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
    ss.str(""); y += delta_y;

    // output number of particles
    ss << "#particles: " << nump_;
    renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
    ss.str(""); y += delta_y;

    // output frame index and total frame time
    ss << "frame: " << loop << "  total time: " << total_time_ << "ms";
    renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
    ss.str(""); y += delta_y;

    // output detailed time
    if (get_detailed_time_)
    {
		ss << "Simulation detailed time: ";
		renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
		ss.str(""); y += delta_y;
		ss << "    preprocessing: " << pre_time_ << "ms";
		renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
		ss.str(""); y += delta_y;
		ss << "    density computation: " << density_time_ << "ms";
		renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
		ss.str(""); y += delta_y;
		ss << "    force computation: " << force_time_ << "ms";
		renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
		ss.str(""); y += delta_y;
		ss << "    total consumption: " << total_time_ << "ms";
		renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
		ss.str(""); y += delta_y;
    }

    glPopMatrix();

    restorePerspectiveProjection();
}

void HybridSystem::resetBuffer(uint nump)
{
    nump_ = 0U;
    buff_capacity_ = nump;

    host_buff_.free();
    device_buff_.free();
    device_buff_temp_.free();
    //device_buff_data_.free();

    if (0 == nump) return;

    
    device_buff_.allocate(nump, kBuffTypeDevice);
    host_buff_.allocate(nump, kBuffTypeHostPinned);
    device_buff_temp_.allocateSubBuffer(&device_buff_);
}
void HybridSystem::addParticle(float3 position, float3 velocity, int colortype)
{
    if (nump_ + 1 > buff_capacity_)
    {
        buff_capacity_ *= 2;
        host_buff_.reallocate(buff_capacity_);
        device_buff_.reallocate(buff_capacity_);
        device_buff_temp_.reallocate(buff_capacity_);
    }

    float4 pos_d;
    pos_d.x = position.x;
    pos_d.y = position.y;
    pos_d.z = position.z;
    pos_d.w = 0;

    host_buff_.get_buff_list().position_d[nump_] = pos_d;
    host_buff_.get_buff_list().velocity[nump_] = velocity;
//    host_buff_.get_buff_list().acceleration[nump_] = make_float3(0.0f, 0.0f, 0.0f);
//    host_buff_.get_buff_list().evaluated_velocity[nump_] = make_float3(0.0f, 0.0f, 0.0f);
//    host_buff_.get_buff_list().density[nump_] = 0.0f;
//    host_buff_.get_buff_list().pressure[nump_] = 0.0f;
    host_buff_.get_buff_list().final_position[nump_] = position * sys_para_.sim_ratio + sys_para_.sim_origin;

    float3 color = make_float3(0.6f, 0.6f, 0.6f);
    
    host_buff_.get_buff_list().color[nump_] = COLORA(color.x, color.y, color.z, 1);
    ++nump_;
}
void HybridSystem::addParticle2(float3 position, float3 velocity, condition phase, float temperature)
{
    if (nump_ + 1 > buff_capacity_)
    {
        buff_capacity_ *= 2;
        host_buff_.reallocate(buff_capacity_);
        device_buff_.reallocate(buff_capacity_);
        device_buff_temp_.reallocate(buff_capacity_);
    }

//    host_buff_.get_buff_list().position[nump_] = position;
    host_buff_.get_buff_list().velocity[nump_] = velocity;
    host_buff_.get_buff_list().final_position[nump_] = position * sys_para_.sim_ratio + sys_para_.sim_origin;

    float3 color = make_float3(0.0f, 0.0f, 1.0f);
    host_buff_.get_buff_list().color[nump_] = COLORA(color.x, color.y, color.z, 1);

    if (phase == FLUID) host_buff_.get_buff_list().color[nump_] = COLORA((position.y / 3) + 0.2, (position.y / 3) + 0.2, (position.y / 3) + 0.2, 1);//COLORA(0.1, 0.6, 0.8, 1);

    ++nump_;
}

void HybridSystem::action1()
{
    static uint step = 0;
    float3 range_min = make_float3(0.48f, 0.6f, 0.48);
    float3 range_max = make_float3(0.52f, 0.7f, 0.52f);
    uint original_nump = nump_;

    ++step;
    //if (step % 2 == 0) return;

    for (float x = range_min.x; x < range_max.x; x += sys_para_.spacing_fluid)
    {
        for (float y = range_min.y; y < range_max.y; y += sys_para_.spacing_fluid)
        {
            for (float z = range_min.z; z < range_max.z; z += sys_para_.spacing_fluid)
            {
                addParticle2(make_float3(x, y, z), make_float3(0, -6, 0), FLUID, 50);
            }
        }
    }
    BuffInit(device_buff_.get_buff_list(), nump_);
    host_buff_.transfer(device_buff_, original_nump, nump_ - original_nump, cudaMemcpyHostToDevice);
    arrangement_->resetNumParticle(nump_);
}

void HybridSystem::registerGraphicsResources()
{
    if (vbo_resources_registered_) return;

    cudaError_t pos_err = cudaGraphicsGLRegisterBuffer(&position_vbo_res_, position_vbo_, cudaGraphicsMapFlagsWriteDiscard);
    cudaError_t col_err = cudaGraphicsGLRegisterBuffer(&color_vbo_res_, color_vbo_, cudaGraphicsMapFlagsWriteDiscard);
    if (pos_err == cudaSuccess && col_err == cudaSuccess)
    {
        vbo_resources_registered_ = true;
    }
    else
    {
        // Registration failed (e.g. no GL context or unsupported config).
        // Clean up any partial registration and fall back to the host-copy path.
        if (pos_err == cudaSuccess) cudaGraphicsUnregisterResource(position_vbo_res_);
        if (col_err == cudaSuccess) cudaGraphicsUnregisterResource(color_vbo_res_);
        position_vbo_res_ = nullptr;
        color_vbo_res_ = nullptr;
    }
}

void HybridSystem::unregisterGraphicsResources()
{
    if (!vbo_resources_registered_) return;
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(position_vbo_res_));
    CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(color_vbo_res_));
    position_vbo_res_ = nullptr;
    color_vbo_res_ = nullptr;
    vbo_resources_registered_ = false;
}

void HybridSystem::createPersistentCudaResources()
{
    // Guarded so repeated calls (e.g. lazy tick-event creation in tick()) do not leak.
    if (!copy_stream_) CUDA_SAFE_CALL(cudaStreamCreate(&copy_stream_));
    if (!compute_done_event_) CUDA_SAFE_CALL(cudaEventCreate(&compute_done_event_));
    if (!copy_done_event_) CUDA_SAFE_CALL(cudaEventCreate(&copy_done_event_));

    if (get_detailed_time_ && !tick_events_created_)
    {
        for (int i = 0; i < 5; ++i)
        {
            CUDA_SAFE_CALL(cudaEventCreate(&tick_events_[i]));
        }
        tick_events_created_ = true;
    }
}

void HybridSystem::destroyPersistentCudaResources()
{
    if (tick_events_created_)
    {
        for (int i = 0; i < 5; ++i)
        {
            if (tick_events_[i]) CUDA_SAFE_CALL(cudaEventDestroy(tick_events_[i]));
            tick_events_[i] = nullptr;
        }
        tick_events_created_ = false;
    }
    if (copy_done_event_) { CUDA_SAFE_CALL(cudaEventDestroy(copy_done_event_)); copy_done_event_ = nullptr; }
    if (compute_done_event_) { CUDA_SAFE_CALL(cudaEventDestroy(compute_done_event_)); compute_done_event_ = nullptr; }
    if (copy_stream_) { CUDA_SAFE_CALL(cudaStreamDestroy(copy_stream_)); copy_stream_ = nullptr; }
}

void HybridSystem::waitForGraphicsCopy()
{
    if (copy_stream_)
    {
        CUDA_SAFE_CALL(cudaStreamSynchronize(copy_stream_));
    }
}
}

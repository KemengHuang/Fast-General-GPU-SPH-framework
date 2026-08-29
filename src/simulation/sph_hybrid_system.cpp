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
#include <iomanip>
#include <sstream>
#include <vector>

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

namespace {

struct BenchmarkStateChecksum {
    double position_x = 0.0;
    double position_y = 0.0;
    double position_z = 0.0;
    double inverse_density = 0.0;
    double weighted_position = 0.0;
};

BenchmarkStateChecksum computeBenchmarkStateChecksum(
    float4 *host_positions, const float4 *device_positions, uint particle_count)
{
    CUDA_SAFE_CALL(cudaMemcpy(host_positions, device_positions,
                              static_cast<size_t>(particle_count) * sizeof(float4),
                              cudaMemcpyDeviceToHost));

    BenchmarkStateChecksum checksum;
    for (uint i = 0; i < particle_count; ++i)
    {
        const float4 position = host_positions[i];
        checksum.position_x += position.x;
        checksum.position_y += position.y;
        checksum.position_z += position.z;
        checksum.inverse_density += position.w;
        checksum.weighted_position += static_cast<double>((i & 1023u) + 1u)
            * (position.x + 3.0 * position.y + 7.0 * position.z);
    }
    return checksum;
}

} // namespace

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




/****************************** HybridSystem ******************************/

HybridSystem::HybridSystem(const float3 &real_world_side, const float3 &sim_origin, bool headless)
#if GSPH_HEADLESS
    : headless_mode_(true)
#else
    : headless_mode_(headless)
#endif
{
#if GSPH_HEADLESS
    (void)headless;
#endif
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

#if !GSPH_HEADLESS
    initializeGraphics();
#endif

    generate_mesh_ = false;
    add_smoke_ = false;
}

HybridSystem::~HybridSystem()
{
#if !GSPH_HEADLESS
    shutdownGraphics();
#endif
    destroyPersistentCudaResources();
    delete arrangement_;
    arrangement_ = nullptr;
    resetBuffer(0);
    releaseKernel();
}
void HybridSystem::tick()
{
    if (!is_running_) return;
    tick_timer_.set_start();
    static int frame_index = 0;
    if (get_detailed_time_ && !tick_events_created_)
    {
        createPersistentCudaResources();
    }
    int *compact_indices = arrangement_->getDevCellIndex();
    int *cell_offsets = arrangement_->getDevCellOffset();
    int *micro_cell_offsets = arrangement_->getDevCellOffsetM();
    int *cell_particle_counts = arrangement_->getDevCellNumP();
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[0]));

    arrangement_->arrangeHybridFrame();

#if HYBRID_DEVICE_GRID_SIZING
    // TRA particles occupy [0, middle) of the compaction index. The physics kernels
    // read the actual split point from device memory, so the upper bound suffices
    // here and no host readback is needed.
    ParticleIdxRange tra_range(0, nump_);
    // Safe upper bound on the SMS task count: one task per 32 particles plus at
    // most one partial task per cell.
    int sms_task_upper_bound = ceil_int(
        static_cast<int>(nump_), kSmsTaskParticles)
        + arrangement_->getNumC();
#else
    // Host-synced sizing: middle_value_ is valid after the arrange sync.
    int middle_host = arrangement_->getTraParticleCount();
    if (middle_host < 0 || middle_host > static_cast<int>(nump_))
        middle_host = nump_;
    ParticleIdxRange tra_range(0, middle_host);
    int sms_task_upper_bound = 0;  // unused on the host-synced path
#endif
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[1]));

    computeDensityHybrid(
        micro_cell_offsets, tra_range, device_buff_.get_buff_list(),
        compact_indices, cell_offsets, cell_particle_counts,
        arrangement_->getSmsTasks(), arrangement_->getSmsTaskCount(),
        arrangement_->getDeviceSmsTaskCount(),
        arrangement_->getDeviceTraParticleCount(),
        sms_task_upper_bound);
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[2]));

    computeForceHybrid(
        micro_cell_offsets, tra_range, device_buff_.get_buff_list(),
        compact_indices, cell_offsets, cell_particle_counts,
        arrangement_->getSmsTasks(), arrangement_->getSmsTaskCount(),
        arrangement_->getDeviceSmsTaskCount(),
        arrangement_->getDeviceTraParticleCount(),
        sms_task_upper_bound);
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[3]));

    advance(device_buff_.get_buff_list(), nump_);
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(tick_events_[4]));

#if !GSPH_HEADLESS
    stageParticleDataForGraphics();
#endif

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
    ++frame_index;
    loop = frame_index;
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

    // The copy and reduction happen after the measured interval.
    const BenchmarkStateChecksum checksum = computeBenchmarkStateChecksum(
        host_buff_.get_buff_list().position_d,
        device_buff_.get_buff_list().position_d, nump_);

    std::cout << "\n========== Headless benchmark (" << measured_frames
              << " measured + " << warmup_frames
              << " warm-up frames) ==========\n";
    std::cout << "Total wall time: " << elapsed_ms << " ms\n";
    std::cout << "Average frame time: " << avg_frame << " ms\n";
    std::cout << "FPS: " << fps << "\n";
    std::cout << std::setprecision(17);
    std::cout << "State checksum: " << checksum.position_x << ", "
              << checksum.position_y << ", " << checksum.position_z << ", "
              << checksum.inverse_density << ", "
              << checksum.weighted_position << "\n";
    std::cout << std::setprecision(6);
    std::cout << "Hybrid split: " << arrangement_->getTraParticleCount()
              << " TRA particles, " << arrangement_->getSmsTaskCount()
              << " SMS tasks\n";
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


void HybridSystem::createPersistentCudaResources()
{
    // Guarded so repeated calls (e.g. lazy tick-event creation in tick()) do not leak.
#if !GSPH_HEADLESS
    createGraphicsCudaResources();
#endif

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
#if !GSPH_HEADLESS
    destroyGraphicsCudaResources();
#endif
}
}

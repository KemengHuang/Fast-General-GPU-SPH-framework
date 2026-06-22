//
// sph_hybrid_system.cpp
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#define _USE_MATH_DEFINES

#include "sph_hybrid_system.h"
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <GL/freeglut.h>
#include "json/json.h"
#include "json/reader.h"
#include "cuda_math.cuh"
#include "sph_kernel.cuh"
#include "sph_marching_cube.h"
#include "pcisph_factor.h"   //sf add
#include "parameters.h"

namespace sph
{

const char *kDefaultSceneFileName = "scene_default.json";
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
//sf 设置SPH系统参数
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

HybridSystem::HybridSystem(const float3 &real_world_side, const float3 &sim_origin)
{
	Scene scene;
    defaultInitializeSPHSysPara(sys_para_, &scene);
    sys_para_.sim_ratio = make_float3(real_world_side.x / sys_para_.world_size.x,
                                      real_world_side.y / sys_para_.world_size.y,
                                      real_world_side.z / sys_para_.world_size.z);
    sys_para_.sim_origin = sim_origin;

	initializeScene(kDefaultSceneFileName, scene);

    initializeKernel();

    // render 
    particle_texture_.loadPNG("ball32.png");
    glGenBuffers(1, &position_vbo_);
    glGenBuffers(1, &color_vbo_);

    get_detailed_time_ = true;
    generate_mesh_ = false;
    add_smoke_ = false;
}

HybridSystem::~HybridSystem()
{
    resetBuffer(0);
    releaseKernel();
}
int tt = 0;
float time = 0;
//sf 模拟过程的主函数
void HybridSystem::tick()
{
    if (!is_running_) return;
	tt++;
    HighResolutionTimerForWin timer;
    timer.set_start();
    static int step = 0;
    cudaEvent_t start, end0, end1, end2, end3; //sf 记录时间
    if (get_detailed_time_)
    {
        CUDA_SAFE_CALL(cudaEventCreate(&start));
        CUDA_SAFE_CALL(cudaEventCreate(&end0));
        CUDA_SAFE_CALL(cudaEventCreate(&end1));
        CUDA_SAFE_CALL(cudaEventCreate(&end2));
        CUDA_SAFE_CALL(cudaEventCreate(&end3));
    }
    static float tot_pres = 0.0f, tot_forc = 0.0f, tot_tot = 0.0f;
    int *d_index = arrangement_->getDevCellIndex();
    int *offset_data = arrangement_->getDevOffsetData();
    int *cell_offset = arrangement_->getDevCellOffset();

	int *cell_offsetM = arrangement_->getDevCellOffsetM();

    int *cell_nump = arrangement_->getDevCellNumP();
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(start));
    int middle = nump_;
    //arrangement_->sortParticles();

    middle = arrangement_->arrangeHybridMode9M();
//    arrangement_->CountingSortCUDA();
//    arrangement_->assignTasksFixedCTA();


    ParticleIdxRange tra_range(0, middle);      // [0, middle)

//      std::cout << "middle value: ******************************************" << middle << std::endl;
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(end0));

	computeDensityHybrid128n(cell_offsetM, tra_range, device_buff_.get_buff_list(), d_index, cell_offset, cell_nump, arrangement_->getBlockTasks(), arrangement_->getNumBlockSMSMode());
//    computeDensitySMS64(device_buff_.get_buff_list(), cell_offset, cell_nump, arrangement_->getBlockTasks(), arrangement_->getNumBlockSMSMode());
//    computeDensityTRA(device_buff_.get_buff_list(), ParticleIdxRange(0, nump_), cell_offset, cell_nump);
    //   std::cout << step << std::endl;
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(end1));

	computeForceHybrid128n(cell_offsetM, tra_range, device_buff_.get_buff_list(), d_index, cell_offset, cell_nump, arrangement_->getBlockTasks(), arrangement_->getNumBlockSMSMode());
//    computeForceSMS64(device_buff_.get_buff_list(), cell_offset, cell_nump, arrangement_->getBlockTasks(), arrangement_->getNumBlockSMSMode());
//    computeForceTRA(device_buff_.get_buff_list(), ParticleIdxRange(0, nump_), cell_offset, cell_nump);
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(end2));

    advance(device_buff_.get_buff_list(), nump_);
	//advanceWave(device_buff_.get_buff_list(), nump_,time);
	time += 0.003;
    if (get_detailed_time_) CUDA_SAFE_CALL(cudaEventRecord(end3));

    //sf 将数据复制回host
    CUDA_SAFE_CALL(cudaMemcpy(host_buff_.get_buff_list().final_position, device_buff_.get_buff_list().final_position, nump_ * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(host_buff_.get_buff_list().color, device_buff_.get_buff_list().color, nump_ * sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    timer.set_end();



    // sf 计算各个步骤的总时间
    if (get_detailed_time_)
    {
        CUDA_SAFE_CALL(cudaEventElapsedTime(&pre_time_, start, end0));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&density_time_, end0, end1));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&force_time_, end1, end2));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&total_time_, start, end3));

        tot_pres += density_time_;
        tot_forc += force_time_;
        tot_tot += total_time_;

        CUDA_SAFE_CALL(cudaEventDestroy(start));
        CUDA_SAFE_CALL(cudaEventDestroy(end0));
        CUDA_SAFE_CALL(cudaEventDestroy(end1));
        CUDA_SAFE_CALL(cudaEventDestroy(end2));
        CUDA_SAFE_CALL(cudaEventDestroy(end3));
    }
    ++step;
    loop = step;
    if (loop > 640) {
  //      exit(0);
    }
}

//sf 初始化场景


void HybridSystem::initializeScene(const std::string &file_name, Scene scene)
{
   
    sys_para_.mass = scene.mass;
    transSysParaToDevice(&sys_para_);

    resetBuffer(scene.recomm_nump);
    particle_interval = scene.interval;
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

    //arrangement_.reset(new Arrangement(device_buff_, device_buff_temp_, nump_, sys_para_.cell_size, sys_para_.grid_size));
    arrangement_ = //new Arrangement(device_buff_, device_buff_temp_, nump_, buff_capacity_, sys_para_.cell_size, sys_para_.grid_size);
        new Arrangement(device_buff_, device_buff_temp_,  nump_, buff_capacity_, sys_para_.cell_size, sys_para_.grid_size);
}




void HybridSystem::initializeScene2(const std::string &file_name)
{
    Scene scene;
    //if (!readSceneFromJsonFile(scene, kDefaultSceneFileName))
    //{
    //    return;
    //}

    //sys_para_.mass = scene.mass;
    transSysParaToDevice(&sys_para_);  //sf 传递系统参数至device端

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
    //arrangement_.reset(new Arrangement(device_buff_, device_buff_temp_, nump_, sys_para_.cell_size, sys_para_.grid_size));
    arrangement_ = new Arrangement(device_buff_, device_buff_temp_, nump_, buff_capacity_, sys_para_.cell_size, sys_para_.grid_size);
}

void HybridSystem::setPause()
{
    is_running_ = !is_running_;
}

bool HybridSystem::isRunning()
{
    return is_running_;
}

//sf 返回粒子总数
uint HybridSystem::getNumParticles()
{
    return nump_;
}

//sf 返回粒子idx的final_position
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

    // Point buffers
	//GLint gsize = size;
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo_);
    glBufferData(GL_ARRAY_BUFFER, nump_*sizeof(float3), host_buff_.get_buff_list().final_position, GL_DYNAMIC_DRAW);
	glVertexPointer(3, GL_FLOAT, 0, 0x0);
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);
    glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(uint), host_buff_.get_buff_list().color, GL_DYNAMIC_DRAW);
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
int cftn = 0;
float pre = 0;
float tot = 0;
float fore = 0;
std::ofstream outtt("combine666666666666666666666666666666666.txt");
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
    float xixi = float(1000.0f / total_time_);

	if (tt > 10&&tt<631){
        cftn++;
        pre += pre_time_;
        fore += (density_time_ + force_time_);
		//outtt << xixi << std::endl;
        if (tt == 630) {
            float outv1 = pre / cftn;
            float outv2 = fore / cftn;
            outtt << outv1 << "   " << outv2 << "   " << outv1 + outv2 << std::endl;
        }
	}
//	std::cout << tt << std::endl;
    ss << "FPS: " << xixi;//time;


    frame_timer_.set_start();
    renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, ss);
    ss.str(""); y += delta_y;

    // output number of particles
    ss << "#particles: " << nump_;
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
}
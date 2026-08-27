// Auto-generated from src/sph_kernel.cu during solver reorganization.

#include "solver/kernel_common.cuh"

namespace sph {
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knInit(ParticleBufferList buff_list, int nump)
{
    unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    if (idx >= nump) return;
    float4 ev;
    ev.x = 0; ev.y = 0; ev.z = 0; ev.w = 0;
    buff_list.evaluated_velocity[idx] = ev;
    //    buff_list.density[idx] = 0.f;
    buff_list.acceleration[idx] = make_float3(0.f, 0.f, 0.f);
    //    buff_list.pressure[idx] = 0.f;
}
__global__
void knManualSetting(ParticleBufferList buff_list, unsigned int nump, int step)
{
    unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);

    if (idx >= nump) return;
}
__global__
void knIntegrateVelocityMix(ParticleBufferList buff_list, unsigned int nump)
{

}
__device__
inline float4 addfloat4(const float4& vec4, const float3 &vec3){
    //  make_float3()
    float4 nVec4;
    nVec4.x = vec4.x + vec3.x;
    nVec4.y = vec4.y + vec3.y;
    nVec4.z = vec4.z + vec3.z;
    nVec4.w = vec4.w;
    return nVec4;
}

__device__
inline float4 floathalf4add3(float3 ra, const float4& vec4){
    float4 v4;
    v4.x = (ra.x + vec4.x)*0.5;;
    v4.y = (ra.y + vec4.y)*0.5;
    v4.z = (vec4.z + ra.z)*0.5;
    v4.w = vec4.w;
    return v4;
}
__device__
inline float3 float4m3(float3 ra, const float4& vec4){

    return make_float3(ra.x*vec4.x, ra.y*vec4.y, vec4.z*ra.z);
}
__device__ 
inline float dotV34(float3 v3, float4 v4){
    return v3.x*v4.x + v3.y*v4.y + v3.z*v4.z;
}
__global__
void knIntegrateVelocitySimWave(ParticleBufferList buff_list, unsigned int nump, float time)
{
    unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);

    if (idx >= nump) return;

    register float4 position = buff_list.position_d[idx];
    register float4 eval_vel = buff_list.evaluated_velocity[idx];
    float3 t_velocity = buff_list.velocity[idx];
    float3 accelerate = buff_list.acceleration[idx];// / t_position.w + kDevSysPara.gravity;
    float diff, speed;
    float3 norm;
    float adj;
    const float epsilon = 0.00001f;
    const float alpha = 20000000;
    const float beta = 200000;
    float slop = 0.09;

    register float kernel = kDevSysPara.kernel + 0.01;

    diff = kernel - (position.y - (kDevSysPara.bound_min.y + (position.x - kDevSysPara.bound_min.x)*slop));
    if (diff > 0.0f)
    {
        norm = make_float3(-slop, 1.0 - slop, 0);
        adj = alpha * diff - beta * dotV34(norm, eval_vel);
        norm *= adj; accelerate += norm;
    }

    diff = kernel - (kDevSysPara.bound_max.y - position.y);
    if (diff > 0.0f)
    {
        accelerate.y -= alpha * diff + beta * eval_vel.y;
    }

    float seq = 3;
    float rd = sin(time * seq);
    diff = kernel - (position.x - (kDevSysPara.bound_min.x + 1.2*(0.5*rd + 0.5)));
    if (diff > 0.0f)
    {
        accelerate.x += (alpha * diff*pow(5000 * (position.y - kDevSysPara.bound_min.y) / (kDevSysPara.bound_max.y - kDevSysPara.bound_min.y), 2) - beta * eval_vel.x);
    }


    diff = kernel - (kDevSysPara.bound_max.x - position.x);
    if (diff > 0.0f)
    {
        accelerate.x -= alpha * diff + beta * eval_vel.x;
    }

    diff = kernel - (position.z - kDevSysPara.bound_min.z);
    if (diff > 0.0f)
    {
        accelerate.z += alpha * diff - beta * eval_vel.z;
    }
    diff = kernel - (kDevSysPara.bound_max.z - position.z);
    if (diff > 0.0f)
    {
        accelerate.z -= alpha * diff + beta * eval_vel.z;
    }

    const float acc_limit = 3000000;    // squared-magnitude limit
    const float vel_limit = 36;         // squared-magnitude limit
    accelerate = (accelerate / position.w + kDevSysPara.gravity);
    speed = accelerate.x * accelerate.x + accelerate.y * accelerate.y + accelerate.z * accelerate.z;
    if (speed > acc_limit)
        accelerate *= sqrtf(acc_limit / speed); // clamp |a| to sqrt(acc_limit)
    t_velocity += accelerate * kDevSysPara.time_step;
    speed = t_velocity.x*t_velocity.x + t_velocity.y*t_velocity.y + t_velocity.z*t_velocity.z;
    if (speed > vel_limit)
        t_velocity *= sqrtf(vel_limit / speed); // clamp |v| to sqrt(vel_limit)

    position = addfloat4(position, t_velocity * kDevSysPara.time_step);

    if (position.x >= kDevSysPara.bound_max.x){
        t_velocity.x = t_velocity.x * kDevSysPara.wall_damping;
        position.x = kDevSysPara.bound_max.x - kernel;
    }
    if (position.x <= kDevSysPara.bound_min.x){
        t_velocity.x = t_velocity.x * kDevSysPara.wall_damping;
        position.x = kDevSysPara.bound_min.x + kernel;
    }
    if (position.y >= kDevSysPara.bound_max.y){
        t_velocity.y = t_velocity.y * kDevSysPara.wall_damping;
        position.y = kDevSysPara.bound_max.y - kernel;
    }
    if (position.y <= kDevSysPara.bound_min.y + (position.x - kDevSysPara.bound_min.x)*slop){
        t_velocity.y = t_velocity.y * kDevSysPara.wall_damping;
        position.y = kDevSysPara.bound_min.y + kernel + (position.x - kDevSysPara.bound_min.x)*slop;
    }
    if (position.z >= kDevSysPara.bound_max.z){
        t_velocity.z = t_velocity.z * kDevSysPara.wall_damping;
        position.z = kDevSysPara.bound_max.z - kernel;
    }
    if (position.z <= kDevSysPara.bound_min.z){
        t_velocity.z = t_velocity.z * kDevSysPara.wall_damping;
        position.z = kDevSysPara.bound_min.z + kernel;
    }

    float denv = (4000 - position.w) / 6000;
    buff_list.color[idx] = COLORA(0.8f * denv, 0.7f * denv + 0.1, 0.8, 1.0);

    buff_list.position_d[idx] = position;
    buff_list.velocity[idx] = t_velocity;
    buff_list.evaluated_velocity[idx] = floathalf4add3(t_velocity, eval_vel);
    buff_list.final_position[idx] = float4m3(kDevSysPara.sim_ratio, position) + kDevSysPara.sim_origin;
}
__global__
void knIntegrateVelocitySim(ParticleBufferList buff_list, unsigned int nump)
{
	unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);

	if (idx >= nump) return;

	register float4 t_position = buff_list.position_d[idx];
	register float4 eval_vel = buff_list.evaluated_velocity[idx];

	//       register float4 t_position = buff_list.position_d[idx];
	//       register float4 eval_vel = buff_list.evaluated_velocity[idx];

	float3 t_velocity = buff_list.velocity[idx];
	float3 acc_temp = buff_list.acceleration[idx] / t_position.w + kDevSysPara.gravity;
	float diff, speed;


	float length = sqrt(acc_temp.x*acc_temp.x + acc_temp.y * acc_temp.y + acc_temp.z * acc_temp.z);
	if (length > (kDevSysPara.kernel / (5 * kDevSysPara.time_step* kDevSysPara.time_step)))
		acc_temp = acc_temp*(kDevSysPara.kernel / (5 * kDevSysPara.time_step* kDevSysPara.time_step)) / length;

	float3 velocity = t_velocity + (acc_temp)* kDevSysPara.time_step;
	float4 position = addfloat4(t_position, velocity * kDevSysPara.time_step);// t_position + velocity * kDevSysPara.time_step;

	const float BOUNDARY = 0.0001f;

	if (position.x >= kDevSysPara.world_size.x - BOUNDARY)
	{
		velocity.x = velocity.x * kDevSysPara.wall_damping;
		position.x = kDevSysPara.world_size.x - BOUNDARY;
	}

	if (position.x < 0.0f + BOUNDARY)
	{
		velocity.x = velocity.x * kDevSysPara.wall_damping;
		position.x = 0.0f + BOUNDARY;
	}

	if (position.y >= kDevSysPara.world_size.y - BOUNDARY)
	{
		velocity.y = velocity.y * kDevSysPara.wall_damping;
		position.y = kDevSysPara.world_size.y - BOUNDARY;
	}

	if (position.y < 0.0f + BOUNDARY)
	{
		velocity.y = velocity.y * kDevSysPara.wall_damping;
		position.y = 0.0f + BOUNDARY;
	}

	if (position.z >= kDevSysPara.world_size.z - BOUNDARY)
	{
		velocity.z = velocity.z * kDevSysPara.wall_damping;
		position.z = kDevSysPara.world_size.z - BOUNDARY;
	}

	if (position.z < 0.0f + BOUNDARY)
	{
		velocity.z = velocity.z * kDevSysPara.wall_damping;
		position.z = 0.0f + BOUNDARY;
	}

	buff_list.position_d[idx] = position;
	buff_list.velocity[idx] = velocity;
	buff_list.evaluated_velocity[idx] = floathalf4add3(velocity, eval_vel);
	buff_list.final_position[idx] = float4m3(kDevSysPara.sim_ratio, position) + kDevSysPara.sim_origin;
}

__global__
void knIntegrateVelocityE(ParticleBufferList buff_list, unsigned int nump)
{
	unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);

	if (idx >= nump) return;

	register float4 t_position = buff_list.position_d[idx];
	register float4 eval_vel = buff_list.evaluated_velocity[idx];

	//       register float4 t_position = buff_list.position_d[idx];
	//       register float4 eval_vel = buff_list.evaluated_velocity[idx];


	float3 t_velocity = buff_list.velocity[idx];
	float3 accelerate = buff_list.acceleration[idx];

	float diff, speed;
	const float alpha = 20000000;
	const float beta = 200000;

	register float kernel = kDevSysPara.kernel + 0.01;

	diff = kernel - (t_position.y - kDevSysPara.bound_min.y);
	if (diff > 0.0f)
	{
		accelerate.y += alpha * diff - beta * t_velocity.y;
	}
	diff = kernel - (kDevSysPara.bound_max.y - t_position.y);
	if (diff > 0.0f)
	{
		accelerate.y -= alpha * diff + beta * t_velocity.y;
	}

	diff = kernel - (t_position.x - kDevSysPara.bound_min.x);
	if (diff > 0.0f)
	{
		accelerate.x += alpha * diff - beta * t_velocity.x;
	}
	diff = kernel - (kDevSysPara.bound_max.x - t_position.x);
	if (diff > 0.0f)
	{
		accelerate.x -= alpha * diff + beta * t_velocity.x;
	}
	// Z-axis
	diff = kernel - (t_position.z - kDevSysPara.bound_min.z);
	if (diff > 0.0f)
	{
		accelerate.z += alpha * diff - beta * t_velocity.z;
	}
	diff = kernel - (kDevSysPara.bound_max.z - t_position.z);
	if (diff > 0.0f)
	{
		accelerate.z -= alpha * diff + beta * t_velocity.z;
	}
	const float acc_limit = 3000000;	// squared-magnitude limit
	const float vel_limit = 36;			// squared-magnitude limit
	accelerate = (accelerate / t_position.w + kDevSysPara.gravity);
	speed = accelerate.x * accelerate.x + accelerate.y * accelerate.y + accelerate.z * accelerate.z;
	if (speed > acc_limit)
		accelerate *= sqrtf(acc_limit / speed);	// clamp |a| to sqrt(acc_limit)
	t_velocity += accelerate * kDevSysPara.time_step;

	speed = t_velocity.x*t_velocity.x + t_velocity.y*t_velocity.y + t_velocity.z*t_velocity.z;
	if (speed > vel_limit)
		t_velocity *= sqrtf(vel_limit / speed);	// clamp |v| to sqrt(vel_limit)
	float4 position = addfloat4(t_position, t_velocity * kDevSysPara.time_step);

	if (position.x >= kDevSysPara.bound_max.x){
		t_velocity.x = t_velocity.x * kDevSysPara.wall_damping;
		position.x = kDevSysPara.bound_max.x - kernel;
	}
	if (position.x <= kDevSysPara.bound_min.x){
		t_velocity.x = t_velocity.x * kDevSysPara.wall_damping;
		position.x = kDevSysPara.bound_min.x + kernel;
	}
	if (position.y >= kDevSysPara.bound_max.y){
		t_velocity.y = t_velocity.y * kDevSysPara.wall_damping;
		position.y = kDevSysPara.bound_max.y - kernel;
	}
	if (position.y <= kDevSysPara.bound_min.y){
		t_velocity.y = t_velocity.y * kDevSysPara.wall_damping;
		position.y = kDevSysPara.bound_min.y + kernel;
	}
	if (position.z >= kDevSysPara.bound_max.z){
		t_velocity.z = t_velocity.z * kDevSysPara.wall_damping;
		position.z = kDevSysPara.bound_max.z - kernel;
	}
	if (position.z <= kDevSysPara.bound_min.z){
		t_velocity.z = t_velocity.z * kDevSysPara.wall_damping;
		position.z = kDevSysPara.bound_min.z + kernel;
	}

	buff_list.position_d[idx] = position;
	buff_list.velocity[idx] = t_velocity;
	buff_list.evaluated_velocity[idx] = floathalf4add3(t_velocity, eval_vel);
	buff_list.final_position[idx] = float4m3(kDevSysPara.sim_ratio, position) + kDevSysPara.sim_origin;
}
__global__
void knIntegrateVelocity(ParticleBufferList buff_list, unsigned int nump)
{

}
}

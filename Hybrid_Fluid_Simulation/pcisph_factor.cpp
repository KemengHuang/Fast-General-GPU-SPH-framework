//
// created by ruanjm on 08/05/15
// Copyright (c) 2023 KMHuang and ruanjm. All right reserved.
//

#include "pcisph_factor.h"
#include "sph_hybrid_system.h"

namespace sph
{



float computeDensityErrorFactorSMS(float mass, float rest_density, float time_step, ParticleBufferList buff_list, int *cell_start, int *cell_end, BlockTask *block_task, int num_block, uint nump)
{
	uint max_num_neighbors = 0;
	uint particle_with_max_num_neighbors = 0;

	sumGrad *particle_host;
	particle_host = (sumGrad *)malloc(sizeof(sumGrad)*nump);
	sumGrad *particle_device;
	CUDA_SAFE_CALL(cudaMalloc((void**)&particle_device, nump * sizeof(sumGrad)));
	//cudaMemset(particle_device, 0, nump);
	computeGradWValuesSimpleSMS(buff_list, cell_start, cell_end, block_task, num_block, particle_device);

	cudaMemcpy(particle_host, particle_device, sizeof(sumGrad)*nump, cudaMemcpyDeviceToHost);//传回主机端
	printf("CUDA_SAFE_CALL(cudaMalloc((void**)&particle_device, nump * sizeof(sumGrad))): %.20f\n", particle_host[0].sumGradWDot);

    //int tds = 256;
    int blocks = (nump + 255)/256;
	for (uint id = 0; id < nump; id++) {
		//if (particle_host[id].ph != LAVA_FLUID) continue;
		if (particle_host[id].num_neigh>max_num_neighbors) {
			max_num_neighbors = particle_host[id].num_neigh;
			particle_with_max_num_neighbors = id;
		}
	}
	float factor = computeFactorSimple(mass, rest_density, time_step, particle_with_max_num_neighbors, particle_host);
	free(particle_host);  
	cudaFree(particle_device);
	return factor;
}


float computeFactorSimple(float mass, float rest_density, float time_step, uint index, sumGrad *particle_host)
{
	float restVol = mass / rest_density;
	float preFactor = 2 * restVol * restVol * time_step * time_step;     //my (delta)t^2*m^2/p0^2   // 是否需要2倍！！！
	float3 temp_plus = particle_host[index].sumGradW;
	float3 temp_minus = temp_plus;
	temp_minus.x *= -1; temp_minus.y *= -1; temp_minus.z *= -1;

//	printf("temp_plus: %.20f %.20f %.20f\n", temp_plus.x, temp_plus.y, temp_plus.z);
//	printf("sumGradWDot: %.20f\n", particle_host[index].sumGradWDot);

	float gradWTerm = (temp_plus.x*temp_minus.x + temp_plus.y*temp_minus.y + temp_plus.z*temp_minus.z) - particle_host[index].sumGradWDot;

	//printf("gradWTerm %f \n", gradWTerm);

	float divisor = preFactor * gradWTerm;

//	printf("preFactor: %.20f\n", preFactor);
//	printf("gradWTerm: %.20f\n", gradWTerm);

	if (divisor == 0)
	{
//		printf("pre-compute densErrFactor: division by 0 /n");
		exit(0);
	}

	return -1.0 / divisor;
}

}
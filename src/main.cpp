#include "core/compile_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "simulation/sph_hybrid_system.h"

#if !GSPH_HEADLESS
#include "render/gui_application.h"
#endif

namespace {

bool initializeCuda()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        std::fprintf(stderr, "There is no CUDA device.\n");
        return false;
    }

    for (int device = 0; device < device_count; ++device)
    {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
            properties.major >= 1)
        {
            if (cudaSetDevice(device) != cudaSuccess) return false;
            std::printf("CUDA initialized.\n");
            return true;
        }
    }

    std::fprintf(stderr, "There is no supported CUDA device.\n");
    return false;
}

int parseBenchmarkFrames(int argc, char **argv)
{
    int frame_count = GSPH_HEADLESS ? 200 : 0;
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::strcmp(argv[i], "--headless") == 0 ||
            std::strcmp(argv[i], "--benchmark") == 0)
        {
            frame_count = std::atoi(argv[++i]);
        }
    }
    return frame_count;
}

int runHeadlessBenchmark(int frame_count)
{
    if (frame_count <= 0)
    {
        std::fprintf(stderr, "Benchmark frame count must be positive.\n");
        return EXIT_FAILURE;
    }
    if (!initializeCuda()) return EXIT_FAILURE;

    const float3 world_origin = make_float3(-40.0f, -40.0f, -40.0f);
    const float3 world_size = make_float3(80.0f, 80.0f, 80.0f);
    sph::HybridSystem system(world_size, world_origin, true);
    system.runBenchmark(frame_count);
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv)
{
    const int benchmark_frames = parseBenchmarkFrames(argc, argv);
    if (benchmark_frames > 0 || GSPH_HEADLESS)
        return runHeadlessBenchmark(benchmark_frames);

#if GSPH_HEADLESS
    return EXIT_FAILURE;
#else
    return runGuiApplication(argc, argv);
#endif
}

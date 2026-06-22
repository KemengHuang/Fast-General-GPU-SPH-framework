cmake_minimum_required(VERSION 3.18)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

project(Hybrid_Fluid_Simulation LANGUAGES CXX CUDA)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA architectures (adjust as needed)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures")
endif()

# Executable
add_executable(gsph)

# Sources – prefer explicit list, but GLOB is okay for small projects
file(GLOB_RECURSE gsph_SOURCES CONFIGURE_DEPENDS
    "Hybrid_Fluid_Simulation/*.cu" "Hybrid_Fluid_Simulation/*.cpp")
target_sources(gsph PRIVATE ${gsph_SOURCES})

# Include directories
target_include_directories(gsph PRIVATE "Hybrid_Fluid_Simulation/")
# CUDAToolkit includes are automatic via target_link_libraries

# Definitions for asset/output paths
target_compile_definitions(gsph PRIVATE
    gsph_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets"
    gsph_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output"
)

# Link libraries
target_link_libraries(gsph PRIVATE
    CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::cudart
    GLUT::GLUT GLEW::GLEW OpenGL::GL Eigen3::Eigen
)

# C++ and CUDA standards
target_compile_features(gsph PRIVATE cxx_std_17 cuda_std_17)

# Separate device code linking
set_target_properties(gsph PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compiler options
target_compile_options(gsph PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        --extended-lambda
        --expt-relaxed-constexpr
        --default-stream=per-thread
    >
)

# MSVC‑specific CUDA flags (disable noisy warnings)
if(MSVC)
    target_compile_options(gsph PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler="/Zc:preprocessor"
            -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
            -Xcudafe "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
        >
    )
endif()

# Source grouping (optional)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${gsph_SOURCES})
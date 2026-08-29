cmake_minimum_required(VERSION 3.18)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Set the project default before CUDA compiler detection. Command-line values
# such as -DCMAKE_CUDA_ARCHITECTURES=120 still take precedence.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures")
endif()

project(Hybrid_Fluid_Simulation LANGUAGES CXX CUDA)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Executable
add_executable(gsph)

# Recursively collect implementation and header files under src/ and third_party/.
file(GLOB_RECURSE gsph_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.cu"
    "src/*.h"
    "src/*.cuh"
    "third_party/lodepng/*.cpp"
    "third_party/lodepng/*.h"
)
target_sources(gsph PRIVATE ${gsph_SOURCES})

option(GSPH_USE_REGISTER_SMS
    "Use register-resident iterator state for the SMS density/force path" ON)
if(GSPH_USE_REGISTER_SMS)
    target_compile_definitions(gsph PRIVATE GSPH_USE_REGISTER_SMS=1)
else()
    target_compile_definitions(gsph PRIVATE GSPH_USE_REGISTER_SMS=0)
endif()

# Include directories
#   src/            project headers
#   third_party/    lodepng (used by gl_texture.h)
target_include_directories(gsph PRIVATE
    src/
    third_party/lodepng/
)

# Link libraries
#   Only CUDA::cudart is required; cublas/cusparse/cusolver and Eigen are unused.
target_link_libraries(gsph PRIVATE
    CUDA::cudart
    GLUT::GLUT
    GLEW::GLEW
    OpenGL::GL
    JsonCpp::JsonCpp
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
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /Zc:preprocessor>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --extended-lambda
        --expt-relaxed-constexpr
        --default-stream=per-thread
        $<$<CONFIG:Debug,RelWithDebInfo>:-lineinfo>
    >
)

# MSVC-specific CUDA flags (disable noisy warnings)
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

# ------------------------------------------------------------------
# Visual Studio convenience
# ------------------------------------------------------------------
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT gsph)
set_property(TARGET gsph PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(OutDir)")

# Copy runtime assets next to the executable so F5 / double-click work.
# Assets are copied into an assets/ subfolder so the hard-coded paths in the source work.
add_custom_command(TARGET gsph POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:gsph>/assets
)
add_custom_command(TARGET gsph POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:gsph>/shaders
)

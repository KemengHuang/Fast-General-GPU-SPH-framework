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

option(GSPH_HEADLESS "Build without OpenGL, GLEW, GLUT, or render sources" OFF)

# Compute dependencies are always required. Graphics dependencies are resolved
# only for the GUI build, so a headless machine needs no OpenGL development SDK.
find_package(jsoncpp CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
if(NOT GSPH_HEADLESS)
    find_package(GLEW REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(OpenGL REQUIRED)
endif()

# Executable
add_executable(gsph)

# Recursively collect the simulation sources. Headless builds exclude the
# complete render subtree; GUI builds additionally compile the PNG loader.
file(GLOB_RECURSE gsph_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.cu"
    "src/*.h"
    "src/*.cuh"
)
if(GSPH_HEADLESS)
    list(FILTER gsph_SOURCES EXCLUDE REGEX "[/\\\\]render[/\\\\]")
else()
    file(GLOB_RECURSE gsph_RENDER_DEPENDENCIES CONFIGURE_DEPENDS
        "third_party/lodepng/*.cpp"
        "third_party/lodepng/*.h"
    )
    list(APPEND gsph_SOURCES ${gsph_RENDER_DEPENDENCIES})
endif()
target_sources(gsph PRIVATE ${gsph_SOURCES})

if(GSPH_HEADLESS)
    target_compile_definitions(gsph PRIVATE GSPH_HEADLESS=1)
else()
    target_compile_definitions(gsph PRIVATE GSPH_HEADLESS=0)
endif()

option(GSPH_USE_REGISTER_SMS
    "Use register-resident iterator state for the SMS density/force path" ON)
if(GSPH_USE_REGISTER_SMS)
    target_compile_definitions(gsph PRIVATE GSPH_USE_REGISTER_SMS=1)
else()
    target_compile_definitions(gsph PRIVATE GSPH_USE_REGISTER_SMS=0)
endif()

option(GSPH_ENABLE_HISTORICAL_ISSAME
    "Run the historical judgeTask isSame classification and coloring" ON)
if(GSPH_ENABLE_HISTORICAL_ISSAME)
    target_compile_definitions(gsph PRIVATE
        GSPH_ENABLE_HISTORICAL_ISSAME=1)
else()
    target_compile_definitions(gsph PRIVATE
        GSPH_ENABLE_HISTORICAL_ISSAME=0)
endif()
option(GSPH_ENABLE_SMS_LOCAL_MERGE
    "Use cooperative shared staging for isSame density pairs" ON)
if(GSPH_ENABLE_SMS_LOCAL_MERGE)
    target_compile_definitions(gsph PRIVATE GSPH_ENABLE_SMS_LOCAL_MERGE=1)
else()
    target_compile_definitions(gsph PRIVATE GSPH_ENABLE_SMS_LOCAL_MERGE=0)
endif()
option(GSPH_ENABLE_SMS_LOCAL_MERGE_FORCE
    "Also consume judgeTask pairs in the register force pass" OFF)
if(GSPH_ENABLE_SMS_LOCAL_MERGE_FORCE)
    target_compile_definitions(gsph PRIVATE
        GSPH_ENABLE_SMS_LOCAL_MERGE_FORCE=1)
else()
    target_compile_definitions(gsph PRIVATE
        GSPH_ENABLE_SMS_LOCAL_MERGE_FORCE=0)
endif()
option(GSPH_ENABLE_SAME_CELL_PAIR_FORCE
    "Compute same-cell force pairs once with shared accumulation" OFF)
if(GSPH_ENABLE_SAME_CELL_PAIR_FORCE)
    target_compile_definitions(gsph PRIVATE
        GSPH_ENABLE_SAME_CELL_PAIR_FORCE=1)
else()
    target_compile_definitions(gsph PRIVATE
        GSPH_ENABLE_SAME_CELL_PAIR_FORCE=0)
endif()
# The simulation headers are always visible; lodepng is a GUI-only dependency.
target_include_directories(gsph PRIVATE
    src/
)
if(NOT GSPH_HEADLESS)
    target_include_directories(gsph PRIVATE third_party/lodepng/)
endif()

# Compute libraries are always linked; graphics libraries are GUI-only.
target_link_libraries(gsph PRIVATE
    CUDA::cudart
    JsonCpp::JsonCpp
)
if(NOT GSPH_HEADLESS)
    target_link_libraries(gsph PRIVATE
        GLUT::GLUT
        GLEW::GLEW
        OpenGL::GL
    )
endif()

# C++ and CUDA standards
target_compile_features(gsph PRIVATE cxx_std_17 cuda_std_17)

option(GSPH_ENABLE_IPO
    "Enable host IPO and CUDA device LTO in optimized configurations" ON)
if(GSPH_ENABLE_IPO)
    set_property(TARGET gsph PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET gsph PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
endif()

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
if(NOT GSPH_HEADLESS)
    add_custom_command(TARGET gsph POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders
            $<TARGET_FILE_DIR:gsph>/shaders
    )
endif()

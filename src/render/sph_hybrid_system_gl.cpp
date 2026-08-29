#include "core/compile_config.h"

#if !GSPH_HEADLESS

#include "simulation/sph_hybrid_system.h"

#include <sstream>

#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "core/cuda_call_check.h"
#include "render/particle_vbo_copy.cuh"

namespace sph
{
namespace
{

void setOrthographicProjection(GLdouble width, GLdouble height)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, width, height, 0);
    glMatrixMode(GL_MODELVIEW);
}

void restorePerspectiveProjection()
{
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void renderBitmapString(float x, float y, float z, void *font,
                        const std::stringstream &text)
{
    const std::string value = text.str();
    glRasterPos3f(x, y, z);
    for (const char *c = value.c_str(); *c != '\0'; ++c)
        glutBitmapCharacter(font, *c);
}

} // namespace

void HybridSystem::initializeGraphics()
{
    if (headless_mode_) return;

    particle_texture_.loadPNG("assets/ball32.png");
    glGenBuffers(1, &position_vbo_);
    glGenBuffers(1, &color_vbo_);

    glBindBuffer(GL_ARRAY_BUFFER, position_vbo_);
    glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(float3), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);
    glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(uint), nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    registerGraphicsResources();
}

void HybridSystem::shutdownGraphics()
{
    waitForGraphicsCopy();
    unregisterGraphicsResources();
}

void HybridSystem::stageParticleDataForGraphics()
{
    if (headless_mode_) return;

    CUDA_SAFE_CALL(cudaEventRecord(compute_done_event_, 0));
    if (vbo_resources_registered_) return;

    CUDA_SAFE_CALL(cudaStreamWaitEvent(copy_stream_, compute_done_event_, 0));
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        host_buff_.get_buff_list().final_position,
        device_buff_.get_buff_list().final_position,
        nump_ * sizeof(float3), cudaMemcpyDeviceToHost, copy_stream_));
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        host_buff_.get_buff_list().color,
        device_buff_.get_buff_list().color,
        nump_ * sizeof(uint), cudaMemcpyDeviceToHost, copy_stream_));
    CUDA_SAFE_CALL(cudaEventRecord(copy_done_event_, copy_stream_));
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
    float quadratic[] = {1.0f, 0.01f, 0.001f};
    glEnable(GL_POINT_DISTANCE_ATTENUATION);
    glPointParameterfvARB(GL_POINT_DISTANCE_ATTENUATION, quadratic);
    glPointSize(static_cast<GLfloat>(size));
    glPointParameterfARB(GL_POINT_SIZE_MAX, 32);
    glPointParameterfARB(GL_POINT_SIZE_MIN, 1.0f);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, particle_texture_.get_texture());
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (vbo_resources_registered_)
    {
        CUDA_SAFE_CALL(cudaStreamWaitEvent(0, compute_done_event_, 0));
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &position_vbo_res_, 0));
        CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &color_vbo_res_, 0));

        size_t position_bytes = 0;
        size_t color_bytes = 0;
        float3 *device_positions = nullptr;
        uint *device_colors = nullptr;
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void **>(&device_positions), &position_bytes,
            position_vbo_res_));
        CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void **>(&device_colors), &color_bytes,
            color_vbo_res_));
        copyParticleDataToVBOs(device_buff_.get_buff_list(), nump_,
                               device_positions, device_colors);

        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &position_vbo_res_, 0));
        CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &color_vbo_res_, 0));
    }
    else
    {
        waitForGraphicsCopy();
    }

    glBindBuffer(GL_ARRAY_BUFFER, position_vbo_);
    if (!vbo_resources_registered_)
    {
        glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(float3),
                     host_buff_.get_buff_list().final_position,
                     GL_DYNAMIC_DRAW);
    }
    glVertexPointer(3, GL_FLOAT, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);
    if (!vbo_resources_registered_)
    {
        glBufferData(GL_ARRAY_BUFFER, nump_ * sizeof(uint),
                     host_buff_.get_buff_list().color, GL_DYNAMIC_DRAW);
    }
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glNormal3f(0, 1, 0.001f);
    glColor4f(1, 1, 1, 1);
    glDrawArrays(GL_POINTS, 0, nump_);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisable(GL_POINT_SPRITE_ARB);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_TEXTURE_2D);
    glDepthMask(GL_TRUE);
}

void HybridSystem::drawInfo(GLdouble width, GLdouble height)
{
    float x = 20;
    float y = 20;
    constexpr float line_height = 20;
    std::stringstream text;
    static unsigned int frame_count = 0;
    static float fps = 0.0f;
    static float accumulated_time = 0.0f;

    setOrthographicProjection(width, height);
    glPushMatrix();
    glLoadIdentity();
    glColor3f(1.0f, 1.0f, 1.0f);
    glDisable(GL_LIGHTING);

    frame_timer_.set_end();
    accumulated_time += frame_timer_.get_millisecond();
    frame_timer_.set_start();
    ++frame_count;
    if (accumulated_time > 100.0f)
    {
        fps = 1000 * frame_count / accumulated_time;
        frame_count = 0;
        accumulated_time = 0.0f;
    }

    const auto write_line = [&](const std::string &line)
    {
        text.str("");
        text.clear();
        text << line;
        renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, text);
        y += line_height;
    };

    text << "FPS: " << fps;
    renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, text);
    y += line_height;
    write_line("#particles: " + std::to_string(nump_));

    text.str("");
    text << "frame: " << loop << "  total time: " << total_time_ << "ms";
    renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, text);
    y += line_height;

    if (get_detailed_time_)
    {
        write_line("Simulation detailed time:");
        text.str("");
        text << "    preprocessing: " << pre_time_ << "ms";
        renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, text);
        y += line_height;
        text.str("");
        text << "    density computation: " << density_time_ << "ms";
        renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, text);
        y += line_height;
        text.str("");
        text << "    force computation: " << force_time_ << "ms";
        renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, text);
        y += line_height;
        text.str("");
        text << "    total consumption: " << total_time_ << "ms";
        renderBitmapString(x, y, 0, GLUT_BITMAP_HELVETICA_12, text);
    }

    glPopMatrix();
    restorePerspectiveProjection();
}

void HybridSystem::registerGraphicsResources()
{
    if (vbo_resources_registered_) return;

    cudaError_t position_error = cudaGraphicsGLRegisterBuffer(
        &position_vbo_res_, position_vbo_, cudaGraphicsMapFlagsWriteDiscard);
    cudaError_t color_error = cudaGraphicsGLRegisterBuffer(
        &color_vbo_res_, color_vbo_, cudaGraphicsMapFlagsWriteDiscard);
    if (position_error == cudaSuccess && color_error == cudaSuccess)
    {
        vbo_resources_registered_ = true;
        return;
    }

    if (position_error == cudaSuccess)
        cudaGraphicsUnregisterResource(position_vbo_res_);
    if (color_error == cudaSuccess)
        cudaGraphicsUnregisterResource(color_vbo_res_);
    position_vbo_res_ = nullptr;
    color_vbo_res_ = nullptr;
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

void HybridSystem::createGraphicsCudaResources()
{
    if (headless_mode_) return;
    if (!copy_stream_) CUDA_SAFE_CALL(cudaStreamCreate(&copy_stream_));
    if (!compute_done_event_)
        CUDA_SAFE_CALL(cudaEventCreate(&compute_done_event_));
    if (!copy_done_event_)
        CUDA_SAFE_CALL(cudaEventCreate(&copy_done_event_));
}

void HybridSystem::destroyGraphicsCudaResources()
{
    if (copy_done_event_)
    {
        CUDA_SAFE_CALL(cudaEventDestroy(copy_done_event_));
        copy_done_event_ = nullptr;
    }
    if (compute_done_event_)
    {
        CUDA_SAFE_CALL(cudaEventDestroy(compute_done_event_));
        compute_done_event_ = nullptr;
    }
    if (copy_stream_)
    {
        CUDA_SAFE_CALL(cudaStreamDestroy(copy_stream_));
        copy_stream_ = nullptr;
    }
}

void HybridSystem::waitForGraphicsCopy()
{
    if (copy_stream_)
        CUDA_SAFE_CALL(cudaStreamSynchronize(copy_stream_));
}

} // namespace sph

#endif // !GSPH_HEADLESS

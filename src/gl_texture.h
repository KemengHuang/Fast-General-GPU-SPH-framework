//
// gl_texture.h
// Hybrid_Parallel_SPH
//
// created by ruanjm on 2016/05/01
// Copyright (c) 2016 ruanjm. All rights reserved.
//

#ifndef _GL_TEXTURE_H
#define _GL_TEXTURE_H

#include <vector>
#include <GL\glew.h>
#include "lodepng.h"

#define IMG_RGB			0
#define IMG_RGBA		1
#define IMG_LUM			2

class PNGTexture
{
public:
    ~PNGTexture(){
        if (data_) free(data_);
    }

    bool loadPNG(const char *path){
        std::vector<unsigned char> out;
        unsigned int w, h;

        unsigned error = lodepng::decode(out, w, h, path);
        if (error)
        {
            printf("can not decode %s\n", path);
            return false;
        }

        x_resolution_ = w;
        y_resolution_ = h;
        size_ = 4 * w * h;
        format_ = IMG_RGBA;

        if (data_) free(data_);
        data_ = (unsigned int*)malloc(size_);
        memcpy(data_, &out[0], size_);

        updateTexture();

        return true;
    }

    GLuint get_texture(){
        return texture_;
    }

private:
    void updateTexture(){
        if (texture_) glDeleteTextures(1, &texture_);

        glGenTextures(1, &texture_);
        glBindTexture(GL_TEXTURE_2D, texture_);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        GLenum fmt;
        int size;
        switch (format_) {
        case IMG_RGB:	fmt = GL_RGB; size = 3;			break;
        case IMG_RGBA:	fmt = GL_RGBA; size = 4;		break;
        case IMG_LUM:	fmt = GL_LUMINANCE; size = 1;	break;
        }

        glTexImage2D(GL_TEXTURE_2D, 0, fmt, x_resolution_, y_resolution_, 0, fmt, GL_UNSIGNED_BYTE, data_);
    }

    GLuint texture_ = 0;
    unsigned int x_resolution_;
    unsigned int y_resolution_;
    unsigned int size_;
    unsigned int format_;
    unsigned int *data_ = nullptr;
};

#endif/*_GL_TEXTURE_H*/

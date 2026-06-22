//
// cuda_call_check.h
// cuda texture tester 
//
// created by ruanjm on 12/03/15
// Copyright (c) 2015 ruanjm. All rights reserved.
//

#ifndef _CUDA_CALL_CHECK_H
#define _CUDA_CALL_CHECK_H

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_SAFE_CALL(err)     cuda_safe_call_(err, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK(err)  cuda_kernel_check_(err, __FILE__, __LINE__)

inline void cuda_safe_call_(cudaError err, const char *file_name, const int num_line)
{
    if (cudaSuccess != err)
    {
        exit(0);
        std::cerr << file_name << "[" << num_line << "]: "
            << "CUDA Running API error[" << (int)err << "]: "
            << cudaGetErrorString(err) << std::endl;
    }
}

inline void cuda_kernel_check_(const char *error_msg, const char *file_name, const int num_line)
{
    cudaError_t err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        exit(0);
        std::cerr << file_name << "[" << num_line << "]: "
            << (error_msg == nullptr ? "NONE" : error_msg)
            << "[" << (int)err << "]: "
            << cudaGetErrorString(err) << std::endl;
    }
}

#endif/*_CUDA_CALL_CHECK_H*/
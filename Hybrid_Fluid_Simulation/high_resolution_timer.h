//
// high_resolution_timer.h
// Heterogeneous_SPH 
//
// created by ruanjm on 09/07/15
// Copyright (c) 2015 ruanjm. All rights reserved.
//

#ifndef _HIGH_RESOLUTION_TIMER_H
#define _HIGH_RESOLUTION_TIMER_H

class HighResolutionTimer
{
public:
    virtual void set_start() = 0;
    virtual void set_end() = 0;
    virtual float get_millisecond() = 0;
};

#ifdef WIN32

#include <windows.h>

class HighResolutionTimerForWin : public HighResolutionTimer
{
public:

    HighResolutionTimerForWin(){
        QueryPerformanceFrequency(&freq_);
        start_.QuadPart = 0;
        end_.QuadPart = 0;
    }

    void set_start(){
        QueryPerformanceCounter(&start_);
    }

    void set_end(){
        QueryPerformanceCounter(&end_);
    }

    float get_millisecond(){
        return static_cast<float>((end_.QuadPart - start_.QuadPart) * 1000 / (float)freq_.QuadPart);
    }

private:
    LARGE_INTEGER freq_;
    LARGE_INTEGER start_, end_;
};

#endif // WIN32

#endif/*_HIGH_RESOLUTION_TIMER_H*/
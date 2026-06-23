//
// screenshot.h
// Heterogeneous_SPH
//
// created by ruanjm on 03/10/15
// Copyright (c) 2015 ruanjm. All right reserved.
//

#ifndef _SCREENSHOT_H
#define _SCREENSHOT_H

#include <string>

bool WriteBitmapFile(int width, int height, const std::string &file_name, unsigned char *bitmapData);
void SaveScreenShot(int width, int height, const std::string &file_name);

#endif/*_SCREENSHOT_H*/

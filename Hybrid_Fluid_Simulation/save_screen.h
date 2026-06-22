//
// save_screen.h
// Heterogeneous_SPH
//
// created by ruanjm on 03/10/15
// Copyright (c) 2015 ruanjm. All right reserved.
//

#ifndef _SAVE_SCREEN_H
#define _SAVE_SCREEN_H

#include <GL\GL.h>
#include <windows.h>
#include <string>

#define BITMAP_ID 0x4D42        // the universal bitmap ID  

BITMAPINFOHEADER    bitmapInfoHeader;

bool WriteBitmapFile(int width, int height, const std::string &file_name, unsigned char *bitmapData)
{ 
    BITMAPFILEHEADER bitmapFileHeader;
    memset(&bitmapFileHeader, 0, sizeof(BITMAPFILEHEADER));
    bitmapFileHeader.bfSize = sizeof(BITMAPFILEHEADER);
    bitmapFileHeader.bfType = 0x4d42;   //BM  
    bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

    BITMAPINFOHEADER bitmapInfoHeader;
    memset(&bitmapInfoHeader, 0, sizeof(BITMAPINFOHEADER));
    bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfoHeader.biWidth = width;
    bitmapInfoHeader.biHeight = height;
    bitmapInfoHeader.biPlanes = 1;
    bitmapInfoHeader.biBitCount = 24;
    bitmapInfoHeader.biCompression = BI_RGB;
    bitmapInfoHeader.biSizeImage = width * abs(height) * 3;

    //////////////////////////////////////////////////////////////////////////  
    FILE * filePtr;        
    unsigned char tempRGB;  
    int imageIdx;

    for (imageIdx = 0; imageIdx < (int)bitmapInfoHeader.biSizeImage; imageIdx += 3)
    {
        tempRGB = bitmapData[imageIdx];
        bitmapData[imageIdx] = bitmapData[imageIdx + 2];
        bitmapData[imageIdx + 2] = tempRGB;
    }

    filePtr = fopen(file_name.c_str(), "wb");
    if (NULL == filePtr)
    {
        return false;
    }

    fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

    fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

    fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);

    fclose(filePtr);
    return true;
}

void SaveScreenShot(int width, int height, const std::string &file_name)
{
    int data_len = height * width * 3;      // bytes
    void *screen_data = malloc(data_len);
    memset(screen_data, 0, data_len);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screen_data);

    WriteBitmapFile(width, height, file_name + ".bmp", (unsigned char*)screen_data);

    free(screen_data);
}


#endif/*_SAVE_SCREEN_H*/
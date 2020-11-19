#ifndef _SD_READER_H
#define _SD_READER_H

#include <stdio.h>
#include <math.h>
#include "nn_feature_provider.h"

class SDDataSet {
public:
    void Append(char* message);    
    void Read(uint32_t file_id);
private:
    
    

    char buffer[FEATURE_WIDTH];
};

void SDDataSet::Append(char* message) {
    FILE* current_dest = 0;

    if (!current_dest) {
        current_dest = fopen("/mnt/sd0a/ACTIVITY.LOG", "a");
    }

    fprintf(current_dest, "[%d] %s\n", 0, message);
    fclose(current_dest);
    current_dest = 0;
}

void SDDataSet::Read(uint32_t file_id) {

}

#endif
#ifndef _NN_FEATURE_PROVIDER_SDCARD_H
#define _NN_FEATURE_PROVIDER_SDCARD_H

#ifndef CONFIG_EXAMPLES_TF_HELLOWORLD_FEATURE_COUNT
#define CONFIG_EXAMPLES_TF_HELLOWORLD_FEATURE_COUNT 4000
#endif

#include "nn_feature_provider.h"
#include "stdio.h"
#include "math.h"

NeuralNetworkFeatureProvider::NeuralNetworkFeatureProvider() {
    //outputs = new int8_t[33025];
    input_number = 0;    
    input_count = CONFIG_EXAMPLES_TF_HELLOWORLD_FEATURE_COUNT;  
    printf("Size of output {%ld}", sizeof(outputs));  
    //exit(0);
}

int8_t* NeuralNetworkFeatureProvider::get_feature() {    
    FILE* current_source;

    char source_filename[FILENAME_MAX];
    uint16_t crop_id = (uint16_t) floor(input_number/10.0);
    
    sprintf(source_filename, "./%s/%04d_%03d.RAW",FEATURE_PATH, input_number, crop_id);

    current_source = fopen(source_filename, "r");
    if(current_source == NULL) {
        printf("Failed to source from %s (%d)\n", source_filename, input_number);
        exit(1);
    }

    printf("Source: %s\n", source_filename);

    // Force read from first position
    fseek(current_source, 0, SEEK_SET);
    feature_size = fread(outputs, 1, g_feature_size, current_source);
    fclose(current_source);
       
    input_number++;

    if (feature_size == 0) {    
        printf("Failed to read file\n");
        exit(2);    
    }

    return outputs;
}

uint32_t NeuralNetworkFeatureProvider::get_feature_length() {
    return feature_size;
}

#endif

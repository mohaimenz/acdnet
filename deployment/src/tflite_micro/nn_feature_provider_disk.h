#ifndef _NN_FEATURE_PROVIDER_SDCARD_H
#define _NN_FEATURE_PROVIDER_SDCARD_H

#ifndef CONFIG_EXAMPLES_TFLITE_MICRO_FEATURE_COUNT
#define CONFIG_EXAMPLES_TFLITE_MICRO_FEATURE_COUNT 4000
#endif

#include "nn_feature_provider.h"
#include "stdio.h"
#include "math.h"

NeuralNetworkFeatureProvider::NeuralNetworkFeatureProvider() {
    input_number = 0;
    input_count = CONFIG_EXAMPLES_TFLITE_MICRO_FEATURE_COUNT;
}

int8_t* NeuralNetworkFeatureProvider::get_feature() {    
    FILE* current_source;

    char source_filename[FILENAME_MAX];
    uint16_t crop_id = (uint16_t) floor(input_number/10.0);
    
    sprintf(source_filename, "%s/%04d_%03d.RAW", FEATURE_PATH, input_number, crop_id);

    current_source = fopen(source_filename, "r");
    if(current_source == NULL) {
        printf("Failed to source from %s (%d)\n", source_filename, input_number);
        exit(1);
    }

    printf("Source: %s\n", source_filename);

    // Force read from first position
    fseek(current_source, 0, SEEK_SET);
    fread(outputs, FEATURE_WIDTH, 1, current_source);
    fclose(current_source);
       
    input_number++;

    return outputs;
}

uint32_t NeuralNetworkFeatureProvider::get_feature_length() {
    return FEATURE_WIDTH;
}

#endif
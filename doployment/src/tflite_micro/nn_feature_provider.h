#ifndef _NN_FEATURE_PROVIDE_H
#define _NN_FEATURE_PROVIDE_H

#include <stdint.h>
#include "model.h"

class NeuralNetworkFeatureProvider {
protected:
    int8_t outputs[FEATURE_WIDTH];

    uint32_t output_length = 0;
    int16_t* inputs = 0;
    uint32_t input_length = 0;

    uint32_t input_count;
    uint32_t input_number;
public:
    NeuralNetworkFeatureProvider();
    int8_t* get_feature();
    uint32_t get_feature_length();    
    uint32_t get_feature_number();
    uint32_t get_feature_count();
};

uint32_t NeuralNetworkFeatureProvider::get_feature_number() {
    return input_number;
}

uint32_t NeuralNetworkFeatureProvider::get_feature_count(){
    return input_count;
}

#endif
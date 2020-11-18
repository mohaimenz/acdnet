#ifndef _NN_FEATURE_PROVIDER_SAMPLE_H
#define _NN_FEATURE_PROVIDER_SAMPLE_H

#include "nn_feature_provider.h"
#include "audio_samples.h"

NeuralNetworkFeatureProvider::NeuralNetworkFeatureProvider(){
    input_number = 0;
    input_count = 1;
}

int8_t* NeuralNetworkFeatureProvider::get_feature() {
    return (int8_t*) audio_samples;
}

uint32_t NeuralNetworkFeatureProvider::get_feature_length() {
    return audio_samples_len;
}

int32_t NeuralNetworkFeatureProvider::get_feature_number() {
    return input_number;
}


#endif
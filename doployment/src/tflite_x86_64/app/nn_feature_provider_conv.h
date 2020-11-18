#ifndef _NN_FEATURE_PROVIDER_CONV_H
#define _NN_FEATURE_PROVIDER_CONV_H

#include <cmath>
#include "nn_feature_provider.h"
#include "audio_samples.h"

NeuralNetworkFeatureProvider::NeuralNetworkFeatureProvider() {
    input_number = 0;
    input_count = 1;
}

// Each signed 16-bit int audio samples must be provided into the 
// feature provider as an array of unsigned chars.
// The feature provider converts the audio waveform
// into an input feature for the Neural Network Manager
int8_t* NeuralNetworkFeatureProvider::get_feature() {  
    inputs = (signed short*) audio_samples;
    input_length = audio_samples_len / 2;    

    signed short max = 0;
    signed short min_threshold = 2000;
    signed long min_threshold_pos = -1;

    for (unsigned long i = 0; i < this->input_length; i++) {
        short abs_value = (inputs[i] ? inputs[i] > 0 : -inputs[i]);

        if (max < abs_value) {
            max = abs_value;            
        }

        if (min_threshold_pos < 0 && abs_value > min_threshold) {
            min_threshold_pos = i;
        }
    }

    for (unsigned long i = 0; i < input_length && i < g_feature_size; i++) {
        unsigned long input_sample_pos = min_threshold_pos + i;
        if (input_sample_pos < input_length){

            float sample = inputs[input_sample_pos] * 127.0 / max;

            if (sample > 127.0) {
                sample = 127.0;
            } else if (sample < -127) {
                // symmetric
                sample = -127.0;
            }

            outputs[i] = (int8_t) floor(sample);
            output_length = i;
        }
    }

    return outputs;
}

uint32_t NeuralNetworkFeatureProvider::get_feature_length() {
    return output_length;
}

#endif
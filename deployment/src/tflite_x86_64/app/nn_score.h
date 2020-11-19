#ifndef _NN_SCORES_H
#define _NN_SCORES_H

#include "model.h"
#include "stdint.h"
#include "stdio.h"

#define CROP_COUNT 10
#define OUTPUT_WIDTH 50

class NeuralNetworkScores {
private:
    int8_t* _output;
    uint8_t _output_count;

    uint8_t _expected_scores[CROP_COUNT];
    int32_t _actual_scores[OUTPUT_WIDTH];
    uint8_t  _crop_index = 0;
    uint32_t _result_index = 0;
    uint32_t _file_offset = 0;
    uint32_t _hit_count = 0;    

public:
    NeuralNetworkScores(int8_t* output, uint8_t output_count);

    bool LoadScores(uint32_t scores_offset, uint32_t score_count);    
    float GetAccuracy();
    void AddOutput();
};

#endif

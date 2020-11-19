#include "nn_score.h"
#include "stdio.h"
#include "stdlib.h"

NeuralNetworkScores::NeuralNetworkScores(int8_t* output, uint8_t output_count) {
    _output = output;
    _output_count = output_count;
}

/*Loads a set number of target scores from a scores.bin file*/
bool NeuralNetworkScores::LoadScores(uint32_t scores_offset, uint32_t score_count) 
{
    FILE* current_source;
    char source_filename[FILENAME_MAX];
        
    sprintf(source_filename, "./%s/scores.bin",FEATURE_PATH);

    current_source = fopen(source_filename, "r");
    if(current_source == NULL) {
        printf("Failed to source from %s\n", source_filename);
        
        return false;
    }

    printf("Source: %s\n", source_filename);

    // Force read from first position
    fseek(current_source, scores_offset, SEEK_SET);
    uint32_t scores_read = fread(_expected_scores, 1, score_count, current_source);

    fclose(current_source);
           
    if (scores_read == 0) {    
        printf("Failed to read file\n");

        return false;
    }

    return true;
}

void NeuralNetworkScores::AddOutput() 
{
    if (_crop_index == 0) {
        for (uint8_t i = 0; i < OUTPUT_WIDTH; i++) {
            _actual_scores[i] = _output[i];
        }
    } else {
        for (uint8_t i = 0; i < OUTPUT_WIDTH; i++) {
            _actual_scores[i] += _output[i];
        }
    }
    
    _crop_index ++;
    _result_index ++;

    if (_crop_index == CROP_COUNT) {
        printf("Reading scores %d\n",_file_offset);
        if (!LoadScores(_file_offset, CROP_COUNT)) return;

        printf("Evaluating scores\n");    
        int32_t argmax = INT32_MIN;
        uint8_t argmax_index = -1;

        printf("[");
        for (uint8_t i = 0; i < OUTPUT_WIDTH; i++) {
            if (_actual_scores[i] > argmax) {
                argmax = _actual_scores[i];
                argmax_index = i;
            }
            printf("%d, ", _actual_scores[i]);
        }
        printf("]\n");

        if (argmax_index == _expected_scores[0]) {
            _hit_count += 1;        
        }

        printf("Result Expected %d Actual %d\n", _expected_scores[0], argmax_index);
        printf("Accuracy: %0.4f\n", GetAccuracy());

        _crop_index = 0;
        _file_offset += CROP_COUNT;       
    }
}
    
float NeuralNetworkScores::GetAccuracy() {
    return _hit_count * CROP_COUNT * 1.0f / _result_index;
}
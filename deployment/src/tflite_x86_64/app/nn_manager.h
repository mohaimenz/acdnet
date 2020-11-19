#ifndef _NN_MANAGER_H
#define _NN_MANAGER_H

#include "model.h"

// #include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"

#include "tensorflow/lite/micro/micro_optional_debug_tools.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {

extern tflite::ErrorReporter* error_reporter;

// Create an area of memory to use for input, output, and intermediate arrays.
// Minimum arena size, at the time of writing. After allocating tensors
// you can retrieve this value by invoking interpreter.arena_used_bytes().
const int kTensorArenaSize = 1024 * 800;

// memory alignment is important when using TFLite
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

}  // namespace

// Wraps the handling of TFLite management into
// a small number of easy calls
class NeuralNetworkManager {
protected:

    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    tflite::MicroProfiler* profiler = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;    
    int inference_count;
    uint32_t inference_time;
    int32_t input_number = 0;

public:
    NeuralNetworkManager();
    
    TfLiteTensor* get_input();
    void set_input(uint32_t input_number, int8_t* buffer, uint32_t buffer_size);
    TfLiteTensor* get_output();
    
    void run_inference();    
    int get_inference_time();
};

#endif

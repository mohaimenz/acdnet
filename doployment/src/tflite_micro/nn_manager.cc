#include <nuttx/time.h>
#include "nn_manager.h"
#include <stdio.h>

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroOpResolver* resolver = nullptr; 
}  // namespace

NeuralNetworkManager::NeuralNetworkManager() {
    // Set up logging. Google style is to avoid globals or statics because of
    // lifetime uncertainty, but since this has a trivial destructor it's okay.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(g_model_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    printf("MODEL:         %s\n", g_model_name);
    printf("MODEL SIZE:    %d\n", g_model_tflite_len);
    printf("FEATURE_WIDTH: %d\n", g_feature_size);
    printf("FEATURE_PATH:  %s\n", FEATURE_PATH);

    if (FEATURE_WIDTH != g_feature_size) {
        printf("Error: Expected FEATURE_WIDTH=%d, got %d", g_feature_size, FEATURE_WIDTH);
        exit(-1);
    }

    printf("Model structure\n");
    tflite::PrintModelData(model, &micro_error_reporter);
    printf("Building resolver\n");

    // Using either a _static_ AllOpsResolver or MicroMutableOpResolver
    // create and register all supported operations as a resolver.    
    // tflite::AllOpsResolver resolver;
    static tflite::MicroMutableOpResolver<8> static_resolver;
    
    static_resolver.AddConv2D();    
    static_resolver.AddDepthwiseConv2D();
    static_resolver.AddFullyConnected();
    static_resolver.AddReshape();
    static_resolver.AddSoftmax();
    static_resolver.AddAveragePool2D();
    static_resolver.AddMaxPool2D();
    static_resolver.AddTranspose();

    resolver = &static_resolver;
    printf("Resolver done\n");
   
    // The Interpreter manages the subgraph and links the nodes and
    // registered operations together in the resolver as an 
    // ordered list, which is then executed in sequence.
    // The interpreter also uses the Greedy memory manager, 
    // to manage the buffers reserved for node calculations
    // along the way to ensure only active 'sequential'
    // memory buffers, effectively releasing unused memory.  

    static tflite::MicroInterpreter static_interpreter(
      model, *resolver, tensor_arena, kTensorArenaSize, error_reporter);

    interpreter = &static_interpreter;
    printf("Interpreter done\n");    

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }
    printf("Allocated tensors\n");

    // We can inspect the final arena size, after the interpreter
    // has analysed the model architecture.  This value should be 
    // the minimum value used when declaring kTensorArenaSize and tensor_arena
    // Note that it is best practice to increase the size where possible
    // to protect against future changes in TFLite API
    printf("Arena used %u bytes\n", interpreter->arena_used_bytes());

    // We can inspect the Interpreter to see how the 
    // buffers are using the memory or the sequence of execution
    printf("Interpreter State\n");
    tflite::PrintInterpreterState(interpreter);
    
    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    printf("Dimensions\n");
    // We can view the input dimensions of the model
    printf("input->dims->size:    %u\n", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        printf("input->dims->data[%d]: %u\n", i, input->dims->data[i]);
    }
    
    // We can observe the expected data type of the input layer
    if (input->type == kTfLiteInt8) printf("int8_t input\n");
    else if (input->type == kTfLiteInt16) printf("int16_t input\n");
    else if (input->type == kTfLiteUInt8) printf("uint8_t input\n");
    else if (input->type == kTfLiteFloat16) printf("float16 input\n");
    else if (input->type == kTfLiteFloat32) printf("float32 input\n");    
    else printf("Unknown input type\n");

    printf("output->dims->size:    %u\n", output->dims->size);
    for (int i = 0; i < output->dims->size; i++) {
        printf("output->dims->data[%d]: %u\n", i, output->dims->data[i]);
    }

    // We can observe the expected data type of the input layer
    if (output->type == kTfLiteInt8) printf("int8_t output\n");
    else if (output->type == kTfLiteInt16) printf("int16_t output\n");
    else if (output->type == kTfLiteUInt8) printf("uint8_t output\n");
    else if (output->type == kTfLiteFloat16) printf("float16 output\n");
    else if (output->type == kTfLiteFloat32) printf("float32 output\n");    
    else printf("Unknown output type\n");

    // Keep track of how many inferences we have performed.
    inference_count = 0;
}

TfLiteTensor* NeuralNetworkManager::get_input() {
    return input;
}

TfLiteTensor* NeuralNetworkManager::get_output() {
    return output;
}

void NeuralNetworkManager::set_input(uint32_t input_number, int8_t* buffer, uint32_t buffer_size) {
    // Push the data from the input feature
    // into the inputs of the first NN node
    this->_input_number = input_number;
    for (uint32_t i = 0; i < buffer_size; i++) {
        input->data.int8[i] = buffer[i];
    }    
}

bool logResults(uint32_t feature_number, TfLiteTensor* output, uint32_t duration_ms) {
    size_t max_i = 0;
    int8_t max_v = -128;

    size_t output_dimensions = 1;

    // printf("output->dims->size:    %u\n", output->dims->size);
    for (int i = 0; i < output->dims->size; i++) {
        // printf("output->dims->data[%d]: %u\n", i, output->dims->data[i]);
        output_dimensions *= output->dims->data[i];
    }

    FILE* log = fopen("/mnt/sd0/activity.log", "a");    
    if (log == NULL) {
        // printf("Failed to open log\n");
        exit(1);
    }

    fprintf(log, "MODEL: %s, ", g_model_name);

    for (size_t i = 0; i < output_dimensions; i++) {
        fprintf(log, "%d, ", output->data.int8[i]);

        if (output->data.int8[i] > max_v) {
            max_v = output->data.int8[i];
            max_i = i;
        }
    }
    printf("No, %d, ArgMax, %d, Value, %d, Duration, %d\n", feature_number, max_i, max_v, duration_ms);    
    fprintf(log, "No, %d, ArgMax, %d, Value, %d, Duration, %d\n", feature_number, max_i, max_v, duration_ms);    
    fflush(log);
    fclose(log);

    return true;
}

void NeuralNetworkManager::run_inference() {
    clock_t t;
    t = clock();
    TfLiteStatus status = interpreter->Invoke();
    inference_time = clock() - t;
    
    switch (status)
    {
        case kTfLiteOk:
        {
            printf("Inference OK\n");            

            logResults(this->_input_number, output, inference_time);

            break;
        }
        case kTfLiteError:
        {
            printf("Inference Error\n");
            break;
        }
        case kTfLiteDelegateError:
        {
            printf("Inference Delegate Error\n");
            break;
        }
        default:
            break;
    }
}

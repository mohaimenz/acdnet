
#include <stdio.h>

#include "nn_manager.h"
#include "nn_feature_provider_disk.h"
#include "nn_score.h"

namespace {
  NeuralNetworkManager* manager = nullptr;
  NeuralNetworkFeatureProvider* featureProvider = nullptr;
  NeuralNetworkScores* scores = nullptr;
}

extern "C" {

int main() {

  // The TFLite management is abstracted from 
  // the application flow, to improve separation of concerns

  manager = new NeuralNetworkManager();
  printf("Created Manager\n");

  featureProvider = new NeuralNetworkFeatureProvider();  
  printf("Created Feature Provider\n");

  // Fetch the output tensor
  TfLiteTensor* output = manager->get_output(); 
  scores = new  NeuralNetworkScores(output->data.int8, OUTPUT_WIDTH);

  while (featureProvider->get_feature_number() < 
    featureProvider->get_feature_count()) {

    // Get data from the Feature Provider
    uint32_t feature_number = featureProvider->get_feature_number();
    int8_t* feature = featureProvider->get_feature();
    uint32_t feature_length = featureProvider->get_feature_length();

    // Populate the TFLite input tensor    
    manager->set_input(feature_number, feature, feature_length);
    printf("Feature %d length: %u\n", feature_number, feature_length);
    
    // Commence inference
    manager->run_inference();

    // Accumlate scores to calculate accuracy
    scores->AddOutput();
  }

  float accuracy = scores->GetAccuracy();
  printf("Final Accuracy: %.04f\n", accuracy);

  return 0;
}

} // EXTERN C
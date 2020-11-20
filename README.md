# ACDNet

## A. Training and Compressing ACDNet

#### A.1 Prerequisits
1. Create python development environment
2. Install `torch 1.3` or higher
2. Install `wavio 0.0.4` python library
3. Install `wget` for downloading ESC-50 over HTTP
4. Install `FFmpeg` for downsampling and upsampling audio recordings

##### Note
* ACDNet is developed and tested in macOS environment. The forthcoming sections assumes that the above libraries/softwares are now installed.*

#### A.2 Dataset preparation
1. Download/clone the repository.
2. Go to the root of ACDNet directory using the terminal.
3. To download and process ESC-50 dataset, run: ```python common/prepare_dataset.py```
4. Prepare the validation data, run: ```python common/val_generator```

*All the required data of ESC-50 for processing `44.1kHz` and `20kHz` are now ready at `datasets/esc50` directory*

#### A.3 Training ACDNet (PyTorch)
*There are pretrained models provided inside `torch/resources/pretrained_models` directory thta can be used instead of training a new model. The model names are self explanatory. There are 5 pretrained ACDNet models validated on 5-folds (5-fold cross validation), 95% Weight pruned and retrained ACDNet model for hybrid pruning, ACDNet20 pruned and fine-tuned (not trained) and ACDNet-20 trained model*

However, to conduct the training of a brand new ACDNet, run: ```python torch\trainer.py```
##### Notes
* Follow on-screen self-explanatory steps
* To train a brand new ACDNet, please select `training from scratch` option and keep the model path `empty` in the next step. 
* The trained models will be saved at `torch/trained_models directory`
* The models will have names `YourGivenName_foldNo` on which it was validated. 
* For five fold cross validation, there will be 5 models named accordingly

#### A.4 Testing ACDNet (PyTorch)
1. To test a trained model, run this command: ```python torch/tester.py```
2. Follow the on-screen self explanatory steps

##### Notes
* A model should always be tested on the fold on which it was validated to reproduce the result. 
* For example, if a model was validated on fold-1, it will reproduce the validation accuracy on that fold. For all other folds (fold 2-5), it will produce approximately 100% prediction accuracy since it was trained on those folds.

#### A.5 Pruning ACDNet (PyTorch)
1. To conduct pruning run: ```python torch/pruning.py```
2. Follow the on-screen self explanatory steps

##### Notes
* To conduct `hybrid` pruning on ACDNet, please run `weight pruning` on ACDNet first and then apply `hybrid pruning` on the weight pruned model. 
* The on-screen steps are easy enough to achive this goal.
* The pruned models will be stored inside `torch/pruned_models` directory

#### A.6 Re-Training ACDNet (PyTorch)
To conduct retraining a pruned model, follow these steps:
1. Run: ```python torch\trainer.py```
2. Choose the training options
3. Provide pruned model path
4. Provide fold number for the model to be validated.

#### A.7 Quantizing ACDNet (PyTorch)
1. For 8-bit post training quantization, run: ```python torch\quanization.py```
2. Provide model path.
3. Provide the fold on which the model was validated.


### B. Rebuilding ACDNet20 in Tensorflow (TF)
For deployment purpose we use Tensorflow and Tensorflow Lite.

#### B.1 Prerequisist:
1. Install tensorflow 2.2.0

#### B.2 Training ACDNet-20
*There is a pretrained ACDNet-20 model inside `tf/resources/pretrained_models` directory. This can be used instead.*

To rebuild ACDNet-20 from scratch in TF, follow these steps:
1. Run: ```python tf/trainer.py```
2. Provide the ACDNet-20 PyTorch model path for it to retrieve the configuration of ACDNet-20 and build an equivalent TF model
3. Follow the on-screen steps for finish the process.

##### Notes
* For this, any fold can be chosen for the model to be validated as it is going to be trained as a brand new ACDNet-20 model in TF.
* The trained model will be saved inside `tf/trained_models` directory
* Once the model training is finished, we are ready to go to the deployment step.

### C. ACDNet-20 on MCU
The model will now be converted to TF Lite, quantized to 8-bit and then using TF Lite Micro, we will push the model to MCU for inferencing.
**Please follow the instructions provided in README.md file inside `deployment` directory**


# 1. Deployment pipeline

## 1.1. Background

### 1.1.1. Background

This project supports:

- the python evaluation of Keras ACDNET20 models
- the conversion of ACDNET20 models to TFLite 
- the deployment of applications to evaluate the accuracy of the models onto both Sony Spresense and x86_64 Ubuntu 20.04

### 1.1.2. Basic steps

Typical use of this project follows the these steps:

1. Pull the source repository (**acdnet**)
2. Review dependencies [(1.2)](#deps)
2. Setup MicroPipeline [(1.3)](#setup)
3. Convert models to TFLite [(1.4)](#convert)
4. Project model deployment
   1. Setup Data sources [(1.5)](#gendata)
   2. [TFLITE_x86_64](./src/tflite_x86_64/README.md)
      1. Copy `model.cc` to target project directory
      2. Configure project
      3. Build project
      4. Execute project
      5. Inspect activity.log file on microSD card to review inference results in detail
   3. [TFLITE_MICRO](./src/tflite_micro/README.md)
      1. Copy `model.cc` to target project directory
      2. Copy `raw` feature data onto microSD card, then insert card into Spresense
      3. Build project
      4. Upload new firmware to Sony Spresense
      5. Connect and execute project
      6. Inspect shell results
      7. Inspect activity.log file on microSD card to review inference results in detail

## 1.2. Dependencies <a name="deps"></a>

This repository has the following git submodule dependencies

| Dependency     | URL                                                   | Branch           | Path                                              | Comment                                              |
| -------------- | ----------------------------------------------------- | ---------------- | ------------------------------------------------- | ---------------------------------------------------- |
| TensorFlow     | https://github.com/tensorflow/tensorflow              | v2.3.1_transpose | ext/tensorflow                                    | Modified to support Transpose TFLite micro operation |
| Flatbuffers    | https://github.com/google/flatbuffers.git             | v1.12.0          | ext/flatbuffers                                   |                                                      |
| Spresense      | https://github.com/spresense/spresense.git            | v2.0.1           | ext/spresense                                     | Supports TFLite micro                                |
| NuttX          | https://github.com/spresense/spresense-nuttx.git      |                  | ext/spresense/nuttx                               |                                                      |
| Nuttx Apps     | https://github.com/spresense/spresense-nuttx-apps.git |                  | ext/spresense/sdk/apps                            |                                                      |
| nnabla Runtime | https://github.com/sony/nnabla-c-runtime              |                  | ext/spresense/externals/nnablart/nnabla-c-runtime |                                                      |

Additional python requirements, which are installed automatically, include:
- h5py==2.10.0
- numpy==1.18.5
- requests==2.24.0
- six==1.15.0
- tensorflow==2.3.1
- tensorflow-estimator==2.3.0

During the build, further dependencies including source audio may be required.

## 1.3. Setup  <a name="setup"></a>

### 1.3.1. Automatic Setup (recommended)

Executing the setup script will perform all necessary preparations outlined below
- creating a python virtual environment
- install required python packages
- downloading and building the dependencies
- installing TensorFlow, spresense_sdk, Nuttx, symlinks and additional libraries

```bash
./setup.sh
```

### 1.3.2. Manual Setup

Recommendation is to use a virtual environment as there are several python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 1.3.2.1. Tensorflow setup

Download the TFLite build dependencies

```bash
(cd ext/tensorflow && ./tensorflow/lite/tools/make/download_dependencies.sh)
```

Apply each Tensorflow patch

```
git apply ext/patches/tensorflow/{.patch} --directory ext/tensorflow
```

Build the tflite library for the local OS

```bash
(cd ext/tensorflow && ./tensorflow/lite/tools/make/build_lib.sh)
```

Create the tflite micro library for the local OS

```bash
(cd ext/tensorflow && make -f tensorflow/lite/micro/tools/make/Makefile)
```

Create the tflite micro library for the micro OS

```
(cd ext/tensorflow && make -f tensorflow/lite/micro/tools/make/Makefile TAGS=cmsis-nn third_party_downloads;)
```



#### 1.3.2.2. Spresense setup

The following steps can be used to manually install the Spresense SDK, and apply the necessary patches

```
sudo usermod -a -G dialout \$USER
git clone --branch=v2.0.1 --depth=1 --recurse-submodules https://github.com/sonydevworld/spresense.git spresense
git clone --depth=1 https://bitbucket.org/acassis/libcxx  ext/libcxx
(cd ext/libcxx; ./install.sh ext/spresense/nuttx)
cp ext/patches/libcxx/optional.cxx ext/spresense/nuttx/libs/libxx/libcxx
```

Apply each Spresense patch

```
git apply ext/patches/spresense/{filename}.patch --directory=ext/spresense
```


## 1.4. Conversion of models <a name="convert"></a>

Execute the python script to automatically convert all models

```bash
./convert_model.py {path/to/model.h5} {fold}
```

*Where*
- model.h5 is a valid pretrained Keras model
- [Optional] fold is a number in range [1 - 5]

*Results*
- TFLite are written automatically to `src/models/{model_file}_{dtype}_{set_name}.tflite` 
- CC are written automatically to `src/models/{model_file}_{dtype}_{set_name}.cc`
- Accuracy results are written automatically to `results/result_{yymmdd}_{hhmm}.npy`

## 1.5. Data sources  <a name="gendata"></a>

Evaluation data used by the model is supplied in the form of compressed Numpy npz files which contain:

- **'x'** the audio sample features
- **'y'** the expected classification output

In the context of this repository there are two sorts of npz files

- 'test' data, used to evaluate accuracy
- 'representation' data used to supply the TFLite conversion with a representative data set

### 1.5.1. Generate Raw Source Data

To create the prerequisite RAW files, containing the audio features needed for both TFLITE_X86_64 and TFLITE_MICRO, it is necessary to run a script.  The script, below, populates the `data/raw` folder with the generated files.

```bash
generate_raw_data.py
```
When copying data to the microSD card, it is recommended to copy the selected dataset of *.RAW files (4000 in total) and scores.bin (1 file) from the generated data path into a `data` folder on the microSD.

## 1.6. Acknowledgements

This project has relied heavily on the following sources:

- TensorFlow Lite [https://www.tensorflow.org/lite/microcontrollers/library]
- PyTorch [https://pytorch.org/docs/stable/quantization.html]
- Sony Spresense [https://developer.sony.com/develop/spresense/docs/home_en.html]
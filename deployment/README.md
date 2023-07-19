# 1. Deployment pipeline

## 1.1. Background

### 1.1.1. Background

This project supports:

- the python evaluation of Keras ACDNET20 models
- the conversion of ACDNET20 models to TFLite 
- the deployment of applications to evaluate the accuracy of the models onto both Sony Spresense and x86_64 Ubuntu 20.04

### 1.1.2. Basic steps

Easy to use demo scripts have been provided for simplicity:

Setup installs the default dependencies
Demo builds using a provided pretrained acdnet model

```bash
./setup.sh
./demo.sh
```

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

1. Create python development environment (Recommend 3.8+)
2. Run the dataset generation scripts from main folder
   ```python common/prepare_dataset.py```
   ```python common/val_generator.py```
3. Install ```xxd``` to support tflite conversion
4. Add user to `dialout` group, then logout and login to activate 
   ```sudo usermod -a -G dialout $USER```
5. Run `setup.sh` in `deployment` folder 
6. Run `demo.sh` in `deployment` folder 

*Notes*: 
- ACDNet is developed and tested in macOS environment. 
- ACDNET deployment is developed and tested on
  - Sony Spresense (US)
  - Ubuntu 20.04 x86_64

- TFLITE_X86_64 requires an Ubuntu 20.04 x86_64 environment
- TFLITE_MICRO requires a Sony Spresense with extension board and microSD card

This repository has the following git submodule dependencies

| Dependency     | URL                                                   | Branch           | Path                                              | Comment                                              |
| -------------- | ----------------------------------------------------- | ---------------- | ------------------------------------------------- | ---------------------------------------------------- |
| Flatbuffers    | https://github.com/google/flatbuffers.git             | v1.12.0          | ext/flatbuffers                                   |                                                      |
| Spresense      | https://github.com/spresense/spresense.git            | v2.0.1           | ext/spresense                                     | Supports TFLite micro                                |
| NuttX          | https://github.com/spresense/spresense-nuttx.git      |                  | ext/spresense/nuttx                               |                                                      |
| Nuttx Apps     | https://github.com/spresense/spresense-nuttx-apps.git |                  | ext/spresense/sdk/apps                            |                                                      |
| nnabla Runtime | https://github.com/sony/nnabla-c-runtime              |                  | ext/spresense/externals/nnablart/nnabla-c-runtime |                                                      |

Additional python requirements, which are installed automatically, include:
- h5py
- numpy
- requests
- six
- tabulate
- tensorflow
- tensorflow-estimator

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

#### 1.3.2.1. TensorFlow setup

Update:
When installing the latest version of `tensorflow`, the `tflite` library and micro library get automatically installed.


#### 1.3.2.2. Spresense setup

The Spresense SDK is required by the TFLITE_MICRO project

The following steps can be used to manually install the Spresense SDK, and apply the necessary patches

Install Build tools

```bash
wget https://raw.githubusercontent.com/sonydevworld/spresense/master/install-tools.sh
chmod 775 install-tools.sh
./install-tools.sh
rm install-tools.sh
```

Install Spresense SDK

```bash
sudo usermod -a -G dialout \$USER
git clone --branch=v2.0.1 --depth=1 --recurse-submodules https://github.com/sonydevworld/spresense.git spresense
git clone --depth=1 https://bitbucket.org/acassis/libcxx  ext/libcxx
(cd ext/libcxx; ./install.sh ext/spresense/nuttx)
cp ext/patches/libcxx/optional.cxx ext/spresense/nuttx/libs/libxx/libcxx
```

Apply each Spresense patch

```bash
(cd ext/spresense && git apply --reject --whitespace=fix ext/patches/spresense/{filename}.patch)
```

Apply each Nuttx patch

```bash
(cd ext/spresense && git apply --reject --whitespace=fix ext/patches/spresense/{filename}.patch)
```

#### Install Flatbuffers

The FlatBuffers library is required by the TFLITE_x86_64 project

```bash
cd ext
git clone --depth=1 --branch v1.12.0 https://github.com/google/flatbuffers.git
```

#### 1.3.2.3. Install Symbolic Links 

The project uses symbolic links to tidy up the folder structure, and avoid duplicate copies of external repositories

```bash
cd ext
ln -s ../src/tflite_micro spresense/examples/tflite_micro
ln -s ../src/tflite_micro_config spresense/sdk/configs/examples/tflite_micro
ln -s tensorflow spresense/examples/tflite_micro/tensorflow
```


## 1.4. Conversion of models <a name="convert"></a>

Execute the python script to automatically convert all models:

```python <path/to/convert_model.py> <path/to/h5_version_of_the_TF_model> --fold <fold_number>```


*Where*
- model.h5 is a valid pretrained Keras model
- fold is a number in range [1 - 5] on which the model was validated

*Results*
- TFLite are written automatically to `src/models/{model_file}_{dataset}_fold{fold}.tflite` 
- CC are written automatically to `src/models/{model_file}_{dtype}_fold{fold}.cc`
- Accuracy results are written automatically to `results/result_{yymmdd}_{hhmm}.npy`

## 1.5. Data sources

Evaluation data used by the model is supplied in the form of compressed Numpy npz files which contain:

- **'x'** the audio sample features
- **'y'** the expected classification output

In the context of this repository there are two sorts of npz files

- 'test' data, used to evaluate accuracy
- 'representation' data used to supply the TFLite conversion with a representative data set
*All the required data are already there inside the `dataset/esc50` directory. No action required at this moment.*

### 1.5.1. Generate Raw Source Data

To create the prerequisite RAW files, containing the audio features needed for both TFLITE_X86_64 and TFLITE_MICRO, it is necessary to run a script.  The script, below, populates the `data/` folder with the generated files.

```bash
generate_raw_data.py {model_path} --fold {fold_number}
```

When copying data to the microSD card, it is recommended to copy the selected dataset of *.RAW files (4000 in total) and scores.bin (1 file) from the generated data path into a `data` folder on the microSD.

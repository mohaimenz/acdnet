# 1. TFLITE_X86_64

## 1.1. TFLITE_X86_64 TFLite project for Linux Desktop

This project aims to quickly assess the model, using a desktop TFLite micro implementation, as the Spresense evaluation is a ~6-8 hr activity per model.

When the application runs, it logs to stdout on the command line, and generates `activity.log` which contains the accuracy evaluation of the model.  
Furthermore accuracy is calculated immediately as the inference completes, and the result of the accuracy can be observed in the stdout response.

_Typical execution time is about 10 minutes._

### 1.1.1. Building

Assumes that:
- the prerequisite model has been created already as a 'cc' file in `src/models`
- the tensorflow library has been created

#### 1.1.1.1. Step 1. Configure build

```bash
cd src/tflite_x86_64

aclocal
automake --add-missing
autoconf
./configure
```

#### 1.1.1.2. Step 2. Set model and compile

- Feature Path is the location of the local RAW feature files
- Feature Width is the size of the array in items of dtype (e.g. int8_t), typically 30225 or 66050

```bash
make copy-model MODEL_PATH=../models/{your_model_filename.cc}
make FEATURE_PATH={path_to_raw_files} FEATURE_WIDTH={30225/66050}
```

***Example:***

```bash
make copy-model MODEL_PATH=../models/20khz_SP41_taylor_l0_full_80_85.25.h5_int8_aug-data-20khz.cpp 
make FEATURE_PATH="../../data/raw/aug-data-20khz/int8" FEATURE_WIDTH=30225
```

### 1.1.2. Execution

This starts the evaluation of the tflite model

```bash
./pipeline
```

### 1.1.3. Results

Whilst accuracy is available immediately from applications the STDOUT shell output, evaluation of accuracy and duration can also be performed on the desktop, using the activity.log file.

1. Wait for tflite_x86)64 to complete.
2. Execute the following command, passing the path to both the activity.log file and source npz file used to generate the RAW files.

```bash
./evaluate_activity_log.py {path_to_activity.log} {path_to_npz_file}
```

***Example***

```bash
./evaluate_activity_log.py activity.log data/raw/aug-data-20khz/int8
```

The `activity.log` file content is a standard csv file, and can be viewed in any standard text editor. 
The `activity.log` standard format of the file:
- Model name [1] *string*
- Scores [50] *int8*
- Feature Index Label [1] *string*
- Feature Index [1] *int*
- ArgMax Label [1] *string*
- ArgMax [1] *int*
- Max Label [1] *string*
- Max Value [1] *int8*
- Duration Label [1] *string*
- Duration [1] *int*
- Target [1] *string*

# TFLITE_MICRO 
*(TFLite Micro project for Sony Spresense)*

## Background - Spresense

The TFLITE_MICRO project uses a Sony Spresense to perform inference.  The application which run the inference is compiled into a Spresense SDK / Nuttx framework and host kernel.

- Main Board
  - Sony’s CXD5602 Processor
  - 8 MB Flash memory, 1.5 MB SRAM
  - PCB with small footprint
  - Dedicated camera connector
  - GNSS (GPS) antenna
  - Pins and LEDs
    - Multiple GPIO (UART, SPI, I2C, I2S)
    - 2 ADC channels
    - Application LED x 4 (Green)
    - Power LED (Blue)
    - USB serial port
- Extension board
  - 3.5 mm headphone jack
  - Micro SD card
  - An extra USB port
  - Multiple microphone pins

[Spresense SDK](https://github.com/sonydevworld/spresense)

[Spresense Documentation](https://developer.sony.com/develop/spresense/docs/home_en.html)

![](doc/Spresense.jpg)

## Installing Spresense SDK development tools

[Sony Spresense SDK Official instructions](https://developer.sony.com/develop/spresense/docs/sdk_set_up_en.html)

**Overview of steps for linux**

Installation of toolchain to default location `${HOME}/spresenseenv/usr/bin`

1. Add user to dialout group
   ```bash
   sudo usermod -a -G dialout <user-name>
   ```
2. Install development tools
   ```bash
   wget https://raw.githubusercontent.com/sonydevworld/spresense/master/install-tools.sh
   bash install-tools.sh
   ```
3. Activate the development tools. This command must run in every terminal window. If you want to skip this step, please add this command in your `${HOME}/.bashrc.`
   *Note: this step has already been added into the build script, to avoid users needing to remember the step*

   ``` bash
   source ~/spresense/setup
   ```

## Building and Upload to Spresense device

Assumptions: 
- models have been converted already
- data generation into RAW files has already been done
- microSD card
  - Fat32 formatted with a single partition,
  - maximum capacity of 16 GB
  - `*.RAW` data files and `scores.bin` must be copied onto the micro SD card, into a folder named `data`, prior to inserting card into the Spresense device
- Sony Spresense SDK development tools are installed.
- Sony Spresense main board device
  - connected to the PC via USB, using the Micro USB port on the Spresense mainboard, not Micro USB port on the expansion board
- Sony Spresense extension board
  - connected securely to the Spresense main board.  Poor connectivity can cause a failure to read/write to the microSD card. 

```bash
cd src/tflite_micro
cp ../models/{model_filename.cc} model.cc
./build.sh
```

On an Ubuntu 20.04, the device will be automatically detected and will be defined at port `/dev/ttyUSB0`

Upload is then achieved by executing the following command.

```bash
./upload.sh
```


## Execution

The application is started from the local desktop command prompt, by typing the command to interact with the board over a serial connection over USB.

```bash
screen /dev/ttyUSB0 115200
```

This will replace the standard command prompt with a pseudo command prompt on the micro controller

```
NuttShell (NSH) NuttX-8.2
nsh> 
```

Standard 'screen' commands can be used
- **Commence logging on PC** Pressing (Ctrl + a) + (Shift + h)
- **Exit shell** Pressing (Ctrl + a) + k, then press y

When this appears, you can interact with the Spresense device, to start the inference as follows.

```
nsh> tflite_micro
```

Note: 
- the microSD card is automounted at `/mnt/sd0`
- during inference, source feature RAW files will be accssible at path `/mnt/sd0/data/`
- during inference, inference results are written to `/mnt/sd0/activity.log`

## Results

Whilst accuracy is available immediately from applications the STDOUT shell output, evaluation of accuracy and duration can also be performed on the desktop, using the activity.log file.

1. Wait for tflite_micro to complete, and the shell prompt to appear.  Unplugging during inference may result in corruption of the microSD card.
2. Turn off the Spresense, by unplugging the USB cable from the computer.
3. Eject the microSD card
4. Mount the microSD card on the PC, by inserting it into a microSD card reader attached to the desktop PC. 
5. Copy the activity.log file into the MicroPipeline directory
6. Execute the following command, passing the path to both the activity.log file and source npz file used to generate the RAW files.

```bash
./evaluate_activity_log.py {path_to_activity.log} {path_to_npz_file}
```

***Example***

```bash
./evaluate_activity_log.py activity.log data/raw/aug-data-20khz/int8
```
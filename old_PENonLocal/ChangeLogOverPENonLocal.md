# Changes over PENonLocal

2022-03-29:
* Fix bug in testing code related to incorrect reshaping of test inputs. See description: https://github.com/JiayongO-O/PENonLocal/issues/6
* Added `training_debug` to over-fit models on smaller train set
* Added `visualize_trained_model_outputs.py` script that loads an input trained model `.pth` file and visualizes the recovered depths  of the train inputs.
* Comment out unused dataloader code.
* Changed how models are loaded. No need to use for loop and copy parameter by parameter. Just use `load_state_dict`
*    


2022-03-23: 
* New setup instructions under new `README.md`. 
  * The original `requirements.txt` file did not work for me, it had a bug setting `skimage==0.0` and also the `numpy` version set would not be compatible with the `matplotlib` version that would be automatically installed.
  * Furthermore, `requirements.txt` would not install `cuda100` which is what is required according to the pytorch documentation.

2022-03-07: 
* Add Lindell et al. 2018 `README.txt` which has better guidelines for data generation. 

2022-03-08:
* Installed OpenCV 3.4.13 from source
  * Why? The new OpenCV 4+ versions do not include opencv/opencv_core
* **Fix Mex Compilation:** Changed `intrinsic_textures/mex/compile.m` command. To include both `-I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib `
* **Fix Invalid Mex File:** Even after fixing compilation, during runtime I would get the message: `Invalid MEX-file '/home/felipe/repos/spatio-temporal-csph/data_gener/TrainData/intrinsic_texture/mex/getGridLLEMatrixNormal.mexa64': /usr/local/MATLAB/R2021b/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: version "GLIBCXX_3.4.26" not found (required by /usr/local/lib/libopencv_core.so.3.4)Found images of size 427x561, filtering at 3 scales.`
  * To fix this I had to add the following environment variable: `export LD_PRELOAD=$LD_PRELOAD:"/usr/lib/x86_64-linux-gnu/libstdc++.so.6"`, and reload MATLAB. This variable forces MATLAB to use the system's `libstdc++.so.6` which has the required version available. See this [answer](https://www.mathworks.com/matlabcentral/answers/329796-issue-with-libstdc-so-6]

2022-03-09:
* **Change libfreenect include statement in get_accel_data.cpp**: To fix `nyuv2_utils` Mex Compilation I had to edit `get_accel_data.cpp`.
  * I installed `libfreenect` using the [Ubuntu instruction here](https://github.com/OpenKinect/libfreenect). First I installed the pre-requisites, and then I used `sudo apt install freenect`. 
  * Since I installed `libfreenect` using `apt`, when compiling a program the include statement should be `#include "libfreenect.h"`, and not `<libfreenect/libfreenect.h>`. So, in `get_accel_data.cpp` I changed the `libfreenect` include statement. 

2022-03-11:
* 

## Libraries that I had to install

### For data generation

* `OpenCV 3.4.13` from source. I had to do this because Ubuntu 20.04 will install OpenCV 4+ by default which does not inlucde the deprecated OpenCV 1. This is needed for the `intrinsic_textures` library
  * I followed this tutorial: `https://docs.opencv.org/3.4.13/d7/d9f/tutorial_linux_install.html`
* `libann`: This one was simple. Simply ran `sudo apt install libann-dev`
* `libfreenect`: This has its own pre-requisites so make sure to follow the [install  instructions]([Ubuntu instruction here](https://github.com/OpenKinect/libfreenect)). For Ubuntu I was able to simply install it using the package manager `apt` (after installing the pre-requisites). 

## Questions for Jiayong

* Does the `ConvertRGBD.m` script take a while to run?
* How much total storage do I need to generate all the datasets?
* How many corrupt files in the NYUv2 dataset? Did you have to download and re-download them multiple times?
* In `ConvertRGBD.m` script why is line 40 skipping every 10 files?
* In testing script under `Fn_Test.py:test_sm()` the code runs out of memory because it keep allocating more gpu memory after each loop iteration.
* What does *=> WARN: Non-local block uses '1' groups* mean?
* 

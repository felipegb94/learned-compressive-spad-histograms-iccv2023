# Changes over PENonLocal

2022-03-07: 
* Add Lindell et al. 2018 `README.txt` which has better guidelines for data generation. 

2022-03-08:
* Installed OpenCV 3.4.13 from source
  * Why? The new OpenCV 4+ versions do not include opencv/opencv_core
* **Fix Mex Compilation:** Changed `intrinsic_textures/mex/compile.m` command. To include both `-I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib `
* **Fix Invalid Mex File:** Even after fixing compilation, during runtime I would get the message: `Invalid MEX-file '/home/felipe/repos/spatio-temporal-csph/data_gener/TrainData/intrinsic_texture/mex/getGridLLEMatrixNormal.mexa64': /usr/local/MATLAB/R2021b/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: version "GLIBCXX_3.4.26" not found (required by /usr/local/lib/libopencv_core.so.3.4)Found images of size 427x561, filtering at 3 scales.`
  * To fix this I had to add the following environment variable: `export LD_PRELOAD=$LD_PRELOAD:"/usr/lib/x86_64-linux-gnu/libstdc++.so.6"`, and reload MATLAB. This variable forces MATLAB to use the system's `libstdc++.so.6` which has the required version available. See this [answer](https://www.mathworks.com/matlabcentral/answers/329796-issue-with-libstdc-so-6]

## Libraries that I had to install

### For data generation

* `OpenCV 3.4.13` from source. I had to do this because Ubuntu 20.04 will install OpenCV 4+ by default which does not inlucde the deprecated OpenCV 1. This is needed for the `intrinsic_textures` library
  * I followed this tutorial: `https://docs.opencv.org/3.4.13/d7/d9f/tutorial_linux_install.html`
* `libann`: This one was simple. Simply ran `sudo apt install libann-dev`

## Questions for Jiayong

* Does the `ConvertRGBD.m` script take a while to run?
* How much total storage do I need to generate all the datasets?


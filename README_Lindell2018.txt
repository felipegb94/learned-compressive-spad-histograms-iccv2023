-----------------------------------------------------------------------------
|            Single-Photon 3D Imaging with Deep Sensor Fusion               |
-----------------------------------------------------------------------------
Code and Models
Stanford Computational Imaging Lab 2018
Contact: lindell@stanford.edu

This folder contains the code implementation and trained models used for the
project. The following describes how to generate training data and perform the
training, evaluation on the Middlebury test dataset, and evaluation on the
captured results. Packaged with the code we include a subset of the captured
and simulated data to keep the file size manageable. All captured and
simulated data can be found at www.computationalimaging.org.


==Generating Training Data==

Steps

1. download NYU V2 data with './simulated_data/download_nyu_dataset.bash'
2. Convert to RGB-D + albedo with './simulated_data/ConvertRGBD.m'
3. Simulate SPAD measurements with './simulated_data/SimulateSpadMeasurements.m'

Commentary

Scripts to generate the training data are found in the ./simulated_data folder.
To generate the training data, first download data from the NYU V2 dataset using
the 'download_nyu_dataset.bash' script. This script requires the 'wget' utility
available on linux. The data will be downloaded to './simulated_data/raw' and
should be unzipped into that same folder after download.

Once the data is unzipped, run the Matlab script 'ConvertRGBD.m', this will
process the raw files into RGB-D images, estimate the albedo using the
implementation of Chen and Koltun (code included in
./simulated_data/intrinsic_texture'), and save '.mat' file outputs to the
'./simulated_data/processed' folder. Be sure to compile the mex files located in the
intrinsic decomposition folder and note that the files require linking to opencv
and libann (approximate nearest neighbor).

With the saved RGBD + albedo files, the SPAD measurements can be simulated using
the SimulateSpadMeasurements.m script. The script will output '.mat' files
containing SPAD measurements and ground truth data used in the training
procedure. These scripts come from the NYU dataset toolkit and require
mex compilation before running.

Note that the code supports generating data for different signal/background
photon levels, which are indexed from 1 to 10. Signal/noise levels 1-9 have
specific average signal and background photon levels per scene, while level 10
indicates that a range of signal/background levels will be simulated between 
different scenes.

==Training==

Steps

1. Install required python packages listed in './requirements.txt' 
2. Run './util/make_train_lists.py' to generate filelists for training 
3. Edit training parameters in './config.ini'
4. Run train.py 

Commentary

The code allows training of four different variations of the network: depth
estimation with an intensity image, depth estimation without an intensity image,
depth estimation with an intensity image and 2x upsampling, and depth estimation
with an intensity image and 8x upsampling. Python packages used for the training
are listed in the './requirements.txt' file. We recommend using anaconda and
installing a new environment with 

$ conda create -n single_photon python=3.7
$ conda activate single_photon

Then install the packages using Pip with the command 

$ pip install -r requirements.txt

After the training data is generated, generate the list of training files by
running the './util/make_train_lists.py' python script. Be sure to edit the
script to specify which signal/background levels were simulated so it knows
which files to look for (see code). 

Then edit the './config.ini' file to set up the training configuration and
specify which varition of the network to run as well as whether to resume from a
checkpoint and which directory to output save checkpoints and tensorboard log
files. See 'config.ini' for further details of training options. The values
originally contained with this file values are the same as those used in the paper.
These options can also be specified over the command line to the 'train.py'
script.

Note that we also provide the trained models used for the paper in the './pth' 
folder. See './config.ini' which files map to which models under the 'resume' 
keywords.

==Evaluation on Middlebury Dataset==

Steps

1. Simulate Middlebury data with './middlebury/SimulateTestMeasurements.m'
2. Edit evaluation parameters in './middlebury.ini'
3. Run evaluate_middlebury.py 

To reproduce the Art scene results from the paper run 'evaluate_middlebury.py'
with no arguments. To run the other scenes, generate the simulated SPAD
measurements for the middlebury scenes using the
'./middlebury/SimulateTestMeasurements.m' script. In the script you can specify
whether to run the high-resolution data used for depth estimation
or the low-resolution data used for depth estimation and upsampling.

Once the data is generated, edit the './middlebury.ini' file to specify which
network variation to run, which signal/background levels to run, which models to
use, and which scenes to run. See the './middlebury.ini' file for more
information. These options can also be specified as command line arguments to
the './evaluate_middlebury.py' processing program.

To generate the depth estimates run './evaluate_middlebury.py'. Results should
appear in a folder called 'results_middlebury'.

==Evaluation on Captured Dataset==

Steps

1. Edit evaluation parameters in './captured.ini'
2. Run evaluate_captured.py 

To reproduce the results for the Elephant scene, run the
'./evaluate_captured.py' Python program. The other scenes can be downloaded from
computationalimaging.org on the paper webpage. To configure the
'./evaluate_captured.py' program to run the other scenes and network variations,
edit the evaluation parameters in './captured.ini'. The output should appear as
'.mat' files in the 'results_captured' folder which the program creates. 

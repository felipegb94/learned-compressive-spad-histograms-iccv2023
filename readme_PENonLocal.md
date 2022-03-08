## [Photon-Efficient 3D Imaging with A Non-Local Neural Network](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5471_ECCV_2020_paper.php)

#### **ECCV2020** [Spotlight]

[Jiayong Peng](https://scholar.google.com/citations?user=cXdZl0wAAAAJ&hl=en), [Zhiwei Xiong](https://scholar.google.com/citations?user=Snl0HPEAAAAJ&hl=en), [Xin Huang](https://quantum.ustc.edu.cn/web/en/node/480), [Zheng-Ping Li](https://quantum.ustc.edu.cn/web/en/node/694), [Dong Liu](https://scholar.google.com/citations?user=lOWByxoAAAAJ&hl=en), [Feihu Xu](https://scholar.google.ca/citations?user=-EZOdMIAAAAJ&hl=en)

#### **Table of Contents**

- [Photon-Efficient 3D Imaging with A Non-Local Neural Network](#photon-efficient-3d-imaging-with-a-non-local-neural-network)
    - [**ECCV2020** [Spotlight]](#eccv2020-spotlight)
    - [**Table of Contents**](#table-of-contents)
    - [1. Introduction](#1-introduction)
    - [2. Citation](#2-citation)
    - [3. Installation](#3-installation)
    - [4. Data generation](#4-data-generation)
    - [5. Training](#5-training)
    - [6. Testing](#6-testing)
    - [7. Results](#7-results)

----

#### 1. Introduction

Photon-efficient imaging has enabled a number of applications relying on single-photon sensors that can capture a 3D image with as few as one photon per pixel. In practice, however, measurements of low photon counts are often mixed with heavy background noise, which poses a great challenge for existing computational reconstruction algorithms. In this paper, we first analyze the long-range correlations in both spatial and temporal dimensions of the measurements. Then we propose a non-local neural network for depth reconstruction by exploiting the long-range correlations. The proposed network achieves decent reconstruction fidelity even under photon counts (and signal-to-background ratio, SBR) as low as 1 photon/pixel (and 0.01), which significantly surpasses the state-of-the-art. Moreover, our non-local network trained on simulated data can be well generalized to different real-world imaging systems, which could extend the application scope of photon-efficient imaging in challenging scenarios with a strict limit on optical flux.![framework](https://github.com/JiayongO-O/Photon-Efficient-3D-Imaging-with-A-Non-Local-Neural-Network/blob/master/data_gener/framework-1.png)

#### 2. Citation

If you find the code useful in your research, please consider citing:



#### 3. Installation

A. Environment

The code is developed using python 3.6 and Pytorch 1.0 on Ubuntu 16.04. NVIDIA 1080Ti GPU is required for training and testing. 

B. Requirements

`pip install -r requirements.txt`

#### 4. Data generation

A. generate training and validation data

- download NYU V2 data

  `bash ./data_gener/TrainData/download_nyu_dataset.bash`

- generate RGB-D and albedo with Matlab 2017b

  `matlab -nodesktop -nosplash -r "run('./data_gener/TrainData/ConvertRGBD.m');exit"`

- simulate SPAD measurements with Matlab 2017b

  `matlab -nodesktop -nosplash -r ./data_gener/TrainData/SimulateTrainMeasurements`

B. generate testing data for Middlebury dataset

- `matlab -nodesktop -nosplash -r ./data_gener/TestData/middlebury/SimulateTestMeasurements`

C. Indoor and outdoor real-world datasets are provided in 

- `./data_gener/TestData/realworld`

#### 5. Training

- make train and validation list

  `python ./training/util/make_train_lists.py`

- edit the corresponding parameters in `./training/config.ini`

- run the main train file

  `python ./training/main.py`

- the training will take about 35 hours on NVIDIA 1080Ti.

#### 6. Testing

- edit the corresponding parameters in `./testing/config.ini`

- run the main test file

  `python ./testing/main.py`

- the test can be conducted on simulated Middlebury dataset, indoor and outdoor real-world dataset by selecting different functions in the `./testing/main.py`

#### 7. Results

We provide some of the visualization results here.

- comparisons on Middlebury dataset
![res_s](https://github.com/JiayongO-O/Photon-Efficient-3D-Imaging-with-A-Non-Local-Neural-Network/blob/master/data_gener/res_s.png)

- comparisons on outdoor real-world scenes
![res_outrw](https://github.com/JiayongO-O/Photon-Efficient-3D-Imaging-with-A-Non-Local-Neural-Network/blob/master/data_gener/res_outrw.png)

- comparisons on indoor real-world scenes
![res_inrw](https://github.com/JiayongO-O/Photon-Efficient-3D-Imaging-with-A-Non-Local-Neural-Network/blob/master/data_gener/res_inrw.png)


































# Learned Compressive Representations for Single-Photon 3D Imaging (ICCV 2023)

This repository contains the code for the [ICCV 2023 paper Learned Compressive Representations for Single-Photon 3D Imaging](https://openaccess.thecvf.com/content/ICCV2023/html/Gutierrez-Barragan_Learned_Compressive_Representations_for_Single-Photon_3D_Imaging_ICCV_2023_paper.html).

> **IMPORTANT NOTE from 2024-03-18:** The code in this repository still needs to be cleaned up and organized for better reproducibility, but we have made it public early. A new release with a simplified codebase will be made available in the next couple months. In the meantime, if you have any questions about the code, please create an issue. There is currently on thread that has been started: https://github.com/felipegb94/learned-compressive-spad-histograms-iccv2023/issues/8

To clone: `git clone --recurse-submodules git@github.com:felipegb94/learned-compressive-spad-histograms-iccv2023.git`

If you use the code in this repository please cite:

```
@inproceedings{gutierrez2023learned,
  title={Learned Compressive Representations for Single-Photon 3D Imaging},
  author={Gutierrez-Barragan, Felipe and Mu, Fangzhou and Ardelean, Andrei and Ingle, Atul and Bruschini, Claudio and Charbon, Edoardo and Li, Yin and Gupta, Mohit and Velten, Andreas},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10756--10766},
  year={2023}
}
```

## Getting started

All scripts and code should be run from the top-level directory, except for the data generation MATLAB code which should be run from the `data_gener` directory.

1. Clone `git clone --recurse-submodules git@github.com:felipegb94/learned-compressive-spad-histograms-iccv2023.git`
2. Go over the setup instructions to create the conda environment (see below). To make sure the environment is setup correctly run: `python csph_layers.py`
3. Download nyuv2 dataset. Run `python scripts/download_nyuv2_simulated_spad_data.py`
4. Try training a model. See `scripts_train/` for example train commands

To re-train a similar model to Peng et al., ECCV 2020 run: `python train.py ++experiment=no_compression_baselines_lr-1e-4 model=DDFN_C64B10 ++model.model_params.input_norm=none ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-4`
`

## Setup

### Step 1: Make sure CUDA drivers work

Verify that the command `nvidia-smi` works on your computer. If it doesn't you may need to install the NVIDIA drivers. One way to do this in Ubuntu is to go to *Software and Updates*, then go to *additional drivers*, select the driver for your GPU, apply changes, and restart the machine. 

### Setting up Environment for New Version

The code has been tested in a machine with CUDA version `11.4` installed.

You can create the conda environment from the `environment.yml` file.

Alternatively, you can manuall create it as follows:

1. Create env: `conda create -n csphenv38 python=3.8`
2. Activate env: `conda activate csphenv38`
3. Install pytorch with cuda support: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
   1. In my case pytorch 1.11 was installed
4. Install Pytorch lightning: `conda install pytorch-lightning -c conda-forge`
5. Install Additional packages: `conda install ipython matplotlib scipy scikit-image`
6. Install Hydra for config files: `pip install hydra-core --upgrade`
7. Install hydra plugins: `pip install hydra_colorlog --upgrade`
8. Install `gdown` for downloading datasets: `pip install gdown`

### Setting up PENonLocal Environments

**Option 1:** Using the `requirements.txt` provided in PENonLocal repository did not work for me. Most likely because of different GPUs + NVIDIA drivers + CUDA toolkit versions. The following steps worked for me instead in my `Ubuntu 20.04` using an `RTX 2070 Super` using `nvidia-driver-470`:

1. Create python 3.6 conda environment: `conda create --name PENonLocalEnv python=3.6`
2. Activate env: `conda activate PENonLocalEnv`
3. Install `pytorch==1.0.0` with `cuda100` using conda as in the [PyTorch documentation](https://pytorch.org/get-started/previous-versions/#v100): `conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch`
4. (Possibly needed) Re-Install mkl tools: `conda install -c anaconda mkl`
   1. This step might not be needed if you are able to `import torch` without any errors. 
   2. I had to perform this step because I got the following error: `ImportError: /home/felipe/miniconda3/envs/test/lib/python3.6/site-packages/torch/lib/libmkldnn.so.0: undefined symbol: cblas_sgemm_alloc` 
5. Install rest of packages: `conda install tensorboardX tqdm scikit-image ipython`
   1. Let conda automatically figure out the required versions here.
   2. Packages like `numpy`, `scipy`, `matplotlib` are automatically installed with the above installation steps.

**Option 2:** Install using the copy of my conda environment in `PENonLocalEnvironment.yml`.

## Citation and Acknowledgements

If you use the code in this repository, in addition to citing our work, please make sure to cite [Peng et al., ECCV 2020](https://github.com/JiayongO-O/PENonLocal) and [Lindell et al., SIGGRAPH 2018](https://davidlindell.com/publications/single-photon-3d). The initial version of the models in this repos were based on Peng et al, and the data generation code was based on Lindell et al.

```
@inproceedings{gutierrez2023learned,
  title={Learned Compressive Representations for Single-Photon 3D Imaging},
  author={Gutierrez-Barragan, Felipe and Mu, Fangzhou and Ardelean, Andrei and Ingle, Atul and Bruschini, Claudio and Charbon, Edoardo and Li, Yin and Gupta, Mohit and Velten, Andreas},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10756--10766},
  year={2023}
}
```




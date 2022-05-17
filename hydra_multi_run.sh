#!/bin/bash

## Run Depth2Depth Very Large
python train.py ++experiment=debug model=DDFN2D_Depth2Depth_01Inputs ++model.n_ddfn_blocks=22 ++train_params.lri=1e-3 ++train_params.batch_size=4 ++train_params.epoch=50 ++train_params.p_tv=1e-5 ++random_seed=1234

## Run Phasor2Depth
python train.py ++experiment=test model=DDFN2D_Phasor2Depth ++train_params.lri=1e-3 ++train_params.batch_size=8 ++train_params.epoch=50 ++train_params.p_tv=1e-5 ++random_seed=1234

## Run CSPH1D TrunFourier
python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=TruncFourier ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-3 ++random_seed=1234

# ## Run CSPH1D CoarseHist
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=CoarseHist ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-4 ++random_seed=1234


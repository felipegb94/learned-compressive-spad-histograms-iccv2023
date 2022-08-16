#!/bin/bash

###### Initialization Experiments
### Parameters:
## * Compression Rate == 128x
## * Encoding Type == separable

## Gray Fourier Initialization + FIXED

# Exp 1: CSPH separable HybridGrayFourier_opt K = 128 | 4x4x1024 (128x compression)
python train.py ++experiment=init_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=128 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=false \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# Exp 2: CSPH separable HybridGrayFourier_opt K = 32 | 4x4x256 (128x compression)
python train.py ++experiment=init_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=32 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=4 \
    ++model.model_params.optimize_tdim_codes=false \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

## Gray Fourier Initialization + Optimization

# Exp 1: CSPH separable HybridGrayFourier_opt K = 128 | 4x4x1024 (128x compression)
python train.py ++experiment=init_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=128 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# Exp 2: CSPH separable HybridGrayFourier_opt K = 32 | 4x4x256 (128x compression)
python train.py ++experiment=init_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=32 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=4 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


## Random Initialization (128x compression)

# Exp 1: CSPH FULL Rand_opt K = 128 | 4x4x1024 (128x compression)
python train.py ++experiment=init_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=Rand \
    ++model.model_params.k=128 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# Exp 2: CSPH FULL Rand_opt K = 32 | 4x4x256 (128x compression)
python train.py ++experiment=init_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=Rand \
    ++model.model_params.k=32 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=4 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

## Random Initialization (256x compression)

# Exp 1: CSPH FULL Rand_opt K = 64 | 4x4x1024 (256x compression)
python train.py ++experiment=init_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=Rand \
    ++model.model_params.k=64 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# Exp 2: CSPH FULL Rand_opt K = 16 | 4x4x256 (256x compression)
python train.py ++experiment=init_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=Rand \
    ++model.model_params.k=16 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=4 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

#!/bin/bash

###### FULL Codes vs. Separable 1D+2D Codes Experiments
### Parameters:
## * Rand init
## * Compression Rate == 128x

## Full 3D Codes

# Exp 1: CSPH FULL Rand_opt K = 128 | 4x4x1024 (128x compression)
python train.py ++experiment=full_vs_separable model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=Rand \
    ++model.model_params.k=128 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=full \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# Exp 2: CSPH FULL Rand_opt K = 32 | 4x4x256 (128x compression)
python train.py ++experiment=full_vs_separable model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=Rand \
    ++model.model_params.k=32 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=4 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=full \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

## Separable 1D + 2D Codes

# Exp 1: CSPH Separable Rand_opt K = 128 | 4x4x1024 (128x compression)
python train.py ++experiment=full_vs_separable model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=Rand \
    ++model.model_params.k=128 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# Exp 2: CSPH Separable Rand_opt K = 32 | 4x4x256 (128x compression)
python train.py ++experiment=full_vs_separable model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=Rand \
    ++model.model_params.k=32 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=4 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


###### How much spatial downsampling?
### Parameters:
## * Compression Rate == 256x
## * Init == Gray Fourier
## * encoding == Separable
## * optimize_tdim_codes == true

# Exp 1: CSPH Separable HybridGrayFourier_opt K = 16 | 2x2x1024 (256x compression)
python train.py ++experiment=spatial_down_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=16 \
    ++model.model_params.spatial_down_factor=2 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# Exp 2: CSPH Separable HybridGrayFourier_opt K = 64 | 4x4x1024 (256x compression)
python train.py ++experiment=spatial_down_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=64 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# Exp 3: CSPH Separable HybridGrayFourier_opt K = 256 | 8x8x1024 (256x compression)
python train.py ++experiment=spatial_down_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=256 \
    ++model.model_params.spatial_down_factor=8 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4
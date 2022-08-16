#!/bin/bash

## NOTE: Had to change random seed to 1235 for Exp 1 to work

###### How much temporal downsampling?
### Parameters:
## * Compression Rate == 256x
## * Init == HybridGrayFourier
## * encoding == Separable
## * optimize_tdim_codes == true

# Exp 1: CSPH Separable HybridGrayFourier_opt k=64 | 4x4x1024 (256x compression)
python train.py ++experiment=temporal_down_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=64 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=1 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4


# Exp 2: CSPH Separable HybridGrayFourier_opt k=32 | 4x4x512 (256x compression)
python train.py ++experiment=temporal_down_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=32 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=2 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# Exp 3: CSPH Separable HybridGrayFourier_opt k=16 | 4x4x256 (256x compression)
python train.py ++experiment=temporal_down_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=16 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=4 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# Exp 4: CSPH Separable HybridGrayFourier_opt k=8 | 4x4x128 (256x compression)
python train.py ++experiment=temporal_down_ablation model=DDFN_C64B10_CSPH \
    ++model.model_params.tblock_init=HybridGrayFourier \
    ++model.model_params.k=8 \
    ++model.model_params.spatial_down_factor=4 \
    ++model.model_params.nt_blocks=8 \
    ++model.model_params.optimize_tdim_codes=true \
    ++model.model_params.encoding_type=separable \
    ++train_params.p_tv=0.0 \
    ++train_params.epoch=30 \
    ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4
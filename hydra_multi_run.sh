#!/bin/bash


# ## STATUS: DONE  Running in compoptics [DONE]
# ## Run CSPH1D TruncFourier K=32 (Increased Learning Rate Experiment) 
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH1D ++model.model_params.init=TruncFourier ++model.model_params.k=32 ++train_params.p_tv=0.0 ++train_params.workers=8 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234

# # ## STATUS: DONE Running in fmu
# # ## Run CSPH1D2D HybridGrayFourier K=128 
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH1D2D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=128 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


# # ## STATUS: Done Running localy
# # ## Run CSPH1D2D HybridGrayFourier K=32 
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH1D2D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=32 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# ## STATUS: DONE Running in localhost
# ## Run CSPH1D HybridFourierGray K=8 (Increased Learning Rate Experiment) 
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH1D ++model.model_params.init=HybridFourierGray ++model.model_params.k=8 ++train_params.p_tv=0.0 ++train_params.workers=8 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# ## STATUS: [DONE] Running in localhost
# ## Run CSPH1D HybridGrayFourier K=8 (Increased Learning Rate Experiment) 
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH1D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=8 ++train_params.p_tv=0.0 ++train_params.workers=8 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


# ## STATUS: Done Running localy
# ## Run CSPH1D2D TruncFourier K=128 
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH1DGlobal2DLocal4xDown ++model.model_params.init=TruncFourier ++model.model_params.k=128 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


# STATUS: [DONE] Running Locally
# CSPH Separable GrayFourier K = 64 | 2x2x1024 (64x compression)
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=64 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# STATUS: [DONE] Running compoptics
# CSPH Separable GrayFourier K = 256 | 4x4x1024 (64x compression)
python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


# STATUS: [DONE] Running compoptics
# CSPH Separable TruncFourier K = 128 | 4x4x1024 (128x compression)
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


# STATUS: [DONE] Running Fangzhou
# CSPH Separable GrayFourier K = 32 | 2x2x1024 (128x compression)
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=32 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


# STATUS: [DONE] Running locally
# CSPH Separable TruncFourier K = 64 | 4x4x512 (128x compression)
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=2 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


# STATUS: [DONE] Running compoptics
# CSPH Separable TruncFourier K = 128 | 4x4x1024 (128x compression)
# python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


############ Tests

# STATUS:  Running locally
# CSPH Separable HybridGrayFourier K = 128 | 4x4x1024 (128x compression)
python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH 
    ++model.model_params.tblock_init=HybridGrayFourier 
    ++model.model_params.k=128 
    ++model.model_params.spatial_down_factor=4 
    ++model.model_params.nt_blocks=1 
    ++model.model_params.optimize_tdim_codes=false 
    ++model.model_params.encoding_type=separable 
    ++train_params.p_tv=0.0 
    ++train_params.epoch=1 
    ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


# STATUS:  Running locally
# CSPH Separable HybridGrayFourier K = 128 | 4x4x1024 (128x compression)
python train.py 
    ++experiment=test_csph model=DDFN_C64B10_CSPH  
    ++model.model_params.tblock_init=Rand  
    ++model.model_params.k=128  
    ++model.model_params.spatial_down_factor=4  
    ++model.model_params.nt_blocks=1  
    ++model.model_params.optimize_tdim_codes=true  
    ++model.model_params.encoding_type=separable  
    ++train_params.p_tv=0.0  ++train_params.epoch=1  ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

# STATUS:  Running locally
# CSPH Separable HybridGrayFourier K = 128 | 4x4x1024 (128x compression)
python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=1 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=1 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4



python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=8 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=8 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# running in compoptics [DONE]
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=16 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=8 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=16 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=16 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# running in compoptics in spatio-temporal-csph_laptop [DONE]
# 
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# running in fmu in spatio-temporal-csph_laptop [DONE]
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4


# Run locally [DONE]
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=16 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [DONE] Run in fmu in spatio-temporal-csph_laptop (compression 64x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [DONE] Run in compoptics in spatio-temporal-csph_laptop (compression 32x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4



### Separable vs. Full Ablation Study

# [PENDING] Run in compoptics in spatio-temporal-csph_laptop (SEPARABLE compression 32x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Run in compoptics in spatio-temporal-csph_laptop (SEPARABLE compression 64x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Run in compoptics in spatio-temporal-csph_laptop (SEPARABLE compression 128x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Run in compoptics in spatio-temporal-csph_laptop (SEPARABLE compression 256x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Run in compoptics in spatio-temporal-csph_laptop (SEPARABLE compression 32x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Run in compoptics in spatio-temporal-csph_laptop (SEPARABLE compression 64x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Run in compoptics in spatio-temporal-csph_laptop (SEPARABLE compression 128x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Run in compoptics in spatio-temporal-csph_laptop (SEPARABLE compression 256x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

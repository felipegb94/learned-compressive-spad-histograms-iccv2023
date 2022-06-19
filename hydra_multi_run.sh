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


# STATUS: Running Locally
# CSPH Separable GrayFourier K = 64 | Down=2 (64x compression)
python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=64 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4


# STATUS: Running compoptics
# CSPH Separable TruncFourier K = 128 | Down=4 (128x compression)
python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=4

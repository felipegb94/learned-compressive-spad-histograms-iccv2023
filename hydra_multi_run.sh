#!/bin/bash


# ## Run Depth2Depth Very Large (Longer training) --> NOT DONE accidentally deleted it (ran compoptics) 
# python train.py ++experiment=debug model=DDFN2D_Depth2Depth_01Inputs ++model.n_ddfn_blocks=22 ++train_params.lri=1e-3 ++train_params.batch_size=4 ++train_params.epoch=50 ++train_params.p_tv=1e-5 ++random_seed=1234

# ## Run Phasor2Depth (Longer training) --> IN PROGRESS (had to restart locally)
# python train.py ++experiment=test model=DDFN2D_Phasor2Depth ++train_params.lri=1e-3 ++train_params.batch_size=8 ++train_params.epoch=50 ++train_params.p_tv=1e-5 ++random_seed=1234
# python train_resume.py ++experiment=test ++train_params.lri=1e-3 ++train_params.batch_size=8 ++train_params.epoch=50 ++train_params.p_tv=1e-5 ++random_seed=1234

# ## Run CSPH1D TruncFourier K=16 (Increased Learning Rate Experiment) --> IN PROGRESS (running on compoptics)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=TruncFourier ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-3 ++random_seed=1234


# ## Run CSPH1D TruncFourier (K=64 experiment) --> DONE (ran locally)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=TruncFourier ++model.model_params.k=64 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-4 ++random_seed=1234

# ## Run CSPH1D HybridGrayFourier (K=64 experiment)  --> DONE (ran locally)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=64 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-4 ++random_seed=1234

# ## Run CSPH1D CoarseHist (K=64 experiment) --> DONE (ran Fangzhou)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=CoarseHist ++model.model_params.k=64 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-4 ++random_seed=1234



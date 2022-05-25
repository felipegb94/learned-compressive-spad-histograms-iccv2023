#!/bin/bash

# ## STATUS: Ran in compoptics and fmu with wrong command (re-running locally)
# ## Run Depth2Depth Layers=22 (Longer training) --> NOT DONE accidentally deleted it (ran compoptics) 
# python train.py ++experiment=debug model=DDFN2D_Depth2Depth_01Inputs ++model.model_params.n_ddfn_blocks=22 ++train_params.lri=1e-3 ++train_params.batch_size=4 ++train_params.epoch=50 ++train_params.p_tv=1e-5 ++random_seed=1234

# ## STATUS: Running locally
# ## Run Depth2Depth Layers=22 NO TV 
# python train.py ++experiment=debug model=DDFN2D_Depth2Depth_01Inputs ++model.model_params.n_ddfn_blocks=22 ++train_params.lri=1e-3 ++train_params.batch_size=4 ++train_params.epoch=50 ++train_params.p_tv=0 ++random_seed=1234

# ## STATUS: Running locally
# ## Run Depth2Depth Layers=16 NO TV 
# python train.py ++experiment=debug model=DDFN2D_Depth2Depth_01Inputs ++model.model_params.n_ddfn_blocks=16 ++train_params.lri=1e-3 ++train_params.batch_size=4 ++train_params.epoch=50 ++train_params.p_tv=0 ++random_seed=1234

## STATUS: Running locally
## Run Depth2Depth Layers=8 NO TV 
# python train.py ++experiment=debug model=DDFN2D_Depth2Depth_01Inputs ++model.model_params.n_ddfn_blocks=8 ++train_params.lri=1e-3 ++train_params.batch_size=8 ++train_params.epoch=50 ++train_params.p_tv=0 ++train_params.cuda=true ++train_params.workers=16 ++train_params.val_workers=16 ++random_seed=1234

# ## STATUS: DONE running locally 
# ## Run Phasor2Depth (Longer training)
# python train.py ++experiment=test model=DDFN2D_Phasor2Depth ++train_params.lri=1e-3 ++train_params.batch_size=8 ++train_params.epoch=50 ++train_params.p_tv=1e-5 ++random_seed=1234
# python train_resume.py ++experiment=test ++train_params.lri=1e-3 ++train_params.batch_size=8 ++train_params.epoch=50 ++train_params.p_tv=1e-5 ++random_seed=1234

# ## STATUS: DONE running in compoptics
# ## Run CSPH1D TruncFourier K=16 (Increased Learning Rate Experiment) --> IN PROGRESS (running on compoptics)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=TruncFourier ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-3 ++random_seed=1234

# ## STATUS: running in fmu
# ## Run CSPH1D HybridGrayFourier K=16 (Increased Learning Rate Experiment) --> IN PROGRESS (running on compoptics)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234

# ## STATUS: running in fmu
# ## Run CSPH1D HybridGrayFourier K=32 (Increased Learning Rate Experiment) --> IN PROGRESS (running on compoptics)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=32 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-3 ++random_seed=1234

# ## STATUS: DONE running locally
# ## Run CSPH1D TruncFourier (K=64 experiment) --> DONE (ran locally)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=TruncFourier ++model.model_params.k=64 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-4 ++random_seed=1234

# ## STATUS: DONE running locally
# ## Run CSPH1D HybridGrayFourier (K=64 experiment)  --> DONE (ran locally)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=64 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-4 ++random_seed=1234

# ## STATUS: DONE running in fmu
# ## Run CSPH1D CoarseHist (K=64 experiment) --> DONE (ran Fangzhou)
# python train.py ++experiment=debug model=DDFN_C64B10_CSPH1D ++model.model_params.init=CoarseHist ++model.model_params.k=64 ++train_params.p_tv=0.0 ++train_params.epoch=10 ++train_params.lri=1e-4 ++random_seed=1234



# ## Run Unet2D CSPH1D HybridGrayFourier (K=16 experiment) 
# python train.py ++experiment=test_unet model=Unet2D_CSPH1D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=50 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=16

# python train.py ++experiment=test_unet model=Unet2D_CSPH1D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=50 ++train_params.lri=1e-4 ++random_seed=1234 ++train_params.batch_size=4


# python train.py ++experiment=test_unet model=Unet2D_CSPH1D2Phasor ++model.model_params.init=HybridGrayFourier ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=50 ++train_params.lri=1e-4 ++random_seed=1234 ++train_params.batch_size=4

# python train.py ++experiment=test_unet model=Unet2D_CSPH1DLinearOut ++model.model_params.init=HybridGrayFourier ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=50 ++train_params.lri=1e-3 ++random_seed=1234 ++train_params.batch_size=16



python train.py ++experiment=test_csph model=DDFN_C64B10_CSPH1D2D ++model.model_params.init=HybridGrayFourier ++model.model_params.k=64 ++train_params.p_tv=0.0 ++train_params.epoch=20 ++train_params.lri=1e-4 ++random_seed=1235 ++train_params.batch_size=4


# python train.py ++experiment=test_unet model=Unet2D_CSPH1D2FFTHist ++model.model_params.init=HybridGrayFourier ++model.model_params.k=16 ++train_params.p_tv=0.0 ++train_params.epoch=50 ++train_params.lri=1e-4 ++random_seed=1234 ++train_params.batch_size=4

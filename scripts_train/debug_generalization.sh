
#### Model at K=32 (32x compression)
## [Stopped] Train in Compoptics and resume locally (1D Temporal CSPH - tblock_init=TruncFourier)
python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3Dv2 ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Models at K=64 (16x compression)

## [Stopped] Train in Compoptics and resume locally (1D Temporal CSPH - tblock_init=TruncFourier)
python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3Dv2 ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=64 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### RUN FOR OLD CSPH3Dv2 model
## [Pending] Run in compoptics (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3Dv2 ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3Dv2 ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=separable ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true


#### 
python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=false ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=separable ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=false ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=separable ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true


############## NEW CSPH3D Models
#### (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)

## [DONE] Run in compoptics (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
## Zero Mean Codes==False, ZNCC Norm = False, Norm X == LinfGlobal, Smooth==False, Account IRF==True, 
python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++model.model_params.zero_mean_tdim_codes=false ++model.model_params.apply_zncc_norm=false ++model.model_params.smooth_tdim_codes=false ++model.model_params.account_irf=true 

## Zero Mean Codes==True, ZNCC Norm = False, Norm X == None, Smooth==False, Account IRF==False, 
python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=none ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=false ++model.model_params.smooth_tdim_codes=false ++model.model_params.account_irf=false

## Zero Mean Codes==True, ZNCC Norm = True, Norm X == None, Smooth==False, Account IRF==False, 
python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=none ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.smooth_tdim_codes=false ++model.model_params.account_irf=false

## Zero Mean Codes==True, ZNCC Norm = True, Norm X == None, Smooth==True, Account IRF==True, 
python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=none ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.smooth_tdim_codes=true ++model.model_params.account_irf=true

## Zero Mean Codes==True, ZNCC Norm = True, Norm X == LinfGlobal, Smooth==False, Account IRF==False, 
python train.py ++experiment=csph3d_debug_generalization model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.smooth_tdim_codes=false ++model.model_params.account_irf=false
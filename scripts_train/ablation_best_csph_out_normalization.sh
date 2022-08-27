#### What is a good normalization to do at the end of the CSPH3D Layer?
# The following commands train CSPH3D models with different normalization strategies
# Experimental configs:
#   - tblock_init: [Rand]
#   - Compression Rates: [128x, 32x]
#   - Block Dims: [1024x4x4]
#   - Optimize Codes: True
#   - Normalizations: [none, Linf, L2, Cmatsize] 
# Questions we want to answer:
#      - What is a good normalization strategy?
#      - Can a good normalization help generalize for more flux levels with sacrificing performance?
################ 


#### Normalization == LinfGlobal + TV regularization
## [Pending] Run in Compoptics (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=1e-5 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [Pending] Run locally (FULL and tblock_init=Rand k=128, codes=(1024x4x4) compression 128x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=1e-5 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Normalization == LinfGlobal
## [Pending] Run in fmu gpu (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Normalization == LinfGlobal + TV reg (Smaller) + smaller lr
## [Not run tet] Run in Compoptics (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=1e-6 ++train_params.epoch=30 ++train_params.lri=3e-4 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Normalization == LinfGlobal + TV reg (medium) + smaller lr
## [Not run tet] Run in Compoptics (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=3e-6 ++train_params.epoch=30 ++train_params.lri=3e-4 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Normalization == Linf + TV reg (medium) + smaller lr
## [Not run tet] Run in Compoptics (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=3e-6 ++train_params.epoch=30 ++train_params.lri=3e-4 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Normalization == LinfGlobal + TV reg (medium) + smaller lr
## [Not run tet] Run in Compoptics (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 128x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=3e-6 ++train_params.epoch=30 ++train_params.lri=3e-4 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true


#### Normalization == LinfGlobal + TV regularization + Smaller lr
## [Pending] Run in fmu gpu (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=1e-5 ++train_params.epoch=30 ++train_params.lri=3e-4 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [Pending] Run in fmu gpu (FULL and tblock_init=Rand k=128, codes=(1024x4x4) compression 128x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=1e-5 ++train_params.epoch=30 ++train_params.lri=3e-4 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Normalization == Linf
## [DONE] Run in Euler (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [DONE] Run in Euler (FULL and tblock_init=Rand k=128, codes=(1024x4x4) compression 128x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Normalization == L2
## [DONE] Run in Euler (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=L2 ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [DONE] Run in Euler (FULL and tblock_init=Rand k=128, codes=(1024x4x4) compression 128x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=L2 ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Normalization == Cmatsize
## [NOT DONE] Run in Euler (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=Cmatsize ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [NOT DONE] Run in Euler (FULL and tblock_init=Rand k=128, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=Cmatsize ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Normalization == None
## Already done. See ablation_csph3d_full_vs_separable
## [DONE] Run in Euler (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=none ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [DONE] Run in Euler (FULL and tblock_init=Rand k=128, codes=(1024x4x4) compression 32x)
python train.py ++experiment=csph3d_good_norm model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++model.model_params.csph_out_norm=none ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true



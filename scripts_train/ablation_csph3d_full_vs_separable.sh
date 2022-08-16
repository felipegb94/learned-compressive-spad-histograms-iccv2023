#### Separable vs. Full Ablation Study
# The following commands train CSPH3D models with full and separable coding matrices
# Experimental configs:
#      - The models with full coding matrices are initialized with random coding matrices
#      - The models with separable coding matrices are initialized with random coding matrices and also with HybridGrayFourier for the temporal coding layer
#
# Questions we want to answer:
#      - Does a separable coding matrix match the performance of full at different compression levels?
#      - Does a good initialization of the temporal coding layer help performance?

############################################################

#### Separable models with Random initialization

# [PENDING] Train in Euler (SEPARABLE and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Train in Euler (SEPARABLE and tblock_init=Rand k=256, codes=(1024x4x4) compression 64x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Train in Euler (SEPARABLE and tblock_init=Rand k=128, codes=(1024x4x4) compression 128x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Train in Euler (SEPARABLE and tblock_init=Rand k=64, codes=(1024x4x4) compression 256x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=Rand ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

############################################################

#### Separable models with HybridGrayFourier initialization

# [PENDING] Train in Euler (SEPARABLE and tblock_init=HybridGrayFourier k=512, codes=(1024x4x4) compression 32x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Train in Euler (SEPARABLE and tblock_init=HybridGrayFourier k=256, codes=(1024x4x4) compression 32x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Train in Euler (SEPARABLE and tblock_init=HybridGrayFourier k=128, codes=(1024x4x4) compression 32x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

# [PENDING] Train in Euler (SEPARABLE and tblock_init=HybridGrayFourier k=64, codes=(1024x4x4) compression 32x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

############################################################

#### Full models with Random initialization

## [DONE] Run in compoptics in spatio-temporal-csph_laptop (FULL and tblock_init=Rand k=512, codes=(1024x4x4) compression 32x)
# python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4
## Model re-trained to double check new csph3d implementation
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

## [DONE] Run in fmu in spatio-temporal-csph_laptop (FULL and tblock_init=Rand k=256, codes=(1024x4x4) compression 64x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.optimize_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

## [DONE] Run in fmu in spatio-temporal-csph_laptop (FULL and tblock_init=Rand k=128, codes=(1024x4x4) compression 128x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4

## [DONE] Run in compoptics in spatio-temporal-csph_laptop (FULL and tblock_init=Rand k=64, codes=(1024x4x4) compression 256x)
python train.py ++experiment=test_csph3d model=DDFN_C64B10_CSPH3Dv1 ++model.model_params.tblock_init=Rand ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_tdim_codes=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4


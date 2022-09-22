#### Separable CSPH3D models
# The following commands train CSPH3D models with coding matrices that have spatio-temporal encoding kernels with dimensions 1024x2x2, and 1024x4x4 whose time and spatial dimensions are separable
# Experimental configs:
#   - Encoding Kernels (Block Dims) = [1024x4x4, 1024x2x2]
#   - Compression Levels = [32x, 64x, 128x]
#   - Encoding Type = separable
#   - Optimize Encoding Kernels = True
#   - tblock_init = [Rand]

# Question we want to answer:
#      - Can separable encoding kernels perform the same as full?


## Models with 1024x4x4 Encoding Kernel

# [STATUS==PENDING] 3D CSPH 1024x4x4 - Compression=32x --> K=512
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==PENDING] 3D CSPH 1024x4x4 - Compression=64x --> K=256
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==PENDING] 3D CSPH 1024x4x4 - Compression=128x --> K=128
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

## Models with 1024x2x2 Encoding Kernel

# [STATUS==PENDING] 3D CSPH 1024x2x2 - Compression=32x --> K=128
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=2 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==PENDING] 3D CSPH 1024x2x2 - Compression=64x --> K=64
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=64 ++model.model_params.spatial_down_factor=2 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==PENDING] 3D CSPH 1024x2x2 - Compression=128x --> K=32
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=2 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

## Models with 1024x8x8 Encoding Kernel

# [STATUS==PENDING] 3D CSPH 1024x8x8 - Compression=32x --> k=2048
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=2048 ++model.model_params.spatial_down_factor=8 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==PENDING] 3D CSPH 1024x8x8 - Compression=64x --> k=1024
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=1024 ++model.model_params.spatial_down_factor=8 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==PENDING] 3D CSPH 1024x8x8 - Compression=128x --> k=512
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=8 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=tru
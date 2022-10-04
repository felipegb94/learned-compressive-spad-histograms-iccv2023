#### What is a good temporal block size
# The following commands train CSPH3D models with coding matrices that have spatio-temporal encoding kernels with dimensions 256x4x4, 64x4x4, 16x4x4 and whose time and spatial dimensions are separable
# Experimental configs:
#   - Encoding Kernels (Block Dims) = [256x4x4, 64x4x4, 16x4x4]
#   - Compression Levels = [32x, 64x, 128x]
#   - Encoding Type = separable
#   - Optimize Encoding Kernels = True
#   - tblock_init = [Rand]

# Question we want to answer:
#      - Can separable encoding kernels perform the same as full?


## Models with 256x4x4 Encoding Kernel

# [STATUS==DONE RAN EULER] 3D CSPH 256x4x4 - Compression=32x --> k=128
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=4 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==DONE RAN EULER] 3D CSPH 256x4x4 - Compression=64x --> k=64
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=4 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==DONE RAN EULER] 3D CSPH 256x4x4 - Compression=128x --> k=32
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=4 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

## Models with 64x4x4 Encoding Kernel

# [STATUS==DONE RAN EULER] 3D CSPH 64x4x4 - Compression=32x --> k=32
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=16 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==FINISHING RUN LOCALLY] 3D CSPH 64x4x4 - Compression=64x --> k=16
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=16 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=16 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==DONE RAN EULER] 3D CSPH 64x4x4 - Compression=128x --> k=8
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=8 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=16 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

## Models with 16x4x4 Encoding Kernel

# [STATUS==DONE RAN EULER] 3D CSPH 16x4x4 - Compression=32x --> k=8
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=8 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=64 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==DONE RAN EULER] 3D CSPH 16x4x4 - Compression=64x --> k=4
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=4 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=64 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==DONE RAN EULER] 3D CSPH 16x4x4 - Compression=128x --> k=2
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=2 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=64 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
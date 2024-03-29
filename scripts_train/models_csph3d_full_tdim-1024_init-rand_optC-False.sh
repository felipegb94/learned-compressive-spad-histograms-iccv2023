#### Spatio-temporal CSPH with full kernel encodings that are NOT Optimized
# The following commands train CSPH3D models with coding matrices that have spatio-temporal encoding kernels with dimensions 1024x4x4, and these kernels are not optimized during training
# Experimental configs:
#   - Encoding Kernels (Block Dims) = [1024x4x4]
#   - Compression Levels = [32x, 64x, 128x]
#   - Encoding Type = full
#   - Optimize Encoding Kernels = FALSE
#   - tblock_init = [Rand]

## Models with 1024x4x4 Encoding Kernel

# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x4x4 - Compression=32x --> k=512
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x4x4 - Compression=64x --> k=256
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x4x4 - Compression=128x --> k=128
python train.py ++experiment=csph3d_models model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=full ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

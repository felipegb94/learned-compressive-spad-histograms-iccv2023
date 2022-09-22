#### What is the best 1D CSPH Baseline
# The following commands train CSPH3D models with coding matrices that only operate on the time dimensions (i.e., 1D CSPH)
# Experimental configs:
#   - Encoding Kernels (Block Dims) = [1024x1x1]
#   - Compression Levels = [32x, 64x, 128x] → K = 32, 16, 8
#   - Encoding Type = csph1d
#   - Optimize Encoding Kernels = FALSE
#   - tblock_init = [CoarseHist, TruncFourier, GrayFourier, Rand, Rand + Opt Codes]

# Questions we want to answer:
#      - Which is the best 1D CSPH baseline to compare with in the main paper?
#      - What is the importance of a well-designed coding matrix?


# #### Models at k=8 (128x compression)

# ## [STATUS==RUNNING LOCALLY] 1D Temporal CSPH - tblock_init=TruncFourier)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# ## [STATUS==RUNNING LOCALLY] 1D Temporal CSPH - tblock_init=HybridGrayFourier)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# ## [STATUS==RUNNING LOCALLY] 1D Temporal CSPH - tblock_init=CoarseHist)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# ## [STATUS==RUNNING LOCALLY] 1D Temporal CSPH - tblock_init=Rand)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# # [STATUS==RUNNING LOCALLY] 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Models at k=16 (64x compression)

## [STATUS==RUNNING IN COMPOPTICS] 1D Temporal CSPH - tblock_init=TruncFourier)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=16 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [STATUS==RUNNING IN COMPOPTICS] 1D Temporal CSPH - tblock_init=HybridGrayFourier)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=16 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [STATUS==RUNNING IN COMPOPTICS] 1D Temporal CSPH - tblock_init=CoarseHist)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=16 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [STATUS==RUNNING IN COMPOPTICS] 1D Temporal CSPH - tblock_init=Rand)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=16 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [STATUS==RUNNING IN COMPOPTICS] 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=16 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

# #### Models at k=32 (32x compression)

# ## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=TruncFourier)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# ## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=HybridGrayFourier)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# ## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=CoarseHist)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# ## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=Rand)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
# ## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
# python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true


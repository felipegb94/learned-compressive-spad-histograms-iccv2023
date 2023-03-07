#### CoarseHistograms at 128x, 64x, 32x, 16x, 8x, and 4x compression

python train.py ++experiment=csph3D_coarsehist_20230306 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=128 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=false ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=csph3D_coarsehist_20230306 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=false ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=csph3D_coarsehist_20230306 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=16 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=false ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=csph3D_coarsehist_20230306 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=256 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=false ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=csph3D_coarsehist_20230306 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=false ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=csph3D_coarsehist_20230306 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=64 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=false ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
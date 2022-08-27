#### What is the best 1D CSPH Baseline
# The following commands train CSPH3D models with coding matrices that only operate on the time dimensions (i.e., 1D CSPH)
# Experimental configs:
#   - tblock_init: [TruncFourier, HybridGrayFourier, HybridFourierGray, CoarseHist, BinaryRand, Rand]
#   - Compression Rates: [128x, 32x] → K = [8, 32]
#   - Block Dims: 1024x1x1
#   - Optimize Codes: False
# Questions we want to answer:
#      - Which is the best 1D CSPH baseline to compare with in the main paper?
#      - What is the importance of a well-designed coding matrix?


#### Models at K=8 (128x compression)

## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=TruncFourier)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=HybridGrayFourier)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=HybridFourierGray)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridFourierGray ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=CoarseHist)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=Rand)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=BinaryRand)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=BinaryRand ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true


#### Models at K=32 (32x compression)

## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=TruncFourier)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=HybridGrayFourier)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=HybridFourierGray)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridFourierGray ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=CoarseHist)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=CoarseHist ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=Rand)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=Linf ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=BinaryRand)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=BinaryRand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Models at K=64 (16x compression)

## [PENDING] Train in Euler (1D Temporal CSPH - tblock_init=TruncFourier)
python train.py ++experiment=csph3D_tdim_baselines model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=TruncFourier ++model.model_params.k=64 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.encoding_type=csph1d ++model.model_params.csph_out_norm=LinfGlobal ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
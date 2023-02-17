#### Validate New Modulo Dataset
# Simulated new dataset with a smaller depth range. This script trains the following models to validate that a dataset with a smaller range does not affect results
#       * 

#### NEW MODULO DATASET GrayFourier Models

## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=HybridGrayFourier)
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=16 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=false ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### NEW MODULO DATASET Learned 1024x1x1 Models

## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=Rand)
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=8 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=16 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=1 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=csph1d ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### New modulo dataset learned 256x4x4 Models
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=32 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=4 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=4 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=4 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### New modulo dataset learned 256x4x4 Models with tdim==HybridGrauFourier
# [STATUS==PENDING] 3D CSPH 256x4x4 - Compression=128x --> k=32
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=32 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=4 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=64 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=4 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=HybridGrayFourier ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=4 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=false ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### New modulo dataset learned 1024x4x4 Separable Models
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### Re-run New modulo dataset learned 1024x4x4 Separable Models
python train.py ++experiment=validate_new_modulo_dataset_20230213_rerun model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1236 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213_rerun model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1236 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213_rerun model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=0.0 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1236 ++train_params.batch_size=4 ++resume_train=true

#### New modulo dataset learned 1024x4x4 Separable Models with TV Regularization
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=1e-5 ++train_params.epoch=30 ++train_params.lri=1e-4 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=1e-5 ++train_params.epoch=30 ++train_params.lri=1e-4 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=1e-5 ++train_params.epoch=30 ++train_params.lri=1e-4 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### New modulo dataset learned 1024x4x4 Separable Models with TV Regularization
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=1e-6 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=256 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=1e-6 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=512 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=1e-6 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

#### No Compression Baselines
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10 ++model.model_params.input_norm=none ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-4
python train.py ++experiment=validate_new_modulo_dataset_20230213 model=DDFN_C64B10_Depth2Depth ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-4


#### Trained in Compoptics New modulo dataset learned 1024x4x4 Separable Models with TV Regularization
## STATUS DONE
# 1024x4x4 separable tv=1e-5 and lri=1e-3
python train.py ++experiment=validate_new_modulo_dataset_20230213_compoptics model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=1e-5 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true

## STATUS DONE
# 1024x4x4 separable tv=1e-6 and lri=1e-3
python train.py ++experiment=validate_new_modulo_dataset_20230213_compoptics model=DDFN_C64B10_CSPH3D ++model.model_params.tblock_init=Rand ++model.model_params.k=128 ++model.model_params.spatial_down_factor=4 ++model.model_params.nt_blocks=1 ++model.model_params.optimize_codes=true ++model.model_params.optimize_tdim_codes=true ++model.model_params.zero_mean_tdim_codes=true ++model.model_params.apply_zncc_norm=true ++model.model_params.encoding_type=separable ++train_params.p_tv=1e-6 ++train_params.epoch=30 ++train_params.lri=1e-3 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true






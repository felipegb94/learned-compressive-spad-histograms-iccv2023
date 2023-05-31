#### Separable CSPH3D models 256x4x4 and truncfourier in time domain
# The models tested here were trained with the commands in scripts_train/models_csph3d_separable_tdim-256_init-truncfourier_optCt-False.sh

## If any command fails exit.
set -e 

## Dataset we are testing with
# test_dataset=middlebury
# test_dataset=middlebury_largedepth
# test_dataset=middlebury_maskedhightimebins
# test_dataset=lindell2018_linospad_min
test_dataset=lindell2018_linospad

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

### Models with 256x4x4 Encoding Kernel and TruncFourier tdim
## Compression=32x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010046
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0196.ckpt # 128 images==0.018838 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## Compression=64x --> k=64
model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010046
ckpt_id=epoch=28-step=100424-avgvalrmse=0.0208.ckpt # 128 images==0.021288  | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## Compression=128x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010112
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0221.ckpt # 128 images==0.0235705 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

### Models with 256x2x2 Encoding Kernel and TruncFourier tdim
## Compression=32x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010116
ckpt_id=epoch=28-step=99559-avgvalrmse=0.0169.ckpt # 128 images==0.0149041 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
### Models with 256x2x2 Encoding Kernel and TruncFourier tdim
## Compression=64x --> k=16
model_name=DDFN_C64B10_CSPH3D/k16_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010116
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0188.ckpt # 128 images==0.019783 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
### Models with 256x2x2 Encoding Kernel and TruncFourier tdim
## Compression=128x --> k=8
model_name=DDFN_C64B10_CSPH3D/k8_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010108
ckpt_id=epoch=29-step=101292-avgvalrmse=0.0208.ckpt # 128 images==0.027000 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


# #### Models with 64x4x4 Encoding Kernel and TruncFourier tdim
# ## Compression=32x --> k=32
# model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt16_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_20230218
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010108
# ckpt_id=epoch=29-step=103887-avgvalrmse=0.0236.ckpt # 128 images==0.025002 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## Compression=64x --> k=16
# model_name=DDFN_C64B10_CSPH3D/k16_down4_Mt16_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_20230218
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010106
# ckpt_id=epoch=29-step=103887-avgvalrmse=0.0265.ckpt # 128 images==0.02853 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## Compression=128x --> k=8
# model_name=DDFN_C64B10_CSPH3D/k8_down4_Mt16_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_20230218
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-23_095038
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0310.ckpt # 128 images==0.03198 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


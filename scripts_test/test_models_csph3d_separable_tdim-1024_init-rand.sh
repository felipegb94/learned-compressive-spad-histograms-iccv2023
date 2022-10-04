#### Separable CSPH3D models
# The models tested here were trained with the commands in scripts_train/models_csph3d_separable_tdim-1024_init-rand.sh

# Question we want to answer:
#      - Can separable encoding kernels perform the same as full?


## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

## Models with 1024x4x4 Separable Encoding Kernel


# 3D CSPH 1024x4x4 - Compression=32x --> k=512
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174558
ckpt_id=epoch=26-step=93500-avgvalrmse=0.0158.ckpt #  | 120 images== | 128 images==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 1024x4x4 - Compression=64x --> k=256
model_name=DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174558
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0167.ckpt #  | 120 images== | 128 images==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 1024x4x4 - Compression=128x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_044702
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0180.ckpt #  | 120 images== | 128 images==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Models with 1024x2x2 Separable Encoding Kernel

# [STATUS==RUNNING IN EULER] 3D CSPH 1024x2x2 - Compression=32x --> K=128
# [STATUS==RUNNING IN EULER] 3D CSPH 1024x2x2 - Compression=64x --> K=64
# [STATUS==RUNNING IN EULER] 3D CSPH 1024x2x2 - Compression=128x --> K=32

## Models with 1024x8x8 Separable Encoding Kernel

# # [STATUS==FINISHING RUN IN COMPOPTICS] 3D CSPH 1024x8x8 - Compression=32x --> k=2048
# # [STATUS==DONE RAN EULER] 3D CSPH 1024x8x8 - Compression=64x --> k=1024
# # [STATUS==STARTED IN EULER -- Still PENDING FINISH] 3D CSPH 1024x8x8 - Compression=128x --> k=512
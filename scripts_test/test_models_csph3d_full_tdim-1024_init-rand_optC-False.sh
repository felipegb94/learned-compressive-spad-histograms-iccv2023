#### Spatio-temporal CSPH with full kernel encodings that are NOT optimized
# The following models were trained using the commands in scripts_train/models_csph3d_full_tdim-1024_init-rand_optC-False.sh

# Question we want to answer:
#      - What is the importance of optimizing the encoding kernels?

## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

## Models with 1024x4x4 Encoding Kernel

# 3D CSPH 1024x4x4 - Compression=32x --> k=512
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=False-optC=False_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_193425
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0273.ckpt #  | 120 images== | 128 images==0.0515
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 1024x4x4 - Compression=64x --> k=256
model_name=DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=False-optC=False_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_162629
ckpt_id=epoch=28-step=100426-avgvalrmse=0.0340.ckpt #  | 120 images== | 128 images==0.1081
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 1024x4x4 - Compression=128x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=False-optC=False_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_074337
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0435.ckpt #  | 120 images== | 128 images==0.2382
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

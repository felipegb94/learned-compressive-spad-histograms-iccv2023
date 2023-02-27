#### Spatio-temporal CSPH with separable kernel encodings with tdim=256 where coding matrix is random
# The following models were trained using the commands in scripts_train/models_csph3d_separable_tdim-256_init-rand_optC-False.sh


## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min
# test_dataset=lindell2018_linospad

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

# Models with 256x4x4 Encoding Kernel (THESE MODELS WERE FROM test_models_csph3d_separable_tdim-varying_init-rand)
## 3D CSPH 256x4x4 - Compression=32x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=False-optC=False_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010306
ckpt_id=epoch=28-step=100424-avgvalrmse=0.0319.ckpt #  128 images==0.06622
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 3D CSPH 256x4x4 - Compression=64x --> k=64
model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt4_Rand-optCt=False-optC=False_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010420
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0433.ckpt #  128 images==0.15015
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 3D CSPH 256x4x4 - Compression=128x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=False-optC=False_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010420
ckpt_id=epoch=29-step=102157-avgvalrmse=0.0577.ckpt #  128 images==0.41653
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

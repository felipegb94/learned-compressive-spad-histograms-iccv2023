#### Spatio-temporal CSPH with FULL kernel encodings with tdim=256
# The following models were trained using the commands in scripts_train/models_csph3d_full_tdim-256_init-rand.sh

# Question we want to answer:
#      - Can spatio-temporal compressive histograms beat temporal only ones?

## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min
# test_dataset=lindell2018_linospad

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

## Models with 256x4x4 Encoding Kernel
## 3D CSPH 256x4x4 - Compression=32x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-21_172418
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0170.ckpt #  128 images==0.015426 | 128 images large depth (9m offset): 0.0952| 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 3D CSPH 256x4x4 - Compression=64x --> k=64
model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-21_172420
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0187.ckpt #  128 images==0.01937 | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 3D CSPH 256x4x4 - Compression=128x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-21_172357
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0207.ckpt #  128 images==0.02336 | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 3D CSPH 256x4x4 - Compression=256x --> k=16
model_name=DDFN_C64B10_CSPH3D/k16_down4_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-21_172357
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0234.ckpt #  128 images==0.02868 | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 3D CSPH 256x4x4 - Compression=512x --> k=8
model_name=DDFN_C64B10_CSPH3D/k8_down4_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-21_172357
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0271.ckpt #  128 images==0.035302 | 128 images large depth (7m offset):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Models with 256x2x2 Encoding Kernel

## 3D CSPH 256x2x2 - Compression=32x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down2_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-21_172357
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0166.ckpt #  128 images==0.014959 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 3D CSPH 256x2x2 - Compression=64x --> k=16
model_name=DDFN_C64B10_CSPH3D/k16_down2_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-21_172357
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0187.ckpt #  128 images==0.018961 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 3D CSPH 256x2x2 - Compression=128x --> k=8
model_name=DDFN_C64B10_CSPH3D/k8_down2_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-20_203738
ckpt_id=epoch=29-step=102157-avgvalrmse=0.0213.ckpt #  128 images==0.02488 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


# # ## Models with 256x8x8 Encoding Kernel [NOT RUN YET...]

## # 3D CSPH 256x8x8 - Compression=32x --> k=512
model_name=DDFN_C64B10_CSPH3D/k512_down8_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-20_203724
ckpt_id=epoch=28-step=99559-avgvalrmse=0.0207.ckpt #  128 images==0.020897 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## # 3D CSPH 256x8x8 - Compression=64x --> k=256
model_name=DDFN_C64B10_CSPH3D/k256_down8_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-20_203724
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0214.ckpt #  128 images==0.023828 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## # 3D CSPH 256x8x8 - Compression=128x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down8_Mt4_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_005700
ckpt_id=epoch=28-step=100424-avgvalrmse=0.0220.ckpt #  128 images==0.0255777 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
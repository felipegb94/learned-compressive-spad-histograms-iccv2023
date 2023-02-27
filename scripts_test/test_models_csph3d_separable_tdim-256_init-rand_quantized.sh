#### Spatio-temporal CSPH with separable kernel encodings with tdim=256
# The following models were trained using the commands in scripts_train/models_csph3d_separable_tdim-256_init-rand.sh

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

## Models with 256x4x4 Encoding Kernel (THESE MODELS WERE FROM test_models_csph3d_separable_tdim-varying_init-rand)
## 3D CSPH 256x4x4 - Compression=32x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174558
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0169.ckpt #  128 images==0.01628714 | 128 images large depth (9m offset): | 128 images large depth (7m offset): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 3D CSPH 256x4x4 - Compression=64x --> k=64
model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_175338
ckpt_id=epoch=27-step=96963-avgvalrmse=0.0185.ckpt #  128 images==0.0197465 | 128 images large depth (7m offset):  | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 3D CSPH 256x4x4 - Compression=128x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_175849
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0208.ckpt #  128 images==0.02417629 | 128 images large depth (7m offset):  | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true

## Models with 256x2x2 Encoding Kernel

## 3D CSPH 256x2x2 - Compression=32x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down2_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010637
ckpt_id=epoch=29-step=102157-avgvalrmse=0.0167.ckpt #  128 images==0.0156196 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 3D CSPH 256x2x2 - Compression=64x --> k=16
model_name=DDFN_C64B10_CSPH3D/k16_down2_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010637
ckpt_id=epoch=28-step=99559-avgvalrmse=0.0186.ckpt #  128 images==0.0194777 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 3D CSPH 256x2x2 - Compression=128x --> k=8
model_name=DDFN_C64B10_CSPH3D/k8_down2_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010637
ckpt_id=epoch=28-step=99559-avgvalrmse=0.0214.ckpt #  128 images==0.02499284 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true


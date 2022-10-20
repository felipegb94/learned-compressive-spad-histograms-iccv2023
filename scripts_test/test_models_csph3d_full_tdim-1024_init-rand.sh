#### Spatio-temporal CSPH with full kernel encodings
# The following models were trained using the commands in scripts_train/models_csph3d_full_tdim-1024_init-rand.sh

# Question we want to answer:
#      - Can spatio-temporal compressive histograms beat temporal only ones?

## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

## Models with 1024x4x4 Encoding Kernel

# # 3D CSPH 1024x4x4 - Compression=32x --> k=512
# model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-30_125727
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0172.ckpt #  | 120 images== | 128 images==0.01644
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # 3D CSPH 1024x4x4 - Compression=64x --> k=256
# model_name=DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174639
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0178.ckpt #  | 120 images== | 128 images==0.01749
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # 3D CSPH 1024x4x4 - Compression=128x --> k=128
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-26_045722
# ckpt_id=epoch=24-step=86574-avgvalrmse=0.0188.ckpt #  | 120 images== | 128 images==0.02035
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 1024x4x4 - Compression=256x --> k=64
model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-17_214734
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0198.ckpt #  | 120 images== | 128 images==0.0241
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 1024x4x4 - Compression=256x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-17_215320
ckpt_id=epoch=28-step=100426-avgvalrmse=0.0217.ckpt #  | 120 images== | 128 images==0.0280
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## Models with 1024x2x2 Encoding Kernel

# # 3D CSPH 1024x2x2 - Compression=32x --> k=128
# model_name=DDFN_C64B10_CSPH3D/k128_down2_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_022542
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0155.ckpt #  | 120 images== | 128 images==0.0175
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # 3D CSPH 1024x2x2 - Compression=64x --> k=64
# model_name=DDFN_C64B10_CSPH3D/k64_down2_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174720
# ckpt_id=epoch=29-step=102158-avgvalrmse=0.0162.ckpt #  | 120 images== | 128 images==0.0356
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # 3D CSPH 1024x2x2 - Compression=128x --> k=32
# model_name=DDFN_C64B10_CSPH3D/k32_down2_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-26_025352
# ckpt_id=epoch=28-step=100426-avgvalrmse=0.0172.ckpt #  | 120 images== | 128 images==0.02219
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# # ## Models with 1024x8x8 Encoding Kernel [NOT RUN YET...]

# # [STATUS==PENDING] 3D CSPH 1024x8x8 - Compression=32x --> k=2048
# # [STATUS==PENDING] 3D CSPH 1024x8x8 - Compression=64x --> k=1024
# # [STATUS==PENDING] 3D CSPH 1024x8x8 - Compression=128x --> k=512
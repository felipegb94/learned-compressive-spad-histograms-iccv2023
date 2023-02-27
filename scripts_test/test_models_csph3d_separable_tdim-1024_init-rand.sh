#### Separable CSPH3D models
# The models tested here were trained with the commands in scripts_train/models_csph3d_separable_tdim-1024_init-rand.sh

# Question we want to answer:
#      - Can separable encoding kernels perform the same as full?


## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=middlebury_largedepth
# test_dataset=middlebury_maskedhightimebins
# test_dataset=lindell2018_linospad_min
# test_dataset=lindell2018_linospad

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

# Models with 1024x4x4 Separable Encoding Kernel

# Re-Run 3D CSPH 1024x4x4 - Compression=32x --> k=512
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_rerun
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_141241
ckpt_id=epoch=28-step=100426-avgvalrmse=0.0156.ckpt #  128 images==0.0147 | 128 images large depth (7m offset)==0.1589 | 128 images masked tbins (9m offset): 0.10447
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# Re-Run 3D CSPH 1024x4x4 - Compression=64x --> k=256
model_name=DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_rerun
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_141241
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0164.ckpt #  128 images==0.0174 | 128 images large depth (7m offset)==0.1131 | 128 images masked tbins (9m offset): 0.1305
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# Re-Run 3D CSPH 1024x4x4 - Compression=128x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_rerun
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_193425
ckpt_id=epoch=23-step=83111-avgvalrmse=0.0176.ckpt #  128 images==0.0213 | 128 images large depth (7m offset)==0.7328 | 128 images masked tbins (9m offset): 0.14975
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# Re-Run 3D CSPH 1024x4x4 - Compression=256x --> k=64
model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_rerun
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-20_221556
ckpt_id=epoch=28-step=100426-avgvalrmse=0.0194.ckpt #  128 images==0.0334 | 128 images large depth (7m offset)==0.1430
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# Re-Run 3D CSPH 1024x4x4 - Compression=256x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_rerun
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-17_205501
ckpt_id=epoch=23-step=83111-avgvalrmse=0.0214.ckpt #  128 images==0.02840 | 128 images large depth (7m offset)==0.10022
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## Models with 1024x2x2 Separable Encoding Kernel

# # 3D CSPH 1024x2x2 - Compression=32x --> K=128
# model_name=DDFN_C64B10_CSPH3D/k128_down2_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_010753
# ckpt_id=epoch=26-step=93500-avgvalrmse=0.0153.ckpt #  128 images==0.0194 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # 3D CSPH 1024x2x2 - Compression=64x --> K=64
# model_name=DDFN_C64B10_CSPH3D/k64_down2_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_154023
# ckpt_id=epoch=27-step=95232-avgvalrmse=0.0160.ckpt #  128 images==0.0217 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # 3D CSPH 1024x2x2 - Compression=128x --> K=32
# model_name=DDFN_C64B10_CSPH3D/k32_down2_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_233248
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0175.ckpt #  128 images==0.0282 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## Models with 1024x8x8 Separable Encoding Kernel

# # 3D CSPH 1024x8x8 - Compression=32x --> k=2048
# model_name=DDFN_C64B10_CSPH3D/k2048_down8_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-05_122605
# ckpt_id=epoch=29-step=102158-avgvalrmse=0.0170.ckpt #  128 images==0.0410 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # 3D CSPH 1024x8x8 - Compression=32x --> k=2048
# model_name=DDFN_C64B10_CSPH3D/k2048_down8_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_rerun
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-12_192900
# ckpt_id=epoch=28-step=100426-avgvalrmse=0.0172.ckpt #  128 images==0.01668 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # 3D CSPH 1024x8x8 - Compression=64x --> k=1024
# model_name=DDFN_C64B10_CSPH3D/k1024_down8_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_081844
# ckpt_id=epoch=28-step=100426-avgvalrmse=0.0176.ckpt #  128 images==0.0201 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # 3D CSPH 1024x8x8 - Compression=128x --> k=512
# model_name=DDFN_C64B10_CSPH3D/k512_down8_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_183828
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0193.ckpt #  128 images==0.0213 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


######################### OLD MODELS ###################################
## Models with 1024x4x4 Separable Encoding Kernel

# # OLD 3D CSPH 1024x4x4 - Compression=32x --> k=512
# model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174558
# # ckpt_id=epoch=26-step=93500-avgvalrmse=0.0158.ckpt #  128 images==0.1351 | 128 images large depth (7m offset)==
# ckpt_id=epoch=20-step=70991-avgvalrmse=0.0161 #  128 images==0.0387 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # OLD 3D CSPH 1024x4x4 - Compression=64x --> k=256
# model_name=DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174558
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0167.ckpt #  128 images==0.0728 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # OLD 3D CSPH 1024x4x4 - Compression=128x --> k=128
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_044702
# ckpt_id=epoch=29-step=102158-avgvalrmse=0.0180.ckpt #  128 images==0.0296 | 128 images large depth (7m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

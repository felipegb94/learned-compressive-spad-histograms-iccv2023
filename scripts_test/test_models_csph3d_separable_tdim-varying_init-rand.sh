#### What is a good temporal block size
# The following commands train CSPH3D models with coding matrices that have spatio-temporal encoding kernels with dimensions 256x4x4, 64x4x4, 16x4x4 and whose time and spatial dimensions are separable
# Experimental configs:
#   - Encoding Kernels (Block Dims) = [256x4x4, 64x4x4, 16x4x4]
#   - Compression Levels = [32x, 64x, 128x]
#   - Encoding Type = separable
#   - Optimize Encoding Kernels = True
#   - tblock_init = [Rand]

# Question we want to answer:
#      - Can separable and small encoding kernels perform the same as full?

## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

## Models with 256x4x4 Encoding Kernel

# 3D CSPH 256x4x4 - Compression=32x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174558
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0169.ckpt #  | 120 images== | 128 images==0.01627
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 256x4x4 - Compression=64x --> k=64
model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_175338
ckpt_id=epoch=27-step=96963-avgvalrmse=0.0185.ckpt #  | 120 images== | 128 images==0.01969
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 256x4x4 - Compression=128x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_175849
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0208.ckpt #  | 120 images== | 128 images==0.0241
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Models with 64x4x4 Encoding Kernel

# 3D CSPH 64x4x4 - Compression=32x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt16_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_174123
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0186.ckpt #  | 120 images== | 128 images==0.0187
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 64x4x4 - Compression=64x --> k=16
model_name=DDFN_C64B10_CSPH3D/k16_down4_Mt16_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-03_192252
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0219.ckpt #  | 120 images== | 128 images==0.0228
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 64x4x4 - Compression=128x --> k=8
model_name=DDFN_C64B10_CSPH3D/k8_down4_Mt16_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_180043
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0270.ckpt #  | 120 images== | 128 images==0.0289
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Models with 16x4x4 Encoding Kernel

# 3D CSPH 16x4x4 - Compression=32x --> k=8
model_name=DDFN_C64B10_CSPH3D/k8_down4_Mt64_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-26_050833
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0251.ckpt #  | 120 images== | 128 images==0.0414
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 16x4x4 - Compression=64x --> k=4
model_name=DDFN_C64B10_CSPH3D/k4_down4_Mt64_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174639
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0333.ckpt #  | 120 images== | 128 images==0.193
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# 3D CSPH 16x4x4 - Compression=128x --> k=2
model_name=DDFN_C64B10_CSPH3D/k2_down4_Mt64_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174608
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0567.ckpt #  | 120 images== | 128 images==0.333
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
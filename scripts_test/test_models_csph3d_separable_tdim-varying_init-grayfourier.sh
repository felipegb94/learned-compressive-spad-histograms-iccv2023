#### Separable CSPH3D models with GrayFourier Initializations
# The following models were trained using the commands in scripts_train/models_csph3d_separable_tdim-varying_init-grayfourier.sh

# Question we want to answer:
#      - Can a good initialization improve performance

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min
test_dataset=lindell2018_linospad

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

## Models with 1024x4x4 Separable Encoding Kernel + HybridGrayFourier Init + Optimized tdim Codes

# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x4x4 - Compression=32x --> K=512
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_203156
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0168.ckpt #  | 120 images== | 128 images==0.02419
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x4x4 - Compression=64x --> K=256
model_name=DDFN_C64B10_CSPH3D/k256_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_202943
ckpt_id=epoch=26-step=91769-avgvalrmse=0.0173.ckpt #  | 120 images== | 128 images==0.01624
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x4x4 - Compression=128x --> K=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_193602
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0179.ckpt #  | 120 images== | 128 images==0.019174
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Models with 1024x4x4 Separable Encoding Kernel + HybridGrayFourier Init + NOT Optimized tdim Codes

# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x4x4 - Compression=32x --> K=512
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_204954
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0175.ckpt #  | 120 images== | 128 images==0.01693
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x4x4 - Compression=64x --> K=256
model_name=DDFN_C64B10_CSPH3D/k256_down4_Mt1_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_074345
ckpt_id=epoch=25-step=88306-avgvalrmse=0.0182.ckpt #  | 120 images== | 128 images==0.01758
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x4x4 - Compression=128x --> K=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_074345
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0189.ckpt #  | 120 images== | 128 images==0.02015
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Models with 1024x2x2 Separable Encoding Kernel + HybridGrayFourier Init + Optimized tdim Codes

# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x2x2 - Compression=32x --> K=128
model_name=DDFN_C64B10_CSPH3D/k128_down2_Mt1_HybridGrayFourier-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_161934
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0151.ckpt #  | 120 images== | 128 images==0.04748
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x2x2 - Compression=64x --> K=64
model_name=DDFN_C64B10_CSPH3D/k64_down2_Mt1_HybridGrayFourier-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_161934
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0155.ckpt #  | 120 images== | 128 images==0.02203
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x2x2 - Compression=128x --> K=32
model_name=DDFN_C64B10_CSPH3D/k32_down2_Mt1_HybridGrayFourier-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_161950
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0169.ckpt #  | 120 images== | 128 images==0.02152
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Models with 1024x2x2 Separable Encoding Kernel + HybridGrayFourier Init + NOT Optimized tdim Codes

# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x2x2 - Compression=32x --> K=128
model_name=DDFN_C64B10_CSPH3D/k128_down2_Mt1_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_162001
ckpt_id=epoch=28-step=98695-avgvalrmse=0.0151.ckpt #  | 120 images== | 128 images==0.01465
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x2x2 - Compression=64x --> K=64
model_name=DDFN_C64B10_CSPH3D/k64_down2_Mt1_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_184802
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0161.ckpt #  | 120 images== | 128 images==0.01927
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 1024x2x2 - Compression=128x --> K=32
model_name=DDFN_C64B10_CSPH3D/k32_down2_Mt1_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_022214
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0180.ckpt #  | 120 images== | 128 images==0.02431
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Models with 64x4x4 Separable Encoding Kernel + HybridGrayFourier Init + Optimized tdim Codes

# [STATUS==DONE RAN IN EULER] 3D CSPH 64x4x4 - Compression=32x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt16_HybridGrayFourier-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_184810
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0200.ckpt #  | 120 images== | 128 images==0.01928
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 64x4x4 - Compression=64x --> k=16
model_name=DDFN_C64B10_CSPH3D/k16_down4_Mt16_HybridGrayFourier-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_184810
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0234.ckpt #  | 120 images== | 128 images==0.02377
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 64x4x4 - Compression=128x --> k=8
model_name=DDFN_C64B10_CSPH3D/k8_down4_Mt16_HybridGrayFourier-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-06_205858
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0286.ckpt #  | 120 images== | 128 images==0.03142
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Models with 64x4x4 Separable Encoding Kernel + HybridGrayFourier Init + NOT Optimized tdim Codes

# [STATUS==DONE RAN IN EULER] 3D CSPH 64x4x4 - Compression=32x --> k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt16_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_074337
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0235.ckpt #  | 120 images== | 128 images==0.02658
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 64x4x4 - Compression=64x --> k=16
model_name=DDFN_C64B10_CSPH3D/k16_down4_Mt16_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_022214
ckpt_id=epoch=28-step=100426-avgvalrmse=0.0266.ckpt #  | 120 images== | 128 images==0.02859
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# [STATUS==DONE RAN IN EULER] 3D CSPH 64x4x4 - Compression=128x --> k=8
model_name=DDFN_C64B10_CSPH3D/k8_down4_Mt16_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_162629
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0314.ckpt #  | 120 images== | 128 images==0.0357
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

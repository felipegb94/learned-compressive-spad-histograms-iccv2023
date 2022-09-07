###### 
## This script tests pre-trained models that used a 3D Compressive Histogram with different normalization approaches
## All CSPH3D models here have the parameters:
##  * Encoding Kernel == Full 1024x4x4
##  * Encoding Kernel Init == Rand
##  * K == [512, 128] --> [32x, 128x] compression --> for K=128 we only have L2 and Linf per-pixel results

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps


################## K=512 Full 1024x4x4 ##################

# ## 32x compression (K=512 1024x4x4)
## No Normalization
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-10_162141
ckpt_id=epoch=28-step=98695-avgvalrmse=0.0171 # 0.0191 | 120 images==0.3567 | 128 images==0.395
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Zero-Mean=True | ZNCC=False | norm=none
model_name: DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-False_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name: csph3d_debug_generalization
model_dirpath: outputs/${.train_dataset}/${.experiment_name}/${.model_name}/run-complete_2022-08-30_125727
ckpt_id: epoch=29-step=103889-avgvalrmse=0.0173.ckpt # 0.0183 | 120 images== | 128 images==0.0257

## Zero-Mean=True | ZNCC=True | norm=none
model_name: DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name: csph3d_debug_generalization
model_dirpath: outputs/${.train_dataset}/${.experiment_name}/${.model_name}/run-complete_2022-08-30_125727
ckpt_id: epoch=29-step=103889-avgvalrmse=0.0172 # 0.0178 | 120 images== | 128 images==0.0171

## Zero-Mean=True | ZNCC=True | norm=LinfGlobal
model_name: DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name: csph3d_debug_generalization
model_dirpath: outputs/${.train_dataset}/${.experiment_name}/${.model_name}/run-complete_2022-08-30_232221
ckpt_id: epoch=29-step=103889-avgvalrmse=0.0175.ckpt # 0.0167 | 120 images== | 128 images==0.01605

## LinfGlobal (Per-image) Normalization (CSPH3Dv2)
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-24_144556
ckpt_id=epoch=24-step=84843-avgvalrmse=0.0177 # 0.0226 | 120 images== | 128 images==0.0337
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## L2 Per-Pixel Normalization (CSPH3Dv2)
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-L2/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2022-08-18_013924
ckpt_id=epoch=22-step=77917-avgvalrmse=0.0174 # 0.0190 | 120 images==0.05288 | 128 images==0.0508
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Linf Per-Pixel Normalization (CSPH3Dv2)
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-Linf/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2022-08-18_011732
ckpt_id=epoch=23-step=81380-avgvalrmse=0.0179 # 0.0206 | 120 images==0.0748 | 128 images==0.07122
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

################## K=128 Full 1024x4x4 ##################

## No Normalization
model_name=DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0
experiment_name=test_csph3d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-07-31_145411
ckpt_id=epoch=26-step=91769-avgvalrmse=0.0180 # 0.0240 | 120 images==0.177 | 128 images==0.169
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## L2 Per-Pixel Normalization (CSPH3Dv2)
model_name=DDFN_C64B10_CSPH3Dv2/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-L2/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2022-08-18_013924
ckpt_id=epoch=24-step=84843-avgvalrmse=0.0185 # 0.0215 | 120 images==0.0489 | 128 images==0.0471
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## Linf Per-Pixel Normalization (CSPH3Dv2)
model_name=DDFN_C64B10_CSPH3Dv2/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-Linf/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-18_011947
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0186 # 0.0219 | 120imgs==0.135 | 128 images==0.1285
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'






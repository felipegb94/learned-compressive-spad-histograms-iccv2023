#### What is the best 1D CSPH Baseline
# Models that were trained using scripts_train/models_csph1d_baselines.sh

# Questions we want to answer:
#      - Which is the best 1D CSPH baseline to compare with in the main paper?
#      - What is the importance of a well-designed coding matrix?

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

# #### Models with 256x1x1 encoding and k=2 (128x compression)
## 1D Temporal CSPH - tblock_init=TruncFourier)
model_name=DDFN_C64B10_CSPH3D/k2_down1_Mt4_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-24_025910
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0417.ckpt # | 128 imgs==0.78150 | 128 imgs large depth (7m offset):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=Rand)
model_name=DDFN_C64B10_CSPH3D/k2_down1_Mt4_Rand-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010932
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0587.ckpt # | 128 imgs==0.8842 | 128 imgs large depth (7m offset):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
model_name=DDFN_C64B10_CSPH3D/k2_down1_Mt4_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010932
ckpt_id=epoch=27-step=96961-avgvalrmse=0.0451.ckpt # | 128 imgs==0.4416 | 128 imgs large depth (7m offset):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# #### Models with 256x1x1 encoding and k=4 (64x compression)
## 1D Temporal CSPH - tblock_init=TruncFourier)
model_name=DDFN_C64B10_CSPH3D/k4_down1_Mt4_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_011009
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0203.ckpt # | 128 imgs==0.032059 | 128 imgs large depth (7m offset):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=Rand)
model_name=DDFN_C64B10_CSPH3D/k4_down1_Mt4_Rand-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_011009
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0393.ckpt # | 128 imgs==0.52002 | 128 imgs large depth (7m offset):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
model_name=DDFN_C64B10_CSPH3D/k4_down1_Mt4_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_011012
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0220.ckpt # | 128 imgs==0.03348 | 128 imgs large depth (7m offset):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

#### Models with 256x1x1 encoding and k=8 (32x compression)

## 1D Temporal CSPH - tblock_init=TruncFourier)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt4_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_011012
ckpt_id=epoch=29-step=102157-avgvalrmse=0.0173.ckpt # | 128 imgs==0.019917 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=Rand)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt4_Rand-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_011022
ckpt_id=epoch=29-step=102157-avgvalrmse=0.0306.ckpt # | 128 imgs==0.178766 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt4_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_011022
ckpt_id=epoch=27-step=96096-avgvalrmse=0.0173.ckpt # | 128 imgs==0.017677 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

#### What is the best 1D CSPH Baseline
# Models that were trained using scripts_train/models_csph1d_baselines.sh

# Questions we want to answer:
#      - Which is the best 1D CSPH baseline to compare with in the main paper?
#      - What is the importance of a well-designed coding matrix?

## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps


# #### Models at k=8 (128x compression)


#### Models at k=16 (64x compression)

## 1D Temporal CSPH - tblock_init=TruncFourier)
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-22_104615
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0162 #  | 120 images== | 128 images==0.0335
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=HybridGrayFourier)
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-28_223938
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0179.ckpt #  | 120 images== | 128 images==0.0255
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=CoarseHist)
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-29_064449
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0219.ckpt #  | 120 images== | 128 images==0.2045
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=Rand)
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_Rand-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-01_122510
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0336.ckpt #  | 120 images== | 128 images==0.1871
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)

#### Models at k=32 (32x compression)


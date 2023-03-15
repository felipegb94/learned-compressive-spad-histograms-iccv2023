#### What is the best 1D CSPH Baseline
# Models that were trained using scripts_train/models_csph1d_baselines.sh

# Questions we want to answer:
#      - Which is the best 1D CSPH baseline to compare with in the main paper?
#      - What is the importance of a well-designed coding matrix?

## If any command fails exit.
set -e 

## Dataset we are testing with
# test_dataset=middlebury
test_dataset=middlebury_largedepth
# test_dataset=middlebury_maskedhightimebins
# test_dataset=lindell2018_linospad_min
# test_dataset=lindell2018_linospad

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps


## CoarseHist K=16
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_coarsehist_20230306
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_162133
ckpt_id=epoch=26-step=93500-avgvalrmse=0.0218.ckpt # | 128 imgs==0.208692 | 128 imgs large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## CoarseHist K=32
model_name=DDFN_C64B10_CSPH3D/k32_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_coarsehist_20230306
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_162133
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0169.ckpt # | 128 imgs==0.101913 | 128 imgs large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## CoarseHist K=64
model_name=DDFN_C64B10_CSPH3D/k64_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_coarsehist_20230306
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_162133
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0145.ckpt # | 128 imgs==0.0403104 | 128 imgs large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## CoarseHist K=128
model_name=DDFN_C64B10_CSPH3D/k128_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_coarsehist_20230306
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_161556
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0136.ckpt # | 128 imgs==0.019057 | 128 imgs large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## CoarseHist K=256
model_name=DDFN_C64B10_CSPH3D/k256_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_coarsehist_20230306
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_162133
ckpt_id=epoch=28-step=98695-avgvalrmse=0.0132.ckpt # | 128 imgs==0.01155 | 128 imgs large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
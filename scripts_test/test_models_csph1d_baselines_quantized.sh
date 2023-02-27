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


#### Models at k=8 (128x compression)

## 1D Temporal CSPH - tblock_init=TruncFourier)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002521
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0193.ckpt # | 128 imgs==0.058306 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 1D Temporal CSPH - tblock_init=HybridGrayFourier)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-05_223706
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0193.ckpt # | 128 imgs==0.044216 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 1D Temporal CSPH - tblock_init=CoarseHist)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-05_191824
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0317.ckpt # | 128 imgs==0.10859 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 1D Temporal CSPH - tblock_init=Rand)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-05_110729
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0441.ckpt # | 128 imgs==0.65700 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-05_115838
ckpt_id=epoch=25-step=88306-avgvalrmse=0.0201.ckpt # | 128 imgs==0.076759 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true


### Models at k=16 (64x compression)

## 1D Temporal CSPH - tblock_init=TruncFourier)
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-22_104615
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0162 # | 128 imgs==0.033550 | 128 imgs large depth (7m offset): 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 1D Temporal CSPH - tblock_init=HybridGrayFourier)
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-28_223938
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0179.ckpt # | 128 imgs==0.0255593 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 1D Temporal CSPH - tblock_init=CoarseHist)
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-29_064449
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0219.ckpt # | 128 imgs==0.2045957 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 1D Temporal CSPH - tblock_init=Rand)
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_Rand-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-01_122510
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0336.ckpt # | 128 imgs==0.2417999 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true
## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002644
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0168.ckpt # | 128 imgs==0.0226053 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true


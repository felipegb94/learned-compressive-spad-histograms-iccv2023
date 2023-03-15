## Apply 8-bit quantization to the coding tensors
# Models without a coding tensor are still tested here so that their depth images are output to the correct folder for post-processing purposes

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


############ COARSE HISTOGRAMS

# ## CoarseHist K=16
# model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_coarsehist_20230306
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_162133
# ckpt_id=epoch=26-step=93500-avgvalrmse=0.0218.ckpt # | 128 imgs==0.208692 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true

# ## CoarseHist K=32
# model_name=DDFN_C64B10_CSPH3D/k32_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_coarsehist_20230306
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_162133
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0169.ckpt # | 128 imgs==0.101913 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true

# ## CoarseHist K=64
# model_name=DDFN_C64B10_CSPH3D/k64_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_coarsehist_20230306
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_162133
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0145.ckpt # | 128 imgs==0.0403104 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true

# ## CoarseHist K=128
# model_name=DDFN_C64B10_CSPH3D/k128_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_coarsehist_20230306
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_161556
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0136.ckpt # | 128 imgs==0.019057 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true

# ## CoarseHist K=256
# model_name=DDFN_C64B10_CSPH3D/k256_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_coarsehist_20230306
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-03-06_162133
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0132.ckpt # | 128 imgs==0.01155 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true

############ Compressive Histograms


# ### Models with 256x2x2 Encoding Kernel and TruncFourier tdim
# ## Compression=128x --> k=8
# model_name=DDFN_C64B10_CSPH3D/k8_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_20230218
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010108
# ckpt_id=epoch=29-step=101292-avgvalrmse=0.0208.ckpt # 128 images==0.027032 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true

### Models with 256x2x2 Encoding Kernel and TruncFourier tdim
## Compression=64x --> k=16
model_name=DDFN_C64B10_CSPH3D/k16_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010116
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0188.ckpt # 128 images==0.019815 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'  ++emulate_int8_quantization=true

# ## 1D Temporal CSPH - tblock_init=TruncFourier)
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002521
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0193.ckpt # | 128 imgs==0.058306 | 128 imgs large depth (7m offset): | 128 images masked tbins (9m):
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true

# ## 1D Temporal CSPH - tblock_init=TruncFourier)
# model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-22_104615
# ckpt_id=epoch=29-step=102158-avgvalrmse=0.0162 # | 128 imgs==0.033550 | 128 imgs large depth (7m offset): 128 images masked tbins (9m):
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"' ++emulate_int8_quantization=true





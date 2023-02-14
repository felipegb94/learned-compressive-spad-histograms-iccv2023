#### No Compression Baselines (Oracle models)
# Models that were trained using the `scripts_train/models_no_compression_baselines.sh` script

# Questions we want to answer:
#      - What hyperparameters should we train the baselines with?

## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=middlebury_largedepth
# test_dataset=middlebury_maskedhightimebins
# test_dataset=lindell2018_linospad
# test_dataset=lindell2018_linospad_min

## Dataset we trained the models with
train_dataset=modulo_nyuv2_64x64x1024_55ps
# train_dataset=nyuv2_64x64x1024_80ps


#### Models at k=8 (128x compression)

# # ## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=HybridGrayFourier)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=validate_new_modulo_dataset_20230207
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-08_090245
ckpt_id=epoch=02-step=8657-avgvalrmse=0.0492.ckpt # 72 images == | 128 images == 0.06494
# ckpt_id=epoch=03-step=13851-avgvalrmse=0.0465.ckpt # 72 images == | 128 images == 0.0575
# ckpt_id=epoch=07-step=25972-avgvalrmse=0.0450.ckpt # 72 images == | 128 images == 0.0625  
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=Rand + Opt Codes)


# ## [STATUS==PENDING] 3D CSPH 1024x4x4 - Compression=128x --> K=128
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230207
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-07_205141
# # ckpt_id=epoch=03-step=13851-avgvalrmse=0.0483.ckpt # 72 images ==  | 128 images == 0.06  
# ckpt_id=epoch=14-step=51944-avgvalrmse=0.0467.ckpt # 72 images ==  | 128 images == 0.15739  
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## [STATUS==PENDING] 3D CSPH 256x4x4 - Compression=128x --> K=32
# model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230207
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-10_095956
# ckpt_id=epoch=26-step=91769-avgvalrmse=0.0468.ckpt # 72 images ==  | 128 images == 0.02525  
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


# # Peng et al., baseline
# model_name=DDFN_C64B10/norm-none/loss-kldiv_tv-1e-05
# experiment_name=validate_new_modulo_dataset_20230207
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-08_221657
# # ckpt_id=epoch=02-step=10388-avgvalrmse=0.0563.ckpt # 72 images == | 128 images == 0.05623  
# ckpt_id=epoch=07-step=25972-avgvalrmse=0.0585.ckpt # 72 images == | 128 images == 0.0188  
# # ckpt_id=epoch=13-step=48481-avgvalrmse=0.0666.ckpt # 72 images == | 128 images == 0.020386  
# # ckpt_id=epoch=17-step=62333-avgvalrmse=0.0705.ckpt # 72 images == | 128 images == 0.02480  
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


############# Re-trained ORIGINAL MODELS
# # Re-trained GrayFourier 1024x1x1 - Compression=128x --> k=8
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230207
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-09_035005
# ckpt_id=epoch=18-step=64065-avgvalrmse=0.0196.ckpt #  128 images==0.04838
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# # Re-trained Learned 1024x1x1 - Compression=128x --> k=8
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230207
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-09_035008
# ckpt_id=epoch=20-step=70991-avgvalrmse=0.0200.ckpt #  128 images==0.11044
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# # Re-trained HybridGrayFourier 256x4x4 - Compression=128x --> k=8
# model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230207
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-09_191440
# ckpt_id=epoch=18-step=65796-avgvalrmse=0.0229.ckpt #  128 images== 0.02395
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# # Re-trained Learned 1024x1x1 - Compression=128x --> k=8
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230207
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-09_191440
# ckpt_id=epoch=18-step=65796-avgvalrmse=0.0186.ckpt #  128 images==0.02892
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

############# ORIGINAL MODELS


# # Re-Run 3D CSPH 1024x4x4 - Compression=128x --> k=128
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_rerun
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-07_193425
# # ckpt_id=epoch=23-step=83111-avgvalrmse=0.0176.ckpt #  128 images==0.0213 | 128 images large depth (7m offset)==0.7328 | 128 images masked tbins (9m offset): 0.14975
# # ckpt_id=epoch=14-step=50213-avgvalrmse=0.0186.ckpt #  128 images==0.02558 | 128 images large depth (7m offset)== | 128 images masked tbins (9m offset): 
# # ckpt_id=epoch=06-step=22509-avgvalrmse=0.0247.ckpt #  128 images==0.02574
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'



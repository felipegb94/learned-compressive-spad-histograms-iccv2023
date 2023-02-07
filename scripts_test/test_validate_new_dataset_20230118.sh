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
train_dataset=nyuv2_64x64x1024_55ps



#### Models at k=8 (128x compression)

# ## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=HybridGrayFourier)
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_dataset_20230118
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-01-18_231501
# ckpt_id=epoch=17-step=62333-avgvalrmse=0.0382.ckpt # 72 images ==  | 128 images == 0.060823 | 128 images large depth (offset 7m):  | 128 images masked tbins (9m): 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## [STATUS==PENDING] 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_dataset_20230118
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-01-18_231501
# ckpt_id=epoch=18-step=64065-avgvalrmse=0.0411.ckpt # 72 images == 0.5785 | 128 images ==  | 128 images large depth (offset 7m):  | 128 images masked tbins (9m): 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## [STATUS==PENDING] 3D CSPH 1024x4x4 - Compression=128x --> K=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=validate_new_dataset_20230118
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-01-18_231454
ckpt_id=epoch=15-step=55407-avgvalrmse=0.0396.ckpt # 72 images == 0.5785 | 128 images ==  | 128 images large depth (offset 7m):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'



# [STATUS==PENDING] 3D CSPH 256x4x4 - Compression=128x --> k=32

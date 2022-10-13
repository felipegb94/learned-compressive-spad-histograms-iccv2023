###### 
## This script tests pre-trained baseline models

## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=middlebury_lowsbr #
# test_dataset=middlebury_highsignal #  
# test_dataset=middlebury_narrowpulse #  
# test_dataset=middlebury_widepulse #  
# test_dataset=lindell2018_linospad_min


## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

##########################################
## Re-trained Peng et al., ECCV 2020 Baselines and Modified Model

## Peng et al., 2020 (Plain Deep Boosting w/out NL) --> Full trained model with gradient decay
model_name=DDFN_C64B10/loss-kldiv_tv-1e-5
experiment_name=baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-04-25_192521
ckpt_id=epoch=05-step=19910-avgvalrmse=0.0287 # 0.0180 | 0.0177 | 0.0203 | 128 images == 0.03095
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## 3D DB Plain Depth2Depth Model --> Full trained model with gradient decay
model_name=DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5
experiment_name=baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-02_085659
ckpt_id=epoch=05-step=19910-avgvalrmse=0.0357 # | 0.0239 | 0.0276
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## 3D DB Plain Depth2Depth Model (No TV) --> Full trained model with gradient decay
model_name=DDFN_C64B10_Depth2Depth/loss-kldiv_tv-0.0
experiment_name=baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-07_171658
ckpt_id=epoch=08-step=28569-avgvalrmse=0.0263 # | 0.0292
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## Peng et al., ECCV 2020 --> Full trained model with gradient decay
# # Middlebury-72=0.0181 | Middlebury-128 
# model_name=DDFN_C64B10_NL_original/loss-kldiv_tv-1e-5
# experiment_name=baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-04-20_185832
# ckpt_id=epoch=05-step=19910-avgvalrmse=0.0281
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## 3D Non-local DB Plain Depth2Depth Model --> Full trained model with gradient decay
# model_name=DDFN_C64B10_NL_Depth2Depth/loss-kldiv_tv-1e-5
# experiment_name=baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-04-22_134732
# ckpt_id=epoch=05-step=19045-avgvalrmse=0.0363 # | | 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


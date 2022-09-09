###### 
## This script tests pre-trained models that used a 3D Compressive Histogram with different total variation parameter
## All CSPH3D models here have the parameters:
##  * Encoding Kernel == Full 1024x4x4
##  * Encoding Kernel Init == Rand
##  * K == [512, 128] --> [32x, 128x] compression
##  * Normalization == LinfGlobal, i.e., using Linf of the full signal

function exit_if_last_cmd_failed(){ 
    if [[ $? = 0 ]]; then echo "success"
    else 
        echo "failure: $?"
        exit 1
    fi
}

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

################## K=512 Full 1024x4x4 ##################

## 0.0 TV + 1e-3LR
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-24_144556
ckpt_id=epoch=24-step=84843-avgvalrmse=0.0177.ckpt # 0.0226 | 120 images== | 128 images==0.0337
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

## 1e-5 TV + 1e-3LR
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-1e-05
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-23_231951
ckpt_id=epoch=01-step=5194-avgvalrmse=0.0417 #  0.0321 | 120 images== | 128 images==0.174
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

## 1e-5 TV + 3e-4LR
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-1e-05
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-26_040500
ckpt_id=epoch=00-step=3462-avgvalrmse=0.0412.ckpt #  0.104 | 120 images== | 128 images==0.744
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

## 1e-6 TV+ 3e-4LR
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-1e-06
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2022-08-24_145125
ckpt_id=epoch=22-step=77917-avgvalrmse=0.0185 # 0.0203 | 120 images== | 128 images==0.0485
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

## 3e-6 TV+ 3e-4LR
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-3e-06
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-26_040500
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0191 # 0.0196 | 120 images== | 128 images==0.0345
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed


################## K=128 Full 1024x4x4 ##################

## 1e-5 TV + 3e-4 LR
model_name=DDFN_C64B10_CSPH3Dv2/k128_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-1e-05
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-26_040500
ckpt_id=epoch=00-step=3462-avgvalrmse=0.0426 #  | 120imgs== | 128 images==0.334
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name exit_if_last_cmd_failed++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

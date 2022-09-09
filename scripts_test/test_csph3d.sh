###### 
## This script tests pre-trained models that used a 3D Compressive Histogram with various hyperparameter combinations. Originally all these models were trained mainly to validate that our CSPH3D implementation was correct
## Some of the models here are also tested in other test scripts

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

################## Variable Compression (Full Kernels CSPH3Dv1 No normalization) #################
##### codes=(1024x4x4) tblock_init=Rand encoding_type=full ##################

## 32x compression (K=512 1024x4x4)
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-10_162141
ckpt_id=epoch=28-step=98695-avgvalrmse=0.0171 # 0.0191 | 120 images==0.3567 | 128 images==0.395
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

## 64x compression (K=256 1024x4x4)
model_name=DDFN_C64B10_CSPH3Dv1/k256_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0
experiment_name=test_csph3d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-08-05_141013
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0175 # 0.0206 |
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

## 128x compression (K=128 1024x4x4)
model_name=DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0
experiment_name=test_csph3d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-07-31_145411
ckpt_id=epoch=26-step=91769-avgvalrmse=0.0180 # 0.0240 | 120 images==0.177 | 128 images==0.169
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

# ## 256x compression (K=64 1024x4x4)
# model_name=DDFN_C64B10_CSPH3Dv1/k64_down4_Mt1_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0
# experiment_name=test_csph3d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-07-25_212205
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0193 # 0.0254 |
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

################## Variable Compression (Full Kernels CSPH3Dv1 No normalization) #################
#### codes=(1024x4x4) tblock_init=Rand encoding_type=separable

## 32x compression (K=512 1024x4x4)
model_name=DDFN_C64B10_CSPH3Dv1/k512_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0
experiment_name=test_csph3d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-08-11_145814
ckpt_id=epoch=28-step=100426-avgvalrmse=0.0143 # 0.0206 | 120 images== | 128 images==0.156
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

# ## 64x compression (K=256 1024x4x4)
# model_name=DDFN_C64B10_CSPH3Dv1/k256_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0
# experiment_name=test_csph3d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-08-11_145814
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0150 # 0.0181 |
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

## 128x compression (K=128 1024x4x4)
model_name=DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0
experiment_name=test_csph3d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-08-11_145813
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0170 # 0.0335 | 120 images== | 128 images==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

# ## 256x compression (K=64 1024x4x4)
# model_name=DDFN_C64B10_CSPH3Dv1/k64_down4_Mt1_Rand-optCt=True-optC=True_separable/loss-kldiv_tv-0.0
# experiment_name=test_csph3d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-08-11_145814
# ckpt_id=epoch=27-step=95232-avgvalrmse=0.0185 # 0.0313 |
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

################## Variable Compression (Separable Kernels CSPH3Dv1 No normalization Init with GrayFourier) #################
#### codes=(1024x4x4) tblock_init=HybridGrayFourier encoding_type=separable

## 32x compression (K=512 1024x4x4)
model_name=DDFN_C64B10_CSPH3Dv1/k512_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0
experiment_name=test_csph3d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-08-10_013956
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0164 # 0.0163 | 120 images== | 128 images==0.0527
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

# ## 64x compression (K=256 1024x4x4)
# model_name=DDFN_C64B10_CSPH3Dv1/k256_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0
# experiment_name=test_csph3d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-08-10_014005
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0170 # 0.01849 |
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

## 128x compression (K=128 1024x4x4)
model_name=DDFN_C64B10_CSPH3Dv1/k128_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0
experiment_name=test_csph3d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-08-10_014115
ckpt_id=epoch=26-step=91769-avgvalrmse=0.0176 # 0.0195 |
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

# ## 256x compression (K=64 1024x4x4)
# model_name=DDFN_C64B10_CSPH3Dv1/k64_down4_Mt1_HybridGrayFourier-optCt=True-optC=True_separable/loss-kldiv_tv-0.0
# experiment_name=test_csph3d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-08-10_014223
# ckpt_id=epoch=25-step=90037-avgvalrmse=0.0187 # 0.0252 |
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed


################## Variable Compression Full and Small Kernels #################
#### codes=(1024x4x4) tblock_init=HybridGrayFourier encoding_type=separable

#### codes=(64x4x4) tblock_init=Rand encoding_type=full

## 32x compression --> mt=16, mr,mc=8, k=32
model_name=DDFN_C64B10_CSPH3Dv1/k32_down4_Mt16_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0
experiment_name=test_csph3d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-07-16_122403
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0146 #  0.01748 |  
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

# ## 64x compression
# model_name=DDFN_C64B10_CSPH3Dv1/k16_down4_Mt16_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0
# experiment_name=test_csph3d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-07-16_142109
# ckpt_id=epoch=24-step=84843-avgvalrmse=0.0169 # 0.02163 | 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

### codes=(128x4x4) tblock_init=Rand encoding_type=full

## 128x compression
model_name=DDFN_C64B10_CSPH3Dv1/k16_down4_Mt8_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0
experiment_name=test_csph3d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-07-13_193650
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0189 # 0.0248 | 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

# ## 256x compression
# model_name=DDFN_C64B10_CSPH3Dv1/k8_down4_Mt8_Rand-optCt=True-optC=True_full/loss-kldiv_tv-0.0
# experiment_name=test_csph3d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-07-13_171326
# ckpt_id=epoch=27-step=96963-avgvalrmse=0.0220 # 0.0319 | 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed






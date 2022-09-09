###### 
## This script tests pre-trained models that used a 1D Compressive Histogram with ZNCC upsampling

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

################## K = 8 (compression = 128x) ##################

## K=8 FourierGray 3D DB Plain CSPH1D Model (No TV | LR=1e-3) --> Full trained model with gradient decay
model_name=DDFN_C64B10_CSPH1D/k8_HybridFourierGray/loss-kldiv_tv-0.0
experiment_name=baselines_csph1d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-26_084337
ckpt_id=epoch=17-step=62333-avgvalrmse=0.0219 # 0.0403 | 0.0356
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

## K=8 HybridGrayFourier 3D DB Plain CSPH1D Model (No TV | LR=1e-3) --> Full trained model with gradient decay
model_name=DDFN_C64B10_CSPH1D/k8_HybridGrayFourier/loss-kldiv_tv-0.0
experiment_name=baselines_csph1d
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-27_181604
ckpt_id=epoch=19-step=67528-avgvalrmse=0.0201 # 0.0442 | 0.0320 | 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
exit_if_last_cmd_failed

# ################## K = 16 (compression = 64x) ##################

# ## K=16 HybridGrayFourier 3D DB Plain CSPH1D Model (No TV, LR=1e-3) --> Full trained model with gradient decay
# model_name=DDFN_C64B10_CSPH1D/k16_HybridGrayFourier/loss-kldiv_tv-0.0
# experiment_name=baselines_csph1d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-20_194518
# ckpt_id=epoch=19-step=69259-avgvalrmse=0.0184 # 0.029 | 0.0203
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

# ## K=16 TruncFourier 3D DB Plain CSPH1D Model (No TV | LR=1e-3) --> Full trained model with gradient decay
# model_name=DDFN_C64B10_CSPH1D/k16_TruncFourier/loss-kldiv_tv-0.0
# experiment_name=baselines_csph1d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-19_142207
# ckpt_id=epoch=19-step=69259-avgvalrmse=0.0170 # 0.0322
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

# ## K=16 CoarseHist 3D DB Plain CSPH1D Model (No TV) --> Full trained model with gradient decay
# model_name=DDFN_C64B10_CSPH1D/k16_CoarseHist/loss-kldiv_tv-0.0
# experiment_name=baselines_csph1d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-15_103103
# ckpt_id=epoch=09-step=34629-avgvalrmse=0.0257 #
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

# ################## K = 32 (compression = 32x) ##################

# ## K=32 HybridGrayFourier 3D DB Plain CSPH1D Model (No TV | LR=1e-3) --> Full trained model with gradient decay
# model_name=DDFN_C64B10_CSPH1D/k32_HybridGrayFourier/loss-kldiv_tv-0.0
# experiment_name=baselines_csph1d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-20_194601
# ckpt_id=epoch=19-step=69259-avgvalrmse=0.0152 # 0.0208 | 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

# ## K=32 TruncFourier 3D DB Plain CSPH1D Model (No TV | LR=1e-3) --> Full trained model with gradient decay
# model_name=DDFN_C64B10_CSPH1D/k32_TruncFourier/loss-kldiv_tv-0.0
# experiment_name=baselines_csph1d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-26_075946
# ckpt_id=epoch=18-step=64065-avgvalrmse=0.0150 # 0.0199
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

# ################## K = 64 (compression = 16x) ##################

# ## K=64 HybridGrayFourier 3D DB Plain CSPH1D Model (No TV) --> Full trained model with gradient decay
# model_name=DDFN_C64B10_CSPH1D/k64_HybridGrayFourier/loss-kldiv_tv-0.0
# experiment_name=baselines_csph1d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-18_050753
# ckpt_id=epoch=09-step=34629-avgvalrmse=0.0183 # 0.0179
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

# ## K=64 TruncFourier 3D DB Plain CSPH1D Model (No TV) --> Full trained model with gradient decay
# model_name=DDFN_C64B10_CSPH1D/k64_TruncFourier/loss-kldiv_tv-0.0
# experiment_name=baselines_csph1d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-17_121426
# ckpt_id=epoch=09-step=34629-avgvalrmse=0.0184 # 0.0177
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

# ## K=64 CoarseHist 3D DB Plain CSPH1D Model (No TV) --> Full trained model with gradient decay
# model_name=DDFN_C64B10_CSPH1D/k64_CoarseHist/loss-kldiv_tv-0.0
# experiment_name=baselines_csph1d
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-17_224020
# ckpt_id=epoch=09-step=32898-avgvalrmse=0.0198 # ~0.0413
# python test.py dataset=$test_dataset ++model_name=$model_name ++experiment_name=$experiment_name ++model_dirpath=$model_dirpath ++ckpt_id=$ckpt_id ++train_dataset=$train_dataset
# exit_if_last_cmd_failed

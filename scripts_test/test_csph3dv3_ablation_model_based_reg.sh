###### 
## This script tests pre-trained models that used a 3D Compressive Histogram with different approaches that we explored to regularize the learned coding matrix. These include:
##      *  Zero-mean Codes: Force all coding matrix rows to be
##      *  ZNCC: Apply Zero-norm normalization to the compressed histogram and also the coding matrix before the unfiltered backprojection step
##      *  Account IRF in Unfilt Backproj.: Smooth the temporal dimension of the coding matrix with the IRF before the unfiltered backrpojection step 
##      *  Smooth tdim codes: Smooth the temporal dimension of the coding matrix using the IRF. Do this before encoding.
##      *  LinfGlobal Normalization: After the unfiltered backprojection, find the Linf of the full 3D signal and divide by it
##
#### All CSPH3D models here have the parameters:
##  * Encoding Kernel == Full 1024x4x4
##  * Encoding Kernel Init == Rand
##  * K == [512] --> [32x] compression

## If any command fails exit.
set -e 

## Dataset we are testing with
# test_dataset=middlebury 
test_dataset=middlebury_lowsbr # [in prog]
# test_dataset=middlebury_highsignal # [pending] 
# test_dataset=middlebury_narrowpulse # [pending] 
# test_dataset=middlebury_widepulse # [pending] 
# test_dataset=nyuv2_min
# test_dataset=lindell2018_linospad_min

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

################## K=512 Full 1024x4x4 ##################

## norm=none | IRF=False | Zero-Mean=False | ZNCC=False | SmoothtdimC=False 
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-10_162141
ckpt_id=epoch=28-step=98695-avgvalrmse=0.0171 # 0.0191 | 120 images==0.3567 | 128 images==0.395--0.3949
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

## norm=LinfGlobal | IRF=False | Zero-Mean=False | ZNCC=False | SmoothtdimC=False 
model_name=DDFN_C64B10_CSPH3Dv2/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal/loss-kldiv_tv-0.0
experiment_name=csph3d_good_norm
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-24_144556
ckpt_id=epoch=24-step=84843-avgvalrmse=0.0177 # 0.0226 | 120 images== | 128 images==0.0337--0.0332
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

## norm=LinfGlobal | IRF=True | Zero-Mean=False | ZNCC=False | SmoothtdimC=False 
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal_irf-True_zn-False_zeromu-False_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_debug_generalization
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-28_230806
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0182 # 0.0222 | 120 images== | 128 images==0.0223
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

## norm=LinfGlobal | IRF=False | Zero-Mean=True | ZNCC=True | SmoothtdimC=False 
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-LinfGlobal_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_debug_generalization
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-30_232221
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0175.ckpt # 0.0167 | 120 images== | 128 images==0.01605
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

## norm=none | IRF=False | Zero-Mean=True | ZNCC=False | SmoothtdimC=False 
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-False_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_debug_generalization
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-30_125727
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0173.ckpt # 0.0183 | 120 images== | 128 images==0.0257 -- 0.0251
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

## norm=none | IRF=False | Zero-Mean=True | ZNCC=True | SmoothtdimC=False 
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_debug_generalization
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-30_125727
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0172 # 0.0178 | 120 images== | 128 images==0.0171 -- 0.01644
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

## norm=none | IRF=True | Zero-Mean=True | ZNCC=True | SmoothtdimC=False 
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-True_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_debug_generalization
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-31_122537
ckpt_id=epoch=28-step=100426-avgvalrmse=0.0171 # 0.01732 | 120 images== | 128 images==0.01737 -- 0.01676 --> 0.01789 using test set IRF
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

## norm=none | IRF=False | Zero-Mean=True | ZNCC=True | SmoothtdimC=True 
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-True/loss-kldiv_tv-0.0
experiment_name=csph3d_debug_generalization
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-09_003946
ckpt_id=epoch=26-step=93500-avgvalrmse=0.0152.ckpt #   | 120 images== | 128 images== -- 0.0169 -->  using test set iRF
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

## norm=none | IRF=True | Zero-Mean=True | ZNCC=True | SmoothtdimC=True 
model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_full_norm-none_irf-True_zn-True_zeromu-True_smoothtdimC-True/loss-kldiv_tv-0.0
experiment_name=csph3d_debug_generalization
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-08-30_232221
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0153 #  0.01601 | 120 images== | 128 images==0.0152 -- 0.0145 --> 0.0157 using test set iRF
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# exit_if_last_cmd_failed

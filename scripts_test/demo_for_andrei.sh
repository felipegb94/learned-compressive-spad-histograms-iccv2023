


#### HISTOGRAM IMAGE INPUT = 1024x32x32
## Dataset we are testing with
test_dataset=middlebury
## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps


# ############ K = 8, Kernel = 1024x1x1 | Compression = Kernel / K = 128 ############
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002521
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0193.ckpt # | 128 imgs==0.0583 | 128 imgs large depth (7m offset): 0.09609 | 128 images masked tbins (9m):  2.13846
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ### Models with 256x2x2 Encoding Kernel and TruncFourier tdim
# ## Compression=128x --> k=8
# model_name=DDFN_C64B10_CSPH3D/k8_down2_Mt4_TruncFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_20230218
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010108
# ckpt_id=epoch=29-step=101292-avgvalrmse=0.0208.ckpt # 128 images==0.0270| 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'



### Models with 256x2x2 Encoding Kernel and TruncFourier tdim
## Compression=128x --> k=8
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt4_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim-256_20230218
model_dirpath=outputs_fmu/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_011012
ckpt_id=epoch=29-step=102157-avgvalrmse=0.0173.ckpt # 128 images==0.0270| 128 images large depth (7m offset)== | 128 images masked tbins (9m offset)==
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# outputs_fmu/nyuv2_64x64x1024_80ps/csph3D_tdim-256_20230218/DDFN_C64B10_CSPH3D/k8_down1_Mt4_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0//checkpoints/

#### Spatio-temporal CSPH with separable kernel encodings with tdim=256
# The following models were trained using the commands in scripts_train/models_csph3d_separable_tdim-256_init-rand.sh

# Question we want to answer:
#      - Can spatio-temporal compressive histograms beat temporal only ones?

## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=lindell2018_linospad_min
# test_dataset=lindell2018_linospad

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps

# ## Models with 256x4x4 Encoding Kernel (THESE MODELS WERE FROM test_models_csph3d_separable_tdim-varying_init-rand)
# ## 3D CSPH 256x4x4 - Compression=32x --> k=128
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-25_174558
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0169.ckpt #  128 images==0.01627 | 128 images large depth (9m offset): 0.0952| 128 images large depth (7m offset): 0.07501 | 128 images masked tbins (9m): 0.0720
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 3D CSPH 256x4x4 - Compression=64x --> k=64
# model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_175338
# ckpt_id=epoch=27-step=96963-avgvalrmse=0.0185.ckpt #  128 images==0.01969 | 128 images large depth (7m offset): 0.07045 | 128 images masked tbins (9m): 0.13429
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 3D CSPH 256x4x4 - Compression=128x --> k=32
# model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-24_175849
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0208.ckpt #  128 images==0.0241 | 128 images large depth (7m offset): 0.0808 | 128 images masked tbins (9m): 0.20777
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 3D CSPH 256x4x4 - Compression=256x --> k=16
# model_name=DDFN_C64B10_CSPH3D/k16_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-17_205501
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0232.ckpt #  128 images==0.02820 | 128 images large depth (7m offset): 0.08616 | 128 images masked tbins (9m): 0.27017
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 3D CSPH 256x4x4 - Compression=512x --> k=8
# model_name=DDFN_C64B10_CSPH3D/k8_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-17_205501
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0273.ckpt #  128 images==0.03769 | 128 images large depth (7m offset):
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


# ## Models with 256x2x2 Encoding Kernel

# ## 3D CSPH 256x2x2 - Compression=32x --> k=32
# model_name=DDFN_C64B10_CSPH3D/k32_down2_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_20230218
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010637
# ckpt_id=epoch=29-step=102157-avgvalrmse=0.0167.ckpt #  128 images==0.0156220 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 3D CSPH 256x2x2 - Compression=64x --> k=16
# model_name=DDFN_C64B10_CSPH3D/k16_down2_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_20230218
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010637
# ckpt_id=epoch=28-step=99559-avgvalrmse=0.0186.ckpt #  128 images==0.019386 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 3D CSPH 256x2x2 - Compression=128x --> k=8
# model_name=DDFN_C64B10_CSPH3D/k8_down2_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3d_models_20230218
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010637
# ckpt_id=epoch=28-step=99559-avgvalrmse=0.0214.ckpt #  128 images==0.025026 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# # ## Models with 256x8x8 Encoding Kernel [NOT RUN YET...]

## # 3D CSPH 256x8x8 - Compression=32x --> k=512
model_name=DDFN_C64B10_CSPH3D/k512_down8_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-23_175206
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0180.ckpt #  128 images==0.0181255 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## # 3D CSPH 256x8x8 - Compression=64x --> k=256
model_name=DDFN_C64B10_CSPH3D/k256_down8_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010643
ckpt_id=epoch=29-step=103887-avgvalrmse=0.0188.ckpt #  128 images==0.0201290 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## # 3D CSPH 256x8x8 - Compression=128x --> k=128
model_name=DDFN_C64B10_CSPH3D/k128_down8_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3d_models_20230218
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-19_010643
ckpt_id=epoch=29-step=103022-avgvalrmse=0.0206.ckpt #  128 images==0.02389 | 128 images large depth (9m offset): | 128 images large depth (7m offset):  | 128 images masked tbins (9m): 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
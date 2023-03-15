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
train_dataset=nyuv2_64x64x1024_80ps



#### Peng et al. ECCV 2020 model

## [STATUS=DONE] LR=1e-4, TV=1e-5, Normalization=None (Same params as original paper)
model_name=DDFN_C64B10/norm-none/loss-kldiv_tv-1e-05
experiment_name=no_compression_baselines_lr-1e-4
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-16_063146
ckpt_id=epoch=06-step=22509-avgvalrmse=0.0261.ckpt # 72 images == 0.01700 | 128 images == 0.01756 | 128 images large depth (offset 7m): 0.08526 | 128 images masked tbins (9m): 0.01757
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## [STATUS=DONE] LR=1e-4, TV=1e-5, Normalization=Linf --> divide each histogram by its maximum
# model_name=DDFN_C64B10/norm-Linf/loss-kldiv_tv-1e-05
# experiment_name=no_compression_baselines_lr-1e-4
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-14_043918
# ckpt_id=epoch=03-step=13851-avgvalrmse=0.0280.ckpt # 72 images == 0.01744 | 128 images == 0.0301 | 128 images large depth (offset 7m):
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## [STATUS=DONE] LR=1e-4, TV=0.0, Normalization=Linf --> divide each histogram by its maximum
# model_name=DDFN_C64B10/norm-Linf/loss-kldiv_tv-0.0
# experiment_name=no_compression_baselines_lr-1e-4
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-13_225106
# ckpt_id=epoch=29-step=102158-avgvalrmse=0.0178.ckpt # 72 images == 0.01538 | 128 images ==0.0761 | 128 images large depth (offset 7m):
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # ## [STATUS=DONE] LR=1e-3, TV=1e-5, Normalization=Linf --> divide each histogram by its maximum
# # model_name=DDFN_C64B10/norm-Linf/loss-kldiv_tv-1e-05
# # experiment_name=no_compression_baselines_lr-1e-3
# # model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-13_051558
# # ckpt_id=epoch=00-step=1731-avgvalrmse=0.0266.ckpt # 72 images == 0.0178 | 128 images == 0.0185 | 128 images large depth (offset 7m):
# # python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# # ## [STATUS=DONE] LR=1e-3, TV=0.0, Normalization=Linf --> divide each histogram by its maximum
# # model_name=DDFN_C64B10/norm-Linf/loss-kldiv_tv-0.0
# # experiment_name=no_compression_baselines_lr-1e-3
# # model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-13_051558
# # ckpt_id=epoch=28-step=98695-avgvalrmse=0.0136.ckpt # 72 images == 0.0128 | 128 images == 0.0330 | 128 images large depth (offset 7m):
# # python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


# ### Argmax Compression + Peng et al. ECCV 2020 model

## [USE THIS ONE. FROM TWO RUNS THIS PERFORMED BEST] LR=1e-4, TV=1e-5 -- 3D DB Plain Depth2Depth Model --> Full trained model with gradient decay
model_name=DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-5
experiment_name=baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/2022-05-02_085659
ckpt_id=epoch=05-step=19910-avgvalrmse=0.0357 # | 0.0239 | 0.0276 | 72 images == 0.02727 | 128 images == 0.02720 | 128 images large depth (offset 7m): 0.1279 | 128 images masked tbins (9m): 0.02890
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## [STATUS=DONE] LR=1e-4, TV=1e-5
model_name=DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-05
experiment_name=no_compression_baselines_lr-1e-4
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-13_051620
ckpt_id=epoch=03-step=12120-avgvalrmse=0.0348.ckpt # 72 images == 0.03755 | 128 images == 0.03453 | 128 images large depth (offset 7m): 0.1219 | 128 images masked tbins (9m): 0.02848
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## [STATUS=DONE] LR=1e-4, TV=0.0
model_name=DDFN_C64B10_Depth2Depth/loss-kldiv_tv-0.0
experiment_name=no_compression_baselines_lr-1e-4
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-13_051623
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0249.ckpt # 72 images == 0.0349 | 128 images == 0.0383 | 128 images large depth (offset 7m): 0.0837 | 128 images masked tbins (9m): 0.04630
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## [STATUS=DONE] LR=1e-3, TV=1e-5
# model_name=DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-05
# experiment_name=no_compression_baselines_lr-1e-3
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-13_051558
# ckpt_id=epoch=01-step=5194-avgvalrmse=0.0347.ckpt # 72 images == 0.0356 | 128 images == 0.0366 | 128 images large depth (offset 7m):
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## [STATUS=DONE] LR=1e-3, TV=0.0
# model_name=DDFN_C64B10_Depth2Depth/loss-kldiv_tv-0.0
# experiment_name=no_compression_baselines_lr-1e-3
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-13_051558
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0212.ckpt # 72 images == 0.02189 | 128 images == 0.02428 | 128 images large depth (offset 7m):
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

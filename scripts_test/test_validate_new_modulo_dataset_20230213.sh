#### 

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

# ## [IN PROGRESS] Learned Separable 256x4x4
# ## k=32
# model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065801
# # ckpt_id=epoch=18-step=65796-avgvalrmse=0.0465.ckpt # 72 images == | 128 images == 0.024639
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0461.ckpt # 72 images == | 128 images == 0.024824
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## k=64
# model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065801
# # ckpt_id=epoch=20-step=72722-avgvalrmse=0.0433.ckpt # 72 images == | 128 images == 0.02054
# ckpt_id=epoch=28-step=100426-avgvalrmse=0.0427.ckpt # 72 images == | 128 images == 0.020208
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## k=128
model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=validate_new_modulo_dataset_20230213
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-17_024851
ckpt_id=epoch=19-step=67527-avgvalrmse=0.0409.ckpt # 72 images == | 128 images == 0.01693
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
outputs_fmu/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k128_down4_Mt4_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0//checkpoints/

# #### [STATUS==IN PROGRESS] HybridGrayFourier Separable 256x4x4
# ## k=32
model_name=DDFN_C64B10_CSPH3D/k32_down4_Mt4_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=validate_new_modulo_dataset_20230213
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-17_022729
ckpt_id=epoch=17-step=59736-avgvalrmse=0.0492.ckpt # 72 images == | 128 images == 0.0267902
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## k=64
# model_name=DDFN_C64B10_CSPH3D/k64_down4_Mt4_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065811
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0458.ckpt # 72 images == | 128 images == 0.02099
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## k=128
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt4_HybridGrayFourier-optCt=False-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065811
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0443.ckpt # 72 images == | 128 images ==  0.02150
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## [IN PROGRESS] Learned Separable 1024x4x4
# ## k=128
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065901
# # ckpt_id=epoch=19-step=67528-avgvalrmse=0.0439.ckpt # 72 images == | 128 images == 0.18305
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0435.ckpt # 72 images == | 128 images == 0.10632
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## k=256
# model_name=DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065901
# # ckpt_id=epoch=17-step=62333-avgvalrmse=0.0419.ckpt # 72 images == | 128 images == 0.26839
# ckpt_id=epoch=28-step=100426-avgvalrmse=0.0413.ckpt # 72 images == | 128 images == 0.27812
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## k=512
# model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065901
# # ckpt_id=epoch=17-step=60602-avgvalrmse=0.0402.ckpt # 72 images == | 128 images == 0.10331
# ckpt_id=epoch=25-step=88306-avgvalrmse=0.0399.ckpt # 72 images == | 128 images == 0.491647
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## [IN PROGRESS] Learned Separable 1024x4x4 with TV=1e-5 LRI=1e-4
# ## k=128
# ## k=256
# model_name=DDFN_C64B10_CSPH3D/k256_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-1e-05
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-14_065906
# ckpt_id=epoch=01-step=5194-avgvalrmse=0.0655.ckpt # 72 images == | 128 images == 0.06447
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## k=512
# model_name=DDFN_C64B10_CSPH3D/k512_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-1e-05
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-14_065906
# ckpt_id=epoch=03-step=12120-avgvalrmse=0.0644.ckpt # 72 images == | 128 images == 0.1950
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## NO Compression Baselines
# ## Peng et al., 2020
# model_name=DDFN_C64B10/norm-none/loss-kldiv_tv-1e-05
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-14_065906
# ckpt_id=epoch=03-step=12120-avgvalrmse=0.0575.ckpt # 72 images == | 128 images == 0.023274
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## Depth2Depth
# model_name=DDFN_C64B10_Depth2Depth/loss-kldiv_tv-1e-05
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-14_065906
# ckpt_id=epoch=04-step=17314-avgvalrmse=0.0679.ckpt # 72 images == | 128 images == 0.06537
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

## [IN PROGRESS] 1D Temporal CSPH - tblock_init=Rand
## k=8
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=validate_new_modulo_dataset_20230213
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065601
# ckpt_id=epoch=24-step=86574-avgvalrmse=0.0469.ckpt # 72 images == | 128 images == 0.46569
ckpt_id=epoch=29-step=103889-avgvalrmse=0.0469.ckpt # 72 images == | 128 images == 0.494878
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## k=16
model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=validate_new_modulo_dataset_20230213
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065716
ckpt_id=epoch=27-step=95232-avgvalrmse=0.0419.ckpt # 72 images == | 128 images == 0.12014 
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## k=32
model_name=DDFN_C64B10_CSPH3D/k32_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=validate_new_modulo_dataset_20230213
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_092750
ckpt_id=epoch=29-step=102158-avgvalrmse=0.0390.ckpt # 72 images == | 128 images == 0.05359
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## [IN PROGRESS] 1D Temporal CSPH - tblock_init=HybridGrayFourier
# ## k=8
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065438
# # ckpt_id=epoch=23-step=83111-avgvalrmse=0.0429.ckpt # 72 images == | 128 images == 0.05233
# ckpt_id=epoch=28-step=100426-avgvalrmse=0.0428.ckpt  # 72 images == | 128 images == 0.05535
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# outputs/modulo_nyuv2_64x64x1024_55ps/validate_new_modulo_dataset_20230213/DDFN_C64B10_CSPH3D/k8_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0/run-complete_2023-02-14_065438/checkpoints/
# ## k=16
# model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065545
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0409.ckpt # 72 images == | 128 images == 0.04601
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## k=32
# model_name=DDFN_C64B10_CSPH3D/k32_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=validate_new_modulo_dataset_20230213
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2023-02-14_065548
# # ckpt_id=epoch=24-step=86574-avgvalrmse=0.0373.ckpt # 72 images == | 128 images == 0.025307
# ckpt_id=epoch=29-step=102158-avgvalrmse=0.0372.ckpt # 72 images == | 128 images == 0.022041
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

##### RAN IN COMPOPTICS 
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-1e-05
# experiment_name=validate_new_modulo_dataset_20230213_compoptics
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-13_193042
# # ckpt_id=epoch=00-step=1731-avgvalrmse=0.0709.ckpt # 72 images == | 128 images == 0.18954
# ckpt_id=epoch=23-step=83111-avgvalrmse=0.1083.ckpt # 72 images == | 128 images == 1.28455
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

# ## lri=1e-3 and tv=1e-06
# model_name=DDFN_C64B10_CSPH3D/k128_down4_Mt1_Rand-optCt=True-optC=True_separable_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-1e-06
# experiment_name=validate_new_modulo_dataset_20230213_compoptics
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2023-02-15_221701
# ckpt_id=epoch=05-step=20777-avgvalrmse=0.0477.ckpt # 72 images == | 128 images == 0.0528
# ckpt_id=epoch=09-step=32898-avgvalrmse=0.0470.ckpt # 72 images == | 128 images == 0.10792
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
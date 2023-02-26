#### What is the best 1D CSPH Baseline
# Models that were trained using scripts_train/models_csph1d_baselines.sh

# Questions we want to answer:
#      - Which is the best 1D CSPH baseline to compare with in the main paper?
#      - What is the importance of a well-designed coding matrix?

## If any command fails exit.
set -e 

## Dataset we are testing with
test_dataset=middlebury
# test_dataset=middlebury_largedepth
# test_dataset=middlebury_maskedhightimebins
# test_dataset=lindell2018_linospad_min
# test_dataset=lindell2018_linospad

## Dataset we trained the models with
train_dataset=nyuv2_64x64x1024_80ps


# #### Models at k=4 (256x compression)

# ## 1D Temporal CSPH - tblock_init=TruncFourier)
# model_name=DDFN_C64B10_CSPH3D/k4_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-17_212101
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0245.ckpt # | 128 imgs==0.115 | 128 imgs large depth (7m offset):0.12844
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=HybridGrayFourier)
# model_name=DDFN_C64B10_CSPH3D/k4_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-17_212547
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0246.ckpt # | 128 imgs==0.113 | 128 imgs large depth (7m offset): 0.1279
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
# model_name=DDFN_C64B10_CSPH3D/k4_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-22_213847
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0258.ckpt # | 128 imgs==0.12787 | 128 imgs large depth (7m offset): 5.44686
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'

#### Models at k=8 (128x compression)

# ## 1D Temporal CSPH - tblock_init=TruncFourier)
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002521
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0193.ckpt # | 128 imgs==0.0583 | 128 imgs large depth (7m offset): 0.09609 | 128 images masked tbins (9m):  2.13846
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=HybridGrayFourier)
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-05_223706
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0193.ckpt # | 128 imgs==0.0442 | 128 imgs large depth (7m offset): 0.0857 | 128 images masked tbins (9m): 1.0409
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=CoarseHist)
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-05_191824
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0317.ckpt # | 128 imgs==0.1085 | 128 imgs large depth (7m offset): 0.4798 | 128 images masked tbins (9m): 0.7751
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=Rand)
# model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-05_110729
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0441.ckpt # | 128 imgs==0.5840 | 128 imgs large depth (7m offset): 2.8278 | 128 images masked tbins (9m): 0.4253
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
model_name=DDFN_C64B10_CSPH3D/k8_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
experiment_name=csph3D_tdim_baselines
model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-05_115838
ckpt_id=epoch=25-step=88306-avgvalrmse=0.0201.ckpt # | 128 imgs==0.08441 | 128 imgs large depth (7m offset): 0.24038 | 128 images masked tbins (9m): 1.3189
python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


# ### Models at k=16 (64x compression)

# ## 1D Temporal CSPH - tblock_init=TruncFourier)
# model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-22_104615
# ckpt_id=epoch=29-step=102158-avgvalrmse=0.0162 # | 128 imgs==0.0335 | 128 imgs large depth (7m offset):0.07662 | 128 images masked tbins (9m): 0.47248
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=HybridGrayFourier)
# model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-28_223938
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0179.ckpt # | 128 imgs==0.0255 | 128 imgs large depth (7m offset):   0.07356 | 128 images masked tbins (9m):  0.3888
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=CoarseHist)
# model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-09-29_064449
# ckpt_id=epoch=29-step=102158-avgvalrmse=0.0219.ckpt # | 128 imgs==0.2045 | 128 imgs large depth (7m offset): 0.22242 | 128 images masked tbins (9m): 1.3501
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=Rand)
# model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_Rand-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-01_122510
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0336.ckpt # | 128 imgs==0.1871 | 128 imgs large depth (7m offset): 1.0446 | 128 images masked tbins (9m): 0.3251
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
# model_name=DDFN_C64B10_CSPH3D/k16_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002644
# ckpt_id=epoch=29-step=102158-avgvalrmse=0.0168.ckpt # | 128 imgs==0.02207 | 128 imgs large depth (7m offset): 0.14843 | 128 images masked tbins (9m): 0.45495
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'


### Models at k=32 (32x compression)

# ## 1D Temporal CSPH - tblock_init=TruncFourier)
# model_name=DDFN_C64B10_CSPH3D/k32_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002644
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0146.ckpt # | 128 imgs==0.02145 | 128 imgs large depth (9m offset) = 0.0818 | 128 imgs large depth (7m offset): 0.06297 | 128 images masked tbins (9m): 0.2474
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=HybridGrayFourier)
# model_name=DDFN_C64B10_CSPH3D/k32_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002644
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0148.ckpt # | 128 imgs==0.01859 | 128 imgs large depth (9m offset) = 0.1351 | 128 imgs large depth (7m offset):  0.11113 | 128 images masked tbins (9m): 1.1300
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=CoarseHist)
# model_name=DDFN_C64B10_CSPH3D/k32_down1_Mt1_CoarseHist-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002644
# ckpt_id=epoch=29-step=103889-avgvalrmse=0.0168.ckpt # | 128 imgs==0.09975   | 128 imgs large depth (7m offset): 0.12912 | 128 images masked tbins (9m): 1.6793
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=Rand)
# model_name=DDFN_C64B10_CSPH3D/k32_down1_Mt1_Rand-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002551
# ckpt_id=epoch=28-step=98695-avgvalrmse=0.0257.ckpt # | 128 imgs==0.0821 | 128 imgs large depth (7m offset): 0.5508 | 128 images masked tbins (9m): 0.1353
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=Rand + Opt Codes)
# model_name=DDFN_C64B10_CSPH3D/k32_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-04_002551
# ckpt_id=epoch=26-step=91769-avgvalrmse=0.0151.ckpt # | 128 imgs==0.02983 | 128 imgs large depth (9m offset) = 0.80546 | 128 imgs large depth (7m offset): 0.22598 | 128 images masked tbins (9m): 0.1602
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'



#### Models at k=2 (512x compression) --> NOTE: k=2 models should not work because the zero_norm operation require k>=3 

# ## 1D Temporal CSPH - tblock_init=TruncFourier)
# model_name=DDFN_C64B10_CSPH3D/k2_down1_Mt1_TruncFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-complete_2022-10-17_210248
# ckpt_id=epoch=25-step=90037-avgvalrmse=0.0660.ckpt # | 128 imgs==1.267
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=HybridGrayFourier)
# model_name=DDFN_C64B10_CSPH3D/k2_down1_Mt1_HybridGrayFourier-optCt=False-optC=False_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2022-10-17_221621
# ckpt_id=epoch=17-step=60602-avgvalrmse=0.0664.ckpt # | 128 imgs==1.313
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
# ## 1D Temporal CSPH - tblock_init=Rand + Opt Codes
# model_name=DDFN_C64B10_CSPH3D/k2_down1_Mt1_Rand-optCt=True-optC=True_csph1d_norm-none_irf-False_zn-True_zeromu-True_smoothtdimC-False/loss-kldiv_tv-0.0
# experiment_name=csph3D_tdim_baselines
# model_dirpath=outputs/${train_dataset}/${experiment_name}/${model_name}/run-latest_2022-10-17_221709
# ckpt_id=epoch=18-step=64065-avgvalrmse=0.2630.ckpt # | 128 imgs==5.791 (Not converged)
# python test.py dataset='"'$test_dataset'"' ++model_name='"'$model_name'"' ++experiment_name=$experiment_name ++model_dirpath='"'$model_dirpath'"' ++ckpt_id='"'$ckpt_id'"' ++train_dataset='"'$train_dataset'"'
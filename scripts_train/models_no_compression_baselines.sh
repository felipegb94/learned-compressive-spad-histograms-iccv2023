#### No Compression Baseline Models
# The following commands train baselines that do no compression and use the same backbone 3D CNN model as the CSPH3D models.

#### Peng et al. ECCV 2020 model

# ## [STATUS=RUNNING IN EULER] LR=1e-4, TV=1e-5, Normalization=None (Same params as original paper)
# python train.py ++experiment=no_compression_baselines_lr-1e-4 model=DDFN_C64B10 ++model.model_params.input_norm=none ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-4

# ## [STATUS=DONE] LR=1e-4, TV=1e-5, Normalization=Linf --> divide each histogram by its maximum
# python train.py ++experiment=no_compression_baselines_lr-1e-4 model=DDFN_C64B10 ++model.model_params.input_norm=Linf ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-4

# ## [STATUS=DONE] LR=1e-4, TV=0.0, Normalization=Linf --> divide each histogram by its maximum
# python train.py ++experiment=no_compression_baselines_lr-1e-4 model=DDFN_C64B10 ++model.model_params.input_norm=Linf ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=0.0 ++train_params.lri=1e-4

# ## [STATUS=DONE] LR=1e-3, TV=1e-5, Normalization=Linf --> divide each histogram by its maximum
# python train.py ++experiment=no_compression_baselines_lr-1e-3 model=DDFN_C64B10 ++model.model_params.input_norm=Linf ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-3

# ## [STATUS=DONE] LR=1e-3, TV=0.0, Normalization=Linf --> divide each histogram by its maximum
# python train.py ++experiment=no_compression_baselines_lr-1e-3 model=DDFN_C64B10 ++model.model_params.input_norm=Linf ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=0.0 ++train_params.lri=1e-3

## [STATUS=PENDING] LR=1e-4, TV=1e-5, Normalization=LinfGlobal --> divide each histogram by its maximum
python train.py ++experiment=no_compression_baselines_lr-1e-4 model=DDFN_C64B10 ++model.model_params.input_norm=LinfGlobal ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-4

## [STATUS=PENDING] LR=1e-4, TV=0.0, Normalization=LinfGlobal --> divide each histogram by its maximum
python train.py ++experiment=no_compression_baselines_lr-1e-4 model=DDFN_C64B10 ++model.model_params.input_norm=LinfGlobal ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=0.0 ++train_params.lri=1e-4

## [STATUS=PENDING] LR=1e-3, TV=1e-5, Normalization=LinfGlobal --> divide each histogram by its maximum
python train.py ++experiment=no_compression_baselines_lr-1e-3 model=DDFN_C64B10 ++model.model_params.input_norm=LinfGlobal ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-3

## [STATUS=PENDING] LR=1e-3, TV=0.0, Normalization=LinfGlobal --> divide each histogram by its maximum
python train.py ++experiment=no_compression_baselines_lr-1e-3 model=DDFN_C64B10 ++model.model_params.input_norm=LinfGlobal ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=0.0 ++train_params.lri=1e-3


#### Argmax Compression + Peng et al. ECCV 2020 model

# ## [STATUS=DONE] LR=1e-4, TV=1e-5
# python train.py ++experiment=no_compression_baselines_lr-1e-4 model=DDFN_C64B10_Depth2Depth ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-4

# ## [STATUS=DONE] LR=1e-4, TV=0.0
# python train.py ++experiment=no_compression_baselines_lr-1e-4 model=DDFN_C64B10_Depth2Depth ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=0.0 ++train_params.lri=1e-4

# ## [STATUS=DONE] LR=1e-3, TV=1e-5
# python train.py ++experiment=no_compression_baselines_lr-1e-3 model=DDFN_C64B10_Depth2Depth ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=1e-5 ++train_params.lri=1e-3

# ## [STATUS=DONE] LR=1e-3, TV=0.0
# python train.py ++experiment=no_compression_baselines_lr-1e-3 model=DDFN_C64B10_Depth2Depth ++train_params.epoch=30 ++random_seed=1235 ++train_params.batch_size=4 ++resume_train=true ++train_params.p_tv=0.0 ++train_params.lri=1e-3

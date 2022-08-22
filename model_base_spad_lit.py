'''
	Base lightning model that contains the share loss functions and training parameters across all models
'''

#### Standard Library Imports
import os
import logging

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import pytorch_lightning as pl
import torchvision
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from losses import criterion_L2, criterion_L1, criterion_KL, criterion_RMSE, criterion_TV
from research_utils.shared_constants import TWOPI, EPSILON
from research_utils.np_utils import calc_mean_percentile_errors
import tof_utils

def make_zeromean_normalized_bins(bins):
	return (2*bins) - 1

def normdepth2phasor(d):
	phase = TWOPI*d
	cos = torch.cos(phase)
	sin = torch.sin(phase)
	return torch.cat((cos, sin), dim=1)

def hist2rec_soft_argmax(hist_img):
	smax = torch.nn.Softmax2d() 
	weights = Variable(torch.linspace(0, 1, steps=hist_img.size()[1]).unsqueeze(1).unsqueeze(1)).to(hist_img.device)
	weighted_smax = weights * smax(hist_img)
	soft_argmax = weighted_smax.sum(1).unsqueeze(1)
	return soft_argmax

class LITBaseSPADModel(pl.LightningModule):
	def __init__(self, 
		backbone_net,
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		data_loss_id = 'kldiv',
		):
		
		super(LITBaseSPADModel, self).__init__()
		
		self.lsmx = torch.nn.LogSoftmax(dim=-3)
		self.smx = torch.nn.Softmax(dim=-3)
		self.print_logger = logging.getLogger(__name__)


		# Train hyperparams		
		self.init_lr = init_lr
		self.lr_decay_gamma = lr_decay_gamma
		self.p_tv = p_tv
		self.data_loss_id = data_loss_id

		self.backbone_net = backbone_net

		self.example_input_array = torch.randn([1, 1, 1024, 32, 32])

		self.test_rmse_all = []

		self.save_hyperparameters(ignore=['backbone_net'])

	def forward(self, x):
		# use forward for inference/predictions
		out = self.backbone_net(x)
		return out

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), self.init_lr)
		lr_scheduler = {
			'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay_gamma, verbose=False)
			, 'name': 'epoch/Adam-lr' # Name for logging in tensorboard (used by lr_monitor callback)
		}
		return [optimizer], [lr_scheduler]

	def get_input_data(self, sample):
		return sample["spad"]
	
	def forward_wrapper(self, sample):
		# Input the correct type of data to the model which should output the recovered histogram and depths
		input_data = self.get_input_data(sample)
		M_mea_re, rec = self(input_data)
		return M_mea_re, rec

	def rec2depth(self, rec):
		'''
			For some models the reconstructed signal are not depths 
		'''
		return rec

	def compute_losses(self, sample, M_mea_re, rec):
		# load data and compute losses
		M_gt = sample["rates"]
		dep = sample["bins"]
		# Get recovered depths normalized between 0,1
		dep_re = self.rec2depth(rec)
		# Normalize
		M_mea_re_lsmx = self.lsmx(M_mea_re).unsqueeze(1)
		# Compute metrics
		loss_kl = criterion_KL(M_mea_re_lsmx, M_gt)
		loss_tv = criterion_TV(dep_re)
		loss_l1 = criterion_L1(dep_re, dep)
		loss_l2 = criterion_L2(dep_re, dep)
		loss_phasorl1 = criterion_L1(normdepth2phasor(dep_re), normdepth2phasor(dep))
		loss_emd = criterion_L1(torch.cumsum(M_gt, dim=-3), torch.cumsum(M_mea_re.unsqueeze(1), dim=-3))
		loss_rmse = criterion_RMSE(dep_re, dep)

		if(self.data_loss_id == 'kldiv'):
			# For the first 100 steps only consider the data loss term
			if(self.global_step < 500): loss = loss_kl
			else: loss = loss_kl + self.p_tv*loss_tv
			# loss = loss_kl + self.p_tv*loss_tv
		elif(self.data_loss_id == 'L1'):
			loss = loss_l1 + self.p_tv*loss_tv
		elif(self.data_loss_id == 'L2'):
			loss = loss_l2 + self.p_tv*loss_tv
		elif(self.data_loss_id == 'EMD'):
			loss = loss_emd + self.p_tv*loss_tv
		elif(self.data_loss_id == 'PhasorL1'):
			loss = loss_phasorl1 + self.p_tv*loss_tv
		elif(self.data_loss_id == 'DirectPhasorL1'):
			loss_directphasorl1 = criterion_L1(rec, normdepth2phasor(dep))
			loss = loss_directphasorl1 + self.p_tv*loss_tv
		elif(self.data_loss_id == 'FFTHistL1'):
			fft_gt_hist = torch.fft.rfft(M_gt.squeeze(1), dim=-3)
			loss_real_l1 = criterion_L1(fft_gt_hist.real, rec[:, 0::2, :])
			loss_imag_l1 = criterion_L1(fft_gt_hist.imag, rec[:, 1::2, :])
			loss = loss_real_l1 + loss_imag_l1 + self.p_tv*loss_tv 
		elif(self.data_loss_id == 'DispL1'):
			loss_displ1 = criterion_L1(rec, 1 / (dep + EPSILON))
			loss = loss_displ1 + self.p_tv*loss_tv
		else:
			assert(False), "Invalid loss id"

		return {'loss': loss, 
				'loss_kl': loss_kl, 
				'loss_l1': loss_l1, 
				'loss_l2': loss_l2, 
				'loss_emd': loss_emd, 
				'loss_tv': loss_tv, 
				'rmse': loss_rmse}

	def training_step(self, sample, batch_idx):
		# Forward pass
		M_mea_re, rec = self.forward_wrapper(sample)
		# Compute Losses
		losses = self.compute_losses(sample, M_mea_re, rec)
		loss = losses['loss']
		loss_l1 = losses['loss_l1']
		loss_l2 = losses['loss_l2']
		loss_kl = losses['loss_kl']
		loss_tv = losses['loss_tv']
		rmse = losses['rmse']

		# Log to logger (i.e., tensorboard), if you want it to be displayed at progress bar, use prog_bar=True
		self.log_dict(
			{
				"loss/train": loss
				, "rmse/train": rmse
				, "train_losses/kldiv": loss_kl
				, "train_losses/l1": loss_l1
				, "train_losses/l2": loss_l2
				, "train_losses/tv": self.p_tv*loss_tv				
			}
			# , prog_bar=True
		)

		# # Log some images every 500 training steps
		# if((batch_idx % 20 == 0)):
		# 	self.log_depth_img(gt_dep=sample["bins"], rec_dep=dep_re, img_title_prefix='Train - ')
		
		if((batch_idx % 250) == 0):
			self.print_logger.info("Train - Epoch {} - Batch {} - Step {}".format(self.current_epoch, batch_idx, self.global_step))
			self.print_logger.info("    RMSE: {:.4f}, L1: {:.4f}, KL: {:.4f}, TV: {:.4f}".format(rmse, loss_l1, loss_kl, self.p_tv*loss_tv))
			
		
		return {'loss': loss}

	def validation_step(self, sample, batch_idx):
		# Forward pass
		M_mea_re, rec = self.forward_wrapper(sample)
		dep_re = self.rec2depth(rec)
		# Compute Losses
		val_losses = self.compute_losses(sample, M_mea_re, rec)
		val_loss = val_losses['loss']
		val_rmse = val_losses['rmse']
		val_loss_l1 = val_losses['loss_l1']
		val_loss_kl = val_losses['loss_kl']
		val_loss_tv = val_losses['loss_tv']

		# Log the losses
		self.log("rmse/avg_val", val_rmse, prog_bar=True)
		# Important NOTE: Newer version of lightning accumulate the val_loss for each batch and then take the mean at the end of the epoch
		self.log_dict(
			{
				"loss/avg_val": val_loss
			}
		)
		# Return depths
		dep = sample["bins"]
		
		if((batch_idx % 300) == 0):
			self.print_logger.info("Validation - Epoch {} - Batch {} - Step {}".format(self.current_epoch, batch_idx, self.global_step))
			self.print_logger.info("    RMSE: {:.4f}, L1: {:.4f}, KL: {:.4f}, TV: {:.4f}".format(val_rmse, val_loss_l1, val_loss_kl, self.p_tv*val_loss_tv))

		
		return {'dep': dep, 'dep_re': dep_re}

	def test_step(self, sample, batch_idx):

		# Forward pass
		M_mea_re, rec = self.forward_wrapper(sample)

		# crop the recovered histogram image and compute rec again
		# we do this because if we test with inputs that are not divisible by the encoding kernel dimensions then they were padded and here we want to remove the padding so that dimensions will match the original input dimensions
		# NOTE: right now this is not done during training and validation but eventually it should be done
		M_gt = sample["rates"] # use M_gt to figure out the original input dimensions
		M_mea_re = M_mea_re[...,0:M_gt.shape[-3],0:M_gt.shape[-2],0:M_gt.shape[-1]]
		# recompute rec
		rec = hist2rec_soft_argmax(M_mea_re)

		# Compute depths
		dep_re = self.rec2depth(rec)

		# Compute Losses
		test_losses = self.compute_losses(sample, M_mea_re, rec)
		test_loss = test_losses['loss']
		test_rmse = test_losses['rmse']

		## Save some model outputs
		# Access dataloader to get some metadata for computation
		dataloader_idx = 0 # If multiple dataloaders are available you need to change this using the input args to the test_step function
		curr_dataloader = self.trainer.test_dataloaders[dataloader_idx]
		# Get tof params to compute depths
		tres = curr_dataloader.dataset.tres_ps*1e-12 
		nt = M_mea_re.shape[-3]
		tau = nt*tres
		### Save model outputs in a folder with the dataset name and with a filename equal to the train data filename
		# First get dataloader to generate the data ids
		spad_data_ids = []
		for idx in sample['idx']:
			spad_data_ids.append(curr_dataloader.dataset.get_spad_data_sample_id(idx))
		out_rel_dirpath = os.path.dirname(spad_data_ids[0])
		if(not os.path.exists(out_rel_dirpath)):
			os.makedirs(out_rel_dirpath, exist_ok=True)
		for i in range(dep_re.shape[0]):
			out_data_fpath = spad_data_ids[i]
			np.savez(out_data_fpath, dep_re=dep_re[i,:].cpu().numpy())

		# Load GT depths
		dep = sample["bins"]

		# Compute depths and RMSE on depths
		rec_depths = tof_utils.bin2depth(dep_re*nt, num_bins=nt, tau=tau)
		gt_depths = tof_utils.bin2depth(dep*nt, num_bins=nt, tau=tau)

		# the following two lines give the same result
		# depths_rmse = torch.sqrt(torch.mean((rec_depths - gt_depths)**2))
		depths_mse = criterion_L2(rec_depths, gt_depths)
		depths_rmse = criterion_RMSE(rec_depths, gt_depths)
		depths_mae = criterion_L1(rec_depths, gt_depths)

		percentiles = [0.5, 0.75, 0.95, 0.99]
		(mean_percentile_errs, _) = calc_mean_percentile_errors(np.abs(rec_depths.cpu().numpy()-gt_depths.cpu().numpy()), percentiles=percentiles)

		# Important NOTE: Newer version of lightning accumulate the test_loss for each batch and then take the mean at the end of the epoch
		# Log results
		self.log_dict(
			{
				"loss/avg_test": test_loss
				, "rmse/avg_test": test_rmse
				, "depths/test_rmse": depths_rmse
				, "depths/test_mae": depths_mae
				, "depths/test_mse": depths_mse
				, "depths/test_mean_abs_perc{:.2f}".format(percentiles[0]): mean_percentile_errs[0]
				, "depths/test_mean_abs_perc{:.2f}".format(percentiles[1]): mean_percentile_errs[1]
				, "depths/test_mean_abs_perc{:.2f}".format(percentiles[2]): mean_percentile_errs[2]
				, "depths/test_mean_abs_perc{:.2f}".format(percentiles[3]): mean_percentile_errs[3]
			}
		)
		return {'dep': dep, 'dep_re': dep_re}

	def log_depth_img(self, gt_dep, rec_dep, img_title_prefix):
		n_img_per_row = gt_dep.shape[0]
		# NOTE: By setting it to global step, we will log more images inside tensorboard, which may require more space
		# If we set global_step to a constant, we will keep overwriting the images.
		grid1 = torchvision.utils.make_grid(gt_dep, nrow=n_img_per_row, value_range=(0,1))
		self.logger.experiment.add_image(img_title_prefix + 'GT Depths', grid1, global_step=self.global_step)
		grid2 = torchvision.utils.make_grid(rec_dep, nrow=n_img_per_row, value_range=(0,1))
		self.logger.experiment.add_image(img_title_prefix + 'Rec. Depths', grid2, global_step=self.global_step)
		## NOTE: The following pause statement helps avoid a random segfault that happens when logging the images inside 
		# training step
		plt.pause(0.1)

	def validation_epoch_end(self, outputs):
		'''
			Important NOTE: In newer lightning versions, for single value metrix like val_loss, we can just add them to the log_dict at val_step
			and lightning will aggregate them correctly.  
		'''
		
		# Stack some of the images from the outputs
		dep = outputs[-1]['dep']
		dep_re = outputs[-1]['dep_re']
		n_samples = min(3, len(outputs))
		dep_all = torch.zeros((n_samples, 1, dep.shape[-2], dep.shape[-1])).type(dep.dtype)
		dep_re_all = torch.zeros((n_samples, 1, dep_re.shape[-2], dep_re.shape[-1])).type(dep_re.dtype)
		for i in range(n_samples):
			dep_all[i,:] = outputs[i]['dep'][0,:] # Grab first img in batch
			dep_re_all[i,:] = outputs[i]['dep_re'][0,:]
		self.log_depth_img(dep_all, dep_re_all, img_title_prefix='')
		# # NOTE: By setting it to global step, we will log more images inside tensorboard, which may require more space
		# # If we set global_step to a constant, we will keep overwriting the images.
		# grid = torchvision.utils.make_grid(dep_all, nrow=n_samples, value_range=(0,1))
		# self.logger.experiment.add_image('GT Depths', grid, global_step=self.global_step)
		# grid = torchvision.utils.make_grid(dep_re_all, nrow=n_samples, value_range=(0,1))
		# self.logger.experiment.add_image('Rec. Depths', grid, global_step=self.global_step)


	def on_train_epoch_end(self) -> None:
		print("")
		return super().on_train_epoch_start()

	def on_validation_epoch_end(self) -> None:
		print("")
		return super().on_validation_epoch_end()
	
	def on_train_start(self):
		# Proper logging of hyperparams and metrics in TB
		# self.logger.log_hyperparams(self.hparams, {"loss/train": 0, "loss/avg_val": 0, "rmse/train": 0, "rmse/avg_val": 0})
		self.logger.log_hyperparams(self.hparams)


class LITL1LossBaseSpadModel(LITBaseSPADModel):
	'''
		Same as BaseSpadModel, but we use L1 instead of KLDiv loss
	'''
	def __init__(self, 		
		backbone_net,
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9):

		super(LITL1LossBaseSpadModel, self).__init__(
			backbone_net=backbone_net,
			init_lr = init_lr,
			p_tv = p_tv, 
			lr_decay_gamma = lr_decay_gamma,
			data_loss_id = 'L1'
			)
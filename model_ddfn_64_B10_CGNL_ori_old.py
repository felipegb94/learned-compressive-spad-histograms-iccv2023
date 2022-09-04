#### Standard Library Imports
import os

#### Library imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import pytorch_lightning as pl
import torchvision
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from losses import criterion_RMSE, criterion_KL, criterion_TV, criterion_L1, criterion_L2
import tof_utils
from research_utils.np_utils import calc_mean_percentile_errors


# feature extraction part
class MsFeat(torch.nn.Module):
	def __init__(self, in_channels):
		outchannel_MS = 2
		super(MsFeat, self).__init__()

		self.conv1 = nn.Sequential(nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1,1,1), padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

		self.conv2 = nn.Sequential(nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1,1,1), padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv2[0].bias, 0.0)

		self.conv3 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv3[0].bias, 0.0)

		self.conv4 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv4[0].bias, 0.0)

	def forward(self, inputs):
		conv1 = self.conv1(inputs)
		conv2 = self.conv2(inputs)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv1)
		return torch.cat((conv1, conv2, conv3, conv4), 1)


class NonLocal(torch.nn.Module):
	def __init__(self, inplanes, use_scale=False, groups=None):
		self.use_scale = use_scale
		self.groups = groups

		super(NonLocal, self).__init__()
		# conv theta
		self.t = nn.Conv3d(inplanes, inplanes//1, kernel_size=1, stride=1, bias=False)
		init.kaiming_normal_(self.t.weight, 0, 'fan_in', 'relu')
		# conv phi
		self.p = nn.Conv3d(inplanes, inplanes//1, kernel_size=1, stride=1, bias=False)
		init.kaiming_normal_(self.p.weight, 0, 'fan_in', 'relu')
		# conv g
		self.g = nn.Conv3d(inplanes, inplanes//1, kernel_size=1, stride=1, bias=False)
		init.kaiming_normal_(self.g.weight, 0, 'fan_in', 'relu')
		# conv z
		self.z = nn.Conv3d(inplanes//1, inplanes, kernel_size=1, stride=1,
						   groups=self.groups, bias=False)
		init.kaiming_normal_(self.z.weight, 0, 'fan_in', 'relu')
		# concat groups
		self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
		init.constant_(self.gn.weight, 0); nn.init.constant_(self.gn.bias, 0)

		if self.use_scale:
			print("=> WARN: Non-local block uses 'SCALE'")
		if self.groups:
			print("=> WARN: Non-local block uses '{}' groups".format(self.groups))

	def kernel(self, t, p, g, b, c, d, h, w):
		"""The linear kernel (dot production).

		Args:
			t: output of conv theata
			p: output of conv phi
			g: output of conv g
			b: batch size
			c: channels number
			d: depth of featuremaps
			h: height of featuremaps
			w: width of featuremaps
		"""
		t = t.view(b, 1, c * d * h * w)
		p = p.view(b, 1, c * d * h * w)
		g = g.view(b, c * d * h * w, 1)

		att = torch.bmm(p, g)

		if self.use_scale:
			att = att.div((c * d * h * w) ** 0.5)

		x = torch.bmm(att, t)
		x = x.view(b, c, d, h, w)

		return x

	def forward(self, x):
		residual = x

		t = self.t(x) #b,ch,d,h,w
		p = self.p(x) #b,ch,d,h,w
		g = self.g(x) #b,ch,d,h,w

		b, c, d, h, w = t.size()

		if self.groups and self.groups > 1:
			_c = int(c / self.groups)

			ts = torch.split(t, split_size_or_sections=_c, dim=1)
			ps = torch.split(p, split_size_or_sections=_c, dim=1)
			gs = torch.split(g, split_size_or_sections=_c, dim=1)

			_t_sequences = []
			for i in range(self.groups):
				_x = self.kernel(ts[i], ps[i], gs[i],
								 b, _c, d, h, w)
				_t_sequences.append(_x)

			x = torch.cat(_t_sequences, dim=1)
		else:
			x = self.kernel(t, p, g,
							b, c, d, h, w)

		x = self.z(x)
		x = self.gn(x) + residual

		return x


# feature integration
class Block(torch.nn.Module):
	def __init__(self, in_channels):
		outchannel_block = 16
		super(Block, self).__init__()
		self.conv1 = nn.Sequential(nn.Conv3d(in_channels, outchannel_block, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

		self.feat1 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat1[0].bias, 0.0)

		self.feat15 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat15[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat15[0].bias, 0.0)

		self.feat2 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat2[0].bias, 0.0)

		self.feat25 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat25[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat25[0].bias, 0.0)

		self.feat = nn.Sequential(nn.Conv3d(24, 8, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.feat[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.feat[0].bias, 0.0)
	# note the channel for each layer
	def forward(self, inputs):

		conv1 = self.conv1(inputs)
		feat1 = self.feat1(conv1)
		feat15 = self.feat15(feat1)
		feat2 = self.feat2(conv1)
		feat25 = self.feat25(feat2)
		feat = self.feat(torch.cat((feat1, feat15, feat2, feat25), 1))
		return torch.cat((inputs, feat), 1)


# build the model
class DeepBoosting(torch.nn.Module):
	def __init__(self, in_channels=1):
		super(DeepBoosting, self).__init__()
		self.msfeat = MsFeat(in_channels)
		self.C1 = nn.Sequential(nn.Conv3d(8,2,kernel_size=1, stride=(1,1,1),bias=True),nn.ReLU(inplace=True))
		init.kaiming_normal_(self.C1[0].weight, 0, 'fan_in', 'relu'); 
		init.constant_(self.C1[0].bias, 0.0)
		self.nl = NonLocal(2, use_scale=False, groups=1)

		self.ds1 = nn.Sequential(nn.Conv3d(2,4,kernel_size=3,stride=(2,1,1),padding=(1,1,1),bias=True),nn.ReLU(inplace=True))
		init.kaiming_normal_(self.ds1[0].weight, 0, 'fan_in', 'relu'); 
		init.constant_(self.ds1[0].bias, 0.0)
		self.ds2 = nn.Sequential(nn.Conv3d(4, 8, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),nn.ReLU(inplace=True))
		init.kaiming_normal_(self.ds2[0].weight, 0, 'fan_in', 'relu'); 
		init.constant_(self.ds2[0].bias, 0.0)
		self.ds3 = nn.Sequential(nn.Conv3d(8,16,kernel_size=3,stride=(2,1,1),padding=(1,1,1),bias=True),nn.ReLU(inplace=True))
		init.kaiming_normal_(self.ds3[0].weight, 0, 'fan_in', 'relu'); 
		init.constant_(self.ds3[0].bias, 0.0)
		self.ds4 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=3, stride=(2, 1, 1), padding=(1, 1, 1), bias=True),nn.ReLU(inplace=True))
		init.kaiming_normal_(self.ds4[0].weight, 0, 'fan_in', 'relu'); 
		init.constant_(self.ds4[0].bias, 0.0)

		self.dfus_block0 = Block(32)
		self.dfus_block1 = Block(40)
		self.dfus_block2 = Block(48)
		self.dfus_block3 = Block(56)
		self.dfus_block4 = Block(64)
		self.dfus_block5 = Block(72)
		self.dfus_block6 = Block(80)
		self.dfus_block7 = Block(88)
		self.dfus_block8 = Block(96)
		self.dfus_block9 = Block(104)
		self.convr = nn.Sequential(
			nn.ConvTranspose3d(112, 56, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),nn.ReLU(inplace=True),
			nn.ConvTranspose3d(56, 28, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),nn.ReLU(inplace=True),
			nn.ConvTranspose3d(28, 14, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),nn.ReLU(inplace=True),
			nn.ConvTranspose3d(14, 7, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False))
		init.kaiming_normal_(self.convr[0].weight, 0, 'fan_in', 'relu')
		init.kaiming_normal_(self.convr[2].weight, 0, 'fan_in', 'relu')
		init.kaiming_normal_(self.convr[4].weight, 0, 'fan_in', 'relu')
		init.normal_(self.convr[6].weight, mean=0.0, std=0.001)

		self.C2 = nn.Sequential(nn.Conv3d(7,1,kernel_size=1, stride=(1,1,1),bias=True),nn.ReLU(inplace=True))
		init.kaiming_normal_(self.C2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.C2[0].bias, 0.0)
		
	def forward(self, inputs):
		smax = torch.nn.Softmax2d()
		msfeat = self.msfeat(inputs) 
		c1 = self.C1(msfeat)
		nlout = self.nl(c1)
		dsfeat1 = self.ds1(nlout)
		dsfeat2 = self.ds2(dsfeat1) 
		dsfeat3 = self.ds3(dsfeat2) 
		dsfeat4 = self.ds4(dsfeat3) 
		b0 = self.dfus_block0(dsfeat4)
		b1 = self.dfus_block1(b0)
		b2 = self.dfus_block2(b1)
		b3 = self.dfus_block3(b2)
		b4 = self.dfus_block4(b3)
		b5 = self.dfus_block5(b4)
		b6 = self.dfus_block6(b5)
		b7 = self.dfus_block7(b6)
		b8 = self.dfus_block8(b7)
		b9 = self.dfus_block9(b8)
		convr = self.convr(b9)
		convr = self.C2(convr)

		denoise_out = torch.squeeze(convr, 1)

		weights = Variable(torch.linspace(0, 1, steps=denoise_out.size()[1]).unsqueeze(1).unsqueeze(1)).to(denoise_out.device)
		weighted_smax = weights * smax(denoise_out)
		soft_argmax = weighted_smax.sum(1).unsqueeze(1)

		return denoise_out, soft_argmax


class LITDeepBoostingOriginal(pl.LightningModule):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1):
		
		super().__init__()
		
		self.lsmx = torch.nn.LogSoftmax(dim=1)
		self.save_hyperparameters()

		# Train hyperparams		
		self.init_lr = init_lr
		self.lr_decay_gamma = lr_decay_gamma
		self.p_tv = p_tv

		self.deep_boosting_model = DeepBoosting(in_channels=in_channels)

		self.example_input_array = torch.randn([1, 1, 1024, 32, 32])

	def forward(self, x):
		# use forward for inference/predictions
		out = self.deep_boosting_model(x)
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
		M_mea_re, dep_re = self(input_data)
		return M_mea_re, dep_re

	def compute_losses(self, sample, M_mea_re, dep_re):
		# load data and compute losses
		M_gt = sample["rates"]
		dep = sample["bins"]
		# Normalize
		M_mea_re_lsmx = self.lsmx(M_mea_re).unsqueeze(1)
		# Compute metrics
		loss_kl = criterion_KL(M_mea_re_lsmx, M_gt)
		loss_tv = criterion_TV(dep_re)
		rmse = criterion_RMSE(dep_re, dep)
		loss = loss_kl + self.p_tv*loss_tv

		return loss, loss_kl, loss_tv, rmse

	def training_step(self, sample, batch_idx):
		# Forward pass
		M_mea_re, dep_re = self.forward_wrapper(sample)
		# Compute Losses
		(loss, loss_kl, loss_tv, rmse) = self.compute_losses(sample, M_mea_re, dep_re)

		# Log to logger (i.e., tensorboard), if you want it to be displayed at progress bar, use prog_bar=True
		self.log_dict(
			{
				"loss/train": loss
				, "rmse/train": rmse
				, "train_losses/kldiv": loss_kl
				, "train_losses/tv": self.p_tv*loss_tv				
			}
			# , prog_bar=True
		)
		return {'loss': loss}


	def validation_step(self, sample, batch_idx):
		# Forward pass
		M_mea_re, dep_re = self.forward_wrapper(sample)
		# Compute Losses
		(val_loss, val_loss_kl, val_loss_tv, val_rmse) = self.compute_losses(sample, M_mea_re, dep_re)
		
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
		return {'dep': dep, 'dep_re': dep_re}


	def test_step(self, sample, batch_idx):
		# Forward pass
		M_mea_re, dep_re = self.forward_wrapper(sample)
		# Compute Losses
		(test_loss, test_loss_kl, test_loss_tv, test_rmse) = self.compute_losses(sample, M_mea_re, dep_re)

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
		# depths_L2 = criterion_RMSE(rec_depths, gt_depths)
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

		# NOTE: By setting it to global step, we will log more images inside tensorboard, which may require more space
		# If we set global_step to a constant, we will keep overwriting the images.
		grid = torchvision.utils.make_grid(dep_all, nrow=n_samples, value_range=(0,1))
		self.logger.experiment.add_image('GT Depths', grid, global_step=self.global_step)
		grid = torchvision.utils.make_grid(dep_re_all, nrow=n_samples, value_range=(0,1))
		self.logger.experiment.add_image('Rec. Depths', grid, global_step=self.global_step)


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

if __name__=='__main__':
	import matplotlib.pyplot as plt
	from model_utils import count_parameters
	# Set random input
	batch_size = 2
	(nr, nc, nt) = (64, 64, 1024) 
	inputs = torch.randn((batch_size, 1, nt, nr, nc))

	# Set compression params

	model = LITDeepBoostingOriginal()

	print("{} Parameters: {}".format(model.__class__.__name__, count_parameters(model)))


	outputs = model(inputs)

	print("inputs shape: {}".format(inputs.shape))
	print("outputs1 shape: {}".format(outputs[0].shape))
	print("outputs2 shape: {}".format(outputs[1].shape))



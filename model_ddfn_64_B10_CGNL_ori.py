#### Standard Library Imports

#### Library imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import pytorch_lightning as pl
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_base_spad_lit import LITBaseSPADModel
from model_ddfn import Block3DGroup, MsFeat3D

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


# build the model
class DeepBoosting(torch.nn.Module):
	def __init__(self, in_channels=1):
		super(DeepBoosting, self).__init__()
		self.msfeat = MsFeat3D(in_channels)
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

		self.dfu_block_group = Block3DGroup(in_channels=32, n_ddfn_blocks=10)
		# self.dfus_block0 = Block(32)
		# self.dfus_block1 = Block(40)
		# self.dfus_block2 = Block(48)
		# self.dfus_block3 = Block(56)
		# self.dfus_block4 = Block(64)
		# self.dfus_block5 = Block(72)
		# self.dfus_block6 = Block(80)
		# self.dfus_block7 = Block(88)
		# self.dfus_block8 = Block(96)
		# self.dfus_block9 = Block(104)

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

		b9 = self.dfu_block_group(dsfeat4)
		# b0 = self.dfus_block0(dsfeat4)
		# b1 = self.dfus_block1(b0)
		# b2 = self.dfus_block2(b1)
		# b3 = self.dfus_block3(b2)
		# b4 = self.dfus_block4(b3)
		# b5 = self.dfus_block5(b4)
		# b6 = self.dfus_block6(b5)
		# b7 = self.dfus_block7(b6)
		# b8 = self.dfus_block8(b7)
		# b9 = self.dfus_block9(b8)

		convr = self.convr(b9)
		convr = self.C2(convr)

		denoise_out = torch.squeeze(convr, 1)

		weights = Variable(torch.linspace(0, 1, steps=denoise_out.size()[1]).unsqueeze(1).unsqueeze(1)).to(denoise_out.device)
		weighted_smax = weights * smax(denoise_out)
		soft_argmax = weighted_smax.sum(1).unsqueeze(1)

		return denoise_out, soft_argmax

class PlainDeepBoosting(torch.nn.Module):
	def __init__(self, in_channels=1):
		super(PlainDeepBoosting, self).__init__()
		self.msfeat = MsFeat3D(in_channels)
		self.C1 = nn.Sequential(nn.Conv3d(8,2,kernel_size=1, stride=(1,1,1),bias=True),nn.ReLU(inplace=True))
		init.kaiming_normal_(self.C1[0].weight, 0, 'fan_in', 'relu'); 
		init.constant_(self.C1[0].bias, 0.0)

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

		self.dfu_block_group = Block3DGroup(in_channels=32, n_ddfn_blocks=10)

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
		
		dsfeat1 = self.ds1(c1)
		dsfeat2 = self.ds2(dsfeat1) 
		dsfeat3 = self.ds3(dsfeat2) 
		dsfeat4 = self.ds4(dsfeat3) 

		b9 = self.dfu_block_group(dsfeat4)

		convr = self.convr(b9)
		convr = self.C2(convr)

		denoise_out = torch.squeeze(convr, 1)

		weights = Variable(torch.linspace(0, 1, steps=denoise_out.size()[1]).unsqueeze(1).unsqueeze(1)).to(denoise_out.device)
		weighted_smax = weights * smax(denoise_out)
		soft_argmax = weighted_smax.sum(1).unsqueeze(1)

		return denoise_out, soft_argmax


class LITDeepBoosting(LITBaseSPADModel):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1):
		
		deep_boosting_model = DeepBoosting(in_channels=in_channels)

		super(LITDeepBoosting, self).__init__(backbone_net=deep_boosting_model,
												init_lr = init_lr,
												p_tv = p_tv, 
												lr_decay_gamma = lr_decay_gamma)
		
		# Overwrite example input array
		self.example_input_array = torch.randn([1, 1, 1024, 32, 32])

class LITPlainDeepBoosting(LITBaseSPADModel):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1):
		
		deep_boosting_model = PlainDeepBoosting(in_channels=in_channels)

		super(LITPlainDeepBoosting, self).__init__(backbone_net=deep_boosting_model,
												init_lr = init_lr,
												p_tv = p_tv, 
												lr_decay_gamma = lr_decay_gamma)
		
		# Overwrite example input array
		self.example_input_array = torch.randn([1, 1, 1024, 32, 32])
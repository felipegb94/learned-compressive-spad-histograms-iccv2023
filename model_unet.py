'''
	Unet code
	Most of the building blocks are borrrowed from: https://github.com/vsitzmann/pytorch_prototyping
'''
#### Standard Library Imports
from collections import OrderedDict
from numpy import outer

#### Library imports
import torch
import torch.nn as nn
import torch.nn.init as init
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


class Conv2dSame(torch.nn.Module):
	'''2D convolution that pads to keep spatial dimensions equal.
	Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
	'''

	def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
		'''
		:param in_channels: Number of input channels
		:param out_channels: Number of output channels
		:param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
		:param bias: Whether or not to use bias.
		:param padding_layer: Which padding to use. Default is reflection padding.
		'''
		super().__init__()
		ka = kernel_size // 2
		kb = ka - 1 if kernel_size % 2 == 0 else ka
		self.net = nn.Sequential(
			padding_layer((ka, kb, ka, kb)),
			nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
		)

		self.weight = self.net[1].weight
		self.bias = self.net[1].bias

	def forward(self, x):
		return self.net(x)

class DownBlock(nn.Module):
	'''A 2D-conv downsampling block following best practices / with reasonable defaults
	(LeakyReLU, kernel size multiple of stride)
	'''

	def __init__(self,
				 in_channels,
				 out_channels,
				 prep_conv=True,
				 middle_channels=None,
				 use_dropout=False,
				 dropout_prob=0.1,
				 norm=nn.BatchNorm2d):
		'''
		:param in_channels: Number of input channels
		:param out_channels: Number of output channels
		:param prep_conv: Whether to have another convolutional layer before the downsampling layer.
		:param middle_channels: If prep_conv is true, this sets the number of channels between the prep and downsampling
								convs.
		:param use_dropout: bool. Whether to use dropout or not.
		:param dropout_prob: Float. The dropout probability (if use_dropout is True)
		:param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
		'''
		super().__init__()

		if middle_channels is None:
			middle_channels = in_channels

		net = list()

		if prep_conv:
			net += [nn.ReflectionPad2d(1),
					nn.Conv2d(in_channels,
							  middle_channels,
							  kernel_size=3,
							  padding=0,
							  stride=1,
							  bias=True if norm is None else False)]

			if norm is not None:
				net += [norm(middle_channels, affine=True)]

			net += [nn.LeakyReLU(0.2, True)]

			if use_dropout:
				net += [nn.Dropout2d(dropout_prob, False)]

		net += [nn.ReflectionPad2d(1),
				nn.Conv2d(middle_channels,
						  out_channels,
						  kernel_size=4,
						  padding=0,
						  stride=2,
						  bias=True if norm is None else False)]

		if norm is not None:
			net += [norm(out_channels, affine=True)]

		net += [nn.LeakyReLU(0.2, True)]

		if use_dropout:
			net += [nn.Dropout2d(dropout_prob, False)]

		self.net = nn.Sequential(*net)

	def forward(self, x):
		return self.net(x)

class UpBlock(nn.Module):
	'''A 2d-conv upsampling block with a variety of options for upsampling, and following best practices / with
	reasonable defaults. (LeakyReLU, kernel size multiple of stride)
	'''

	def __init__(self,
				 in_channels,
				 out_channels,
				 post_conv=True,
				 use_dropout=False,
				 dropout_prob=0.1,
				 norm=nn.BatchNorm2d,
				 upsampling_mode='transpose'):
		'''
		:param in_channels: Number of input channels
		:param out_channels: Number of output channels
		:param post_conv: Whether to have another convolutional layer after the upsampling layer.
		:param use_dropout: bool. Whether to use dropout or not.
		:param dropout_prob: Float. The dropout probability (if use_dropout is True)
		:param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
		:param upsampling_mode: Which upsampling mode:
				transpose: Upsampling with stride-2, kernel size 4 transpose convolutions.
				bilinear: Feature map is upsampled with bilinear upsampling, then a conv layer.
				nearest: Feature map is upsampled with nearest neighbor upsampling, then a conv layer.
				shuffle: Feature map is upsampled with pixel shuffling, then a conv layer.
		'''
		super().__init__()

		net = list()

		if upsampling_mode == 'transpose':
			net += [nn.ConvTranspose2d(in_channels,
									   out_channels,
									   kernel_size=4,
									   stride=2,
									   padding=1,
									   bias=True if norm is None else False)]
		elif upsampling_mode == 'bilinear':
			net += [nn.UpsamplingBilinear2d(scale_factor=2)]
			net += [
				Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
		elif upsampling_mode == 'nearest':
			net += [nn.UpsamplingNearest2d(scale_factor=2)]
			net += [
				Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
		elif upsampling_mode == 'shuffle':
			net += [nn.PixelShuffle(upscale_factor=2)]
			net += [
				Conv2dSame(in_channels // 4, out_channels, kernel_size=3,
						   bias=True if norm is None else False)]
		else:
			raise ValueError("Unknown upsampling mode!")

		if norm is not None:
			net += [norm(out_channels, affine=True)]

		net += [nn.ReLU(True)]

		if use_dropout:
			net += [nn.Dropout2d(dropout_prob, False)]

		if post_conv:
			net += [Conv2dSame(out_channels,
							   out_channels,
							   kernel_size=3,
							   bias=True if norm is None else False)]

			if norm is not None:
				net += [norm(out_channels, affine=True)]

			net += [nn.ReLU(True)]

			if use_dropout:
				net += [nn.Dropout2d(0.1, False)]

		self.net = nn.Sequential(*net)

	def forward(self, x, skipped=None):
		if skipped is not None:
			input = torch.cat([skipped, x], dim=1)
		else:
			input = x
		return self.net(input)


class UnetSkipConnectionBlock(nn.Module):
	'''Helper class for building a 2D unet.
	'''

	def __init__(self,
				 outer_nc,
				 inner_nc,
				 upsampling_mode,
				 norm=nn.BatchNorm2d,
				 submodule=None,
				 use_dropout=False,
				 dropout_prob=0.1):
		super().__init__()

		if submodule is None:
			model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
					 UpBlock(inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
							 upsampling_mode=upsampling_mode)]
		else:
			model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
					 submodule,
					 UpBlock(2 * inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
							 upsampling_mode=upsampling_mode)]

		self.model = nn.Sequential(*model)

	def forward(self, x):
		forward_passed = self.model(x)
		return torch.cat([x, forward_passed], 1)


class Unet(nn.Module):
	'''A 2d-Unet implementation with sane defaults.
	'''

	def __init__(self,
				 in_channels,
				 out_channels,
				 nf0,
				 num_down,
				 max_channels,
				 use_dropout,
				 upsampling_mode='transpose',
				 dropout_prob=0.1,
				 norm=nn.BatchNorm2d,
				 outermost_linear=False):
		'''
		:param in_channels: Number of input channels
		:param out_channels: Number of output channels
		:param nf0: Number of features at highest level of U-Net
		:param num_down: Number of downsampling stages.
		:param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
		:param use_dropout: Whether to use dropout or no.
		:param dropout_prob: Dropout probability if use_dropout=True.
		:param upsampling_mode: Which type of upsampling should be used. See "UpBlock" for documentation.
		:param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
		:param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
		'''
		super().__init__()

		assert (num_down > 0), "Need at least one downsampling layer in UNet."

		# Define the in block
		self.in_layer = [Conv2dSame(in_channels, nf0, kernel_size=3, bias=True if norm is None else False)]
		if norm is not None:
			self.in_layer += [norm(nf0, affine=True)]
		self.in_layer += [nn.LeakyReLU(0.2, True)]

		if use_dropout:
			self.in_layer += [nn.Dropout2d(dropout_prob)]
		self.in_layer = nn.Sequential(*self.in_layer)

		# Define the center UNet block
		self.unet_block = UnetSkipConnectionBlock(min(2 ** (num_down-1) * nf0, max_channels),
												  min(2 ** (num_down-1) * nf0, max_channels),
												  use_dropout=use_dropout,
												  dropout_prob=dropout_prob,
												  norm=None, # Innermost has no norm (spatial dimension 1)
												  upsampling_mode=upsampling_mode)

		for i in list(range(0, num_down - 1))[::-1]:
			self.unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),
													  min(2 ** (i + 1) * nf0, max_channels),
													  use_dropout=use_dropout,
													  dropout_prob=dropout_prob,
													  submodule=self.unet_block,
													  norm=norm,
													  upsampling_mode=upsampling_mode)

		# Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
		# automatically receives the output of the in_layer and the output of the last unet layer.
		self.out_layer = [Conv2dSame(2 * nf0,
									 out_channels,
									 kernel_size=3,
									 bias=outermost_linear or (norm is None))]

		if not outermost_linear:
			if norm is not None:
				self.out_layer += [norm(out_channels, affine=True)]
			self.out_layer += [nn.ReLU(True)]

			if use_dropout:
				self.out_layer += [nn.Dropout2d(dropout_prob)]
		self.out_layer = nn.Sequential(*self.out_layer)

		self.out_layer_weight = self.out_layer[0].weight

	def forward(self, x):
		in_layer = self.in_layer(x)
		unet = self.unet_block(in_layer)
		out_layer = self.out_layer(unet)
		return out_layer


if __name__=='__main__':
	import matplotlib.pyplot as plt
	from model_utils import count_parameters

	# Set random input
	batch_size = 3
	(nr, nc, nt) = (32, 32, 64) 
	in_ch  = 16
	inputs2D = torch.ones((batch_size, in_ch, nr, nc))

	## Init unet
	unet_model = Unet(in_channels=in_ch, out_channels=1, nf0=64, num_down=3, max_channels=512, upsampling_mode='nearest', use_dropout=False, outermost_linear=False)

	outputs = unet_model(inputs2D)
	print("Unet model: {} params".format(count_parameters(unet_model)))
	print("		inputs2D: {}".format(inputs2D.shape))
	print("		outputs: {}".format(outputs.shape))





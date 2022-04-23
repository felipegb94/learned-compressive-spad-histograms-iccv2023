#### Standard Library Imports
import os

#### Library imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
import torchvision
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


	
class Compressive3DLayer(nn.Module):
	def __init__(self, k=2):
		# Init parent class
		super(Compressive3DLayer, self).__init__()
		
		kernel_dims = (16,4,4)
		self.compressive_layer = nn.Conv3d(in_channels=1, 
										out_channels=k, 
										kernel_size=kernel_dims, 
										stride=kernel_dims, 
										padding=0, 
										bias=False)
		init.kaiming_normal_(self.compressive_layer.weight, 0, 'fan_in', 'relu') 
		# init.constant_(self.compressive_layer.bias, 0.0)
		self.relu1 = nn.ReLU(inplace=True)

		self.ups1conv =  nn.Sequential(
			nn.Conv3d(k, k//2, kernel_size=(5,3,3), stride=(1,1,1), padding=(2,1,1), dilation=1, bias=True), 
			nn.ReLU(inplace=True)
		)
		init.kaiming_normal_(self.ups1conv[0].weight, 0, 'fan_in', 'relu'); 
		init.constant_(self.ups1conv[0].bias, 0.0)

		self.ups2conv =  nn.Sequential(
			nn.Conv3d(k//2, 1, kernel_size=(5,3,3), stride=(1,1,1), padding=(2,1,1), dilation=1, bias=True), 
			nn.ReLU(inplace=True)
		)
		init.kaiming_normal_(self.ups2conv[0].weight, 0, 'fan_in', 'relu'); 
		init.constant_(self.ups2conv[0].bias, 0.0)


	def upsample4x2x2x(self, x): 
		return torch.nn.functional.interpolate(x, scale_factor=(4,2,2), mode='nearest')

	def forward(self, inputs):
		#### Compressive Layer Bottleneck downsampling
		## Apply compressive layer
		B = self.compressive_layer(inputs)
		## Apply non-linearity after compressive layer
		B_relu = self.relu1(B)

		#### Upsample the inputs again to have the same dimension as inputs
		## Do nearest neighbor interp, and then conv3D
		ups1_out = self.upsample4x2x2x(B_relu)
		ups1conv_out = self.ups1conv(ups1_out)
		ups2_out = self.upsample4x2x2x(ups1conv_out)
		ups2conv_out = self.ups2conv(ups2_out)

		return ups2conv_out

class Compressive3DLayerWithBias(Compressive3DLayer):
	def __init__(self, k=2):
		# Init parent class
		super(Compressive3DLayerWithBias, self).__init__(k=k)
		
		## Overwrite compressive 3D layer to have the bias term 
		kernel_dims = (16,4,4)
		self.compressive_layer = nn.Conv3d(in_channels=1, 
										out_channels=k, 
										kernel_size=kernel_dims, 
										stride=kernel_dims, 
										padding=0, 
										bias=True)
		init.kaiming_normal_(self.compressive_layer.weight, 0, 'fan_in', 'relu') 
		init.constant_(self.compressive_layer.bias, 0.0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Set random input
	batch_size = 1
	(nr, nc, nt) = (32, 32, 1024) 
	inputs = torch.randn((batch_size, 1, nt, nr, nc))

	# Set compression params
	k = 16
	kernel_dims = (16,4,4)

	compressive_layer = Compressive3DLayer(k=k)
	compressive_layer_withbias = Compressive3DLayerWithBias(k=k)

	print("Compressive Layer N Params: {}".format(count_parameters(compressive_layer)))
	print("Compressive Layer With Bias N Params: {}".format(count_parameters(compressive_layer_withbias)))

	outputs = compressive_layer(inputs)

	print("Input Shape: {}".format(inputs.shape))
	print("Output Shape: {}".format(outputs.shape))

	# ups1 = nn.ConvTranspose3d(k, k//2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=True)
	# ups1_out = ups1(B_relu)
	# ups1 = torch.nn.functional.interpolate(B_relu, scale_factor=(4,2,2), mode='trilinear')
	
	compressed_input = compressive_layer.compressive_layer(inputs) 
	relu_compressed_input = compressive_layer.relu1(compressed_input) 
	ups1_out = compressive_layer.upsample4x2x2x(relu_compressed_input)
	ups1conv_out = compressive_layer.ups1conv(ups1_out)
	ups2_out = compressive_layer.upsample4x2x2x(ups1conv_out)
	ups2conv_out = compressive_layer.ups2conv(ups2_out)

	print("    Compressed Input Shape: {}".format(compressed_input.shape))
	print("    Ups1 Out Shape: {}".format(ups1_out.shape))
	print("    Ups1Conv Out Shape: {}".format(ups1conv_out.shape))
	print("    Ups2 Out Shape: {}".format(ups2_out.shape))
	print("    Ups2Conv Out Shape: {}".format(ups2conv_out.shape))

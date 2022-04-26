'''
	Building Blocks for a 2D and 3D Deep Boosting Model
	These blocks are adapted from Peng et al., ECCV 2020
	Simply run this file to look at the input and outputs dimensions of each block
'''
#### Standard Library Imports
from collections import OrderedDict

#### Library imports
import torch
import torch.nn as nn
import torch.nn.init as init
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


# feature extraction part (in 3D)
class MsFeat3D(torch.nn.Module):
	'''
		The feature extraction block outputs the concatenation of the convolutuion and dilated conv applied to inptus
		For each convolution and dilated convolution 2 features will be extracted. 
		In other words:
			Input == (B, IN_CH, D1, D2, D3)
			Output == (B, 8, D1, D2, D3)
	'''
	def __init__(self, in_channels, outchannel_MS = 2):
		super(MsFeat3D, self).__init__()

		self.conv1 = nn.Sequential(nn.Conv3d(in_channels, outchannel_MS, kernel_size=3, stride=(1,1,1), padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv1[0].bias, 0.0)

		self.conv2 = nn.Sequential(nn.Conv3d(in_channels, outchannel_MS, kernel_size=3, stride=(1,1,1), padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv2[0].bias, 0.0)

		self.conv3 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, kernel_size=3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv3[0].bias, 0.0)

		self.conv4 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, kernel_size=3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
		init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu'); init.constant_(self.conv4[0].bias, 0.0)

	def forward(self, inputs):
		conv1 = self.conv1(inputs)
		conv2 = self.conv2(inputs)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv1)
		return torch.cat((conv1, conv2, conv3, conv4), 1)

# feature integration
class Block3D(torch.nn.Module):
	'''
		The output of this block will add 8 feature channels to the input
		These feature channels will be derived from convolutions and dilated convolutions on the inputs
		Finally, the output will be the input concatenated with the output 8 feature channel output
		In other words:
			Input == (B, IN_CH, D1, D2, D3)
			Output == (B, IN_CH+8, D1, D2, D3)
	'''
	def __init__(self, in_channels):
		outchannel_block = 16
		super(Block3D, self).__init__()
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


class Block3DGroup(torch.nn.Module):
	def __init__(self, in_channels, n_ddfn_blocks=10):
		super(Block3DGroup, self).__init__()
		self.n_ddfn_blocks = n_ddfn_blocks
		curr_in_channels = in_channels
		dfu3D_blocks_dict = OrderedDict()
		for i in range(self.n_ddfn_blocks):
			dfu3D_blocks_dict['dfu3D_block{}'.format(i)] = Block3D(in_channels=curr_in_channels) 
			curr_in_channels += 8
		self.n_added_feature_channels = 8*self.n_ddfn_blocks
		self.dfu3D_blocks = nn.Sequential(dfu3D_blocks_dict)

	def forward(self, inputs):
		return self.dfu3D_blocks(inputs)


if __name__=='__main__':
	import matplotlib.pyplot as plt

	# Set random input
	batch_size = 4
	(nr, nc, nt) = (32, 32, 64) 
	inputs = torch.ones((batch_size, 2, nt, nr, nc))

	# Set compression params

	block3D_group10_model = Block3DGroup(in_channels=2, n_ddfn_blocks=10)

	outputs = block3D_group10_model(inputs)

	print("block3D_group10 input shape: {}".format(inputs.shape))
	print("block3D_group10 output shape: {}".format(outputs.shape))

	loss = torch.mean(torch.abs(inputs - outputs[:,2:4,:]))

	print(loss)
	loss.backward()

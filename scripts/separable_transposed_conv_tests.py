#### Standard Library Imports

#### Library imports
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


if __name__=='__main__':
	pl.seed_everything(2)
	## Generate inputs
	k=51
	batch_size = 3
	(nr, nc, nt) = (32, 32, 1024)
	inputs2d = torch.randn((batch_size, 1, nr, nc))
	kernel2d = (4, 4)
	inputs3d = torch.randn((batch_size, 1, nt, nr, nc))
	kernel3d = (1024, 4, 4)
	kernel3d_size = kernel3d[0]*kernel3d[1]*kernel3d[2]

	## Get 2D and 3D conv layers
	conv2d = torch.nn.Conv2d(in_channels=1
										, out_channels=k
										, kernel_size=kernel2d
										, stride=kernel2d
										, padding=0, dilation=1, bias=False)
	conv3d = torch.nn.Conv3d(in_channels=1
										, out_channels=k
										, kernel_size=kernel3d
										, stride=kernel3d
										, padding=0, dilation=1, bias=False)

	conv2dtrans = torch.nn.ConvTranspose2d(in_channels=k
										, out_channels=1
										, kernel_size=kernel2d
										, stride=kernel2d
										# , stride=1
										, padding=0, dilation=1, bias=False)
	# conv2dtrans = torch.nn.Conv2d(in_channels=k
	# 									, out_channels=1
	# 									, kernel_size=kernel2d
	# 									, stride=1
	# 									, padding=4, dilation=1, bias=False)
	conv2dtrans_inv = torch.nn.ConvTranspose2d(in_channels=k
										, out_channels=1
										, kernel_size=kernel2d
										, stride=kernel2d
										, padding=0, dilation=1, bias=False)
	conv2dtrans_inv.weight.data = conv2d.weight.data
	# conv2dtrans_inv.weight.data = conv2d.weight.data.transpose(-1,-2)
	# conv2dtrans_inv.weight.data = torch.rot90(conv2d.weight.data, 2, [2,3])

	conv2d_unfold = torch.nn.Unfold(kernel_size=kernel2d
									, stride=kernel2d
									, dilation=1, padding=0)
	conv2d_w = conv2d.weight
	conv2d_w_mat = conv2d_w.view(k,-1)
	conv2d_w_mat_t = conv2d_w_mat.transpose(-1,-2) 
	conv3d_w = conv3d.weight
	conv3d_w_mat = conv3d_w.view(k,-1)
	conv3d_w_mat_t = conv3d_w_mat.transpose(-1,-2) 
	print("Dims for Kernels and Matrices")
	print("    conv2d_w.shape: {}".format(conv2d_w.shape))
	print("    conv2d_w_mat.shape: {}".format(conv2d_w_mat.shape))
	print("    conv3d_w.shape: {}".format(conv3d_w.shape))
	print("    conv3d_w_mat.shape: {}".format(conv3d_w_mat.shape))

	conv2dtrans_w = conv2dtrans.weight
	print("    conv2dtrans_w.shape: {}".format(conv2dtrans_w.shape))


	## Testing in 2D first
	# Regular convolution and tansposed convolution outputs
	conv2d_out = conv2d(inputs2d)
	conv2dtrans_out = conv2dtrans(conv2d_out)
	conv2dtrans_inv_out = conv2dtrans_inv(conv2d_out)
	print("Dims for Regular Conv Outputs")
	print("    inputs2d.shape: {}".format(inputs2d.shape))
	print("    conv2d_out.shape: {}".format(conv2d_out.shape))
	print("    conv2dtrans_out.shape: {}".format(conv2dtrans_out.shape))
	print("    conv2dtrans_inv_out.shape: {}".format(conv2dtrans_inv_out.shape))
	# Matrix Mult Convolution --> Unfold + Matmul + Fold
	inputs2d_unf_vec = conv2d_unfold(inputs2d)
	print("inputs2d_unf_vec.shape: {}".format(inputs2d_unf_vec.shape))
	# Do convolution
	conv2d_matmul_unf = conv2d_w_mat.matmul(inputs2d_unf_vec)
	conv2d_matmul_out = conv2d_matmul_unf.view(batch_size, k, int(nr/kernel2d[-2]), int(nc/kernel2d[-1]))
	print("conv2d_matmul_out - conv2d_out = {}".format(torch.abs(conv2d_matmul_out-conv2d_out).mean()))
	print(torch.all(torch.isclose(conv2d_matmul_out, conv2d_out)))


	# Transposed convolution with matmul
	conv2dtrans_matmul_unf = conv2d_w_mat_t.matmul(conv2d_matmul_unf)

	conv2dtrans_matmul_out = conv2dtrans_matmul_unf.view(batch_size, 1, nr, nc)

	fold2d = torch.nn.Fold((nr,nc), kernel_size=kernel2d, dilation=1, padding=0, stride=kernel2d)
	conv2dtrans_matmul_out = fold2d(conv2dtrans_matmul_unf)


	print("conv2dtrans_matmul_out - conv2dtrans_inv_out = {}".format(torch.abs(conv2dtrans_matmul_out-conv2dtrans_inv_out).mean()))
	print(torch.all(torch.isclose(conv2dtrans_matmul_out, conv2dtrans_inv_out)))

	print("conv2dtrans_matmul_out.min() = {}".format(conv2dtrans_matmul_out.min()))
	print("conv2dtrans_matmul_out.max() = {}".format(conv2dtrans_matmul_out.max()))
	print("conv2dtrans_matmul_out.mean() = {}".format(conv2dtrans_matmul_out.mean()))
	print("conv2dtrans_inv_out.min() = {}".format(conv2dtrans_inv_out.min()))
	print("conv2dtrans_inv_out.max() = {}".format(conv2dtrans_inv_out.max()))
	print("conv2dtrans_inv_out.mean() = {}".format(conv2dtrans_inv_out.mean()))

	plt.clf()
	plt.hist(conv2dtrans_matmul_out.detach().cpu().numpy().flatten(), alpha=0.5, bins=100)
	plt.hist(conv2dtrans_inv_out.detach().cpu().numpy().flatten(), alpha=0.5, bins=100)


	# ## Testing in 3D 
	# conv3d_out = conv3d(inputs3d)


	# ## 3D conv with matmul
	# inputs3d_unf_orig = inputs3d.unfold(-3, kernel3d[-3], kernel3d[-3]).unfold(-2, kernel3d[-2], kernel3d[-2]).unfold(-1, kernel3d[-1], kernel3d[-1])
	# unfold_shape = inputs3d_unf_orig.size()
	# inputs3d_unf_orig = inputs3d_unf_orig.permute(0, 2, 3, 4, 1, 5, 6, 7)
	
	# inputs3d_unf = inputs3d_unf_orig.contiguous().view(batch_size, -1, kernel3d[-3], kernel3d[-2], kernel3d[-1])
	
	# import unfoldNd
	# unfold3d_op = unfoldNd.UnfoldNd(kernel_size=kernel3d, dilation=1, padding=0, stride=kernel3d)
	# inputs3d_unf_vec = unfold3d_op(inputs3d) 

	# # inputs3d_unf_vec = inputs3d_unf_orig.contiguous().view(batch_size, -1, kernel3d_size).transpose(-1,-2)
	# # inputs3d_unf_vec = inputs3d_unf.view(batch_size, -1, kernel3d_size).transpose(-1,-2)
	
	# # print(torch.all(torch.isclose(inputs3d_unf_vec, inputs3d_unf2)))


	# # conv3d_matmul_unf = conv3d_w_mat.matmul(inputs3d_unf_vec)
	# conv3d_matmul_unf = torch.matmul(conv3d_w_mat, inputs3d_unf_vec)
	
	# conv3d_matmul_out = conv3d_matmul_unf.view(batch_size, k, int(nt/kernel3d[-3]), int(nr/kernel3d[-2]), int(nc/kernel3d[-1]))
	# conv3d_matmul_out_vec = conv3d_matmul_out.view(batch_size, k, -1)  

	# print("inputs3d_unf.shape: {}".format(inputs3d_unf.shape))
	# print("inputs3d_unf_vec.shape: {}".format(inputs3d_unf_vec.shape))
	# print("conv3d_matmul_unf.shape: {}".format(conv3d_matmul_unf.shape))
	# print("conv3d_matmul_out.shape: {}".format(conv3d_matmul_out.shape))

	# print("conv3d_matmul_out - conv3d_out = {}".format(torch.abs(conv3d_matmul_out-conv3d_out).mean()))
	# print(torch.all(torch.isclose(conv3d_matmul_out, conv3d_out)))

	





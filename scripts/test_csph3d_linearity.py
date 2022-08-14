'''
@author: Felipe Gutierrez-Barragan
'''
import sys
import os
sys.path.append('../')
sys.path.append('./')

## Library Imports
import matplotlib.pyplot as plt
import numpy as np
import torch

## Local Imports
from csph_layers import CSPH3DLayer


if __name__=='__main__':

	# Select device
	use_gpu = True
	if(torch.cuda.is_available() and use_gpu): device = torch.device("cuda:0")
	else: device = torch.device("cpu")
	
	# Set CSPH3D layer parameters input
	k=32
	tblock_init = 'HybridGrayFourier'
	optimize_tdim_codes = True
	optimize_codes = True
	batch_size = 4
	nt_blocks = 1
	spatial_down_factor = 4		
	(nr, nc, nt) = (32, 32, 1024)
	
	## Generate different inputs combinations
	rand_inputs1 = torch.randint(low=0, high=5, size=(batch_size, 1, nt, nr, nc), dtype=torch.float32).to(device)
	rand_inputs2 = torch.randint(low=0, high=10, size=(batch_size, 1, nt, nr, nc), dtype=torch.float32).to(device)
	rand_inputs2[rand_inputs2 < 7] = 0 
	rand_inputs3 = torch.randint(low=0, high=58, size=(batch_size, 1, nt, nr, nc), dtype=torch.float32).to(device)

	## Create a full and separable CSPH layer
	full_csph_layer = CSPH3DLayer(k=k, num_bins=nt, tblock_init=tblock_init, optimize_codes=True, nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, encoding_type='full')
	full_csph_layer.to(device)
	separable_csph_layer = CSPH3DLayer(k=k, num_bins=nt, tblock_init=tblock_init, optimize_codes=True, nt_blocks=nt_blocks, spatial_down_factor=spatial_down_factor, encoding_type='separable')
	separable_csph_layer.to(device)

	## Test full layer
	out_full_csph1 = full_csph_layer.csph_layer_full(rand_inputs1)
	out_full_csph2 = full_csph_layer.csph_layer_full(rand_inputs2)
	out_full_csph3 = full_csph_layer.csph_layer_full(rand_inputs3)
	
	out_full_csph1plus2 = full_csph_layer.csph_layer_full(rand_inputs1 + rand_inputs2)
	out_full_csph1plus3 = full_csph_layer.csph_layer_full(rand_inputs1 + rand_inputs3)
	out_full_csph2plus3 = full_csph_layer.csph_layer_full(rand_inputs2 + rand_inputs3)

	diff1 = torch.abs(out_full_csph1plus2 - out_full_csph1 - out_full_csph2)
	diff2 = torch.abs(out_full_csph1plus3 - out_full_csph1 - out_full_csph3)
	diff3 = torch.abs(out_full_csph2plus3 - out_full_csph2 - out_full_csph3)

	print("diff1.max() = {}".format(diff1.max()))
	print("diff1.mean() = {}".format(diff1.mean()))

	print("diff2.max() = {}".format(diff2.max()))
	print("diff2.mean() = {}".format(diff2.mean()))

	print("diff3.max() = {}".format(diff3.max()))
	print("diff3.mean() = {}".format(diff3.mean()))


	## Test full layer
	out_separable_csph1 = separable_csph_layer.csph_layer_separable(rand_inputs1)
	out_separable_csph2 = separable_csph_layer.csph_layer_separable(rand_inputs2)
	out_separable_csph3 = separable_csph_layer.csph_layer_separable(rand_inputs3)
	
	out_separable_csph1plus2 = separable_csph_layer.csph_layer_separable(rand_inputs1 + rand_inputs2)
	out_separable_csph1plus3 = separable_csph_layer.csph_layer_separable(rand_inputs1 + rand_inputs3)
	out_separable_csph2plus3 = separable_csph_layer.csph_layer_separable(rand_inputs2 + rand_inputs3)

	diff1 = torch.abs(out_separable_csph1plus2 - out_separable_csph1 - out_separable_csph2)
	diff2 = torch.abs(out_separable_csph1plus3 - out_separable_csph1 - out_separable_csph3)
	diff3 = torch.abs(out_separable_csph2plus3 - out_separable_csph2 - out_separable_csph3)

	print("diff1.max() = {}".format(diff1.max()))
	print("diff1.mean() = {}".format(diff1.mean()))

	print("diff2.max() = {}".format(diff2.max()))
	print("diff2.mean() = {}".format(diff2.mean()))

	print("diff3.max() = {}".format(diff3.max()))
	print("diff3.mean() = {}".format(diff3.mean()))




#### Standard Library Imports
from cmath import nan
import sys
sys.path.append('../')

#### Library imports
import torch
import matplotlib.pyplot as plt

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils.timer import Timer
from unfilt_backproj_3D import *

def time_full_unfilt_backproj(backproj_layer, y, W):
	try:
		with Timer(backproj_layer.__class__.__name__) as timer_obj:
			H_hat_full = backproj_layer(y=y, W=W)
			torch.cuda.synchronize()
		H_hat_full_np = H_hat_full.data.cpu().detach().numpy()
		plt.hist(H_hat_full_np.flatten(), bins=1000, alpha=1.0, label='H_hat_full - {}'.format(backproj_layer.__class__.__name__))
		wall_time = timer_obj.elapsed
	except RuntimeError as e:
		print("Error caugh - " + str(e))
		wall_time = -1
		H_hat_full_np = -1
	return wall_time, H_hat_full_np

if __name__=='__main__':
	# pl.seed_everything(2)
	## Generate inputs
	k=64
	(bt, br, bc) = (128, 4, 4)
	batch_size = 4
	(nt, nr, nc) = (1024, 32, 32)
	use_gpu = True
	if(torch.cuda.is_available() and use_gpu): device = torch.device("cuda:0")
	else: device = torch.device("cpu")

	# Full 3d histogram image
	H = torch.randn((batch_size, 1, nt, nr, nc))
	H = 5*torch.rand((batch_size, 1, nt, nr, nc)) + torch.randn((batch_size, 1, nt, nr, nc)) 
	H = 5 + torch.randn((batch_size, 1, nt, nr, nc))

	H = H.to(device)

	# Set block/kernel dimensions
	(nt_blocks, nr_blocks, nc_blocks) = (int(nt/bt), int(nr/br), int(nc/bc))
	kernel3d = (bt, br, bc)
	stride3d = kernel3d
	kernel3d_size = kernel3d[0]*kernel3d[1]*kernel3d[2]
	print("Parameters:")
	print("    Size of 3D Hist. Image: {}".format((nt,nr,nc)))
	print("    Size of 3D Hist. Block: {}".format((bt,br,bc)))
	print("    Num of 3D Hist. Block: {}".format((nt_blocks,nr_blocks,nc_blocks)))
	print("    K: {}".format(k))
	print("    use_gpu: {}".format(use_gpu))

	# Conv kernel for time dim
	kernel3d_t = (kernel3d[0], 1, 1)
	stride3d_t = kernel3d_t
	conv3d_t = torch.nn.Conv3d(in_channels=1
										, out_channels=k
										, kernel_size=kernel3d_t
										, stride=stride3d_t
										, padding=0, dilation=1, bias=False, device=device)
	# Conv kernel for spatial dims
	kernel3d_xy = (1, kernel3d[1], kernel3d[1])
	stride3d_xy = kernel3d_xy
	conv3d_xy = torch.nn.Conv3d(in_channels=k
										, out_channels=k
										, kernel_size=kernel3d_xy
										, stride=stride3d_xy
										, groups=k
										, padding=0, dilation=1, bias=False, device=device)

	W_t = conv3d_t.weight
	W_xy = conv3d_xy.weight
	# Computes outer product appropriately --> It is more like a batched element-wise multiplication
	W_txy = W_t*W_xy
	# print("W_t.shape = {}, type = {}, type(W_t.data) = {}".format(W_t.shape, type(W_t), type(W_t.data)))
	# print("W_xy.shape = {}, type = {}, type(W_xy.data) = {}".format(W_xy.shape, type(W_xy), type(W_xy.data)))
	# print("W_txy.shape = {}, type = {}, type(W_txy.data) = {}".format(W_txy.shape, type(W_txy), type(W_txy.data)))

	# Create 3D conv kernel with the above weights
	conv3d_txy = torch.nn.Conv3d(in_channels=1
										, out_channels=k
										, kernel_size=kernel3d
										, stride=stride3d
										, padding=0, dilation=1, bias=False, device=device)
	conv3d_txy.weight.data = torch.clone(W_txy.data)
	W_txy = conv3d_txy.weight


	## Forward pass
	# Full filter
	with Timer("3D Conv FULL:"):
		csph_full = conv3d_txy(H)
		torch.cuda.synchronize()
	# Separable filter
	with Timer("3D Conv SEPARABLE:"):
		csph_separable_t = conv3d_t(H)
		csph_separable = conv3d_xy(csph_separable_t)
		torch.cuda.synchronize()
	# Compare outputs
	plt.clf()
	plt.subplot(3,1,1)
	plt.hist(csph_full.cpu().data.numpy().flatten(), bins=100, alpha=1.0, label='csph_full')
	plt.hist(csph_separable.cpu().data.numpy().flatten(), bins=100, alpha=0.5, label='csph_separable')
	plt.title("CSPH Full vs. Separable");plt.legend()

	# Visualize outputs
	plt.subplot(3,1,2)

	## Unfiltered backprojection FULL (for loop implementation)
	unfilt_backproj_forloop_layer = UnfiltBackproj3DForLoop()
	(unfilt_backproj_full_forloop_time, H_hat_full_forloop_np) = time_full_unfilt_backproj(unfilt_backproj_forloop_layer, y=csph_full, W=conv3d_txy.weight)
	
	## Unfiltered backprojection FULL (transposed 3d conv implementation)
	unfilt_backproj_tconv_layer = UnfiltBackproj3DTransposeConv()
	(unfilt_backproj_full_tconv_time, H_hat_full_tconv_np) = time_full_unfilt_backproj(unfilt_backproj_tconv_layer, y=csph_full, W=conv3d_txy.weight)


	# ## Unfiltered backprojection FULL (for loop implementation)
	# with Timer("Unfiltered Backproj FULL (for loop):"):
	# 	unfilt_backproj_layer = UnfiltBackproj3DForLoop()
	# 	H_hat_full_forloop = unfilt_backproj_layer(y=csph_full, W=conv3d_txy.weight)
	# 	torch.cuda.synchronize()
	# H_hat_full_forloop_np = H_hat_full_forloop.data.cpu().detach().numpy()
	# plt.hist(H_hat_full_forloop_np.flatten(), bins=1000, alpha=1.0, label='H_hat_full_forloop')

	# ## Unfiltered backprojection (transposed 3D conv implementation)
	# with Timer("Unfiltered Backproj FULL (transposed conv):"):
	# 	unfilt_backproj_tconv_layer = UnfiltBackproj3DTransposeConv()
	# 	H_hat_full_tconv = unfilt_backproj_tconv_layer(y=csph_full, W=conv3d_txy.weight)
	# 	torch.cuda.synchronize()
	# H_hat_full_tconv_np = H_hat_full_tconv.data.cpu().detach()
	# plt.hist(H_hat_full_tconv_np.flatten(), bins=1000, alpha=0.5, label='H_hat_full_tconv')

	# nn_up = torch.nn.Upsample(scale_factor=kernel3d, mode='nearest')

	# with Timer("Unfiltered Backproj FULL (batch):"):
	# 	unfilt_backproj_batch_layer = UnfiltBackproj3DBatch()
	# 	H_hat_full_batch = unfilt_backproj_batch_layer(y=csph_full, W=conv3d_txy.weight)
	# 	torch.cuda.synchronize()
	# H_hat_full_batch_np = H_hat_full_batch.data.cpu().detach()
	# plt.hist(H_hat_full_batch_np.flatten(), bins=1000, alpha=0.5, label='H_hat_full_batch')
	# plt.title("H_hat Outputs");
	# plt.legend()


	# with Timer("Unfiltered Backproj SEPARABLE (for loop, t first):"):
	# 	H_hat_separable = torch.zeros_like(H)
	# 	for t in range(nt_blocks):
	# 		(start_t_idx, end_t_idx) = get_nonoverlapping_start_end_idx(t, bt)  
	# 		for y in range(nr_blocks):
	# 			(start_y_idx, end_y_idx) = get_nonoverlapping_start_end_idx(y, br)  
	# 			for x in range(nc_blocks):
	# 				(start_x_idx, end_x_idx) = get_nonoverlapping_start_end_idx(x, bc)
	# 				h_hat_separable_curr = torch.sum((csph_full[:,:,t:t+1,y:y+1,x:x+1]*W_t.squeeze(1).unsqueeze(0))*W_xy.squeeze(1).unsqueeze(0), dim=1, keepdim=True)
	# 				H_hat_separable[:, :, start_t_idx:end_t_idx, start_y_idx:end_y_idx, start_x_idx:end_x_idx] = h_hat_separable_curr
	# 	torch.cuda.synchronize()
	# H_hat_separable_np = H_hat_separable.data.cpu().detach()
	# plt.hist(H_hat_separable_np.flatten(), bins=1000, alpha=0.3, label='H_hat_separable (t first)')

	# with Timer("Unfiltered Backproj SEPARABLE (for loop, xy first):"):
	# 	H_hat_separable = torch.zeros_like(H)
	# 	for t in range(nt_blocks):
	# 		(start_t_idx, end_t_idx) = get_nonoverlapping_start_end_idx(t, bt)  
	# 		for y in range(nr_blocks):
	# 			(start_y_idx, end_y_idx) = get_nonoverlapping_start_end_idx(y, br)  
	# 			for x in range(nc_blocks):
	# 				(start_x_idx, end_x_idx) = get_nonoverlapping_start_end_idx(x, bc)
	# 				h_hat_separable_curr = torch.sum((csph_full[:,:,t:t+1,y:y+1,x:x+1]*W_xy.squeeze(1).unsqueeze(0))*W_t.squeeze(1).unsqueeze(0), dim=1, keepdim=True)
	# 				H_hat_separable[:, :, start_t_idx:end_t_idx, start_y_idx:end_y_idx, start_x_idx:end_x_idx] = h_hat_separable_curr
	# 	torch.cuda.synchronize()
	# H_hat_separable_np = H_hat_separable.data.cpu().detach()
	# plt.hist(H_hat_separable_np.flatten(), bins=1000, alpha=0.3, label='H_hat_separable (xy first)')

	# with Timer("Unfiltered Backproj SEPARABLE (transposed conv, t first):"):
	# 	H_hat_separable_tconv_t = F.conv_transpose3d(csph_full, weight=conv3d_t.weight, bias=None, stride=stride3d_t, groups=k)
	# 	H_hat_separable_tconv = F.conv_transpose3d(H_hat_separable_tconv_t, weight=conv3d_xy.weight, bias=None, stride=stride3d_xy)
	# 	torch.cuda.synchronize()
	# H_hat_separable_tconv_np = H_hat_separable_tconv.data.cpu().detach()
	# plt.hist(H_hat_separable_tconv_np.flatten(), bins=1000, alpha=0.3, label='H_hat_separable_tconv (t first)')
	# del H_hat_separable_tconv_t
	# torch.cuda.empty_cache()

	# with Timer("Unfiltered Backproj SEPARABLE (transposed conv, xy first):"):
	# 	H_hat_separable_tconv_xy = F.conv_transpose3d(csph_full, weight=conv3d_xy.weight, bias=None, stride=stride3d_xy, groups=k)
	# 	H_hat_separable_tconv = F.conv_transpose3d(H_hat_separable_tconv_xy, weight=conv3d_t.weight, bias=None, stride=stride3d_t)
	# 	torch.cuda.synchronize()
	# H_hat_separable_tconv_np = H_hat_separable_tconv.data.cpu().detach()
	# plt.hist(H_hat_separable_tconv_np.flatten(), bins=1000, alpha=0.3, label='H_hat_separable_tconv (xy first)')
	# del H_hat_separable_tconv_xy
	# torch.cuda.empty_cache()

	# ## Unfiltered backprojection (transposed 3D conv implementation)
	# with Timer("Unfiltered Backproj FULL (transposed conv):"):
	# 	# H_hat_full_tconv = F.conv_transpose3d(csph_full, weight=W_t*W_xy, bias=None, stride=stride3d)
	# 	H_hat_full_tconv = F.conv_transpose3d(csph_full, weight=conv3d_txy.weight, bias=None, stride=stride3d)
	# 	torch.cuda.synchronize()
	# H_hat_full_tconv_np = H_hat_full_tconv.data.cpu().detach()
	# plt.hist(H_hat_full_tconv_np.flatten(), bins=1000, alpha=0.5, label='H_hat_full_tconv')

	# nn_tdim_up = torch.nn.Upsample(scale_factor=kernel3d_t, mode='nearest')
	# nn_xydim_up = torch.nn.Upsample(scale_factor=kernel3d_xy, mode='nearest')
	# try:
	# 	with Timer("Unfiltered Backproj SEPARABLE (new, t first):"):
	# 		csph_separable_t_up = nn_tdim_up(csph_separable)
	# 		W_t_up = W_t.repeat((1, 1, nt_blocks, 1, 1))
	# 		H_hat_separable_new_t = csph_separable_t_up*W_t_up.squeeze(1).unsqueeze(0)
	# 		W_xy_up = W_xy.repeat((1, 1, 1, nr_blocks, nc_blocks))
	# 		H_hat_separable_new = torch.sum(nn_xydim_up(H_hat_separable_new_t)*W_xy_up.squeeze(1).unsqueeze(0), dim=1, keepdim=True)
	# 		torch.cuda.synchronize()
	# 	H_hat_separable_new_np = H_hat_separable_new.data.cpu().detach()
	# 	plt.hist(H_hat_separable_new_np.flatten(), bins=1000, alpha=0.5, label='H_hat_separable_new (t first)')
	# 	del csph_separable, csph_separable_t_up, W_t_up, W_xy_up
	# except RuntimeError as e:  
	# 	print("Error caugh - " + str(e))
	# torch.cuda.empty_cache()





	# # assert(H_hat_full_tconv.shape == H_hat_full.shape), "Dimension mismatch between H_hat_full and H_hat_tconv"


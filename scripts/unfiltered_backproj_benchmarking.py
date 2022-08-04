#### Standard Library Imports
import sys
import os
sys.path.append('../')
sys.path.append('./')

#### Library imports
import torch
import matplotlib.pyplot as plt

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils.timer import Timer
from research_utils import io_ops
from unfilt_backproj_3D import *

def time_full_unfilt_backproj(backproj_layer, y, W):
	try:
		with Timer(backproj_layer.__class__.__name__) as timer_obj:
			H_hat_full = backproj_layer(y=y, W=W)
			torch.cuda.synchronize()
		H_hat_full_np = H_hat_full.data.cpu().detach().numpy()
		plt.hist(H_hat_full_np.flatten(), bins=1000, alpha=1.0, label='H_hat_full - {}'.format(backproj_layer.__class__.__name__), histtype=u'step', density=True)
		wall_time = timer_obj.elapsed
	except RuntimeError as e:
		print("Error caugh - " + str(e))
		wall_time = -1
		H_hat_full_np = -1
	return wall_time, H_hat_full_np

def time_separable_unfilt_backproj(backproj_layer, y, W1d, W2d):
	try:
		with Timer(backproj_layer.__class__.__name__) as timer_obj:
			H_hat_separable = backproj_layer(y=y, W1d=W1d, W2d=W2d)
			torch.cuda.synchronize()
		H_hat_separable_np = H_hat_separable.data.cpu().detach().numpy()
		plt.hist(H_hat_separable_np.flatten(), bins=1000, alpha=1.0, label='H_hat_separable - {}'.format(backproj_layer.__class__.__name__),histtype=u'step', density=True)
		wall_time = timer_obj.elapsed
	except RuntimeError as e:
		print("Error caugh - " + str(e))
		wall_time = -1
		H_hat_separable_np = -1
	return wall_time, H_hat_separable_np

def append_benchmark_data(benchmark_data, method_id, params_id, wall_time):
	if(not (method_id in benchmark_data.keys())):
		benchmark_data[method_id] = {}
	if(not (params_id in benchmark_data[method_id].keys())):
		benchmark_data[method_id][params_id] = []
	benchmark_data[method_id][params_id].append(wall_time)

def compose_params_id_string(k, bt, br, bc): return 'k={}_bt={}_br={}_bc={}'.format(k, bt, br, bc)
def compose_input_size_string(batch_size, nt, nr, nc): return 'batch={}_{}x{}x{}'.format(batch_size, nt, nr, nc)

def parse_params_id_string(params_id):
	k = int(params_id)

if __name__=='__main__':
	# pl.seed_everything(2)
	## Generate inputs configurtations

	# ## Experiment 1: bt vs. runtime
	# k_all=[64]
	# bt_all=[32,64,128,256,512,1024]
	# brbc_all=[(4,4)]

	# ## Experiment 2: K vs. runtime
	# k_all=[32,64,128,256,512]
	# bt_all=[1024]
	# brbc_all=[(4,4)]

	## Experiment 3: spatial vs. runtime
	k_all=[128]
	bt_all=[1024]
	brbc_all=[(1,1), (2,2), (4,4), (8,8)]

	params_configs_all = []
	for k in k_all:
		for bt in bt_all:
			for brbc in brbc_all:
				params_configs_all.append((k,bt,brbc[0],brbc[1]))

	for use_gpu in [True, False]:
		for params_config in params_configs_all:
			# Inputs parameters
			(k, bt, br, bc) = params_config
			params_id = compose_params_id_string(k, bt, br, bc)
			print("-----------------------")
			print("Running {}".format(params_id))
			print("-----------------------")
			batch_size = 4
			(nt, nr, nc) = (1024, 32, 32)
			input_size_id = compose_input_size_string(batch_size, nt, nr, nc)

			if(torch.cuda.is_available() and use_gpu): device = torch.device("cuda:0")
			else: device = torch.device("cpu")

			device_name = torch.cuda.get_device_name(0).replace(" ", "")
			benchmark_fname = 'unfilt_backproj_benchmark_'+device_name+'_'+input_size_id+'.json'
			benchmark_dirpath = './scripts/benchmark_results'
			if(os.path.basename(os.getcwd()) == 'scripts'): benchmark_dirpath = './benchmark_results'
			os.makedirs(benchmark_dirpath, exist_ok=True)
			benchmark_fpath =  os.path.join(benchmark_dirpath, benchmark_fname)
			
			# Load benchmark data
			if(os.path.exists(benchmark_fpath)): benchmark_data = io_ops.load_json(benchmark_fpath)
			else: benchmark_data = {}

			if(not (device.type in benchmark_data.keys())): benchmark_data[device.type] = {}

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

			#### Unfiltered Backprojection FULL Implementations
			## Unfiltered backprojection FULL (for loop implementation)
			unfilt_backproj_forloop_layer = UnfiltBackproj3DForLoop()
			(unfilt_backproj_full_forloop_time, H_hat_full_forloop_np) = time_full_unfilt_backproj(unfilt_backproj_forloop_layer, y=csph_full, W=conv3d_txy.weight)
			append_benchmark_data(benchmark_data[device.type], unfilt_backproj_forloop_layer.__class__.__name__, params_id, unfilt_backproj_full_forloop_time)
			## Unfiltered backprojection FULL (transposed 3d conv implementation)
			unfilt_backproj_tconv_layer = UnfiltBackproj3DTransposeConv()
			(unfilt_backproj_full_tconv_time, H_hat_full_tconv_np) = time_full_unfilt_backproj(unfilt_backproj_tconv_layer, y=csph_full, W=conv3d_txy.weight)
			append_benchmark_data(benchmark_data[device.type], unfilt_backproj_tconv_layer.__class__.__name__, params_id, unfilt_backproj_full_tconv_time)
			## Unfiltered backprojection FULL (batch implementation)
			unfilt_backproj_batch_layer = UnfiltBackproj3DBatch()
			(unfilt_backproj_full_batch_time, H_hat_full_batch_np) = time_full_unfilt_backproj(unfilt_backproj_batch_layer, y=csph_full, W=conv3d_txy.weight)
			append_benchmark_data(benchmark_data[device.type], unfilt_backproj_batch_layer.__class__.__name__, params_id, unfilt_backproj_full_batch_time)

			#### Unfiltered Backprojection SEPARABLE Implementations
			## Unfiltered backprojection SEPARABLE-tfirst (for loop implementation)
			unfilt_backproj_separable1D2D_forloop_layer = UnfiltBackproj3DSeparable1D2DForLoop(apply_W1d_first=True)
			(unfilt_backproj_separable1D2D_forloop_time, H_hat_separable1D2D_forloop_np) = time_separable_unfilt_backproj(unfilt_backproj_separable1D2D_forloop_layer, y=csph_full, W1d=conv3d_t.weight, W2d=conv3d_xy.weight)
			append_benchmark_data(benchmark_data[device.type], unfilt_backproj_separable1D2D_forloop_layer.__class__.__name__+'_t1st', params_id, unfilt_backproj_separable1D2D_forloop_time)
			## Unfiltered backprojection SEPARABLE-xyfirst (for loop implementation)
			unfilt_backproj_separable2D1D_forloop_layer = UnfiltBackproj3DSeparable1D2DForLoop(apply_W1d_first=False)
			(unfilt_backproj_separable2D1D_forloop_time, H_hat_separable2D1D_forloop_np) = time_separable_unfilt_backproj(unfilt_backproj_separable2D1D_forloop_layer, y=csph_full, W1d=conv3d_t.weight, W2d=conv3d_xy.weight)
			append_benchmark_data(benchmark_data[device.type], unfilt_backproj_separable2D1D_forloop_layer.__class__.__name__+'_xy1st', params_id, unfilt_backproj_separable2D1D_forloop_time)
			## Unfiltered backprojection SEPARABLE-tfirst (for loop implementation)
			unfilt_backproj_separable1D2D_tconv_layer = UnfiltBackproj3DSeparable1D2DTransposeConv(apply_W1d_first=True)
			(unfilt_backproj_separable1D2D_tconv_time, H_hat_separable1D2D_tconv_np) = time_separable_unfilt_backproj(unfilt_backproj_separable1D2D_tconv_layer, y=csph_full, W1d=conv3d_t.weight, W2d=conv3d_xy.weight)
			append_benchmark_data(benchmark_data[device.type], unfilt_backproj_separable1D2D_tconv_layer.__class__.__name__+'_t1st', params_id, unfilt_backproj_separable1D2D_tconv_time)
			## Unfiltered backprojection SEPARABLE-xyfirst (for loop implementation)
			unfilt_backproj_separable2D1D_tconv_layer = UnfiltBackproj3DSeparable1D2DTransposeConv(apply_W1d_first=False)
			(unfilt_backproj_separable2D1D_tconv_time, H_hat_separable2D1D_tconv_np) = time_separable_unfilt_backproj(unfilt_backproj_separable2D1D_tconv_layer, y=csph_full, W1d=conv3d_t.weight, W2d=conv3d_xy.weight)
			append_benchmark_data(benchmark_data[device.type], unfilt_backproj_separable2D1D_tconv_layer.__class__.__name__+'_xy1st', params_id, unfilt_backproj_separable2D1D_tconv_time)

			plt.legend()

			io_ops.write_json(benchmark_fpath, benchmark_data)


			### OLD batch implementation (not very fast)
			# nn_tdim_up = torch.nn.Upsample(scale_factor=kernel3d_t, mode='nearest')
			# nn_xydim_up = torch.nn.Upsample(scale_factor=kernel3d_xy, mode='nearest')
			# try:
			# 	with Timer("Unfiltered Backproj SEPARABLE (batch, t first):"):
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

			# assert(H_hat_full_tconv.shape == H_hat_full.shape), "Dimension mismatch between H_hat_full and H_hat_tconv"



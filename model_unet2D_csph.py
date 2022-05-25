#### Standard Library Imports

#### Library imports
import torch
from torch.autograd import Variable
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_base_spad_lit import LITBaseSPADModel
from csph_layers import CSPH1DGlobalEncodingLayer
from model_unet import Unet
from layers_parametric1D import Gaussian1DLayer, IRF1DLayer
from tof_utils import linearize_phase, TWOPI



class LITUnet2DCSPH1D(LITBaseSPADModel):
	def __init__(self 
		, init_lr = 1e-4
		, p_tv = 1e-5 
		, lr_decay_gamma = 0.9
		, data_loss_id = 'L1'
		, k=16
		, num_bins=1024
		, init = 'HybridGrayFourier'
		, h_irf = None
		, optimize_csph = False
		):
		unet_model = Unet(
						in_channels=k, 
						out_channels=self.get_out_channels(), 
						nf0=64, 
						num_down=3, 
						max_channels=512, 
						upsampling_mode='nearest', 
						use_dropout=False, 
						outermost_linear=self.get_outermost_linear()
						)

		# Init parent class
		super(LITUnet2DCSPH1D, self).__init__(
			backbone_net=unet_model,
			init_lr=init_lr, 
			p_tv=p_tv, 
			lr_decay_gamma=lr_decay_gamma,
			data_loss_id = data_loss_id,
			)

		self.csph1D_layer = CSPH1DGlobalEncodingLayer(k = k, num_bins = num_bins, init = init, h_irf = h_irf, optimize_weights=optimize_csph)

		## Replace the hist reconstruction layer
		gauss_layer = Gaussian1DLayer(gauss_len=num_bins, out_dim=-3)
		if(h_irf is None):
			self.hist_rec_layer = torch.nn.Sequential(
				gauss_layer
			)
		else:
			irf_layer = IRF1DLayer(h_irf, conv_dim=0)
			self.hist_rec_layer = torch.nn.Sequential(
				gauss_layer, 
				irf_layer
			)

	def get_out_channels(self): return 1

	def get_outermost_linear(self): return False

	def forward(self, x):
		B = self.csph1D_layer(x.squeeze(1))
		# use forward for inference/predictions
		out_depths = self.backbone_net(B)
		out_hist = self.hist_rec_layer(out_depths)
		return (out_hist.squeeze(1), out_depths)

class LITUnet2DCSPH1D2Phasor(LITUnet2DCSPH1D):
	def __init__(self, *args, **kwargs):
		# Init parent class
		super(LITUnet2DCSPH1D2Phasor, self).__init__(*args, **kwargs)

	def get_out_channels(self): return 2

	def get_outermost_linear(self): return True

	def rec2depth(self, rec):
		'''
			For some models the reconstructed signal are not depths 
		'''
		phase = torch.arctan2(rec[:, 1, :], rec[:, 0, :]).unsqueeze(1) # Imag/Sin part goes on numerator
		linear_phase = linearize_phase(phase)
		return linear_phase / TWOPI

	def forward(self, x):
		B = self.csph1D_layer(x.squeeze(1))
		# use forward for inference/predictions
		out_phasor = self.backbone_net(B)
		out_depths = self.rec2depth(out_phasor)
		out_hist = self.hist_rec_layer(out_depths)
		return (out_hist.squeeze(1), out_phasor)

class LITUnet2DCSPH1D2FFTHist(LITUnet2DCSPH1D):
	def __init__(self, *args, num_bins=1024, **kwargs):
		# Init parent class
		self.num_bins = num_bins
		super(LITUnet2DCSPH1D2FFTHist, self).__init__(*args, **kwargs)
		self.smax_op = torch.nn.Softmax2d()

	def get_out_channels(self): return 2*((self.num_bins // 2) + 1)

	def get_outermost_linear(self): return True

	def rec2depth(self, rec):
		'''
			For some models the reconstructed signal are not depths 
		'''
		real = rec[:, 0::2, :, :]
		imag = rec[:, 1::2, :, :]
		rec_hist = torch.fft.irfft(real+1j*imag, dim=-3, n=self.num_bins)
		weights = Variable(torch.linspace(0, 1, steps=self.num_bins).unsqueeze(1).unsqueeze(1)).to(rec_hist.device)
		weighted_smax = weights * self.smax_op(200*rec_hist)
		soft_argmax = weighted_smax.sum(1).unsqueeze(1)
		# if(self.global_step == 300):
		# 	breakpoint()
		return soft_argmax

	def forward(self, x):
		B = self.csph1D_layer(x.squeeze(1))
		# use forward for inference/predictions
		out_ffthist = self.backbone_net(B)
		out_depths = self.rec2depth(out_ffthist)
		out_hist = self.hist_rec_layer(out_depths)
		return (out_hist.squeeze(1), out_ffthist)
class LITUnet2DCSPH1DLinearOut(LITUnet2DCSPH1D):
	def __init__(self, *args, **kwargs):
		super(LITUnet2DCSPH1DLinearOut, self).__init__(*args, **kwargs)

	def get_outermost_linear(self): return True

if __name__=='__main__':
	import matplotlib.pyplot as plt
	from model_utils import count_parameters
	from model_base_spad_lit import normdepth2phasor

	# Set random input
	batch_size = 2
	(nr, nc, nt) = (64, 64, 1024) 
	inputs = torch.randn((batch_size, 1, nt, nr, nc))

	simple_hist_input = torch.zeros((2, 1, nt, 32, 32))
	simple_hist_input[0, 0, 100, 0, 0] = 3
	simple_hist_input[0, 0, 200, 0, 0] = 1
	simple_hist_input[0, 0, 50, 0, 0] = 1
	simple_hist_input[0, 0, 540, 0, 0] = 1

	simple_hist_input[1, 0, 300, 0, 0] = 2
	simple_hist_input[1, 0, 800, 0, 0] = 1
	simple_hist_input[1, 0, 34, 0, 0] = 1
	simple_hist_input[1, 0, 900, 0, 0] = 1

	# Set compression params
	k = 16
	model = LITUnet2DCSPH1D(k=k, num_bins=nt)
	outputs = model(inputs)
	print("LITUnet2DCSPH1D model: {} params".format(count_parameters(model)))
	print("		inputs shape: {}".format(inputs.shape))
	print("		outputs1 shape: {}".format(outputs[0].shape))
	print("		outputs2 shape: {}".format(outputs[1].shape))
	print("		outputs2 min: {}".format(outputs[1].min()))
	print("		outputs2 max: {}".format(outputs[1].max()))
	print("		outputs2 mean: {}".format(outputs[1].mean()))

	## Test phasor estimates
	model = LITUnet2DCSPH1D2Phasor(k=k, num_bins=nt)
	print("LITUnet2DCSPH1D model: {} params".format(count_parameters(model)))
	outputs = model(inputs)
	test_depths = torch.rand((1, 1, 2, 2))
	test_phasor = normdepth2phasor(test_depths)
	test_rec_depths = model.rec2depth(test_phasor)
	print("		inputs shape: {}".format(inputs.shape))
	print("		outputs1 shape: {}".format(outputs[0].shape))
	print("		outputs2 shape: {}".format(outputs[1].shape))
	print("		outputs2 min: {}".format(outputs[1].min()))
	print("		outputs2 max: {}".format(outputs[1].max()))
	print("		outputs2 mean: {}".format(outputs[1].mean()))
	print('		test_depths: {}'.format(test_depths))
	# print('		test_phasor: {}'.format(test_phasor))
	print('		test_rec_depths: {}'.format(test_rec_depths))

	## Test IFFT 
	model = LITUnet2DCSPH1D2FFTHist(k=k, num_bins=nt)
	print("LITUnet2DCSPH1D model: {} params".format(count_parameters(model)))
	outputs = model(inputs)
	# test_depths = torch.rand((1, 1, 2, 2))
	# test_phasor = normdepth2phasor(test_depths)
	# test_rec_depths = model.rec2depth(test_phasor)
	# print("		inputs shape: {}".format(inputs.shape))
	# print("		outputs1 shape: {}".format(outputs[0].shape))
	# print("		outputs2 shape: {}".format(outputs[1].shape))
	# print("		outputs2 min: {}".format(outputs[1].min()))
	# print("		outputs2 max: {}".format(outputs[1].max()))
	# print("		outputs2 mean: {}".format(outputs[1].mean()))
	# print('		test_depths: {}'.format(test_depths))
	# # print('		test_phasor: {}'.format(test_phasor))
	# print('		test_rec_depths: {}'.format(test_rec_depths))


	# ## Test with simple inputs
	# out, zncc, B = model.forward_test(simple_hist_input)
	# print("Input Shape: {}".format(simple_hist_input.shape))
	# print("ZNCC Out Shape: {}".format(zncc.shape))
	# print("B Out Shape: {}".format(B.shape))
	
	# # Look at outputs
	# plt.clf()
	# plt.plot(simple_hist_input[0,0,:,0,0], '--', label="Inputs 2")
	# plt.plot(zncc[0,0,:,0,0], label='ZNCC Outputs 2')
	# plt.plot(simple_hist_input[1,0,:,0,0], '--', label="Inputs 3")
	# plt.plot(zncc[1,0,:,0,0], label='ZNCC Outputs 3')
	# plt.legend()






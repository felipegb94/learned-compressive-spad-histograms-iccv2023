# The SPAD data pre-process function

#### Standard Library Imports
import os
from glob import glob

#### Library imports
import numpy as np
import scipy
import scipy.io
import torch
import torch.utils.data
import torch.nn.functional as F

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

def normalize_matlab_depth_bins(range_bins, num_bins):
	# make range 0-n_bins-1 instead of 1-n_bins
	norm_range_bins = (range_bins - 1) / (num_bins - 1) 
	return norm_range_bins

def bins2hist(range_bins, num_bins):
	range_bins = range_bins.squeeze()
	(nt, nr, nc) = (num_bins,) + range_bins.shape
	hist = np.zeros((nt,nr,nc))
	for i in range(nr):
		for j in range(nc):
			bin_idx = range_bins[i,j]
			hist[bin_idx, i,j] = 1.
	return hist

def pad_tdim(inputs, tblock_len):
	'''
		temporal dim padding:
			- use circular pad if inputs are larger than kernels since it is a periodic signal
				* We will crop things after recovery
			- use zero pad if inputs are smaller
				* We will crop things after recovery
			- do nothing if inputs are correct size
	'''
	num_tbins = inputs.shape[-3]
	if((num_tbins % tblock_len) == 0): tdim_pad = 0
	else: tdim_pad =  tblock_len - (num_tbins % tblock_len)
	## set padding mode
	# if inputs are larger in time dimension - circular pad, otherwise pad with 0's
	pad_mode = "constant" #
	if(num_tbins > tblock_len): pad_mode = "circular"
	## set padding mode
	if(tdim_pad == 0):
		# do nothing
		return inputs
	else:
		if(len(inputs.shape) == 4):
			## Padding for 3D inputs expects a 5D tensor, so we have to unsqueeze for it to work
			## See: https://discuss.pytorch.org/t/problems-using-torch-nn-functional-pad/85326
			return F.pad(inputs.unsqueeze(0), (0, 0, 0, 0, 0, tdim_pad), mode=pad_mode).squeeze(0)
		elif(len(inputs.shape) == 5):
			return F.pad(inputs, (0, 0, 0, 0, 0, tdim_pad), mode=pad_mode)
		else:
			assert(False), "inputs should be 4D or 5D tensors"

def pad_xydim(inputs, num_kernel_rows, num_kernel_cols):
	'''
		Use reflection padding on the spatial boundaries, and only pad on one side of the dimension since we will crop later the outputs to end up with a smaller input
	'''
	assert(inputs.shape[-1] > num_kernel_cols), "num cols should be larger than kernel"
	assert(inputs.shape[-2] > num_kernel_rows), "num rows should be larger than kernel"
	assert(len(inputs.shape) >= 4), "to pad rows and cols inputs should be a 4D tensor"
	if((inputs.shape[-1] % num_kernel_cols) == 0): col_dim_pad = 0
	else: col_dim_pad =  num_kernel_cols - (inputs.shape[-1] % num_kernel_cols)
	if((inputs.shape[-2] % num_kernel_rows) == 0): row_dim_pad = 0
	else: row_dim_pad =  num_kernel_rows - (inputs.shape[-2] % num_kernel_rows)
	return F.pad(inputs, (0, col_dim_pad, 0, row_dim_pad), mode="reflect")


class SpadDataset(torch.utils.data.Dataset):
	def __init__(self, datalist_fpath, noise_idx=None, output_size=None, disable_rand_crop=False):
		"""__init__
		:param datalist_fpath: path to text file with list of spad data files
		:param noise_idx: the noise index list to include in the dataset (e.g., 1 or 2
		:param output_size: the output size after random crop. If set to None it will output the full image
		"""

		with open(datalist_fpath) as f: 
			self.spad_data_fpaths_all = f.read().split()

		self.datalist_fpath = datalist_fpath
		self.datalist_fname = os.path.splitext(os.path.basename(datalist_fpath))[0]

		self.noise_idx = noise_idx
		self.spad_data_fpaths = []
		if(noise_idx is None):
			self.spad_data_fpaths = self.spad_data_fpaths_all
		else:
			if(isinstance(noise_idx, int)):
				noise_idx = [noise_idx]
			assert(isinstance(noise_idx, list)), "Input noise idx should be an int or list"
			for fpath in self.spad_data_fpaths_all:
				for idx in noise_idx:
					# If noise_idx in current fpath, add it, and continue to next fpath
					if('_p{}'.format(idx) in os.path.basename(fpath)):
						self.spad_data_fpaths.append(fpath)
						break

		## Get the base dirpath of the whole dataset. Sometimes the dataset will have multiple folder levels
		dataset_base_dirpath = os.path.dirname(os.path.commonprefix(self.spad_data_fpaths))

		## Try to load PSF vector --> We may use it in some models
		## Find all files that start with PSF
		pattern = os.path.join(dataset_base_dirpath, 'PSF_used_for_simulation*.mat')
		psf_fpaths = glob(pattern)
		if(len(psf_fpaths) >= 1):
			# If file exists load it and use it
			if(len(psf_fpaths) > 1):
				print("Warning: More than one PSF/IRF avaiable. Choosing {}".format(psf_fpaths[0]))
			psf_data = scipy.io.loadmat(psf_fpaths[0])
			self.psf = psf_data['psf'].mean(axis=-1)
		else:
			# If file does not exist, just generate simple psf
			print("No PSF/IRF available. Generating a simple narrow Gaussian")
			from research_utils.signalproc_ops import gaussian_pulse
			psf_len = 11
			psf_mu = psf_len // 2
			self.psf = gaussian_pulse(time_domain=np.arange(0,psf_len), mu=psf_mu, width=0.5)

		if(isinstance(output_size, int)): self.output_size = (output_size, output_size)
		else: self.output_size = output_size
		self.disable_rand_crop = disable_rand_crop

		(_, _, _, tres_ps) = self.get_spad_data_sample_params(idx=0)
		self.tres_ps = tres_ps # time resolution in picosecs

		print("SpadDataset with {} files".format(len(self.spad_data_fpaths)))

	def __len__(self):
		return len(self.spad_data_fpaths)

	def get_spad_data_sample_id(self, idx):
		# Create a unique identifier for this file so we can use it to save model outputs with a filename that contains this ID
		spad_data_fname = self.spad_data_fpaths[idx]
		spad_data_id = self.datalist_fname + '/' + os.path.splitext(os.path.basename(spad_data_fname))[0]
		return spad_data_id

	def get_spad_data_sample_params(self, idx):
		'''
			Load the first sample and look at some of the parameters of the simulation
		'''
		# load spad data
		spad_data_fname = self.spad_data_fpaths[idx]
		spad_data = scipy.io.loadmat(spad_data_fname)
		# SBR = spad_data['SBR'].squeeze()
		# mean_signal_photons = spad_data['mean_signal_photons'].squeeze()
		# mean_background_photons = spad_data['mean_background_photons'].squeeze()
		(nr, nc, nt) = spad_data['rates'].shape
		tres_ps = spad_data['bin_size'].squeeze()*1e12
		return (nr, nc, nt, tres_ps)

	def tryitem(self, idx):
		'''
			Try to load the spad data sample.
			* All normalization and data augmentation happens here.
		'''
		# load spad data
		spad_data_fname = self.spad_data_fpaths[idx]
		spad_data = scipy.io.loadmat(spad_data_fname)
		
		# normalized pulse as GT histogram
		rates = np.asarray(spad_data['rates']).astype(np.float32)
		(nr, nc, n_bins) = rates.shape
		rates = rates[np.newaxis,: ]
		rates = np.transpose(rates, (0, 3, 1, 2))
		rates = rates / (np.sum(rates, axis=-3, keepdims=True) + 1e-8)

		# simulated spad measurements
		# Here we need to swap the rows and cols because matlab saves dimensions in a different order.
		spad = np.asarray(scipy.sparse.csc_matrix.todense(spad_data['spad'])).astype(np.float32)
		spad = spad.reshape((nc, nr, n_bins))
		spad = spad[np.newaxis, :]
		spad = np.transpose(spad, (0, 3, 2, 1))

		# # ground truth depths in units of bins
		bins = np.asarray(spad_data['bin']).astype(np.float32)
		bins = bins[np.newaxis, :]
		bins = normalize_matlab_depth_bins(bins, n_bins)

		# # Estimated argmax depths from spad measurements
		est_bins_argmax = np.asarray(spad_data['est_range_bins_argmax'])
		# Generate hist from bin indeces
		est_bins_argmax_hist = bins2hist(est_bins_argmax-1, num_bins=n_bins).astype(np.float32)
		est_bins_argmax_hist = est_bins_argmax_hist[np.newaxis,:]
		# Normalize the bin indeces
		est_bins_argmax = est_bins_argmax[np.newaxis,:].astype(np.float32)
		est_bins_argmax = normalize_matlab_depth_bins(est_bins_argmax, n_bins)

		# Compute random crop if neeeded
		(h, w) = (nr, nc)
		if(self.output_size is None):
			new_h = h
			new_w = w
		else:
			new_h = self.output_size[0]
			new_w = self.output_size[1]

		if(self.disable_rand_crop):
			top = 0
			left = 0        
		else:
			# add 1 because randint produces between low <= x < high 
			top = np.random.randint(0, h - new_h + 1) 
			left = np.random.randint(0, w - new_w + 1)

		rates = rates[..., top:top + new_h, left:left + new_w]
		spad = spad[..., top:top + new_h, left: left + new_w]
		bins = bins[..., top: top + new_h, left: left + new_w]
		est_bins_argmax = est_bins_argmax[..., top: top + new_h, left: left + new_w]
		est_bins_argmax_hist = est_bins_argmax_hist[..., top: top + new_h, left: left + new_w]
		rates = torch.from_numpy(rates)
		spad = torch.from_numpy(spad)
		bins = torch.from_numpy(bins)
		est_bins_argmax = torch.from_numpy(est_bins_argmax)
		est_bins_argmax_hist = torch.from_numpy(est_bins_argmax_hist)
		# make sure these parameters have a supported data type by dataloader (uint16 is not supported)
		spad_data['SBR'] = spad_data['SBR'].astype(np.float32)
		spad_data['mean_signal_photons'] = spad_data['mean_signal_photons'].astype(np.int32)
		spad_data['mean_background_photons'] = spad_data['mean_background_photons'].astype(np.int32)

		sample = {
			'rates': rates 
			, 'spad': spad 
			, 'bins': bins 
			, 'est_bins_argmax': est_bins_argmax 
			, 'est_bins_argmax_hist': est_bins_argmax_hist 
			, 'SBR': spad_data['SBR'][0,0]
			, 'mean_signal_photons': spad_data['mean_signal_photons'][0,0]
			, 'mean_background_photons': spad_data['mean_background_photons'][0,0]
			, 'idx': idx
			}

		return sample

	def __getitem__(self, idx):
		try:
			sample = self.tryitem(idx)
		except Exception as e:
			print(idx, e)
			idx = idx + 1
			sample = self.tryitem(idx)
		return sample


class Lindell2018LinoSpadDataset(torch.utils.data.Dataset):
	def __init__(self, datalist_fpath, dims=(1536,256,256), tres_ps=26, encoding_kernel_dims=None):
		"""__init__
		:param datalist_fpath: path to text file with list of spad data files
		:param encoding_kernel_dims: a triplet with the dimensions of the encoding kernel used in the CSPH encoding layer. This is used to calculate padding
		"""

		self.datalist_fpath = datalist_fpath
		self.datalist_fname = os.path.splitext(os.path.basename(datalist_fpath))[0]


		self.tres_ps = tres_ps # time resolution in picosecs (used at test time for computing depths)
		assert(len(dims) == 3), "Input dims should have 3 elements"
		(self.max_nt, self.max_nr, self.max_nc) = dims
		self.encoding_kernel_dims = encoding_kernel_dims
		self.pad_inputs = True
		if(self.encoding_kernel_dims is None): self.pad_inputs = False
		else: 
			assert(len(encoding_kernel_dims) == 3), "Encoding kernel dims should have 3 elements"
			# only pad if input dims are not divisible encoding kernel dims
			if(((self.max_nc % self.encoding_kernel_dims[-1]) == 0) and ((self.max_nr % self.encoding_kernel_dims[-2]) == 0) and ((self.max_nt % self.encoding_kernel_dims[-3]) == 0)):
				self.pad_inputs = False

		## Get all the filepaths
		with open(datalist_fpath) as f: 
			self.spad_data_fpaths_all = f.read().split()
		self.spad_data_fpaths = self.spad_data_fpaths_all

		## Get the base dirpath of the whole dataset. Sometimes the dataset will have multiple folder levels
		dataset_base_dirpath = os.path.dirname(os.path.commonprefix(self.spad_data_fpaths))

		## Try to load PSF vector --> We may use it in some models
		## If there is not PSF just generate a very narrow one
		## Find all files that start with PSF
		pattern = os.path.join(dataset_base_dirpath, 'PSF_used_for_simulation*.mat')
		psf_fpaths = glob(pattern)
		if(len(psf_fpaths) >= 1):
			# If file exists load it and use it
			if(len(psf_fpaths) > 1):
				print("Warning: More than one PSF/IRF avaiable. Choosing {}".format(psf_fpaths[0]))
			psf_data = scipy.io.loadmat(psf_fpaths[0])
			self.psf = psf_data['psf'].mean(axis=-1)
		else:
			# If file does not exist, just generate simple psf
			print("No PSF/IRF available. Generating a simple narrow Gaussian")
			from research_utils.signalproc_ops import gaussian_pulse
			psf_len = 11
			psf_mu = psf_len // 2
			self.psf = gaussian_pulse(time_domain=np.arange(0,psf_len), mu=psf_mu, width=0.5)

		print("SpadDataset with {} files".format(len(self.spad_data_fpaths)))

	def __len__(self):
		return len(self.spad_data_fpaths)

	def get_spad_data_sample_id(self, idx):
		# Create a unique identifier for this file so we can use it to save model outputs with a filename that contains this ID
		spad_data_fname = self.spad_data_fpaths[idx]
		spad_data_id = self.datalist_fname + '/' + os.path.splitext(os.path.basename(spad_data_fname))[0]
		return spad_data_id

	def tryitem(self, idx):
		'''
			Try to load the spad data sample.
			* All normalization and data augmentation happens here.
		'''
		# load spad data
		spad_data_fname = self.spad_data_fpaths[idx]
		spad_data = scipy.io.loadmat(spad_data_fname)
		frame_num = 0 # frame number to use (some data files have multiple frames in them)

		# ## load intensity img
		# intensity_imgs = np.asarray(spad_data['cam_processed_data'])[0]
		# intensity_img = np.asarray(intensity_imgs[frame_num]).astype(np.float32)

		# spad measurements
		spad_sparse_data = np.asarray(spad_data['spad_processed_data'])[0]
		spad = np.asarray(scipy.sparse.csc_matrix.todense(spad_sparse_data[frame_num])).astype(np.float32)
		spad = spad.reshape((self.max_nt, self.max_nr, self.max_nc)).swapaxes(-1,-2)
		spad = spad[np.newaxis, :]

		# crop spatial dimension to avoid out of memory errors
		# only needed in the high-resolution dataset
		if((self.max_nr == 256) or (self.max_nc == 256)):
			# spad = spad[:, :, 40:216, 40:216]
			# spad = spad[:, :, 32:224, 32:224]
			# spad = spad[:, :, 0::2, 0::2]
			spad = spad[:, :, 0::4, 0::4]
		# no gt available here so just use spad measurmeents
		rates = np.array(spad)
		rates = rates / (np.sum(rates, axis=-3, keepdims=True) + 1e-8)

		# Estimated argmax depths from spad measurements
		est_bins_argmax = np.argmax(spad, axis=-3)
		# Generate hist from bin indeces
		est_bins_argmax_hist = bins2hist(est_bins_argmax, num_bins=self.max_nt).astype(np.float32)
		est_bins_argmax_hist = est_bins_argmax_hist[np.newaxis,:]
		# Normalize the bin indeces
		est_bins_argmax = est_bins_argmax.astype(np.float32) / spad.shape[-3]

		# ground truth depths in units of bins
		bins = np.array(est_bins_argmax).astype(np.float32)

		# crop and convert to tensor
		rates = torch.from_numpy(rates)
		spad = torch.from_numpy(spad)
		bins = torch.from_numpy(bins)
		est_bins_argmax = torch.from_numpy(est_bins_argmax)
		est_bins_argmax_hist = torch.from_numpy(est_bins_argmax_hist)

		# # Only pad inputs. Do not pad outputs. The outputs shape is what will tell us later how to crop the inputs after the CSPH encoding and decoding steps
		if(self.pad_inputs): 
			spad = pad_xydim(pad_tdim(spad, self.encoding_kernel_dims[-3]), self.encoding_kernel_dims[-2], self.encoding_kernel_dims[-1]) 

		# set temp values for sbr, signal and bkg params 
		sbr = 0.
		mean_signal_photons = 0.
		mean_background_photons = 0.

		sample = {
			'rates': rates 
			, 'spad': spad 
			, 'bins': bins 
			, 'est_bins_argmax': est_bins_argmax 
			, 'est_bins_argmax_hist': est_bins_argmax_hist 
			, 'SBR': sbr
			, 'mean_signal_photons': mean_signal_photons
			, 'mean_background_photons': mean_background_photons
			, 'idx': idx
			}

		return sample

	def __getitem__(self, idx):
		try:
			sample = self.tryitem(idx)
		except Exception as e:
			print(idx, e)
			idx = idx + 1
			sample = self.tryitem(idx)
		return sample

if __name__=='__main__':
	import matplotlib.pyplot as plt
	'''
		Comment/Uncomment the dataset that you want to load a few samples from. Datasets implemented:
		* middlebury test dataset
		* nyuv2 val dataset
		* Linospad test dataset from Lindell et al., 2018
	'''


	# ## Middlebury Test dataset
	# datalist_fpath = './datalists/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0.txt'
	# noise_idx = None
	# spad_dataset = SpadDataset(datalist_fpath, noise_idx=noise_idx, output_size=None)

	# ## nyuv2 val dataset
	# datalist_fpath = './datalists/nyuv2_val_SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1.txt'
	# noise_idx = [1]
	# spad_dataset = SpadDataset(datalist_fpath, noise_idx=noise_idx, output_size=None)

	## Lindell2018 LinoSpad test dataset
	datalist_fpath = './datalists/test_lindell_linospad_captured.txt'
	noise_idx = None
	spad_dataset = Lindell2018LinoSpadDataset(datalist_fpath)

	batch_size = 1

	loader = torch.utils.data.DataLoader(spad_dataset, batch_size=1, shuffle=True, num_workers=0)
	iter_loader = iter(loader)
	for i in range(5):
		spad_sample = iter_loader.next()
		
		spad = spad_sample['spad']
		rates = spad_sample['rates']
		bins = spad_sample['bins']
		est_bins_argmax = spad_sample['est_bins_argmax']
		idx = spad_sample['idx']
		print(idx)

		spad = spad.cpu().detach().numpy()[0,:].squeeze()
		rates = rates.cpu().detach().numpy()[0,:].squeeze()
		bins = bins.cpu().detach().numpy()[0,:].squeeze()
		est_bins_argmax = est_bins_argmax.cpu().detach().numpy()[0,:].squeeze()
		# idx = idx.cpu().detach().numpy().squeeze()
		# 
		print("     Rates: {}", rates.shape)        
		print("     SPAD dims: {}", spad.shape)        

		plt.clf()
		plt.subplot(2,2,1)
		plt.imshow(bins)
		plt.subplot(2,2,2)
		plt.imshow(np.argmax(spad, axis=0) / spad.shape[0], vmin=bins.min(), vmax=bins.max())
		plt.subplot(2,2,3)
		plt.imshow(np.log(1+np.sum(spad, axis=0)))
		plt.subplot(2,2,4)
		plt.imshow(est_bins_argmax, vmin=bins.min(), vmax=bins.max())
		plt.pause(0.2)

	if(not ('lindell' in datalist_fpath)):
		plt.clf()
		plt.plot(spad_dataset.psf)
		plt.title("PSF used to simulate thiis dataset")
		plt.pause(0.1)


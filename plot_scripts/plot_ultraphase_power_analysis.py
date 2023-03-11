'''
	This script creates lines plots of compression vs. different test set metrics
'''
#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from IPython.core import debugger
from torch import norm
breakpoint = debugger.set_trace

#### Local imports
from tof_utils import *
from research_utils import plot_utils
from analyze_test_results_utils import get_hydra_io_dirpaths

'''
	processing_tclk: how many clock cycles are needed by one core to process one timestamp
	readout_tclk: how many clock cycles are needed to read out one core
	data_bytes: how many bytes are read per core
	data_words: how many memory locations are read per core
'''

'''
	processing_power: average power consumed when processing all photons in all pixels at the specified frame rate
	est_swisspad2_readout_power: Estimated power dissipation in the I/O interface assuming the same readout scheme as SwissSPAD2 and operating at the specific framerate
'''

if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_ultraphase_power')
	
	## Ultraphase numbers
	# 2.38 nanoseconds
	Tclk = 2.38e-9
	# 94.6 GOps/W
	core_power = 94.6e9 # Ops per wayy
	# instruction power
	inst_power = 1 / (core_power)

	# ## Input Parameters
	n_photons = 500
	n_pixels = 4
	fps = 30
	out_fname = 'ultraphase_power_nphotons-{}_npixels-{}_fps-{}'.format(n_photons, n_pixels, fps)

	df = pd.DataFrame(columns=['name', 'n_pixels', "n_photon_timestamps_per_frame", "fps", "processing_power", "est_swisspad2_readout_power"])

	# ## Data for: 2x2 + 210 photons
	if(n_photons == 210 and n_pixels == 4 and fps == 30):
		fourier_k8_data = {
			"name": ["fourier_k8"]
			, "processing_power": [1.96e-09]
			, "est_swisspad2_readout_power": [8.36e-07]
		}
		hybrid_learned_fourier_k8_data = {
			"name": ["hybrid_learned_fourier_k8"]
			, "processing_power": [2.60e-09]	
			, "est_swisspad2_readout_power": [8.36e-07]
		}

		coarsehist_k16_data = {
			"name": ["coarsehist_k16"]
			, "processing_power": [3.99e-11]
			, "est_swisspad2_readout_power": [8.36e-07]
		}
		coarsehist_k64_data = {
			"name": ["coarsehist_k64"]
			, "processing_power": [3.24e-11]
			, "est_swisspad2_readout_power": [3.35e-06]		
		}
	elif(n_photons == 500 and n_pixels == 4 and fps == 30):
		## Data for: 2x2 + 500 photons + 30 fps
		fourier_k8_data = {
			"name": ["fourier_k8"]
			, "processing_power": [1.11e-08]
			, "est_swisspad2_readout_power": [8.36e-07]
		}
		hybrid_learned_fourier_k8_data = {
			"name": ["hybrid_learned_fourier_k8"]
			, "processing_power": [1.46E-08]	
			, "est_swisspad2_readout_power": [8.36e-07]
		}

		coarsehist_k16_data = {
			"name": ["coarsehist_k16"]
			, "processing_power": [2.26E-10]
			, "est_swisspad2_readout_power": [8.36e-07]
		}
		coarsehist_k64_data = {
			"name": ["coarsehist_k64"]
			, "processing_power": [1.45E-10]
			, "est_swisspad2_readout_power": [3.35e-06]		
		}
	else: assert(False),"no data for input params"


	fourier_k8_data = pd.DataFrame.from_dict(fourier_k8_data)
	hybrid_learned_fourier_k8_data = pd.DataFrame.from_dict(hybrid_learned_fourier_k8_data)
	coarsehist_k16_data = pd.DataFrame.from_dict(coarsehist_k16_data)
	coarsehist_k64_data = pd.DataFrame.from_dict(coarsehist_k64_data)
	data = pd.concat([fourier_k8_data, hybrid_learned_fourier_k8_data,coarsehist_k16_data,coarsehist_k64_data])
	data.loc[:, 'processing_power'] *= 1e6
	data.loc[:, 'est_swisspad2_readout_power'] *= 1e6

	model_names = data.loc[:, 'name'].values
	processing_power = data.loc[:, 'processing_power'].values
	est_swisspad2_readout_power = data.loc[:, 'est_swisspad2_readout_power'].values


	df = pd.DataFrame(
			{
				'processing_power': processing_power
				, 'est_swisspad2_readout_power': est_swisspad2_readout_power
		    }, index = model_names
			)
	
	plt.clf()
	fig = plt.gcf()
	ax = plt.gca()
	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	plot_utils.update_fig_size(fig, height=2,width=8)
	width = 0.8
	df.plot( kind= 'bar', ax=ax , secondary_y= 'est_swisspad2_readout_power', rot=0, width=width)
	ax.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=True,      # ticks along the bottom edge are off
		top=False,         # ticks along the top edge are off
		labelbottom=False) # labels along the bottom edge are off
	plot_utils.set_ticks(ax, fontsize=13)
	ax.tick_params(axis='y', colors=colors[0])
	
	ax2 = plt.gca()
	plot_utils.set_ticks(ax2, fontsize=13)
	ax2.tick_params(axis='y', colors=colors[1])
	ax.get_legend().remove()


	out_dirpath = 'results/raw_figures'
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname, file_ext='svg')


	# fig = plt.gcf()
	# plot_utils.update_fig_size(height=2,width=6)
	# ax = fig.add_subplot(111) # Create matplotlib axes
	# ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
	
	# data.plot.bar(color=colors[0], x='name', y='est_swisspad2_readout_power', width=width, ax=ax, position=1)
	# ax.tick_params(axis='y', colors=colors[0])
	# ax.get_legend().remove()

	# data.plot.bar(color=colors[1], x='name', y='processing_power', width=width, ax=ax2, position=0)
	# ax2.tick_params(axis='y', colors=colors[1])
	# ax2.get_legend().remove()

	# ax.tick_params(
	# 	axis='x',          # changes apply to the x-axis
	# 	which='both',      # both major and minor ticks are affected
	# 	bottom=True,      # ticks along the bottom edge are off
	# 	top=False,         # ticks along the top edge are off
	# 	labelbottom=False) # labels along the bottom edge are off
	

	# plt.clf()
	# ax = plt.gca()
	# # ax.bar(model_names, processing_power)
	# ax.bar(model_names, est_swisspad2_readout_power)
	# plt.ylim(0, est_swisspad2_readout_power.max())

	# ax2 = ax.twinx()
	# width = 0.4
	# ax2.bar(model_names, processing_power)

	# plt.yscale("log")




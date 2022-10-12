'''
	This script plots the test set results for models that were trained with different settings introduced in CSPH3D model
	The goal of this visualization is to find a good CSPH3D design that provides a good trade-off between performance and model size.
'''
#### Standard Library Imports
import os

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import io
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from plot_scripts import plot_scripts_utils
from tof_utils import *
from research_utils import plot_utils, np_utils, io_ops
from spad_dataset import SpadDataset
from analyze_test_results_utils import init_model_metrics_dict, process_middlebury_test_results, get_hydra_io_dirpaths


if __name__=='__main__':

	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_csph3d_test_set_metrics')

	## Add high flux test results
	plot_high_flux = True

	## output dirpaths
	experiment_name = 'middlebury/test_set_metrics'
	# out_dirpath = os.path.join(io_dirpaths.results_weekly_dirpath, experiment_name)
	out_dirpath = os.path.join(io_dirpaths.results_figures_dirpath, experiment_name)
	os.makedirs(out_dirpath, exist_ok=True)

	## Scene ID and Params
	test_set_id = 'test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0'
	middlebury_test_set_info = plot_scripts_utils.middlebury_test_set_info
	scene_ids = middlebury_test_set_info['scene_ids']
	sbr_params_low_flux = middlebury_test_set_info['sbr_params_low_flux']
	sbr_params_high_flux = middlebury_test_set_info['sbr_params_high_flux']
	sbr_params = sbr_params_low_flux + sbr_params_high_flux

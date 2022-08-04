'''
Plot the results generated by the unfiltered_backproj_benchmarking.py script
Make sure to run it 
'''

#### Standard Library Imports
import sys
import os
sys.path.append('../')
sys.path.append('./')

#### Library imports
import numpy as np
import matplotlib.pyplot as plt
import torch

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from research_utils import io_ops, plot_utils
from unfiltered_backproj_benchmarking import compose_input_size_string, compose_params_id_string

if __name__=='__main__':

	## Load benchmark data we want to look at
	benchmark_data_fpath = 'scripts/benchmark_results/unfilt_backproj_benchmark_QuadroP5000_batch=4_1024x32x32.json'
	# benchmark_data_fpath = 'scripts/benchmark_results/unfilt_backproj_benchmark_TITANXp_batch=4_1024x32x32.json'
	## Load benchmark data
	assert(os.path.exists(benchmark_data_fpath)), "benchmark data does not exist" 
	benchmark_data = io_ops.load_json(benchmark_data_fpath)

	benchmark_data_id = benchmark_data_fpath.split('/')[-1].split('unfilt_backproj_benchmark_')[-1].split('.json')[0] 
	out_dirpath = 'results/week_2022-08-01/unfilt_backproj_benchmark/{}'.format(benchmark_data_id)


	## get the cpu and gpu data
	cpu_benchmark_data = benchmark_data['cpu']
	gpu_benchmark_data = benchmark_data['cuda']

    ############ Plot block t-size vs. runtime #################
	k=64
	(bt, br, bc) = (64, 4, 4)
	brbc = (br,bc)
	bt_all = [32, 64, 128, 256, 512, 1024]
	fixed_params_id = 'k={}_btx{}x{}'.format(k,br,bc)
	out_fname = 'tblock_vs_runtime_{}'.format(fixed_params_id)
	
	def init_benchmark_fig():
		plt.clf()
		fig = plt.gcf()
		fig.set_size_inches(w=14,h=7)
		plt.suptitle(benchmark_data_id, fontsize=16)

	def finalize_benchmark_fig(xlabel):
		plt.subplot(1,2,1)
		plt.title("CPU Performance - {}".format(fixed_params_id),fontsize=14)
		plt.xlabel(xlabel, fontsize=14)
		plt.ylabel("Log Runtime Perf (secs)", fontsize=14)
		plot_utils.set_ticks(fontsize=12)
		plt.legend()
		plt.subplot(1,2,2)
		plt.title("GPU Performance - {}".format(fixed_params_id),fontsize=14)
		plt.xlabel(xlabel, fontsize=14)
		plt.ylabel("Log Runtime Perf (secs)", fontsize=14)
		plot_utils.set_ticks(fontsize=12)
		plt.legend()

	init_benchmark_fig()
	plt.subplot(1,2,1)
	for method_key in cpu_benchmark_data.keys():
		if(method_key == 'UnfiltBackproj3DBatch'): continue
		curr_benchmark_data = cpu_benchmark_data[method_key]
		runtime_perf = []
		for bt in bt_all:
			params_id = compose_params_id_string(k, bt, brbc[0], brbc[1])
			runtime_perf.append(np.log10(np.mean(cpu_benchmark_data[method_key][params_id])))
		plt.plot(bt_all, runtime_perf, '-*', linewidth=3, label=method_key)
	plt.subplot(1,2,2)
	for method_key in cpu_benchmark_data.keys():
		if(method_key == 'UnfiltBackproj3DBatch'): continue
		curr_benchmark_data = gpu_benchmark_data[method_key]
		runtime_perf = []
		for bt in bt_all:
			params_id = compose_params_id_string(k, bt, brbc[0], brbc[1])
			runtime_perf.append(np.log10(np.mean(gpu_benchmark_data[method_key][params_id])))
		plt.plot(bt_all, runtime_perf, '-*', linewidth=3, label=method_key)
	finalize_benchmark_fig(xlabel="Temporal Block Dim Size (bt)")

	plot_utils.save_currfig_png(dirpath=out_dirpath, filename=out_fname)

    ############ Plot block t-size vs. runtime #################
	k=64
	(bt, br, bc) = (1024, 4, 4)
	brbc = (br,bc)
	k_all = [32,64,128,256,512]
	fixed_params_id = '{}x{}x{}'.format(bt,br,bc)
	out_fname = 'k_vs_runtime_{}'.format(fixed_params_id)
	
	init_benchmark_fig()
	plt.subplot(1,2,1)
	for method_key in cpu_benchmark_data.keys():
		if(method_key == 'UnfiltBackproj3DBatch'): continue
		curr_benchmark_data = cpu_benchmark_data[method_key]
		runtime_perf = []
		for k in k_all:
			params_id = compose_params_id_string(k, bt, brbc[0], brbc[1])
			runtime_perf.append(np.log10(np.mean(cpu_benchmark_data[method_key][params_id])))
		plt.plot(k_all, runtime_perf, '-*', linewidth=3, label=method_key)
	plt.subplot(1,2,2)
	for method_key in cpu_benchmark_data.keys():
		if(method_key == 'UnfiltBackproj3DBatch'): continue
		curr_benchmark_data = gpu_benchmark_data[method_key]
		runtime_perf = []
		for k in k_all:
			params_id = compose_params_id_string(k, bt, brbc[0], brbc[1])
			runtime_perf.append(np.log10(np.mean(gpu_benchmark_data[method_key][params_id])))
		plt.plot(k_all, runtime_perf, '-*', linewidth=3, label=method_key)
	finalize_benchmark_fig(xlabel="Number of Filters (K)")
	plot_utils.save_currfig_png(dirpath=out_dirpath, filename=out_fname)

    ############ Plot block xy-size vs. runtime #################
	k=128
	(bt, br, bc) = (1024, 4, 4)
	brbc = (br,bc)
	brbc_all=[(1,1), (2,2), (4,4), (8,8)]
	brbc_all_labels = ['{}x{}'.format(brbc[0], brbc[1]) for brbc in brbc_all]
	fixed_params_id = 'k={}_{}xbrxbc'.format(k,bt)
	out_fname = 'spatial_vs_runtime_{}'.format(fixed_params_id)
	
	init_benchmark_fig()
	plt.subplot(1,2,1)
	for method_key in cpu_benchmark_data.keys():
		if(method_key == 'UnfiltBackproj3DBatch'): continue
		curr_benchmark_data = cpu_benchmark_data[method_key]
		runtime_perf = []
		for brbc in brbc_all:
			params_id = compose_params_id_string(k, bt, brbc[0], brbc[1])
			runtime_perf.append(np.log10(np.mean(cpu_benchmark_data[method_key][params_id])))
		plt.plot(brbc_all_labels, runtime_perf, '-*', linewidth=3, label=method_key)
	plt.subplot(1,2,2)
	for method_key in cpu_benchmark_data.keys():
		if(method_key == 'UnfiltBackproj3DBatch'): continue
		curr_benchmark_data = gpu_benchmark_data[method_key]
		runtime_perf = []
		for brbc in brbc_all:
			params_id = compose_params_id_string(k, bt, brbc[0], brbc[1])
			runtime_perf.append(np.log10(np.mean(gpu_benchmark_data[method_key][params_id])))
		plt.plot(brbc_all_labels, runtime_perf, '-*', linewidth=3, label=method_key)
	finalize_benchmark_fig(xlabel="Spatial Block Dim Size (brxbc)")
	plot_utils.save_currfig_png(dirpath=out_dirpath, filename=out_fname)



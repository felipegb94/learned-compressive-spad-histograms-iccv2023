'''
	This script plots the compression ratio for different coding tensor designs.
'''
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from research_utils import plot_utils
from analyze_test_results_utils import get_hydra_io_dirpaths


def nparr2strarr(a): return np.char.mod('%d', a)

def annotate_heatmap(x, y, z, ax=None):
	if(ax is None): ax = plt.gca()
	for i in range(len(y)):
		for j in range(len(x)):
			z_val = np.round(z[i, j], decimals=1)
			text = ax.text(j, i, z_val, ha="center", va="center", color="w", fontsize=12)

if __name__=='__main__':
	
	## load all dirpaths without creating a job 
	io_dirpaths = get_hydra_io_dirpaths(job_name='plot_csph3d_test_set_metrics')

	out_dirpath = os.path.join(io_dirpaths.results_figures_dirpath, 'compression_ratios')

	## Set min max compression for visualization purposes
	min_compression = 1
	max_compression = 10000

	## Conventional histogram parameters
	# Number of elements in full histogram
	n = 10**np.arange(6,10)
	# Total number of elements in conventional histogram
	H_size = n
	# H_size = n_pixels*n_t
	## Conventional histogram data transfer and in sensor storage
	conv_hist_r = H_size.reshape((H_size.size, 1)) 
	conv_hist_s = H_size.reshape((H_size.size, 1))

	## Compressive Histogram
	k = 512
	N_b = 10**np.arange(0,7) # Number of signal blocks
	print("First dimension (n): {}".format(n))
	print("Second dimension (N_b): {}".format(N_b))
	M = n.reshape((n.size,1)) / N_b.reshape((1, N_b.size)) # Coding Tensor Size
	B_size = k*N_b
	C_size = k*M
	comp_hist_r = B_size.reshape((1, N_b.size))
	comp_hist_s = B_size.reshape((1, N_b.size)) + C_size

	print("Data Transfer Rate and Storage as a function of number of pixels:")
	print("    Coding Tensor Size (M): {}".format(M))
	print("    k: {}".format(k))
	print("    Number of Signal Blocks: {}".format(N_b))

	## In-sensor storage compression ration
	CR_s = conv_hist_s / comp_hist_s
	CR_r = conv_hist_r / comp_hist_r


	out_fname_base = 'K-{}'.format(k)

	plt.close('all')
	plt.figure()
	fig = plt.gcf()
	ax = plt.gca()
	ax.imshow(CR_s,norm=colors.LogNorm(vmin=min_compression, vmax=max_compression)); 
	plt.title("In-sensor Storage Compression Ratios (K={})".format(k), fontsize=16)
	annotate_heatmap(N_b, n, CR_s, ax=ax)
	ax.set_xticks(np.arange(len(N_b)))
	ax.set_xticklabels(N_b)
	# ax.set_xlabel("Number of Signal Blocks (N_b)", fontsize=14)
	ax.set_yticks(np.arange(len(n)))
	ax.set_yticklabels(n)
	# ax.set_ylabel("Histogram Image Size (n)", fontsize=14)
	plot_utils.update_fig_size(height=4, width=8)
	plot_utils.set_ticks(fontsize=12)
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname_base+'_storage_encoding-full')

	plt.figure()
	fig = plt.gcf()
	ax = plt.gca()
	ax.imshow(CR_r,norm=colors.LogNorm(vmin=min_compression, vmax=max_compression)); 
	plt.title("Off-sensor Data Transfer Compression Ratios (K={})".format(k), fontsize=16)
	annotate_heatmap(N_b, n, CR_r, ax=ax)
	ax.set_xticks(np.arange(len(N_b)))
	ax.set_xticklabels(N_b)
	# ax.set_xlabel("Number of Signal Blocks (N_b)", fontsize=14)
	ax.set_yticks(np.arange(len(n)))
	ax.set_yticklabels(n)
	# ax.set_ylabel("Histogram Image Size (n)", fontsize=14)
	plot_utils.update_fig_size(height=4, width=8)
	plot_utils.set_ticks(fontsize=12)
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname_base+'_transmission_encoding-full')



	################ Compression ratios using separable coding matrices

	## Compressive Histogram
	M = np.sqrt(n.reshape((n.size,1)) / N_b.reshape((1, N_b.size))) # Coding Tensor Size
	B_size = k*N_b
	C_size = k*M
	comp_hist_r = B_size.reshape((1, N_b.size))
	comp_hist_s = B_size.reshape((1, N_b.size)) + C_size

	print("Data Transfer Rate and Storage as a function of number of pixels:")
	print("    Coding Tensor Size (M): {}".format(M))
	print("    k: {}".format(k))
	print("    Number of Signal Blocks: {}".format(N_b))

	## In-sensor storage compression ration
	CR_s = conv_hist_s / comp_hist_s
	CR_r = conv_hist_r / comp_hist_r

	plt.figure()
	fig = plt.gcf()
	ax = plt.gca()
	ax.imshow(CR_s,norm=colors.LogNorm(vmin=min_compression, vmax=max_compression)); 
	plt.title("In-sensor Storage Compression Ratios (K={})".format(k), fontsize=16)
	annotate_heatmap(N_b, n, CR_s, ax=ax)
	ax.set_xticks(np.arange(len(N_b)))
	ax.set_xticklabels(N_b)
	# ax.set_xlabel("Number of Signal Blocks (N_b)", fontsize=12)
	ax.set_yticks(np.arange(len(n)))
	ax.set_yticklabels(n)
	# ax.set_ylabel("Histogram Image Size (n)", fontsize=12)
	plot_utils.update_fig_size(height=4, width=8)
	plot_utils.set_ticks(fontsize=12)
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname_base+'_storage_encoding-separable')

	plt.figure()
	fig = plt.gcf()
	ax = plt.gca()
	ax.imshow(CR_r,norm=colors.LogNorm(vmin=min_compression, vmax=max_compression)); 
	plt.title("Off-sensor Data Transfer Compression Ratios (K={})".format(k), fontsize=16)
	annotate_heatmap(N_b, n, CR_r, ax=ax)
	ax.set_xticks(np.arange(len(N_b)))
	ax.set_xticklabels(N_b)
	# ax.set_xlabel("Number of Signal Blocks (N_b)", fontsize=12)
	ax.set_yticks(np.arange(len(n)))
	ax.set_yticklabels(n)
	# ax.set_ylabel("Histogram Image Size (n)", fontsize=12)
	plot_utils.update_fig_size(height=4, width=8)
	plot_utils.set_ticks(fontsize=12)
	plot_utils.save_currfig(dirpath=out_dirpath, filename=out_fname_base+'_transmission_encoding-separable')

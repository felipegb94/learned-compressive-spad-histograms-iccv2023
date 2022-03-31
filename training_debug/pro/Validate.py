import os
import numpy as np 
import torch
from tqdm import tqdm
from pro.Loss import criterion_KL,criterion_TV,criterion_L2

lsmx = torch.nn.LogSoftmax(dim=1)
dtype = torch.cuda.FloatTensor

## For debugging
from IPython.core import debugger
breakpoint = debugger.set_trace
import matplotlib.pyplot as plt

def save_currfig( dirpath = '.', filename = 'curr_fig', file_ext = 'png', use_imsave=False  ):
	# Create directory to store figure if it does not exist
	os.makedirs(dirpath, exist_ok=True)
	# Pause to make sure plot is fully rendered and not warnings or errors are thown
	plt.pause(0.02)
	# If filename contains file extension then ignore the input file ext
	# Else add the input file etension
	if('.{}'.format(file_ext) in  filename): filepath = os.path.join(dirpath, filename)
	else: filepath = os.path.join(dirpath, filename) + '.{}'.format(file_ext)
	plt.savefig(filepath, 
				dpi=None, 
				# facecolor='w', 
				# edgecolor='w',
				# orientation='portrait', 
				# papertype=None, 
				transparent=True, 
				bbox_inches='tight', 
				# pad_inches=0.1,
				# metadata=None 
				format=file_ext
				)

def validate(model, val_loader, n_iter, val_loss, params, logWriter):

	model.eval()
	
	l_all = []
	l_kl = []
	l_tv = []
	l_rmse = []

	for sample in tqdm(val_loader):
		M_mea = sample["spad"].type(dtype)
		M_gt = sample["rates"].type(dtype)
		dep = sample["bins"].type(dtype)
		idx = sample["idx"]

		M_mea_re, dep_re = model(M_mea)
		M_mea_re_lsmx = lsmx(M_mea_re).unsqueeze(1)
		loss_kl = criterion_KL(M_mea_re_lsmx, M_gt).data.cpu().numpy()
		loss_tv = criterion_TV(dep_re).data.cpu().numpy()
		rmse = criterion_L2(dep_re, dep).data.cpu().numpy()
		loss = loss_kl + params["p_tv"]*loss_tv

		l_all.append(loss)
		l_kl.append(loss_kl)
		l_tv.append(loss_tv)
		l_rmse.append(rmse)

		# for i in range(len(idx)):
		# 	C = 3e8
		# 	Tp = 100e-9			

		# 	# if(n_iter < 5):
		# 	# 	plt.clf()
		# 	# 	plt.imshow(dep.data.cpu().numpy()[i, 0, :, :] * Tp * C / 2, vmin=0, vmax=5)
		# 	# 	plt.title("GT, idx: {}, iter: {}".format(idx[i], n_iter))
		# 	# 	plt.colorbar()
		# 	# 	model_id = params['log_file'].split('/')[-1]
		# 	# 	out_dirpath = '/home/felipe/research_projects/2022_compressive-sp3Dc/results/week_2022-03-28/overfitting_20-ex_validate/'+model_id
		# 	# 	out_fname = "gt_idx-{}".format(idx[i], n_iter)
		# 	# 	save_currfig(dirpath = out_dirpath, filename = out_fname, file_ext = 'png', use_imsave=False)
		# 	# 	plt.pause(0.05)

		# 	plt.clf()
		# 	plt.subplot(1,2,1)
		# 	plt.imshow(dep.data.cpu().numpy()[i, 0, :, :] * Tp * C / 2, vmin=0, vmax=5)
		# 	plt.title("GT, idx: {}, iter: {}".format(idx[i], n_iter))
		# 	plt.subplot(1,2,2)
		# 	plt.imshow(dep_re.data.cpu().numpy()[i, 0, :, :] * Tp * C / 2, vmin=0, vmax=5)
		# 	plt.title("Rec, idx: {}, iter: {}".format(idx[i], n_iter))
		# 	plt.colorbar()
		# 	model_id = params['log_file'].split('/')[-1]
		# 	out_dirpath = '/home/felipe/research_projects/2022_compressive-sp3Dc/results/week_2022-03-28/overfitting_20-ex_validate/'+model_id
		# 	out_fname = "rec_idx-{}_itr-{}".format(idx[i], n_iter)
		# 	save_currfig(dirpath = out_dirpath, filename = out_fname, file_ext = 'png', use_imsave=False)
		# 	plt.pause(0.05)



	# log the val losses
	logWriter.add_scalar("loss_val/all", np.mean(l_all), n_iter)
	logWriter.add_scalar("loss_val/kl", np.mean(l_kl), n_iter)
	logWriter.add_scalar("loss_val/tv", np.mean(l_tv), n_iter)
	logWriter.add_scalar("loss_val/rmse", np.mean(l_rmse), n_iter)
	val_loss["KL"].append(np.mean(l_kl))
	val_loss["TV"].append(np.mean(l_tv))
	val_loss["RMSE"].append(np.mean(l_rmse))

	return val_loss, logWriter


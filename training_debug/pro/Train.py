# The train function
import os 
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import scipy.io as scio
from pro.Validate import validate
from util.SaveChkp import save_checkpoint
from pro.Loss import criterion_KL,criterion_TV,criterion_L2

## For debugging
from IPython.core import debugger
breakpoint = debugger.set_trace
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

cudnn.benchmark = True
lsmx = torch.nn.LogSoftmax(dim=1)
dtype = torch.cuda.FloatTensor

def train(model, train_loader, val_loader, optimer, epoch, n_iter,
			train_loss, val_loss, params, logWriter):
	train_val_loader = copy.deepcopy(train_loader)
	for sample in tqdm(train_loader):
		# configure model state
		model.train()

		# load data and train the network
		M_mea = sample["spad"].type(dtype)
		M_gt = sample["rates"].type(dtype)
		dep = sample["bins"].type(dtype)

		M_mea_re, dep_re = model(M_mea)

		M_mea_re_lsmx = lsmx(M_mea_re).unsqueeze(1)
		loss_kl = criterion_KL(M_mea_re_lsmx, M_gt)
		loss_tv = criterion_TV(dep_re)
		rmse = criterion_L2(dep_re, dep)

		if(params["add_tv_loss"]):
			loss = loss_kl + params["p_tv"]*loss_tv
		else:
			loss = loss_kl

		optimer.zero_grad()
		loss.backward()
		optimer.step()
		n_iter += 1

		logWriter.add_scalar("loss_train/all", loss, n_iter)
		logWriter.add_scalar("loss_train/kl", loss_kl, n_iter)
		logWriter.add_scalar("loss_train/tv", loss_tv, n_iter)
		logWriter.add_scalar("loss_train/rmse", rmse, n_iter)
		train_loss["KL"].append(loss_kl.data.cpu().numpy())
		train_loss["TV"].append(loss_tv.data.cpu().numpy())
		train_loss["RMSE"].append(rmse.data.cpu().numpy())

		print("Train Loss Itr: {}".format(n_iter))
		print("    Loss: {}".format(loss))
		print("    Loss KL: {}".format(loss_kl))
		print("    Loss TV: {}".format(loss_tv))
		print("    RMSE: {}".format(rmse))
		
		if n_iter % params["save_every"] == 0:
			print("Start validation...")
			C = 3e8
			Tp = 100e-9
			with torch.no_grad():
				for sample in tqdm(train_val_loader):
					val_M_mea = sample["spad"].type(dtype)
					val_M_gt = sample["rates"].type(dtype)
					val_dep = sample["bins"].type(dtype)
					val_idx = sample["idx"]

					val_M_mea_re, val_dep_re = model(val_M_mea)
					# val_M_mea_re_lsmx = lsmx(val_M_mea_re).unsqueeze(1)
					# val_loss_kl = criterion_KL(val_M_mea_re_lsmx, val_M_gt).data.cpu().numpy()
					# val_loss_tv = criterion_TV(val_dep_re).data.cpu().numpy()
					# val_rmse = criterion_L2(val_dep_re, val_dep).data.cpu().numpy()
					# val_loss = val_loss_kl + params["p_tv"]*val_loss_tv

					for i in range(len(val_idx)):
						C = 3e8
						Tp = 100e-9			
						plt.clf()
						plt.subplot(1,2,1)
						plt.imshow(val_dep.data.cpu().numpy()[i, 0, :, :] * Tp * C / 2, vmin=0, vmax=5)
						plt.title("GT, idx: {}, iter: {}".format(val_idx[i], n_iter))
						plt.subplot(1,2,2)
						plt.imshow(val_dep_re.data.cpu().numpy()[i, 0, :, :] * Tp * C / 2, vmin=0, vmax=5)
						plt.title("Rec, idx: {}, iter: {}".format(val_idx[i], n_iter))
						plt.colorbar()
						model_id = params['log_file'].split('/')[-1]
						out_dirpath = '/home/felipe/research_projects/2022_compressive-sp3Dc/results/week_2022-03-28/overfitting_20-ex_validate/'+model_id
						out_fname = "rec_idx-{}_itr-{}".format(val_idx[i], n_iter)
						save_currfig(dirpath = out_dirpath, filename = out_fname, file_ext = 'png', use_imsave=False)
					del val_M_mea_re
					del val_dep_re
					torch.cuda.empty_cache()


			val_loss, logWriter = validate(model, val_loader, n_iter, val_loss, params, logWriter)

			scio.savemat(file_name=params["log_file"]+"/train_loss.mat", mdict=train_loss)
			scio.savemat(file_name=params["log_file"]+"/val_loss.mat", mdict=val_loss)
			# save model states
			print("Validation complete! \nSaving checkpoint...")
			save_checkpoint(n_iter, epoch, model, optimer,
				file_path=params["log_file"]+"/epoch_{}_{}.pth".format(epoch, n_iter))
			print("Checkpoint saved!")
	
	return model, optimer, n_iter, train_loss, val_loss, logWriter

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
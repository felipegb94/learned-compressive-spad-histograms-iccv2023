# The train function
import numpy as np 
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import scipy.io as scio
from pro.Validate import validate
from util.SaveChkp import save_checkpoint
from pro.Loss import criterion_KL,criterion_TV,criterion_L2

cudnn.benchmark = True
lsmx = torch.nn.LogSoftmax(dim=1)
dtype = torch.cuda.FloatTensor

def train(model, train_loader, val_loader, optimer, epoch, n_iter,
            train_loss, val_loss, params, logWriter):

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

        loss = loss_kl + params["p_tv"]*loss_tv

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

        if n_iter % params["save_every"] == 0:
            print("Sart validation...")
            val_loss, logWriter = validate(model, val_loader, n_iter, val_loss, params, logWriter)

            scio.savemat(file_name=params["log_file"]+"/train_loss.mat", mdict=train_loss)
            scio.savemat(file_name=params["log_file"]+"/val_loss.mat", mdict=val_loss)
            # save model states
            print("Validation complete! \nSaving checkpoint...")
            save_checkpoint(n_iter, epoch, model, optimer,
                file_path=params["log_file"]+"/epoch_{}_{}.pth".format(epoch, n_iter))
            print("Checkpoint saved!")
    
    return model, optimer, n_iter, train_loss, val_loss, logWriter


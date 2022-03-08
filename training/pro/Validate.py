import numpy as np 
import torch
from tqdm import tqdm
from pro.Loss import criterion_KL,criterion_TV,criterion_L2

lsmx = torch.nn.LogSoftmax(dim=1)
dtype = torch.cuda.FloatTensor

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

    # log the val losses
    logWriter.add_scalar("loss_val/all", np.mean(l_all), n_iter)
    logWriter.add_scalar("loss_val/kl", np.mean(l_kl), n_iter)
    logWriter.add_scalar("loss_val/tv", np.mean(l_tv), n_iter)
    logWriter.add_scalar("loss_val/rmse", np.mean(l_rmse), n_iter)
    val_loss["KL"].append(np.mean(l_kl))
    val_loss["TV"].append(np.mean(l_tv))
    val_loss["RMSE"].append(np.mean(l_rmse))

    return val_loss, logWriter


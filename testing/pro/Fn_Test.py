import numpy as np
import torch
import torch.nn as nn
from glob import glob
import pathlib
import scipy
import os
import scipy.io as scio
import time
import h5py

dtype = torch.cuda.FloatTensor

# test function for Middlebury dataset 
def test_sm(model, opt, outdir_m):

    rmse_all = []
    time_all = []

    for name_test in glob(opt["testDataDir"] + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
        name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

        print("Loading data {} and processing...".format(name_test_id))

        dep = scio.loadmat(name_test)["depth"]
        dep = np.asarray(dep).astype(np.float32)
        h, w = dep.shape
        
        M_mea = scio.loadmat(name_test)["spad"]
        M_mea = scipy.sparse.csc_matrix.todense(M_mea)
        M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, h, w, -1])
        M_mea = torch.from_numpy(np.transpose(M_mea, (0, 1, 4, 2, 3))).type(dtype)

        t_s = time.time()
        M_mea_re, dep_re = model(M_mea)
        t_e = time.time()
        time_all.append(t_e - t_s)

        C = 3e8
        Tp = 100e-9

        dist = dep_re.data.cpu().numpy()[0, 0, :, :] * Tp * C / 2
        rmse = np.sqrt(np.mean((dist - dep)**2))
        rmse_all.append(rmse)

        scio.savemat(name_test_save, {"data":dist, "rmse":rmse})
        print("The RMSE: {}".format(rmse))

    return np.mean(rmse_all), np.mean(time_all)


# test function for outdoor real-world dataset
def test_outrw(model, opt, outdir_m):
    rmse_all = [0,0]
    time_all = []
    base_pad = 16
    step = 32
    grab = 32
    dim = 64

    for name_test in glob(opt["testDataDir"] + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
        name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

        print("Loading data {} and processing...".format(name_test_id))
        M_mea_raw = np.asarray(scipy.io.loadmat(name_test)['y'])
        
        tp_s = 4770 
        t_inter =1024
        M_mea = M_mea_raw[:, :,tp_s:tp_s+t_inter]
        M_mea = M_mea.transpose((2, 0, 1))
        M_mea = torch.from_numpy(M_mea).unsqueeze(0).unsqueeze(0).type(dtype)

        out = np.zeros((M_mea.shape[1], M_mea.shape[2]))
        M_mea = torch.nn.functional.pad(M_mea, (base_pad, 0, base_pad, 0, 0, 0)) # pad only on edge

        t_s = time.clock()
        for i in tqdm(range(4)):
            for j in range(4):
                M_mea_input = M_mea[:, :, :, i*step:(i)*step+dim, j*step:(j)*step+dim]
                print("Size of input:{}".format(M_mea_input.shape))
                M_mea_re, dep_re = model(M_mea_input)
                M_mea_re = M_mea_re.data.cpu().numpy().squeeze()
                tile_out = np.argmax(M_mea_re, axis=0)  
                if i <= 14:
                    out[i*step:(i+1)*step, j*step:(j+1)*step] = tile_out[step:step+step, step:step+step]
                else:
                    out[i*step:(i+1)*step, j*step:(j+1)*step] = tile_out[16:16+step, 16:16+step]
        
        t_e = time.clock()
        time_all.append(t_e - t_s)

        dist = out.astype(np.float32) * 0.15

        scio.savemat(name_test_save, {"data":dist})
    
    return np.mean(rmse_all), np.mean(time_all)


# test function for indoor real-world data
def test_inrw(model, opt, outdir_m):

    rmse_all = [0,0]
    time_all = []
    base_pad = 16
    step = 16
    grab = 32
    dim = 64

    for name_test in glob(opt["testDataDir"] + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
        name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

        print("Loading data {} and processing...".format(name_test_id))
        M_mea = scio.loadmat(name_test)["spad_processed_data"]
        M_mea = scipy.sparse.csc_matrix.todense(M_mea)
        M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, 1536, 256, 256])
        M_mea = M_mea.transpose((0,1,2,4,3)).type(dtype)

        out = np.zeros((M_mea.shape[3], M_mea.shape[4]))
        M_mea = torch.nn.functional.pad(M_mea, (base_pad, 0, base_pad, 0, 0, 0))

        t_s = time.time()
        for i in tqdm(range(16)):
            for j in range(16):
                M_mea_input = M_mea[:, :, :, i*step:(i)*step+dim, j*step:(j)*step+dim]
                M_mea_re, dep_re = model(M_mea_input)
                M_mea_re = M_mea_re.data.cpu().numpy().squeeze()
                tile_out = np.argmax(M_mea_re, axis=0)
                out[i*step:(i+1)*step, j*step:(j+1)*step] = \
                    tile_out[step//2:step//2+step, step//2:step//2+step]

        t_e = time.time()
        time_all.append(t_e - t_s)
        
        dist = out * 6 / 1536.

        scio.savemat(name_test_save, {"data":dist})

    return np.mean(rmse_all), np.mean(time_all)

   


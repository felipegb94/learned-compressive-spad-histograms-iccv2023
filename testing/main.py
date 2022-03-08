# The test file for network
# Based on pytorch 1.0
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from tensorboardX import SummaryWriter
from datetime import datetime
import skimage.io
import scipy.io as scio
from glob import glob
import pathlib

from pro.Fn_Test import test_sm, test_inrw, test_outrw
from models import model_ddfn_64_B10_CGNL_ori
from util.ParseArgs import parse_args

    
def main():
    
    # parse arguments
    opt = parse_args("./config.ini")
    print("Number of available GPUs: {} {}".format(torch.cuda.device_count(), \
        torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Test Model Path: {}".format(opt["testModelsDir"]))
    print("Test Data Path: {}".format(opt["testDataDir"]))
    print("Test Output Path: {}".format(opt["testOutDir"]))
    # list all the test models
    file_list = sorted(glob(opt["testModelsDir"]))
    print("+++++++++++++++++++++++++++++++++++++++++++")

    # configure network
    model = model_ddfn_64_B10_CGNL_ori.DeepBoosting()
    model.cuda()
    model.eval()
    print(model)

    for iter, pre_model in enumerate(file_list):

        print('The total number of test models are: {}'.format(len(file_list)))

        filename, _ = os.path.splitext(os.path.split(pre_model)[1])
        outdir_m = opt["testOutDir"] + '/Model_'+ filename
        pathlib.Path(outdir_m).mkdir(parents=True, exist_ok=True)

        print('=> Loading checkpoint {}'.format(pre_model))
        ckpt = torch.load(pre_model) 
        model_dict = model.state_dict()
        try:
            ckpt_dict = ckpt["state_dict"]
        except KeyError:
            print('Key error loading state_dict from checkpoint; assuming checkpoint contains only the state_dict')
            ckpt_dict = ckpt

        for key_iter, k in enumerate(ckpt_dict.keys()): # to update the model using the pretrained models
            # model_dict.update({k[7:]: ckpt_dict[k]}) # use this for multi-GPU trained model
            model_dict.update({k: ckpt_dict[k]})  # use this for single-GPU trained model
            if key_iter == (len(ckpt_dict.keys()) - 1):
                print('Model Parameter Update!')
        model.load_state_dict(model_dict)

        # the test function for middlebury dataset
        rmse, runtime = test_sm(model, opt, outdir_m)
        # the test function for real-world indoor and outdoor dataset
        # rmse, runtime = test_inrw(model, opt, outdir_m)
        # rmse, runtime = test_outrw(model, opt, outdir_m)

        print("Model: {} and Performance: {} with run time of: {}".format(filename, rmse, runtime))


if __name__=="__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Execuating code...")
    main()




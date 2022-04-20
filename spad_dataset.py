# The SPAD data pre-process function

#### Standard Library Imports
import os

#### Library imports
import numpy as np
import scipy
import scipy.io
import torch
import torch.utils.data

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


class SpadDataset(torch.utils.data.Dataset):
    def __init__(self, datalist_fpath, noise_idx=None, output_size=None, disable_rand_crop=False, tres_ps=1000):
        """__init__
        :param datalist_fpath: path to text file with list of spad data files
        :param noise_idx: the noise index list to include in the dataset (e.g., 1 or 2
        :param output_size: the output size after random crop. If set to None it will output the full image
        """

        with open(datalist_fpath) as f: 
            self.spad_data_fpaths_all = f.read().split()

        self.tres_ps = tres_ps # time resolution in picosecs
        self.noise_idx = noise_idx
        self.spad_data_fpaths = []
        if(noise_idx is None):
            self.spad_data_fpaths = self.spad_data_fpaths_all
        else:
            if(isinstance(noise_idx, int)):
                noise_idx = [noise_idx]
            assert(isinstance(noise_idx, list)), "Input noise idx should be an int or list"
            for fpath in self.spad_data_fpaths_all:
                for idx in noise_idx:
                    # If noise_idx in current fpath, add it, and continue to next fpath
                    if('_p{}'.format(idx) in os.path.basename(fpath)):
                        self.spad_data_fpaths.append(fpath)
                        break

        self.spad_data_fpaths =  self.spad_data_fpaths
        # self.spad_data_fpaths.extend([intensity.replace('intensity', 'spad')
        #                             .replace('.mat', '_p{}.mat'.format(noise_idx))
        #                             for intensity in self.intensity_files])

        if(isinstance(output_size, int)): self.output_size = (output_size, output_size)
        else: self.output_size = output_size
        self.disable_rand_crop = disable_rand_crop

        print("SpadDataset with {} files".format(len(self.spad_data_fpaths)))

    def __len__(self):
        return len(self.spad_data_fpaths)

    def tryitem(self, idx):
        '''
            Try to load the spad data sample.
            * All normalization and data augmentation happens here.
        '''
        # load spad data
        spad_data = scipy.io.loadmat(self.spad_data_fpaths[idx])

        # normalized pulse as GT histogram
        rates = np.asarray(spad_data['rates']).astype(np.float32)
        (nr, nc, n_bins) = rates.shape
        rates = rates[np.newaxis,: ]
        rates = np.transpose(rates, (0, 3, 1, 2))
        rates = rates / (np.sum(rates, axis=-3, keepdims=True) + 1e-8)

        # simulated spad measurements
        # Here we need to swap the rows and cols because matlab saves dimensions in a different order.
        spad = np.asarray(scipy.sparse.csc_matrix.todense(spad_data['spad'])).astype(np.float32)
        spad = spad.reshape((nc, nr, n_bins))
        spad = spad[np.newaxis, :]
        spad = np.transpose(spad, (0, 3, 2, 1))

        # # ground truth depths in units of bins
        bins = np.asarray(spad_data['bin']).astype(np.float32)
        bins = bins[np.newaxis, :]
        bins = (bins - 1) / (n_bins - 1) # make range 0-n_bins-1 instead of 1-n_bins

        (h, w) = (nr, nc)
        if(self.output_size is None):
            new_h = h
            new_w = w
        else:
            new_h = self.output_size[0]
            new_w = self.output_size[1]

        if(self.disable_rand_crop):
            top = 0
            left = 0        
        else:
            # add 1 because randint produces between low <= x < high 
            top = np.random.randint(0, h - new_h + 1) 
            left = np.random.randint(0, w - new_w + 1)

        rates = rates[..., top:top + new_h, left:left + new_w]
        spad = spad[..., top:top + new_h, left: left + new_w]
        bins = bins[..., top: top + new_h, left: left + new_w]

        rates = torch.from_numpy(rates)
        spad = torch.from_numpy(spad)
        bins = torch.from_numpy(bins)

        sample = {'rates': rates, 'spad': spad, 'bins': bins, 'idx': idx, 'tres_ps': self.tres_ps}

        return sample

    def __getitem__(self, idx):
        try:
            sample = self.tryitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self.tryitem(idx)
        return sample


if __name__=='__main__':
    import matplotlib.pyplot as plt

    ## Try test dataset
    datalist_fpath = './datalists/test_middlebury_SimSPADDataset_nr-72_nc-88_nt-1024_tres-98ps_dark-0_psf-0.txt'
    noise_idx = None
    spad_dataset = SpadDataset(datalist_fpath, noise_idx=noise_idx, output_size=None)

    # ## Load val dataset
    # datalist_fpath = './datalists/val_nyuv2_SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1.txt'
    # noise_idx = [1]
    # spad_dataset = SpadDataset(datalist_fpath, noise_idx=noise_idx, output_size=60)

    batch_size = 1

    loader = torch.utils.data.DataLoader(spad_dataset, batch_size=1, shuffle=True, num_workers=0)
    iter_loader = iter(loader)
    for i in range(10):
        spad_sample = iter_loader.next()
        
        spad = spad_sample['spad']
        rates = spad_sample['rates']
        bins = spad_sample['bins']
        idx = spad_sample['idx']
        print(idx)

        spad = spad.cpu().detach().numpy()[0,:].squeeze()
        rates = rates.cpu().detach().numpy()[0,:].squeeze()
        bins = bins.cpu().detach().numpy()[0,:].squeeze()
        # idx = idx.cpu().detach().numpy().squeeze()
        # 
        print("     Rates: {}", rates.shape)        
        print("     SPAD dims: {}", spad.shape)        

        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(bins)
        plt.subplot(2,2,2)
        plt.imshow(np.argmax(spad, axis=0) / spad.shape[0], vmin=bins.min(), vmax=bins.max())
        plt.subplot(2,2,3)
        plt.imshow(np.log(1+np.sum(spad, axis=0)))
        # plt.imshow(spad.cpu().detach().numpy().squeeze()[0,:].sum(axis=0))
        plt.pause(0.5)





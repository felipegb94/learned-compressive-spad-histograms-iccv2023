# The SPAD data pre-process function
import torch
import torch.utils.data
import scipy.io
import numpy as np
import skimage.transform


# class ToTensor(object):
#     """Transfer the tensor to torch.tensor
#     """

#     def __init__(self):
#         pass

#     def __call__(self, sample):
#         rates, spad, bins = sample['rates'],\
#                             sample['spad'],\
#                             sample['bins']

#         rates = torch.from_numpy(rates)
#         spad = torch.from_numpy(spad)
#         bins = torch.from_numpy(bins)

#         return {'rates': rates, 'spad': spad, 'bins': bins}


# class RandomCrop(object):
#     """Crop randomly the image in a sample.
#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """

#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         rates, spad, bins = sample['rates'],\
#                             sample['spad'],\
#                             sample['bins']

#         h, w = spad.shape[2:]
#         new_h = self.output_size
#         new_w = self.output_size

#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)

#         rates = rates[:, :, top: top + new_h,
#                       left: left + new_w]
#         spad = spad[:, :, top: top + new_h,
#                     left: left + new_w]
#         bins = bins[:, top: top + new_h,
#                     left: left + new_w]

#         return {'rates': rates, 'spad': spad, 'bins': bins}


class SpadDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, noise_idx=1, output_size=32, disable_rand_crop=False):
        """__init__
        :param datapath: path to text file with list of
                        training files (intensity files)
        :param noise_idx: the noise index 1 or 2
        :param output_size: the output size after random crop
        """

        with open(datapath) as f:
            self.intensity_files = f.read().split()
        self.spad_files = []
        self.spad_files.extend([intensity.replace('intensity', 'spad')
                                    .replace('.mat', '_p{}.mat'.format(noise_idx))
                                    for intensity in self.intensity_files])
        self.output_size = output_size
        self.disable_rand_crop = disable_rand_crop

    def __len__(self):
        return len(self.spad_files)

    def tryitem(self, idx):
        # simulated spad measurements
        spad = np.asarray(scipy.sparse.csc_matrix.todense(scipy.io.loadmat(
            self.spad_files[idx])['spad'])).reshape([1, 64, 64, -1]) # 1,64,64,1024
        spad = np.transpose(spad, (0, 3, 2, 1))

        # normalized pulse as GT histogram
        rates = np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['rates']).reshape([1, 64, 64, -1])
        rates = np.transpose(rates, (0, 3, 1, 2))
        rates = rates / np.sum(rates, axis=1)[None, :, :, :] 

        bins = (np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['bin']).astype(
                np.float32).reshape([64, 64])-1)[None, :, :] / 1023

        h, w = spad.shape[2:]
        new_h = self.output_size
        new_w = self.output_size

        if(self.disable_rand_crop):
            top = 16
            left = 16        
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

        rates = rates[:, :, top: top + new_h,
                      left: left + new_w]
        spad = spad[:, :, top: top + new_h,
                    left: left + new_w]
        bins = bins[:, top: top + new_h,
                    left: left + new_w]

        rates = torch.from_numpy(rates)
        spad = torch.from_numpy(spad)
        bins = torch.from_numpy(bins)

        sample = {'rates': rates, 'spad': spad, 'bins': bins, 'idx': idx}

        return sample

    def __getitem__(self, idx):
        try:
            sample = self.tryitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self.tryitem(idx)
        return sample

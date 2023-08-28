'''
    This script downloads the simulated spad nyuv2 dataset used for training
    Make sure to run this script from the top-level folder of the repository
    i.e.,
        python scripts/download_nyuv2_simulated_spad_data.py
'''

#### Standard Library Imports
import zipfile
import os
import sys
sys.path.append('.')

#### Library imports
import gdown

#### Local imports
from research_utils.io_ops import get_filepaths_in_dir

# def download_and_extract_zip(url, data_base_dirpath, scene_id):
#     zip_fpath = os.path.join(data_base_dirpath, '{}.zip'.format(scene_id))
#     print("Downloading: {}".format(zip_fpath))
#     if(os.path.exists(zip_fpath)):
#         print("{} already exists. Aborting download. If you wish to overwrite delete the file. ".format(zip_fpath))
#     else:
#         gdown.download(url, zip_fpath, fuzzy=True)
#         print('Extracting ... this may take a few minutes..')
#         with zipfile.ZipFile(zip_fpath, 'r') as f:
#             f.extractall(data_base_dirpath)

def get_dataset_url(dataset_id):
    if(dataset_id == 'ModuloSimSPADDataset_nr-64_nc-64_nt-1024_tres-55ps_dark-1_psf-1'):
        gdrive_dataset_folder_url = 'https://drive.google.com/drive/folders/1cwHxX0Qt-rg0GF7Z2OMWc3r5IStBlNYp'
    elif(dataset_id == 'SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1'):
        gdrive_dataset_folder_url = 'https://drive.google.com/drive/folders/1fxUxFwg3frDHCi8jkHdDRhaRG5JrZq8Q'
    elif(dataset_id == 'SimSPADDataset_nr-64_nc-64_nt-1024_tres-60ps_dark-1_psf-1'):
        gdrive_dataset_folder_url = 'https://drive.google.com/drive/folders/1D6Tq-Z37xb37yBw6chn-rgrVthxDbsU4'
    elif(dataset_id == 'SimSPADDataset_nr-64_nc-64_nt-1024_tres-55ps_dark-1_psf-1'):
        gdrive_dataset_folder_url = 'https://drive.google.com/drive/folders/1AyqhLjsPOyQjTnsA1IdOPqYVXVCVjwni'
    elif(dataset_id == 'SimSPADDataset_min'):
        gdrive_dataset_folder_url = 'https://drive.google.com/drive/folders/17W987nnzLnqCgV_Z3waLxha3gs1IY7kO'
    else: assert(False), "Invalid dataset ID"
    return gdrive_dataset_folder_url 

if __name__=='__main__':

    ## Output dir where to store the dataset (leaving default should work)
    out_dir = './data_gener/TrainData'
    os.makedirs(out_dir, exist_ok=True)
    
    ## dataset folder ID
    dataset_id = 'SimSPADDataset_nr-64_nc-64_nt-1024_tres-80ps_dark-1_psf-1'
    #dataset_id = 'SimSPADDataset_nr-64_nc-64_nt-1024_tres-55ps_dark-1_psf-1'
    #dataset_id = 'ModuloSimSPADDataset_nr-64_nc-64_nt-1024_tres-55ps_dark-1_psf-1'
    # dataset_id = 'SimSPADDataset_min'
    dataset_dir = os.path.join(out_dir, dataset_id)

    ## Get the url 
    gdrive_dataset_folder_url = get_dataset_url(dataset_id)

    ## Only download if folder does not exist
    if(os.path.exists(dataset_dir)):
        print("Dataset has already been downloaded. If you want to overwrite, remove {} folder and run this script again".format(dataset_dir))
        zip_fpaths = get_filepaths_in_dir(dataset_dir, match_str_pattern='*')
    else:
        zip_fpaths = gdown.download_folder(url=gdrive_dataset_folder_url, output=dataset_dir, quiet=False)

    print(zip_fpaths)

    ## Unzip all downloaded files
    for fpath in zip_fpaths:
        if('.zip' in fpath):
            print('Extracting {}... this may take a few minutes..'.format(fpath))
            with zipfile.ZipFile(fpath, 'r') as f:
                f.extractall(dataset_dir)

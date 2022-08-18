'''
    This script downloads the test data used in Lindell et al., 2018 SIGGRAPH paper
    If you use this data please make sure to cite their work. See: https://www.computationalimaging.org/publications/single-photon-3d-imaging-with-deep-sensor-fusion/

    The data for this paper is stored in our own shared drive so we can control the sharing aspects of it

    Make sure to run this script from the top-level folder of the  repository
    i.e.,
        python scripts/download_2018Siggraph_lindell_test_data.py
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


if __name__=='__main__':

    ## Output dir where to store the dataset (leaving default should work)
    out_dir = './'
    os.makedirs(out_dir, exist_ok=True)

    ## dataset dirpath
    dataset_id = '2018SIGGRAPH_lindell_test_data'
    dataset_dir = os.path.join(out_dir, dataset_id)

    ## Get the url 
    gdrive_dataset_folder_url = 'https://drive.google.com/drive/folders/1TEnd3Cr8VORf7ojuF9vVeQcitqC8ipZX'

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

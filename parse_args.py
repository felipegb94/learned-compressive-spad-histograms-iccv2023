#### Standard Library Imports


#### Library imports
import configargparse

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

# def add_io_dirpaths_args(p):
#     p.add_argument('--datalists_dir', type=str, help='Directory where datalists for train, val, and test datasets should be stored')
#     p.add_argument('--results_dir', type=str, help='Directory where results should be stored')
#     p.add_argument('--results_data_dir', type=str, help='Directory where results data should be stored')
#     p.add_argument('--logdir_dir', type=str, help='Log directory')
#     p.add_argument('--nyuv2_raw_dataset_dir', type=str, help='Dirpath to nyuv2 raw dataset')
#     p.add_argument('--nyuv2_processed_dataset_dir', type=str, help='Dirpath to nyuv2 processed dataset')
#     p.add_argument('--train_spad_dataset_dir', type=str, help='Dirpath to directory containing the train spad dataset')
#     p.add_argument('--test_spad_dataset_dir', type=str, help='Dirpath to directory containing the test spad dataset')

# def add_all_args(p):
#     add_io_dirpaths_args(p)
    

if __name__=='__main__':
    p = configargparse.ArgumentParser()
    p.add('-c', '--config', default='config.ini', is_config_file=True, help='Path to config file.')
    # add_io_dirpaths_args(p)

    opt = p.parse_args()

    print(opt)

#### Standard Library Imports


#### Library imports
import configargparse

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

def add_train_args(p):
    # General training options
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--gpu_num', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epoch', type=int, default=8)
    p.add_argument('--save_every', type=int, default=100)
    p.add_argument('--lri', type=float, default=1e-4)
    p.add_argument('--p_tv', type=float, default=1e-5)
    p.add_argument('--optimizer', type=str, default='Adam')
    p.add_argument('--noise_idx', type=int, default=1)

    # p.add_argument('--model_name', type=str, default='DDFN_C64B10_NL') = DDFN_C64B10_NL
    # p.add_argument('--log_dir', type=, default=) = ./output/logfile
    # p.add_argument('--log_file', type=, default=) = ${log_dir}/${model_name}
    # p.add_argument('--util_dir', type=, default=) = ./util
    # p.add_argument('--train_file', type=, default=) = ${util_dir}/train_intensity.txt
    # p.add_argument('--val_file', type=, default=) = ${util_dir}/val_intensity.txt
    # p.add_argument('--resume', type=, default=) = False
    # p.add_argument('--resume_fpt', type=, default=) = ${log_dir}/rsm
    # p.add_argument('--resume_mod', type=, default=) = ${resume_fpt}/xxx.pth
    # p.add_argument('--train_loss', type=, default=) = ${resume_fpt}/xxx.mat
    # p.add_argument('--val_loss', type=, default=) = ${resume_fpt}/xxxx.mat


if __name__=='__main__':
    p = configargparse.ArgumentParser()
    p.add('-c', '--config', default='config.ini', is_config_file=True, help='Path to config file.')
    # add_io_dirpaths_args(p)

    opt = p.parse_args()

    print(opt)

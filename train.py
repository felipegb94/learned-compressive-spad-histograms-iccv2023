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
    p.add_argument('--noise_idx', type=int, default=1)

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

add_train_args(p)


opt = p.parse_args()

print(opt)


# if __name__=='__main__':
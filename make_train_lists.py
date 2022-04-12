#### Standard Library Imports


#### Library imports
import configargparse
import numpy as np

from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

if __name__=='__main__':
    p = configargparse.ArgumentParser()
    p.add('-c', '--config', default='io_dirpaths.conf', is_config_file=True, help='Path to config file.')

    opt = p.parse_args()

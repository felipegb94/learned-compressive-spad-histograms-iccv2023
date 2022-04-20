#### Standard Library Imports


#### Library imports
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports


C = 3e8

def bin2tof(b, num_bins, tau):
    '''
        b == bin
        num_bins == number of bins in histogram
        tau == period
    '''
    return (b / num_bins) * tau

def tof2depth(tof):
    return tof * C / 2.

def bin2depth(b, num_bins, tau):
    return tof2depth(bin2tof(b, num_bins, tau))
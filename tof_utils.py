#### Standard Library Imports


#### Library imports
import numpy as np
from IPython.core import debugger
breakpoint = debugger.set_trace


#### Local imports
from research_utils.shared_constants import SPEED_OF_LIGHT, TWOPI


C = SPEED_OF_LIGHT

def bin2tof(b, num_bins, tau):
    '''
        b == bin
        num_bins == number of bins in histogram
        tau == period
    '''
    return (b / num_bins) * tau

def tof2depth(tof):
    return tof * C / 2.

def bin2phase(bin, num_bins):
    return TWOPI*(bin/num_bins)

def bin2depth(b, num_bins, tau):
    return tof2depth(bin2tof(b, num_bins, tau))

def linearize_phase(phase):
	# If phase  < 0 then we need to add 2pi.
	corrected_phase = phase + (TWOPI*(phase < 0))
	return corrected_phase
	
def phase2depth(phase, repetition_tau):
	return time2depth(phase2time(phase, repetition_tau))

def phase2time(phase, repetition_tau):
	'''
		Assume phase is computed with np.atan2
	'''
	# If phase  < 0 then we need to add 2pi.
	corrected_phase = linearize_phase(phase)
	return (corrected_phase*repetition_tau / TWOPI )

def time2depth(time):
	return (SPEED_OF_LIGHT * time) / 2.

def freq2depth(freq):
	return (SPEED_OF_LIGHT * (1./freq)) / 2.

def depth2time(depth):
	return (2*depth /  SPEED_OF_LIGHT)

def phasor2time(phasor, repetition_tau):
	phase = np.angle(phasor)
	return phase2time(phase, repetition_tau)
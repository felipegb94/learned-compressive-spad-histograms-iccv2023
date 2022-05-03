#### Standard Library Imports

#### Library imports
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports
from model_ddfn_64_B10_CGNL_ori import LITPlainDeepBoosting
from model_ddfn_64_B10_CGNL_ori_old import LITDeepBoostingOriginal



class LITDeepBoostingDepth2Depth(LITDeepBoostingOriginal):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1):
		# Init parent class
		super(LITDeepBoostingDepth2Depth, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		
	def get_input_data(self, sample):
		return sample["est_bins_argmax_hist"]
	
class LITPlainDeepBoostingDepth2Depth(LITPlainDeepBoosting):
	def __init__(self, 
		init_lr = 1e-4,
		p_tv = 1e-5, 
		lr_decay_gamma = 0.9,
		in_channels=1):
		# Init parent class
		super(LITPlainDeepBoostingDepth2Depth, self).__init__(init_lr=init_lr,p_tv=p_tv,lr_decay_gamma=lr_decay_gamma,in_channels=in_channels)
		
	def get_input_data(self, sample):
		return sample["est_bins_argmax_hist"]
	
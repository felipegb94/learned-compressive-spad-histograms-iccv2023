#### Standard Library Imports


#### Library imports
from omegaconf import DictConfig, OmegaConf
import hydra
from IPython.core import debugger
breakpoint = debugger.set_trace

#### Local imports

@hydra.main(config_path="./conf", config_name="train")
def train(cfg):
	print(OmegaConf.to_yaml(cfg))

if __name__=='__main__':
	train()
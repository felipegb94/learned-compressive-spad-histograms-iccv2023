# The parse argments
import os
import sys
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime

def parse_args(config_path = "./config.ini"):
    if os.path.exists(config_path):
        print("Reading config file from {} and parse args...".format(config_path))
        opt = {}
        config = ConfigParser(interpolation=ExtendedInterpolation())     # use ConfigParser realize a instance
        config.read(config_path)    # read the config file
        config_bk = ConfigParser()
        
        # get returns the "str" type
        opt["testModelsDir"] = config.get("params", "testModelsDir")
        opt["testDataDir"] = config.get("params", "testDataDir")
        opt["testOutDir"] = config.get("params", "testOutDir")
        
        return opt
    else:
        print("No file exist named {}".format(config_path))
        sys.exit("NO FILE ERROR")
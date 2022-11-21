
import argparse
from yacs.config import CfgNode as CN


arg_parser = argparse.ArgumentParser(description="singular parser")
arg_parser.add_argument('--config-file', type=str,
                        default="configs/baseline.py")


def load_config(args):

    config = CN._load_cfg_py_source(args.config_file)
    # Any arg based changes can be done here!

    return config


import argparse
from yacs.config import CfgNode as CN


arg_parser = argparse.ArgumentParser(description="singular parser")
arg_parser.add_argument('--config-file', type=str,
                        default="../configs/baseline.py")
arg_parser.add_argument('--debug', action='store_true',
                        help="Enables DEBUG mode; for visualizer computes mesh independetly per envelope")
arg_parser.add_argument('--save-loc', type=str,
                        default="../results/single_shape")


def load_config(args):

    config = CN._load_cfg_py_source(args.config_file)
    # Any arg based changes can be done here!

    return config

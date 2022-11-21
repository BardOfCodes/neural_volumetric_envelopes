from yacs.config import CfgNode as CN
from .subconfigs.nve_model import config as model_config
from .subconfigs.dataset import config as dataset_config
cfg = CN()

cfg.MODEL = model_config.clone()

cfg.OPT = CN()
cfg.OPT.LR = 0.003

cfg.DATASET = dataset_config.clone()

cfg.DATALOADER = CN()
cfg.DATALOADER.BATCH_SIZE = 2048
cfg.DATALOADER.NUM_WORKERS = 8

cfg.TRAINER = CN()

cfg.EVALUATOR = CN()

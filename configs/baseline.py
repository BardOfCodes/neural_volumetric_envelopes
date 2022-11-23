from yacs.config import CfgNode as CN
from configs.subconfigs.nve_model import config as model_config
from configs.subconfigs.dataset import config as dataset_config
cfg = CN()

cfg.EXP_NAME = "Baseline"

cfg.MODEL = model_config.clone()

cfg.OPT = CN()
cfg.OPT.LR = 0.003

cfg.DATASET = dataset_config.clone()

cfg.DATALOADER = CN()
cfg.DATALOADER.BATCH_SIZE = 512
cfg.DATALOADER.NUM_WORKERS = 2

cfg.TRAINER = CN()
cfg.TRAINER.EXP_NAME = cfg.EXP_NAME

cfg.EVALUATOR = CN()

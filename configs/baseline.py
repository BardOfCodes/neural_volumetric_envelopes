from yacs.config import CfgNode as CN
from configs.subconfigs.nve_model import config as model_config
from configs.subconfigs.dataset import config as dataset_config 
cfg = CN()

cfg.EXP_NAME = "codebook"

n_surface_points = 1024

cfg.MACHINE_SPEC = CN()
## USE ALIAS/LINKS
cfg.MACHINE_SPEC.DATA_DIR = "../data/nve/sdf_data/cuboid/cuboid_envelopes.pkl"
cfg.MACHINE_SPEC.SAVE_DIR = "weights/" + cfg.EXP_NAME
cfg.MACHINE_SPEC.LOG_DIR = "logs/" + cfg.EXP_NAME

cfg.MODEL = model_config.clone()

cfg.OPT = CN()
cfg.OPT.LR = 0.0001

cfg.DATASET = dataset_config.clone()
cfg.DATASET.PATH = cfg.MACHINE_SPEC.DATA_DIR
cfg.DATASET.N_SURFACE_POINTS = n_surface_points

cfg.DATALOADER = CN()
cfg.DATALOADER.BATCH_SIZE = 8
cfg.DATALOADER.NUM_WORKERS = 0

cfg.TRAINER = CN()
cfg.TRAINER.EXP_NAME = cfg.EXP_NAME
cfg.TRAINER.FEATURE_TRANSFORM_WEIGHT = 0.001
cfg.TRAINER.COMMIT_LOSS_WEIGHT = 0
cfg.TRAINER.N_EPOCHS = 10000
cfg.TRAINER.SAVE_EPOCH = 500
cfg.TRAINER.EVAL_EPOCH = 5
cfg.TRAINER.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR
cfg.TRAINER.RESUME_CHECKPOINT = False
cfg.TRAINER.LOG_INTERVAL = 50
# For any pretrained weights
cfg.TRAINER.INIT_WEIGHTS = ""

cfg.EVALUATOR = CN()
cfg.EVALUATOR.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR

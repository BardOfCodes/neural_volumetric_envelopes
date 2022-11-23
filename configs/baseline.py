from yacs.config import CfgNode as CN
from configs.subconfigs.nve_model import config as model_config
from configs.subconfigs.dataset import config as dataset_config
cfg = CN()

cfg.EXP_NAME = "Baseline"

n_surface_points = 1024

cfg.MACHINE_SPEC = CN()
## USE ALIAS/LINKS
cfg.MACHINE_SPEC.DATA_DIR = "data/nve/sdf_data/cuboid/cuboid_envelopes.pkl"
cfg.MACHINE_SPEC.SAVE_DIR = "weights/" + cfg.EXP_NAME
cfg.MACHINE_SPEC.LOG_DIR = "logs/" + cfg.EXP_NAME

cfg.MODEL = model_config.clone()
cfg.MODEL.E2F.INPUT_DIM = n_surface_points

cfg.OPT = CN()
cfg.OPT.LR = 0.003

cfg.DATASET = dataset_config.clone()
cfg.DATASET.PATH = cfg.MACHINE_SPEC.DATA_DIR
cfg.DATASET.N_SURFACE_POINTS = n_surface_points

cfg.DATALOADER = CN()
cfg.DATALOADER.BATCH_SIZE = 16
cfg.DATALOADER.NUM_WORKERS = 2

cfg.TRAINER = CN()
cfg.TRAINER.EXP_NAME = cfg.EXP_NAME
cfg.TRAINER.L2_WEIGHT = 0.0005
cfg.TRAINER.N_EPOCHS = 100
cfg.TRAINER.SAVE_EPOCH = 20
cfg.TRAINER.EVAL_EPOCH = 5
cfg.TRAINER.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR
cfg.TRAINER.RESUME_CHECKPOINT = True
cfg.TRAINER.LOG_INTERVAL = 10
# For any pretrained weights
cfg.TRAINER.INIT_WEIGHTS = ""

cfg.EVALUATOR = CN()
cfg.EVALUATOR.SAVE_DIR = cfg.MACHINE_SPEC.SAVE_DIR

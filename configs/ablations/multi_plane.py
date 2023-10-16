from configs.ablations.codebook_baseline import cfg as base_cfg

cfg = base_cfg.clone()

cfg.EXP_NAME = "main"
cfg.MACHINE_SPEC.DATA_DIR = "data/nve/sdf_data/planes/"
cfg.DATASET.PATH = cfg.MACHINE_SPEC.DATA_DIR
cfg.DATASET.MODE = "MULTIPLE"
cfg.MACHINE_SPEC.SAVE_DIR = "weights/"
cfg.MACHINE_SPEC.LOG_DIR = "logs/"
cfg.DATASET.N_SHAPES = 50
cfg.MODEL.CODEBOOK_SIZE = 256

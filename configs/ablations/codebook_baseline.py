from configs.baseline import cfg as base_cfg

cfg = base_cfg.clone()

cfg.EXP_NAME = "CBNVE"

cfg.MODEL.TYPE = "CodeBookNVE" 
cfg.MODEL.CODEBOOK_SIZE = 2
cfg.TRAINER.COMMIT_LOSS_WEIGHT = 0.5
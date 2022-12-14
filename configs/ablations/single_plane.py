from configs.ablations.codebook_baseline import cfg as base_cfg

cfg = base_cfg.clone()

cfg.EXP_NAME = "main"
# cfg.EXP_NAME = "CBNVE-MultiPlane-10"
cfg.MACHINE_SPEC.DATA_DIR = "../data/nve/sdf_data/planes/1304ef60fe9793f685e0a2007a11e92f/models/envelopes.pkl"
# cfg.MACHINE_SPEC.DATA_DIR = "data/nve/sdf_data/planes/130d3f27fb083eebc0909d98a1ff2b4/models/envelopes.pkl"
cfg.DATASET.PATH = cfg.MACHINE_SPEC.DATA_DIR
cfg.MACHINE_SPEC.SAVE_DIR = "weights/"
cfg.MACHINE_SPEC.LOG_DIR = "logs/"
cfg.DATASET.N_SHAPES = 50
cfg.MODEL.CODEBOOK_SIZE = 256

from yacs.config import CfgNode as CN

config = CN()
config.TYPE = "NVEModel"

config.E2F = CN()
config.E2F.INPUT_DIM = 3  # Either 6 (e2f input contains surface normals) or 3 (no normals)
config.E2F.NUM_LATENTS = 8
config.E2F.LATENT_DIM = 16
config.E2F.FEATURE_TRANSFORM = True

config.F2P = CN()
config.F2P.INPUT_DIM = config.E2F.NUM_LATENTS * config.E2F.LATENT_DIM + 3
config.F2P.OUTPUT_DIM = 1
config.F2P.HIDDEN_DIM = 256
config.F2P.NUM_LAYERS = 4

config.CODEBOOK_SIZE = 0
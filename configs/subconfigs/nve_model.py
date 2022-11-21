from yacs.config import CfgNode as CN

config = CN()

config.E2F = CN()
config.E2F.INPUT_DIM = 1024
config.E2F.NUM_LATENTS = 8
config.E2F.LATENT_DIM = 128

config.F2P = CN()
config.F2P.INPUT_DIM = 128 * 8 + 3
config.F2P.OUTPUT_DIM = 1
config.F2P.HIDDEN_DIM = 256
config.F2P.NUM_LAYERS = 2
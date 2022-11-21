"""
High level mode which combines e2f and f2p. 
To be used for training, as well as mesh extraction etc. 
"""
import torch.nn as nn
import torch as th
from .base_e2f import PointNetLatent
from .base_f2p import FeatureToPoint


class NVEModel(nn.modules):
    def __init__(self, model_config):

        self.e2f = PointNetLatent(model_config.E2F.INPUT_DIM, model_config.E2F.NUM_LATENTS,
                                  model_config.E2F.LATENT_DIM, feature_transform=False, normalize_latents=False)

        self.f2p = FeatureToPoint(model_config.F2P.INPUT_DIM, model_config.F2P.OUTPUT_DIM,
                                  model_config.F2P.HIDDEN_DIM, model_config.F2P.NUM_LAYERS)
        # Add positional encoding to the points?
        # Add Flatten
        self.flatten = nn.Flatten()

    def forward(self, envelope):
        # B X D
        input_obj = envelope['input_obj']
        # B X 8 X L
        feats = self.e2f.forward(input_obj)
        # Expand the features for point prediction
        target_points = envelope['target_points']
        expanded_features = []
        for ind, cur_points in enumerate(target_points):
            expanded_feature = feats[ind: ind +1].expand(cur_points.shape[0], -1)
            expanded_features.append(expanded_feature)
        # B(E) X 8 X L
        expanded_features = th.cat(expanded_features[0])
        # B(E) X 3
        target_points = th.cat(target_points, 0)
        # Flatten and cat
        flattened_input = self.flatten(expanded_feature)
        f2p_input = th.cat(flattened_input, target_points, 1)
        pred_values = self.f2p.forward(f2p_input)
        return pred_values

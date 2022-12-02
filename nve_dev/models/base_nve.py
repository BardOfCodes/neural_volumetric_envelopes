"""
High level mode which combines e2f and f2p. 
To be used for training, as well as mesh extraction etc. 
"""
import torch.nn as nn
import torch as th
from .base_e2f import PointNetLatent
from .base_f2p import FeatureToPoint
import numpy as np
# import rff # pip install random-fourier-features-pytorch

class NVEModel(nn.Module): 
    def __init__(self, model_config):
        super(NVEModel, self).__init__()
        self.e2f = PointNetLatent(model_config.E2F.INPUT_DIM, model_config.E2F.NUM_LATENTS,
                                  model_config.E2F.LATENT_DIM, model_config.E2F.FEATURE_TRANSFORM, normalize_latents=False)

        self.f2p = FeatureToPoint(model_config.F2P.INPUT_DIM, model_config.F2P.OUTPUT_DIM,
                                  model_config.F2P.HIDDEN_DIM, model_config.F2P.NUM_LAYERS)
        
        # Add positional encoding to the points?
        # self.positinal_encoding = rff.layers.PositionalEncoding(sigma=1.0, m=10)
        self.positinal_encoding = None

        # Add Flatten
        self.flatten = nn.Flatten()

    # Helper function for visualization; Forward pass to predict sdf
    # sdf points, N x 3 
    # surface points, B X D
    @th.no_grad()
    def predict_sdf(self, sdf_points, surface_points) :  
        
        # feats -B X NUM_LATENTS X LATENT_DIM
        feats, _, _ = self.e2f.forward(surface_points)

        # Expand the features for point prediction
        expanded_features = []
        sdf_points = [sdf_points]
        for ind, cur_points in enumerate(sdf_points):
            expanded_feature = feats[ind: ind + 1].expand(cur_points.shape[0], -1, -1)
            expanded_features.append(expanded_feature)
        # B(E) X 8 X L
        expanded_features = th.cat(expanded_features, 0)
        # B(E) X 3
        sdf_points = th.cat(sdf_points, 0)
        
        if self.positinal_encoding is not None :
            sdf_points = self.positinal_encoding(sdf_points)
        
        # Flatten and cat
        flattened_input = self.flatten(expanded_features)
        f2p_input = th.cat([flattened_input, sdf_points], 1)
        pred_values = self.f2p.forward(f2p_input)
        return pred_values
    
    def forward(self, input_data):
        # B X D
        input_obj = input_data['surface_points']
        
        # feats -B X NUM_LATENTS X LATENT_DIM
        feats, _, trans_feat = self.e2f.forward(input_obj)

        # Expand the features for point prediction
        training_points = input_data['training_points']
        expanded_features = []
        for ind, cur_points in enumerate(training_points):
            expanded_feature = feats[ind: ind + 1].expand(cur_points.shape[0], -1, -1)
            expanded_features.append(expanded_feature)
        # B(E) X 8 X L
        expanded_features = th.cat(expanded_features, 0)
        # B(E) X 3
        training_points = th.cat(training_points, 0)

        if self.positinal_encoding is not None :
            training_points = self.positinal_encoding(training_points)

        # Flatten and cat
        flattened_input = self.flatten(expanded_features)
        f2p_input = th.cat([flattened_input, training_points], 1)

        pred_values = self.f2p.forward(f2p_input)
        return pred_values, trans_feat

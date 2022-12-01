"""
High level mode which combines e2f and f2p. 
To be used for training, as well as mesh extraction etc. 
"""
import torch.nn as nn
import torch as th
from .base_e2f import PointNetLatent
from .base_f2p import FeatureToPoint


class NVEModel(nn.Module): 
    def __init__(self, model_config):
        super(NVEModel, self).__init__()
        self.e2f = PointNetLatent(model_config.E2F.INPUT_DIM, model_config.E2F.NUM_LATENTS,
                                  model_config.E2F.LATENT_DIM, feature_transform=False, normalize_latents=False)

        self.f2p = FeatureToPoint(model_config.F2P.INPUT_DIM, model_config.F2P.OUTPUT_DIM,
                                  model_config.F2P.HIDDEN_DIM, model_config.F2P.NUM_LAYERS)
        # Add positional encoding to the points?
        # Add Flatten
        self.flatten = nn.Flatten()

    # Helper function for visualization; Forward pass to predict sdf
    # sdf points, N x 3 
    # surface points, B X D
    @th.no_grad()
    def predict_sdf(self, sdf_points, surface_points) :  
        # TODO: Double check this logic
        # B X 8 X L
        pointnet_output = self.e2f.forward(surface_points)
        feats = pointnet_output[0]
        
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
        # Flatten and cat
        flattened_input = self.flatten(expanded_features)
        f2p_input = th.cat([flattened_input, sdf_points], 1)
        pred_values = self.f2p.forward(f2p_input)
        return pred_values
    
    def forward(self, input_data):
        # B X D
        input_obj = input_data['surface_points']
        
        # B X 8 X L
        pointnet_output = self.e2f.forward(input_obj)
        feats = pointnet_output[0]
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
        # Flatten and cat
        flattened_input = self.flatten(expanded_features)
        f2p_input = th.cat([flattened_input, training_points], 1)
        pred_values = self.f2p.forward(f2p_input)
        return pred_values

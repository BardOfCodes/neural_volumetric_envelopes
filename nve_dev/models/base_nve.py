"""
High level mode which combines e2f and f2p. 
To be used for training, as well as mesh extraction etc. 
"""
import torch.nn as nn
import torch as th
from .base_e2f import PointNetLatent
from .base_f2p import FeatureToPoint
import numpy as np
from .vqvae import VectorQuantize
# import rff # pip install random-fourier-features-pytorch

class NVEModel(nn.Module): 
    def __init__(self, model_config):
        super(NVEModel, self).__init__()
        self.e2f = PointNetLatent(model_config.E2F.INPUT_DIM, model_config.E2F.NUM_LATENTS,
                                  model_config.E2F.LATENT_DIM, model_config.E2F.FEATURE_TRANSFORM, normalize_latents=False)

        self.f2p = FeatureToPoint(model_config.F2P.INPUT_DIM, model_config.F2P.OUTPUT_DIM,
                                  model_config.F2P.HIDDEN_DIM, model_config.F2P.NUM_LAYERS)
        
        self.use_surface_normals = model_config.E2F.INPUT_DIM == 6
        
        # Add positional encoding to the points?
        # self.positional_encoding = rff.layers.PositionalEncoding(sigma=1.0, m=10)
        self.positional_encoding = None

        # Add Flatten
        self.flatten = nn.Flatten()
    
    # Helper function for visualization; Forward pass to predict sdf
    # sdf points, N x 3 
    # surface points, B X D
    @th.no_grad()
    def predict_sdf(self, sdf_points, surface_points, surface_normals=th.tensor([])) :  
        if th.numel(surface_normals):
            e2f_input = th.cat((surface_points, surface_normals), axis=2)
        else:
            e2f_input = surface_points
        e2f_input = th.permute(e2f_input, (0,2,1)) # swap last two dimensions
        # feats -B X NUM_LATENTS X LATENT_DIM
        feats, additionals = self.get_e2f_features(e2f_input)

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
        
        if self.positional_encoding is not None :
            sdf_points = self.positional_encoding(sdf_points)
        
        # Flatten and cat
        flattened_input = self.flatten(expanded_features)
        f2p_input = th.cat([flattened_input, sdf_points], 1)
        pred_values = self.f2p.forward(f2p_input)
        return pred_values, additionals
    
    def get_e2f_features(self, input_obj):
        feats, _, trans_feat = self.e2f.forward(input_obj)
        additionals = dict(trans_feat=trans_feat)
        return feats, additionals
        
    def forward(self, input_data):
        # E2F input: B x INPUT_DIM (3 or 6) X n_surface_points (per envelope)
        if self.use_surface_normals:
            input_obj = th.cat((input_data['surface_points'], input_data['surface_normals']), axis=2)
        else: 
            input_obj = input_data['surface_points']
        input_obj = th.permute(input_obj, (0,2,1)) # swap last two dimensions


        # feats -B X NUM_LATENTS X LATENT_DIM
        feats, additionals = self.get_e2f_features(input_obj)

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

        if self.positional_encoding is not None :
            training_points = self.positional_encoding(training_points)

        # Flatten and cat
        flattened_input = self.flatten(expanded_features)
        f2p_input = th.cat([flattened_input, training_points], 1)

        pred_values = self.f2p.forward(f2p_input)
        return pred_values, additionals

class CodeBookNVE(NVEModel):
    
    def __init__(self, model_config):
        super(CodeBookNVE, self).__init__(model_config)
        # Create the VQVAE:
        self.vq = VectorQuantize(dim=model_config.E2F.LATENT_DIM,
                                 codebook_size=model_config.CODEBOOK_SIZE)
    
    def get_e2f_features(self, input_obj):
        feats, additionals = super(CodeBookNVE, self).get_e2f_features(input_obj)
        feats, codebook_indices, commit_loss = self.vq(feats)
        additionals['commit_loss'] = commit_loss
        additionals['codebook_indices'] = codebook_indices
        # additionals['code'] = feats.detach().reshape(-1)
        return feats, additionals
"""
Evaluator:

1> Init test dataset.
2> Expose `evaluate`, a function to evaluate an input model in on the test dataset.

## Eventually we can branch it out if need be.
"""
import numpy as np
import os
from .utils.train_utils import save_all_weights
from .trainer import Trainer

class Evaluator(Trainer):

    def __init__(self, eval_config):
        self.eval_score = -np.inf
        self.best_score = -np.inf
        # Only care about test set MSE
        self.l2_weight = 0
        self.feature_transform_weight = 0
        self.commit_loss_weight = 0
        
        self.save_dir = eval_config.SAVE_DIR
    
    def evaluate(self, model, optimizer, train_state, eval_dataloader):
        loss_list = []
        for iteration_ind, batch in enumerate(eval_dataloader):
            loss, _ = self.calculate_loss(model, batch)
        
            loss_list.append(loss.item())
        avg_loss = np.nanmean(loss_list)
        
        self.eval_score = -avg_loss
        
        if self.eval_score > self.best_score:
            # Save best model:
            self.best_score = self.eval_score
        
            save_path = os.path.join(
                self.save_dir, "best_model.ptpkl")
            save_all_weights(model, optimizer, train_state, save_path)
        
        

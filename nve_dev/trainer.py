"""
A Trainer class, which: 
1> Instantiates all objects (dataloaders, models, evaluator, optimizer, logger)
2> Exposes a "train" function - can be epoch-level, iter-level, or even loss level. 
3> exposes a function to log each iteration stat.
4> Store and reload a "train-state" for restarting from a checkpoint.

## Eventually we can branch it out if need be.
"""

from .utils.logger import WandbLogger
from .utils.train_utils import resume_checkpoint_filename, load_all_weights, save_all_weights
from .train_state import TrainState
from datetime import datetime
import torch as th
from pathlib import Path
import os
from .models.base_e2f import feature_transform_regularizer

class Trainer():

    def __init__(self, train_config):
        # Instantiate logger.
        datetime_tag = datetime.now().strftime("Exp: %m/%d/%Y-%H:%M:%S")
        exp_name = "_".join([train_config.EXP_NAME, datetime_tag])
        # Train config is expected to be a Python dictionary {param: value}
        self.logger = WandbLogger(
            project_name='NVE', entity='csci2951-i', exp_name=exp_name, train_config=train_config)

        # Hyps
        self.feature_transform_weight = train_config.FEATURE_TRANSFORM_WEIGHT
        self.commit_loss_weight = train_config.COMMIT_LOSS_WEIGHT
        self.n_epochs = train_config.N_EPOCHS
        self.save_epoch = train_config.SAVE_EPOCH
        self.eval_epoch = train_config.EVAL_EPOCH
        self.save_dir = train_config.SAVE_DIR
        self.log_interval = train_config.LOG_INTERVAL

        if train_config.RESUME_CHECKPOINT:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            self.init_model_path = resume_checkpoint_filename(self.save_dir)
            if self.init_model_path is None:
                self.init_model_path = train_config.INIT_WEIGHTS
        else:
            self.init_model_path = train_config.INIT_WEIGHTS

    def train(self, model, optimizer, train_dataloader, eval_dataloader, evaluator):

        # Init train state
        train_state = TrainState()

        # Should model weight be loaded outside/before train?
        if self.init_model_path:
            model, optimizer, train_state = load_all_weights(
                model, optimizer, train_state, self.init_model_path)

        self.logger.watch(model)
        model.train()

        for epoch in range(train_state.cur_epoch, self.n_epochs):
            print("Epoch", epoch)
            train_state.cur_epoch = epoch
            for iteration_ind, batch in enumerate(train_dataloader):
                loss, stats_dict = self.calculate_loss(model, batch)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                train_state.n_steps += 1
                self.log_training_details(stats_dict, train_state)
            # Epoch level log?

            if (epoch + 1) % self.eval_epoch == 0:
                # perform evaluation
                model.eval()
                # TBD: Don't send optimizer etc. 
                # Save "best Model" in a differnt way.
                evaluator.evaluate(model, optimizer, train_state, eval_dataloader)
                model.train()
                # Set score:
                train_state.cur_score = evaluator.eval_score
                if train_state.cur_score > train_state.best_score:
                    train_state.best_score = train_state.cur_score
                    train_state.best_epoch = train_state.cur_epoch

            if (epoch + 1) % self.save_epoch == 0:
                # Commenting this out for now as its unclear 
                # how to save more than the model with wandb.
                # Also, should we even store weights through wandb?
                # I prefer outside of it.
                # self.logger.log_model(model, step=iteration_ind)
                save_path = os.path.join(
                    self.save_dir, "weights_%d.ptpkl" % train_state.cur_epoch)
                save_all_weights(model, optimizer, train_state, save_path)

    def calculate_loss(self, model, input_data):
        # Forward and losses
        pred_values, additionals  = model.forward(input_data)
        gt_distances = th.cat(input_data['gt_distances'], 0)
        mse_loss = th.nn.functional.mse_loss(pred_values, gt_distances)

        feature_transform_loss = 0
        if model.e2f.feature_transform :
            trans_feat = additionals['trans_feat']
            feature_transform_loss = feature_transform_regularizer(trans_feat)

        loss = mse_loss + feature_transform_loss * self.feature_transform_weight
        # And Commit Loss
        if self.commit_loss_weight > 0:
            loss += self.commit_loss_weight * additionals['commit_loss'][0]
        
        stats_dict = dict(
            loss=loss.item(),
            mse_loss=mse_loss.item(),
        )
        return loss, stats_dict

    def log_training_details(self, stats_dict, train_state):

        stats_dict_it = train_state.get_state_stats()
        if train_state.n_steps % self.log_interval == 0:

            self.logger.log(stats_dict, train_state.n_steps)

            self.logger.log(stats_dict_it, train_state.n_steps)

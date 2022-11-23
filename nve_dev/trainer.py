"""
A Trainer class, which: 
1> Instantiates all objects (dataloaders, models, evaluator, optimizer, logger)
2> Exposes a "train" function - can be epoch-level, iter-level, or even loss level. 
3> exposes a function to log each iteration stat.
4> Store and reload a "train-state" for restarting from a checkpoint.

## Eventually we can branch it out if need be.
"""

from .utils.logger import WandbLogger
from datetime import datetime

class Trainer():
    
    def __init__(self, train_config):
        # Instantiate logger.
        datetime_tag = datetime.now().strftime("Exp: %m/%d/%Y-%H:%M:%S")
        exp_name = "_".join([train_config.EXP_NAME, datetime_tag])
        # Train config is expected to be a Python dictionary {param: value}
        self.logger = WandbLogger(project_name='NVE', entity='csci2951-i', exp_name=exp_name, train_config=train_config)
        # Also train state
    
    def train(self, train_settings):
        
        self.logger.watch(model)
        model.train()
        for epoch in range(n_epochs):
            
            for iteration_ind, batch in enumerate(dataloader):
                
                input_data = format_data(batch)
                loss = self.calculate_loss(model, input_data)
                
                self.logger.log(metrics={"loss": loss}, step=iteration_ind)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                self.log_training_details()
            
        
        if (epoch + 1) % eval_epoch == 0:
            # perform evaluation
            model.eval()
            self.perform_evaluation()
            model.train()
        
        if (epoch + 1) % save_iter == 0:
            self.checkpoint()
            self.logger.log_model(model, step=iteration_ind)



                
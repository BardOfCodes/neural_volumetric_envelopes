"""
A Trainer class, which: 
1> Instantiates all objects (dataloaders, models, evaluator, optimizer, logger)
2> Exposes a "train" function - can be epoch-level, iter-level, or even loss level. 
3> exposes a function to log each iteration stat.
4> Store and reload a "train-state" for restarting from a checkpoint.

## Eventually we can branch it out if need be.
"""

class Trainer():
    
    def __init__(self, train_config):
        pass
        # Instantiate logger.
    
    def train(self, train_settings):
        
        model.train()
        for epoch in range(n_epochs):
            
            for iteration_ind, batch in enumerate(dataloader):
                
                input_data = format_data(batch)
                loss = self.calculate_loss(model, input_data)
                
                
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



                
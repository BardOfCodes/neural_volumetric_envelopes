"""
A Trainer class, which: 
1> Instantiates all objects (dataloaders, models, evaluator, optimizer, logger)
2> Exposes a "train" function - can be epoch-level, iter-level, or even loss level. 
3> exposes a function to log each iteration stat.
4> Store and reload a "train-state" for restarting from a checkpoint.

## Eventually we can branch it out if need be.
"""
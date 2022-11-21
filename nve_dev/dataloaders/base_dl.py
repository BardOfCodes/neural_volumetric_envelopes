"""
A torch dataset.
Strongly linked to Envelope extraction.
"""
import torch as th
import math

class EnvelopeDataset(th.utils.data.IterableDataset):
    
    def __init__(self, dataset_config):
        
        self.path = dataset_config.PATH
        # TBD: preload everything to the memory.
        
    def __len__(self):
        # Must be size of Epoch X Batch Size
        return self.n_iters
    
    def __getitem__(self, idx):
        # TBD
        envelope_idx = None
        return envelope_idx
        
        
        
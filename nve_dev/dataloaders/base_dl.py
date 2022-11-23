"""
A torch dataset.
Strongly linked to Envelope extraction.
"""
from typing import Dict
import torch as th
import math
import pickle
import numpy as np

class EnvelopeDataset(th.utils.data.Dataset):
    
    def __init__(self, dataset_config):
        
        self.path = dataset_config.PATH   
        self.n_surface_points = dataset_config.N_SURFACE_POINTS
        
        self.min_surface_points = float("inf")
        self.max_surface_points = -1
        self.min_training_points = float("inf")
        self.max_training_points = -1

        self.num_envelopes = 0
        # Map from envelope_id in .pkl file -> id in EnvelopeDataset, useful for final visulization
        self.id_to_envelope_id : Dict[int, str] = {}
        self.envelope_id_to_id : Dict[str, int] = {}

        with open(self.path, "rb") as f :
            loaded_dict = pickle.load(f)
            for envelope_id, envelope_data in loaded_dict.items() :
                self.id_to_envelope_id[self.num_envelopes] = envelope_id
                self.envelope_id_to_id[envelope_id] = self.num_envelopes

                num_surface_points = envelope_data["surface_points"].shape[0]
                self.min_surface_points = min(self.min_surface_points, num_surface_points)
                self.max_surface_points = max(self.max_surface_points, num_surface_points)
                
                num_training_points = envelope_data["training_points"].shape[0]
                self.min_training_points = min(self.min_training_points, num_training_points)
                self.max_training_points = max(self.max_training_points, num_training_points)
                
                self.num_envelopes += 1
        
        # Adding multiple data workers compatability
        self.start = 0
        self.end = self.num_envelopes

        print("Envelope Dataset Intialized w/", self.num_envelopes, "envelopes")
        print("Max/Min Number of Surface Points:", self.max_surface_points, self.min_surface_points)
        print("Max/Min Number of Training Points:", self.max_training_points, self.min_training_points)

        self.load_points(loaded_dict)
        
        
    def load_points(self, loaded_dict):        
        # TODO: Add information on envelopes vertices; Needs to be incorporated in .pkl
        # TODO: Other surface point padding options: Random Sample, Surface point mask

        self.surface_points = np.zeros((self.num_envelopes, self.max_surface_points, 3), dtype=np.float32)
        self.training_points = np.zeros((self.num_envelopes, self.max_training_points, 3), dtype=np.float32)
        self.gt_distances = self.loss_masks = np.zeros((self.num_envelopes, self.max_training_points, 1), dtype=np.int32)

        idx = 0
        for _, envelope_data in loaded_dict.items() :
            # Currently padding with first surface points
            sp = envelope_data["surface_points"]
            self.surface_points[idx] = np.concatenate((sp, sp[:self.max_surface_points - sp.shape[0]]))
            
            num_training_points = envelope_data["training_points"].shape[0]
            self.training_points[idx, : num_training_points] = envelope_data["training_points"]

            self.gt_distances[idx, : num_training_points] = envelope_data["gt_distances"]
            self.loss_masks[idx, : num_training_points] = 1             
            
            idx += 1

    def __len__(self):
        return self.num_envelopes
    
    def __getitem__(self, idx):
        # loss_mask represents the valid indices in training_points/gt_distances
        input_data = {
            "surface_points": self.surface_points[idx], 
            "training_points": self.training_points[idx], 
            "gt_distances": self.gt_distances[idx], 
            "loss_masks": self.loss_masks[idx]
        }
        return input_data
    
# Example from https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
def worker_init_fn(worker_id):
    worker_info = th.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

if __name__ == '__main__':
    print("Testing dataset")
    dataset = EnvelopeDataset(dataset_config = None)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=10, worker_init_fn = worker_init_fn, num_workers = 2, shuffle = True)

    for iteration_idx, batch in enumerate(dataloader) :        
        surface_points, training_points, gt_distances, loss_masks = batch

        # print(surface_points.shape)
        # print(training_points.shape)
        # print(gt_distances.shape)
        # print(loss_masks.shape)
        #  exit()
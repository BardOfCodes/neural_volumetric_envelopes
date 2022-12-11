"""
A torch dataset.
Strongly linked to Envelope extraction.
"""
from typing import Dict
import torch as th
import math
import pickle
import numpy as np

from .base_dl import EnvelopeDataset, worker_init_fn


class NoMaskDataset(EnvelopeDataset):
    """ 
    This version contains no `loss_masks` tensors. 
    Instead, all training points (and gt_distances) are stored as varying length lists. 
    Note: Should be used with collate function.
    """

    def load_points(self, loaded_dict):

        self.surface_points = np.zeros(
            (self.num_envelopes, self.n_surface_points, 3), dtype=np.float32)
        self.surface_normals = np.zeros(
            (self.num_envelopes, self.n_surface_points, 3), dtype=np.float32)
        self.training_points = []
        self.gt_distances = []
        idx = 0
        for envelope_id, envelope_data in loaded_dict.items():
            
            # Subsample self.n_surface_points
            sp = envelope_data["surface_points"]
            sn = envelope_data["surface_point_normals"]
            
            # TODO: Revert to random sel_index with set seed for consistency across training/visualization
            # sel_index = np.arange(0, sp.shape[0])
            # sel_index = np.random.choice(sel_index, size=self.n_surface_points)
            
            if sp.shape[0] > self.n_surface_points:
                # grab the first n_surface_points (points in each enevelope are already randomly permuted)
                sel_index = np.arange(0, self.n_surface_points) 
            else:
                # if this envelope has fewer than n_surface_points, repeat the surface points until self.n_surface_points elements are chosen
                sel_index = np.array(self.repeat([*range(sp.shape[0])], self.n_surface_points))  

            self.surface_points[idx] = sp[sel_index]
            self.surface_normals[idx] = sn[sel_index]
            self.training_points.append(envelope_data["training_points"].astype(np.float32))
            self.gt_distances.append(envelope_data["gt_distances"].astype(np.float32))

            idx += 1
            # if idx == 1 :
            #     break
    
    def repeat(self, seq, length):
        multiple, remainder = divmod(length, len(seq))
        return seq * multiple + seq[:remainder]
    def __getitem__(self, idx):
        # loss_mask represents the valid indices in training_points/gt_distances
        # Dict for clarity:
        input_data = {
            "surface_points": self.surface_points[idx],
            "surface_normals": self.surface_normals[idx],
            "training_points": self.training_points[idx],
            "gt_distances": self.gt_distances[idx],
            "envelope_vertices" : self.envelope_vertices[idx],
        }
        return input_data

def no_mask_collate(batch, device="cuda"):
    surface_points = []
    surface_normals = []
    training_points = []
    gt_distances = []
    for cur_batch in batch:
        surface_points.append(cur_batch['surface_points'])
        surface_normals.append(cur_batch['surface_normals'])
        value = cur_batch['training_points']
        value = th.tensor(value).to(device, non_blocking=True)
        training_points.append(value)
        value = cur_batch['gt_distances']
        value = th.tensor(value).to(device, non_blocking=True)
        gt_distances.append(value)
    
    surface_points = np.stack(surface_points, 0)
    surface_points = th.tensor(surface_points).to(device, non_blocking=True)
    surface_normals = np.stack(surface_normals, 0)
    surface_normals = th.tensor(surface_normals).to(device, non_blocking=True)
    
    batch = {
        "surface_points": surface_points,
        "surface_normals": surface_normals,
        "training_points": training_points,
        "gt_distances": gt_distances,
    }
    return batch

if __name__ == '__main__':
    print("Testing dataset")
    dataset = EnvelopeDataset(dataset_config=None)
    dataloader = th.utils.data.DataLoader(
        dataset, batch_size=10, worker_init_fn=worker_init_fn, num_workers=2, shuffle=True, collate_fn=no_mask_collate)

    for iteration_idx, batch in enumerate(dataloader):
        for key, value in batch.items():
            if isinstance(value, th.Tensor):
                print(key, value.shape)
            else:
                print(key, len(value), value[0].shape)

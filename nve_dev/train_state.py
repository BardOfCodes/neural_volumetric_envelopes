
import numpy as np

class TrainState():

    def __init__(self, epoch=0):

        self.best_score = -np.inf
        self.cur_score = -np.inf
        self.best_epoch = 0
        self.cur_epoch = epoch

        self.n_forwards = 0
        self.n_updates = 0
        self.n_steps = 0
        self.state_attrs = ["best_score", "cur_score", "best_epoch", "cur_epoch", "n_steps"]

    def get_state_stats(self,):
        stats_dict_it = {}
        stats_dict_it['cur epoch'] = self.cur_epoch
        stats_dict_it['best epoch'] = self.best_epoch
        stats_dict_it['cur score'] = self.cur_score
        stats_dict_it['best score'] = self.best_score
        
        return stats_dict_it

    def get_state(self):
        state_dict = {}
        for key in self.state_attrs:
            value = getattr(self, key)
            state_dict[key] = value
        return state_dict
    
    def set_state(self, state_dict):
        for key, value in state_dict.items():
            setattr(self, key, value)
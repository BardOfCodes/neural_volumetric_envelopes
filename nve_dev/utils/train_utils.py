import os
from pathlib import Path
import _pickle as cPickle

def resume_checkpoint_filename(save_dir):
    """ Return latest save file.
    """
    ckpt_dir = save_dir
    ckpts = [os.path.join(ckpt_dir, x) for x in os.listdir(ckpt_dir) if x.split('.')[-1] == 'ptpkl' and x.split('_')[0] == 'weights']
    # select latest checkpoint:
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = ckpts[-1]
    else:
        latest_checkpoint = None
        print("No Checkpoint files found yet!")
    return latest_checkpoint


def save_all_weights(model, optimizer, train_state, save_path):
    save_dir = os.path.dirname(save_path)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    dictionary_items = {
        "model_weights": model.state_dict(),
        "optimizer_weights": optimizer.state_dict(),
        # TBD: LR SCHEDULER
        "epoch": train_state.cur_epoch,
        "score": train_state.cur_score,
        "train_state": train_state.get_state(),
    }        
    with open(save_path, 'wb') as f:
        cPickle.dump(dictionary_items, f)
    
def load_all_weights(model, optimizer, train_state, 
                     load_path, strict=True, 
                     load_optim=True, load_train_state=True):
        
    print("loading weights %s" % load_path)
    with open(load_path, "rb") as f:
        dictionary_items = cPickle.load(f)
    model_weights = dictionary_items["model_weights"]
    model.load_state_dict(model_weights, strict=strict)
    if load_optim:
        optimizer_weights = dictionary_items['optimizer_weights']
        optimizer.load_state_dict(optimizer_weights)
        # TBD: LR Scheduler
    if load_train_state:
        cur_train_state = dictionary_items['train_state']
        train_state.set_state(cur_train_state)
    del dictionary_items

    return model, optimizer, train_state


import torch as th 

from nve_dev.utils.generic_utils import arg_parser, load_config
from nve_dev.dataloaders import EnvelopeDataset, NoMaskDataset
# Push these inside dataloaders as well.
from nve_dev.dataloaders.base_dl import worker_init_fn
from nve_dev.dataloaders.no_mask_dl import no_mask_collate
from nve_dev.models import NVEModel
from nve_dev.trainer import Trainer
from nve_dev.evaluator import Evaluator
from nve_dev.utils.train_utils import load_all_weights
from nve_dev.utils.viz_utils import save_mesh_V2

def main():
    th.backends.cudnn.benchmark = True
    try:
        th.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # Load config
    args = arg_parser.parse_args()
    config = load_config(args)

    # Instantiate Model, optimizer
    model = NVEModel(config.MODEL)
    model.cuda()
    model.eval()

    # Instantiate DataLoader
    dataset = NoMaskDataset(config.DATASET)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, pin_memory=False,
                                          num_workers=0, worker_init_fn=worker_init_fn,
                                          shuffle=False)
    
    # Load model weights
    model_path = "weights/Baseline_no_weight_penality_lr_0.0003/weights_99.ptpkl" # TODO: put a path, make config or arg
    model, _, _ = load_all_weights(
                model, 
                optimizer = None, 
                train_state = None, 
                load_path = model_path,
                load_optim = False, 
                load_train_state = False)
    
    save_mesh_V2(model, dataloader, file_name = "no_weight_penalty_lr_sparse_bounds_0.5_dense", num_samples_per_envelope = 2**24)
    
    print("Visualization Done!")


if __name__ == "__main__":
    main()

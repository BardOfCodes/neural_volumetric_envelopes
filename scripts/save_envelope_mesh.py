
import torch as th 

from nve_dev.utils.generic_utils import arg_parser, load_config
from nve_dev.dataloaders import EnvelopeDataset, NoMaskDataset
# Push these inside dataloaders as well.
from nve_dev.dataloaders.base_dl import worker_init_fn
from nve_dev.dataloaders.no_mask_dl import no_mask_collate
import nve_dev.models as models
from nve_dev.trainer import Trainer
from nve_dev.evaluator import Evaluator
from nve_dev.utils.train_utils import load_all_weights
from nve_dev.utils.viz_utils import save_each_envelope

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
    model_class = getattr(models, config.MODEL.TYPE)
    model = model_class(config.MODEL)
    model.cuda()
    model.eval()

    # Instantiate DataLoader
    dataset = NoMaskDataset(config.DATASET)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, pin_memory=False,
                                          num_workers=0, worker_init_fn=worker_init_fn,
                                          shuffle=False)
    
    # Load model weights
    # model_path = "%s/%s/best_model.ptpkl" % (config.MACHINE_SPEC.SAVE_DIR, config.EXP_NAME)
    model_path = "%s/mikhail/best_model_32.ptpkl" % (config.MACHINE_SPEC.SAVE_DIR)
    model, _, _ = load_all_weights(
                model, 
                optimizer = None, 
                train_state = None, 
                load_path = model_path,
                load_optim = False, 
                load_train_state = False)

    save_each_envelope(model, dataloader, output_directory=args.save_loc, num_samples_per_envelope = 2**18)

    print("Visualization Done!")


if __name__ == "__main__":
    main()

import torch as th

from nve_dev.utils.generic_utils import arg_parser, load_config
from nve_dev.dataloaders import EnvelopeDataset, NoMaskDataset
# Push these inside dataloaders as well.
from nve_dev.dataloaders.base_dl import worker_init_fn
from nve_dev.dataloaders.no_mask_dl import no_mask_collate
from nve_dev.models import NVEModel
from nve_dev.trainer import Trainer
from nve_dev.evaluator import Evaluator


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
    optimizer = th.optim.AdamW(model.parameters(), lr=config.OPT.LR)
    # TO ADD LR Scheduler

    # Instantiate DataLoader
    train_dataset = NoMaskDataset(config.DATASET)
    train_dataloader = th.utils.data.DataLoader(train_dataset, batch_size=config.DATALOADER.BATCH_SIZE, pin_memory=False,
                                          num_workers=config.DATALOADER.NUM_WORKERS, worker_init_fn=worker_init_fn,
                                          shuffle=True, collate_fn=no_mask_collate)
    
    eval_dataset = NoMaskDataset(config.DATASET)
    eval_dataloader = th.utils.data.DataLoader(eval_dataset, batch_size=config.DATALOADER.BATCH_SIZE, pin_memory=False,
                                          num_workers=0, worker_init_fn=worker_init_fn,
                                          shuffle=False, collate_fn=no_mask_collate)
    # Instantiate Training and Evaluator
    trainer = Trainer(config.TRAINER)
    evaluator = Evaluator(config.EVALUATOR)
    trainer.train(model, optimizer, train_dataloader, eval_dataloader, evaluator)
    print("Training finished!")


if __name__ == "__main__":
    main()

import torch as th

from nve_dev.utils.generic_utils import arg_parser, load_config
from nve_dev.dataloaders import EnvelopeDataset
from nve_dev.models import NVEModel
from nve_dev.trainer import Trainer
from nve_dev.evaluator import Evaluator


def main():
    # Load config

    args = arg_parser.parse_args()
    config = load_config(args)

    # Instantiate Model, optimizer
    model = NVEModel(config.MODEL)
    optimizer = th.optim.Adam(model.paramters(), lr=config.OPT.LR)

    # Instantiate DataLoader
    dataset = EnvelopeDataset(config.DATASET)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=config.DATALOADER.BATCH_SIZE, pin_memory=False,
                                          num_workers=config.DATALOADER.NUM_WORKERS, shuffle=True)
    # Instantiate Training and Evaluator
    trainer = Trainer(config.TRAINER)
    evaluator = Evaluator(config.EVALUATOR)

    trainer.train(model, optimizer, dataloader, evaluator, training_settings)
    print("Training finished!")
    # save final model:
    save_model("final_model.pkl", model, optimizer, trainer.train_state)


if __name__ == "__main__":
    main()

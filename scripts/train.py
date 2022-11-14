from nve_dev.utils.generic_utils import arg_parser, load_config 

def main():
    # Load config
    
    args = arg_parser.parse_args()
    config = load_config(args)
    
    # Instantiate Trainer
    
    # trainer.train()
    pass
    
    
    
if __name__ == "__main__":
    main()
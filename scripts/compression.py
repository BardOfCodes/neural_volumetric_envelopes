
import torch as th 
import os
import pickle
import numpy as np

from nve_dev.utils.generic_utils import arg_parser, load_config
import nve_dev.models as models

def main():
    th.backends.cudnn.benchmark = True
    try:
        th.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # Load config
    args = arg_parser.parse_args()
    config = load_config(args)
    
    codebook_ind_bitlen = 8 # for 256 codebooks size
    voxel_ind_bitlen = 5 # for 32 division grid
    codelength = 64

    # Instantiate DataLoader
    path = config.DATASET.PATH
    n_shapes = config.DATASET.N_SHAPES
    directories = os.listdir(path)
    directories = [x for x in directories if os.path.exists(os.path.join(path, x, "models/envelopes.pkl"))]
    directories = directories[:n_shapes]
    
    compression_list = []
    
    across_data_new_mem = 0
    across_data_old_mem = 0
    
    for dir in directories:
        filepath = os.path.join(path, dir, "models/envelopes.pkl")
        
        with open(filepath, "rb") as f :
            new_loaded_dict = pickle.load(f)
            n_envelopes = len(list(new_loaded_dict.values()))
        
        envelope_desc_mem = n_envelopes * 3 * voxel_ind_bitlen
        codebook_desc_mem = n_envelopes * 8 * codebook_ind_bitlen
        
        new_mem = codebook_desc_mem + envelope_desc_mem
        across_data_new_mem += new_mem
        
        # Original mem:
        filepath = os.path.join(path, dir, "models/model_normalized_manifold.obj")
        old_mem = os.path.getsize(filepath) * 8 # since it returns bytes
        
        across_data_old_mem += old_mem
        
        compression_ratio = old_mem/float(new_mem)
        
        compression_list.append(compression_ratio)
    
    compression_np = np.array(compression_list)
    print("AVG. Compression", np.mean(compression_np), "+-", np.std(compression_np))
    print("Median. Compression", np.mean(compression_np))
        

    # Add the mem of codebook:
    across_data_new_mem += codelength * 2 **(codebook_ind_bitlen)
    # also the decoder:
    model_class = getattr(models, config.MODEL.TYPE)
    model = model_class(config.MODEL)
    model_weight_mem = sum(p.numel() for p in model.f2p.parameters())
    across_data_new_mem += model_weight_mem
    
    # Overall Compression:
    across_data_compression = across_data_old_mem/float(across_data_new_mem)
    print("Overall Compression", across_data_compression)
    

if __name__ == "__main__":
    main()

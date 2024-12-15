# import public package
import torch
import numpy

def fix_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    numpy.random.seed(SEED)

    return True
import torch
import numpy as np
import logging


def set_seed():
    temp = 30
    np.random.seed(temp)
    torch.manual_seed(temp)
    torch.cuda.manual_seed_all(temp)


def set_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',  # Optional: to include timestamps and log levels
        filename='sandbox.log',  # Specify the file where logs should be saved
        filemode='w'  # 'a'  append, 'w' overwrite
    )

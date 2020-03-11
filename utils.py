import torch

def get_device_type():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    return dev
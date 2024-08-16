import numpy as np
import torch
import os
import sys
sys.path.append(".")

def load_constant(path):
    constant = np.load(os.path.join(path, "constant.npz"))
    constant_mean = np.load(os.path.join(path, "constant_mean.npz"))
    constant_std = np.load(os.path.join(path, "constant_std.npz"))
    vars = ["land_sea_mask", "orography", "latitude", "longitude"]
    out_constant = {}
    for f in vars:
        out_constant[f] = (constant[f] - constant_mean[f]) / constant_std[f]
    out_constant = np.concatenate([out_constant[f] for f in vars], axis=1).astype(np.float32)

    return out_constant

if __name__ == "__main__":
    load_constant("../../data/train_pred")
#!/usr/bin/env python3
from pathlib import Path
import argparse,json,platform,time
import numpy as np
import torch
from timer_runtime import load_timer
from c4_runtime import load_c4
H=48

def sync(d):
    if d.type=='cuda': torch.cuda.synchronize(d)

def main(model, arr, idx, device):
    x = torch.from_numpy(
        arr["x"][idx:idx+1]
    ).to(
        device=device,
        dtype=torch.float32,
    )

    p = model(x).logits

    if p.shape[-1] < H:
        raise RuntimeError(
            f"Timer logits too short: {tuple(p.shape)}"
        )

    return p[:, :H]
if __name__=='__main__': main()

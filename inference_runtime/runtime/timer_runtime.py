#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM
H=48

def load_timer(root,device):
    m=AutoModelForCausalLM.from_pretrained(str(root/'checkpoints/timer_base_84m'),trust_remote_code=True,local_files_only=True)
    m.to(device); m.eval()
    if hasattr(m.config,'use_cache'): m.config.use_cache=False
    return m

@torch.inference_mode()
def infer(m,z,device,batch_size):
    out=[]; x=z['x']; mu=z['mu']; sd=z['std']
    for s in range(0,len(x),batch_size):
        e=min(len(x),s+batch_size)
        xb=torch.from_numpy(x[s:e]).to(device=device,dtype=torch.float32)
        p=m(xb).logits
        if p.shape[-1] < H:
            raise RuntimeError(f"Timer logits too short: {tuple(p.shape)}")
        p=p[:,:H]
        p=p.float().cpu().numpy()*sd[s:e,None]+mu[s:e,None]
        out.append(p.astype(np.float32))
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--batch_size',type=int,default=1); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; dev=torch.device(a.device)
    p=infer(load_timer(root,dev),np.load(a.input,allow_pickle=False),dev,a.batch_size)
    a.output.parent.mkdir(parents=True,exist_ok=True); np.save(a.output,p); print('saved',a.output,p.shape,p.dtype)
if __name__=='__main__': main()

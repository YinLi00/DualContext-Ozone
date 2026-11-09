#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from c4_model import DualContextAblationForecaster, InputAdapter, time_feature_dim

def load_c4(root,device):
    meta=json.loads((root/'metadata/meta.json').read_text(encoding='utf-8'))
    timer=AutoModelForCausalLM.from_pretrained(str(root/'checkpoints/timer_base_84m'),trust_remote_code=True,local_files_only=True)
    try: ck=torch.load(root/'checkpoints/c4_seed2026_best.pt',map_location='cpu',weights_only=False)
    except TypeError: ck=torch.load(root/'checkpoints/c4_seed2026_best.pt',map_location='cpu')
    a=ck['args']; cities=list(ck.get('cities',meta['cities'])); variables=list(ck.get('variables',meta['variables']))
    m=DualContextAblationForecaster(timer=timer,n_vars=len(variables),n_cities=len(cities),InputAdapter=InputAdapter,time_feature_dim=time_feature_dim,variant='c4_full_dualcontext',city_names=cities,input_hidden=int(a['input_hidden']),output_hidden=int(a['output_hidden']),context_hidden=int(a['context_hidden']),emb_dim=int(a['emb_dim']),city_emb_dim=int(a['city_emb_dim']),dropout=float(a['dropout']),o3_idx=int(meta['o3_idx']),corr_temperature=float(a['corr_temperature']))
    missing,unexpected=m.load_state_dict(ck['adapter_state'],strict=False)
    bad=[k for k in missing if not k.startswith('timer.')]
    if unexpected or bad: raise RuntimeError(f'load mismatch unexpected={list(unexpected)} non_timer_missing={bad}')
    m.to(device); m.eval(); return m

@torch.inference_mode()
def infer(m,z,device,batch_size):
    out=[]; mu=z['mu']; sd=z['std']
    for s in range(0,len(z['x']),batch_size):
        e=min(len(z['x']),s+batch_size)
        p=m(torch.from_numpy(z['x'][s:e]).to(device=device,dtype=torch.float32),torch.from_numpy(z['x_multi'][s:e]).to(device=device,dtype=torch.float32),torch.from_numpy(z['time_feat'][s:e]).to(device=device,dtype=torch.float32),torch.from_numpy(z['var_idx'][s:e]).to(device=device,dtype=torch.long),torch.from_numpy(z['city_id'][s:e]).to(device=device,dtype=torch.long))
        p=p.float().cpu().numpy()*sd[s:e,None]+mu[s:e,None]
        out.append(p.astype(np.float32))
    return np.concatenate(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--batch_size',type=int,default=1); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; dev=torch.device(a.device)
    p=infer(load_c4(root,dev),np.load(a.input,allow_pickle=False),dev,a.batch_size)
    a.output.parent.mkdir(parents=True,exist_ok=True); np.save(a.output,p); print('saved',a.output,p.shape,p.dtype)
if __name__=='__main__': main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',default='timer,p4,c4,qv4,tsmixer')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--base-root',type=Path,default=None)
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/benchmark'))
    ap.add_argument('--smoke-only',action='store_true')
    ap.add_argument('--throughput-batch',type=int,default=32)
    ap.add_argument('--refresh-chunk-size',type=int,default=295)
    a=ap.parse_args()
    root=Path(__file__).resolve().parent
    bench=root/'runtime'/'benchmark_model.py'
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for model in [x.strip() for x in a.models.split(',') if x.strip()]:
        out=a.output_dir/f'{model}_benchmark.json'
        cmd=[sys.executable,str(bench),'--model',model,'--mode','smoke' if a.smoke_only else 'all','--device',a.device,'--output-json',str(out),'--throughput-batch',str(a.throughput_batch),'--refresh-chunk-size',str(a.refresh_chunk_size)]
        if a.base_root is not None: cmd += ['--base-root',str(a.base_root)]
        print('\n>>>',' '.join(cmd),flush=True)
        subprocess.run(cmd,check=True)
    print('\nALL REQUESTED MODELS PASS')
if __name__=='__main__': main()

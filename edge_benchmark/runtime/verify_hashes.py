#!/usr/bin/env python3
from __future__ import annotations
import hashlib
from pathlib import Path

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parents[1]
    man=root/'manifests/sha256_manifest.tsv'
    if not man.is_file(): raise FileNotFoundError(man)
    n=0
    for line in man.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        rel,size,digest=line.split('\t')
        p=root/rel
        if not p.is_file(): raise RuntimeError(f'MISSING {rel}')
        if int(size)!=p.stat().st_size: raise RuntimeError(f'SIZE MISMATCH {rel}')
        got=sha256(p)
        if got!=digest: raise RuntimeError(f'HASH MISMATCH {rel}: {got} != {digest}')
        n+=1
    print(f'HASH PASS {n} files')
if __name__=='__main__': main()

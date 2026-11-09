#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
exp=json.loads((root/'manifests/file_hashes_sha256.json').read_text())
def h(p):
    x=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): x.update(b)
    return x.hexdigest()
bad=[]
for r,e in exp.items():
    p=root/r
    if not p.exists() or h(p)!=e: bad.append(r)
if bad: print('HASH FAIL',bad); sys.exit(1)
print('HASH PASS',len(exp),'files')

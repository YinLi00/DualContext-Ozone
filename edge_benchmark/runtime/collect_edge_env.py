#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, platform, shutil, subprocess
from pathlib import Path
import torch
from edge_common import environment_report, resolve_device, write_json


def cmd(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as e:
        return {"error": repr(e)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-json',type=Path,required=True); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    d=resolve_device(a.device)
    rep=environment_report(d)
    rep['hostname']=platform.node()
    rep['jetson_files']={}
    for p in ['/etc/nv_tegra_release','/etc/nv_boot_control.conf']:
        path=Path(p)
        if path.is_file():
            try: rep['jetson_files'][p]=path.read_text(errors='replace')[:10000]
            except Exception as e: rep['jetson_files'][p]=repr(e)
    if shutil.which('nvpmodel'):
        rep['nvpmodel_q']=cmd(['nvpmodel','-q'])
    if shutil.which('jetson_clocks'):
        rep['jetson_clocks_show']=cmd(['jetson_clocks','--show'])
    write_json(a.output_json,rep); print(json.dumps(rep,indent=2,ensure_ascii=False))
if __name__=='__main__': main()

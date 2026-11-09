from pathlib import Path
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "processed" / "stage1_indices_compact"
DST = ROOT / "processed" / "stage1_indices_compact_qc"

L = 96
EPS = 1e-6

DST.mkdir(parents=True, exist_ok=True)

meta = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))
(DST / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

array_paths = meta["array_paths"]
variables = meta["variables"]

cache = {}

def load_values(array_id: int):
    if array_id not in cache:
        path = ROOT / array_paths[array_id]
        cache[array_id] = np.load(path, allow_pickle=True)["values"].astype(np.float32)
    return cache[array_id]

for split in ["train", "val", "train_all", "test"]:
    print(f"\n==== filtering {split} ====")
    data = np.load(SRC / f"{split}.npz")

    array_id = data["array_id"]
    city_id = data["city_id"]
    var_idx = data["var_idx"]
    start_idx = data["start_idx"]

    n = len(array_id)
    keep = np.ones(n, dtype=bool)

    bad_count_by_var = {v: 0 for v in variables}

    for i in range(n):
        aid = int(array_id[i])
        vid = int(var_idx[i])
        st = int(start_idx[i])

        values = load_values(aid)
        x = values[st:st+L, vid]

        # invalid if input is non-finite or completely constant
        if (not np.isfinite(x).all()) or (float(np.nanmax(x) - np.nanmin(x)) <= EPS):
            keep[i] = False
            bad_count_by_var[variables[vid]] += 1

        if (i + 1) % 500000 == 0:
            print(f"  checked {i+1:,}/{n:,}, removed={(~keep[:i+1]).sum():,}", flush=True)

    out = DST / f"{split}.npz"

    np.savez_compressed(
        out,
        array_id=array_id[keep],
        city_id=city_id[keep],
        var_idx=var_idx[keep],
        start_idx=start_idx[keep],
    )

    print(f"original: {n:,}")
    print(f"kept    : {keep.sum():,}")
    print(f"removed : {(~keep).sum():,}")
    print("removed by variable:")
    for v, c in bad_count_by_var.items():
        if c > 0:
            print(f"  {v}: {c}")

print("\nDone. QC compact indices saved to:")
print(DST)

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = ROOT / "processed" / "stage1_indices"
OUT_ROOT = ROOT / "processed" / "stage1_indices_compact"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

splits = ["train", "val", "train_all", "test"]

# First pass: collect global mappings.
array_paths = []
cities = []
variables = []

for split in splits:
    fp = INDEX_ROOT / f"stage1_index_{split}.csv"
    print(f"Scanning {fp}")
    usecols = ["array_path", "city", "var_name"]
    for chunk in pd.read_csv(fp, usecols=usecols, chunksize=500_000):
        array_paths.extend(chunk["array_path"].unique().tolist())
        cities.extend(chunk["city"].unique().tolist())
        variables.extend(chunk["var_name"].unique().tolist())

array_paths = sorted(set(array_paths))
cities = sorted(set(cities))
# Keep variable order consistent with stage1_city_var_stats.json / saved npz arrays.
stats_path = ROOT / "processed" / "stage1_city_var_stats.json"
stats_meta = json.loads(stats_path.read_text(encoding="utf-8"))
variables = stats_meta["variables"]

array_to_id = {p: i for i, p in enumerate(array_paths)}
city_to_id = {c: i for i, c in enumerate(cities)}
var_to_id = {v: i for i, v in enumerate(variables)}

meta = {
    "array_paths": array_paths,
    "cities": cities,
    "variables": variables,
    "array_to_id": array_to_id,
    "city_to_id": city_to_id,
    "var_to_id": var_to_id,
    "history_length": 96,
    "prediction_length": 48,
}

with open(OUT_ROOT / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("\nMappings:")
print("  n_arrays:", len(array_paths))
print("  n_cities:", len(cities))
print("  variables:", variables)

# Second pass: write compact npz for each split.
for split in splits:
    fp = INDEX_ROOT / f"stage1_index_{split}.csv"
    print(f"\nCompacting {split}: {fp}")

    array_id_list = []
    city_id_list = []
    var_idx_list = []
    start_idx_list = []

    usecols = ["array_path", "city", "var_idx", "start_idx"]

    n = 0
    for chunk in pd.read_csv(fp, usecols=usecols, chunksize=500_000):
        array_id = chunk["array_path"].map(array_to_id).to_numpy(dtype=np.int32)
        city_id = chunk["city"].map(city_to_id).to_numpy(dtype=np.int16)
        var_idx = chunk["var_idx"].to_numpy(dtype=np.int8)
        start_idx = chunk["start_idx"].to_numpy(dtype=np.int32)

        array_id_list.append(array_id)
        city_id_list.append(city_id)
        var_idx_list.append(var_idx)
        start_idx_list.append(start_idx)

        n += len(chunk)
        print(f"  read {n:,} rows", end="\r")

    array_id = np.concatenate(array_id_list)
    city_id = np.concatenate(city_id_list)
    var_idx = np.concatenate(var_idx_list)
    start_idx = np.concatenate(start_idx_list)

    out = OUT_ROOT / f"{split}.npz"
    np.savez_compressed(
        out,
        array_id=array_id,
        city_id=city_id,
        var_idx=var_idx,
        start_idx=start_idx,
    )

    print(f"\n  saved: {out}")
    print(f"  n={len(array_id):,}")

print("\nDone.")

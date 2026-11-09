#!/usr/bin/env python3

from pathlib import Path
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = ROOT / "experiments/results"
INDEX_ROOT = ROOT / "experiments/data_v2/indices"
GLOBAL_ARCHIVE = ROOT / "experiments/results/02_global_baselines/00_global11_v2_array_archive"
OUT = ROOT / "experiments/exports/dualcontext_ablation_raw_by_station_v2"

SEEDS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_N = 106200
EXPECTED_H = 48
EXPECTED_STATIONS = 295
EXPECTED_VARIANTS = 9


def safe_name(x):
    return re.sub(r'[<>:"/\\|?*]', "_", str(x).strip())


def resolve_path(raw):
    p = Path(str(raw))
    if not p.is_absolute():
        p = ROOT / p
    return p.expanduser().resolve()


def station_name(meta, aid):
    stations = meta.get("stations")

    if isinstance(stations, list) and aid < len(stations):
        item = stations[aid]

        if isinstance(item, dict):
            for key in [
                "station_id", "station", "site_id",
                "site_code", "station_code", "code", "name"
            ]:
                v = item.get(key)
                if v is not None and str(v).strip():
                    return safe_name(v)

        elif str(item).strip():
            return safe_name(item)

    return safe_name(resolve_path(meta["array_paths"][aid]).stem)


def bad_path(p):
    s = str(p).lower()
    return any(x in s for x in [
        "00_smoke",
        "smoke_cpu",
        "smoke_gpu",
        "/backups/",
        "/backup/",
        "/logs/",
    ])


# ============================================================
# 1. 自动寻找 DualContext formal result root
# ============================================================

candidates = []

for p in RESULTS_ROOT.iterdir():
    if not p.is_dir():
        continue

    name = p.name.lower()

    if "dualcontext" in name and "ablation" in name:
        candidates.append(p)

if len(candidates) != 1:
    print("DualContext result-root candidates:")
    for x in candidates:
        print(" ", x)
    raise RuntimeError(
        f"Expected exactly 1 DualContext ablation root, got {len(candidates)}"
    )

SOURCE = candidates[0]

print("=" * 80)
print("DUALCONTEXT ABLATION RAW EXPORT")
print("=" * 80)
print("Source:", SOURCE)


# ============================================================
# 2. 自动发现9个正式变体
#    正式结构要求：
#    .../<variant>/seed2026/pred.npy
# ============================================================

variant_runs = {}

for pred_path in SOURCE.rglob("pred.npy"):

    if bad_path(pred_path):
        continue

    run_dir = pred_path.parent

    m = re.fullmatch(r"seed(\d+)", run_dir.name)

    if not m:
        continue

    seed = int(m.group(1))

    if seed not in SEEDS:
        continue

    true_path = run_dir / "true.npy"

    if not true_path.exists():
        continue

    try:
        pred = np.load(
            pred_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        true = np.load(
            true_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except Exception:
        continue

    if pred.shape != (EXPECTED_N, EXPECTED_H):
        continue

    if true.shape != (EXPECTED_N, EXPECTED_H):
        continue

    variant_dir = run_dir.parent

    rel_variant = variant_dir.relative_to(SOURCE)
    export_name = safe_name(variant_dir.name)

    key = str(rel_variant)

    variant_runs.setdefault(
        key,
        {
            "variant_dir": variant_dir,
            "export_name": export_name,
            "runs": {},
        }
    )

    if seed in variant_runs[key]["runs"]:
        raise RuntimeError(
            f"Duplicate run: {key} seed={seed}"
        )

    variant_runs[key]["runs"][seed] = run_dir


# 只保留5个seed全部完整的正式variant
complete = {}

for key, info in variant_runs.items():
    if sorted(info["runs"].keys()) == SEEDS:
        complete[key] = info


print()
print("===== DISCOVERED FORMAL VARIANTS =====")

for i, (key, info) in enumerate(
    sorted(complete.items()),
    start=1,
):
    print(
        f"[{i:02d}] {key} "
        f"-> {info['export_name']} "
        f"seeds={sorted(info['runs'])}"
    )

if len(complete) != EXPECTED_VARIANTS:
    raise RuntimeError(
        f"Found {len(complete)} complete variants, "
        f"expected {EXPECTED_VARIANTS}"
    )

print()
print(
    f"Preflight PASS: "
    f"{EXPECTED_VARIANTS} variants × "
    f"{len(SEEDS)} seeds = "
    f"{EXPECTED_VARIANTS * len(SEEDS)} runs"
)


# ============================================================
# 3. revision-v2 test sample identity
# ============================================================

meta = json.loads(
    (INDEX_ROOT / "meta.json").read_text(
        encoding="utf-8"
    )
)

with np.load(
    INDEX_ROOT / "test.npz",
    allow_pickle=False,
) as z:
    array_id = np.asarray(
        z["array_id"],
        dtype=np.int64,
    )

    city_id = np.asarray(
        z["city_id"],
        dtype=np.int64,
    )

if len(array_id) != EXPECTED_N:
    raise RuntimeError(
        f"test N={len(array_id)}, expected={EXPECTED_N}"
    )

cities = [
    safe_name(x)
    for x in meta["cities"]
]

groups = {}

for aid in sorted(set(array_id.tolist())):

    rows = np.flatnonzero(
        array_id == aid
    )

    if len(rows) != 360:
        raise RuntimeError(
            f"array_id={aid}: "
            f"{len(rows)} samples, expected 360"
        )

    cids = np.unique(
        city_id[rows]
    )

    if len(cids) != 1:
        raise RuntimeError(
            f"array_id={aid}: multiple cities"
        )

    cid = int(cids[0])

    groups[aid] = {
        "rows": rows,
        "city": cities[cid],
        "station": station_name(meta, aid),
    }

if len(groups) != EXPECTED_STATIONS:
    raise RuntimeError(
        f"stations={len(groups)}, "
        f"expected={EXPECTED_STATIONS}"
    )


# ============================================================
# 4. canonical TRUE
# ============================================================

canonical_candidates = sorted(
    GLOBAL_ARCHIVE.glob(
        "*/seed2026/true.npy"
    )
)

if not canonical_candidates:
    raise FileNotFoundError(
        "Cannot locate canonical global true.npy"
    )

canonical_true = np.load(
    canonical_candidates[0],
    mmap_mode="r",
    allow_pickle=False,
)

if canonical_true.shape != (
    EXPECTED_N,
    EXPECTED_H,
):
    raise RuntimeError(
        f"canonical true shape="
        f"{canonical_true.shape}"
    )

print()
print("Cities   :", len(cities))
print("Stations :", len(groups))
print("N        :", EXPECTED_N)


# ============================================================
# 5. Export
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

run_count = 0
manifest_runs = []

# 防止不同上级目录存在同名variant
export_names = [
    x["export_name"]
    for x in complete.values()
]

if len(set(export_names)) != len(export_names):
    raise RuntimeError(
        "Duplicate variant directory names detected."
    )


for variant_key, info in sorted(
    complete.items()
):

    export_name = info["export_name"]

    print()
    print("=" * 80)
    print(export_name)
    print("=" * 80)

    # 每个variant每站保存一份true
    for aid, station_info in groups.items():

        dest = (
            OUT
            / export_name
            / station_info["city"]
            / station_info["station"]
        )

        dest.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            dest / "true.npy",
            np.asarray(
                canonical_true[
                    station_info["rows"]
                ],
                dtype=np.float32,
            ),
            allow_pickle=False,
        )

    for seed in SEEDS:

        run_dir = info["runs"][seed]

        pred = np.load(
            run_dir / "pred.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        true = np.load(
            run_dir / "true.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        if pred.shape != (
            EXPECTED_N,
            EXPECTED_H,
        ):
            raise RuntimeError(
                f"{export_name} seed={seed}: "
                f"pred={pred.shape}"
            )

        if true.shape != (
            EXPECTED_N,
            EXPECTED_H,
        ):
            raise RuntimeError(
                f"{export_name} seed={seed}: "
                f"true={true.shape}"
            )

        if not np.allclose(
            true,
            canonical_true,
            rtol=0.0,
            atol=1e-6,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"{export_name} seed={seed}: "
                "true differs from canonical"
            )

        if not np.isfinite(pred).all():
            raise RuntimeError(
                f"{export_name} seed={seed}: "
                "pred contains NaN/Inf"
            )

        for aid, station_info in groups.items():

            station_pred = np.asarray(
                pred[
                    station_info["rows"]
                ],
                dtype=np.float32,
            )

            if station_pred.shape != (
                360,
                EXPECTED_H,
            ):
                raise RuntimeError(
                    f"{export_name} "
                    f"seed={seed}: "
                    f"station shape="
                    f"{station_pred.shape}"
                )

            dest = (
                OUT
                / export_name
                / station_info["city"]
                / station_info["station"]
            )

            np.save(
                dest
                / f"pred_seed{seed}.npy",
                station_pred,
                allow_pickle=False,
            )

        run_count += 1

        manifest_runs.append({
            "variant": variant_key,
            "export_name": export_name,
            "seed": seed,
            "source_run": str(run_dir),
        })

        print(
            f"[{run_count:02d}/45] "
            f"{export_name} "
            f"seed={seed} COMPLETE"
        )


# ============================================================
# 6. Manifest
# ============================================================

manifest = {
    "protocol": "revision-v2",
    "experiment_group": "dualcontext_ablation",
    "source_root": str(SOURCE),
    "output_root": str(OUT),
    "n_variants": len(complete),
    "n_seeds": len(SEEDS),
    "n_runs": run_count,
    "seeds": SEEDS,
    "n_cities": len(cities),
    "n_stations": len(groups),
    "n_test_samples": EXPECTED_N,
    "samples_per_station": 360,
    "forecast_horizon": EXPECTED_H,
    "station_array_shape": [360, EXPECTED_H],
    "contains_metrics": False,
    "layout": (
        "<ablation>/<city>/<station>/"
        "true.npy + pred_seedXXXX.npy"
    ),
    "runs": manifest_runs,
}

(
    OUT / "EXPORT_MANIFEST.json"
).write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)
print("Output   :", OUT)
print("Runs     :", run_count)
print("Variants :", len(complete))
print("Stations :", len(groups))

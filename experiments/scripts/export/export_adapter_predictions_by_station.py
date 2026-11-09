#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    ROOT /
    "experiments/results/10_adapter_time_ablation_v2"
)

INDEX_ROOT = (
    ROOT /
    "experiments/data_v2/indices"
)

GLOBAL_ARCHIVE = (
    ROOT /
    "experiments/results/02_global_baselines/"
    "00_global11_v2_array_archive"
)

OUT = (
    ROOT /
    "experiments/exports/"
    "adapter_time_ablation_raw_by_station_v2"
)

H = 48
EXPECTED_N = 106200
EXPECTED_STATIONS = 295
SEEDS = [2026, 2027, 2028, 2029, 2030]

# 内部variant -> 导出文件夹名
VARIANTS = {
    "p1_o3_output": "P1_O3-output_adapter",
    "p2_context": "P2_Context_adapter",
    "p3_time_context": "P3_Time-context_adapter",
    "p4_io_time_context": "P4_IO-Time-Context_adapter",
    "t0_none": "T0_No_temporal_features",
    "t1_hour": "T1_Hour_only",
    "t2_doy": "T2_DOY_only",
}


def safe_name(x):
    x = str(x).strip()
    return re.sub(r'[<>:"/\\|?*]', "_", x)


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
                "station_id",
                "station",
                "site_id",
                "site_code",
                "station_code",
                "code",
                "name",
            ]:
                value = item.get(key)
                if value is not None and str(value).strip():
                    return safe_name(value)

        elif str(item).strip():
            return safe_name(item)

    return safe_name(
        resolve_path(meta["array_paths"][aid]).stem
    )


def is_smoke_or_backup(path):
    low = str(path).lower()
    return any(x in low for x in [
        "00_smoke",
        "/backups/",
        "/logs/",
    ])


def find_run_dir(variant, seed):
    """
    自动寻找唯一正式 pred.npy/true.npy。
    不假设formal结果具体在01/02哪个子目录。
    """
    candidates = []

    for pred_path in SOURCE.rglob("pred.npy"):
        if is_smoke_or_backup(pred_path):
            continue

        run_dir = pred_path.parent
        text = str(run_dir).lower()

        if variant.lower() not in text:
            continue

        if f"seed{seed}" not in run_dir.parts:
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

        if (
            pred.shape == (EXPECTED_N, H)
            and true.shape == (EXPECTED_N, H)
        ):
            candidates.append(run_dir)

    candidates = sorted(set(candidates))

    if len(candidates) != 1:
        print()
        print(
            f"ERROR variant={variant} seed={seed}: "
            f"formal candidates={len(candidates)}"
        )
        for x in candidates:
            print("  ", x)
        raise RuntimeError(
            "Expected exactly one formal run."
        )

    return candidates[0]


# ============================================================
# Official revision-v2 sample identity
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
    rows = np.flatnonzero(array_id == aid)

    if len(rows) != 360:
        raise RuntimeError(
            f"array_id={aid}: "
            f"{len(rows)} samples, expected 360"
        )

    cids = np.unique(city_id[rows])

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
        f"station groups={len(groups)}, "
        f"expected={EXPECTED_STATIONS}"
    )

# ============================================================
# Canonical TRUE
# ============================================================

canonical_candidates = list(
    GLOBAL_ARCHIVE.glob("*/seed2026/true.npy")
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

if canonical_true.shape != (EXPECTED_N, H):
    raise RuntimeError(
        f"canonical true shape={canonical_true.shape}"
    )

print("=" * 80)
print("ADAPTER / TIME ABLATION RAW EXPORT")
print("=" * 80)
print("Variants :", len(VARIANTS))
print("Seeds    :", len(SEEDS))
print("Runs     :", len(VARIANTS) * len(SEEDS))
print("Cities   :", len(cities))
print("Stations :", len(groups))
print("N        :", EXPECTED_N)

# ============================================================
# Preflight: 先确认35个正式run全部存在
# ============================================================

run_dirs = {}

print()
print("===== PREFLIGHT: FIND 35 FORMAL RUNS =====")

for variant in VARIANTS:
    for seed in SEEDS:
        run_dir = find_run_dir(
            variant,
            seed,
        )

        run_dirs[(variant, seed)] = run_dir

        print(
            f"[FOUND] {variant} seed={seed}"
        )

if len(run_dirs) != 35:
    raise RuntimeError(
        f"found {len(run_dirs)} runs, expected 35"
    )

print("Preflight PASS: 35/35")

# ============================================================
# Export
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

count = 0
source_manifest = []

for variant, export_name in VARIANTS.items():

    print()
    print("=" * 80)
    print(export_name)
    print("=" * 80)

    # TRUE每个variant/站点只保存一次
    for aid, info in groups.items():

        dest = (
            OUT /
            export_name /
            info["city"] /
            info["station"]
        )

        dest.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            dest / "true.npy",
            np.asarray(
                canonical_true[info["rows"]],
                dtype=np.float32,
            ),
            allow_pickle=False,
        )

    for seed in SEEDS:

        run_dir = run_dirs[(variant, seed)]

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

        if pred.shape != (EXPECTED_N, H):
            raise RuntimeError(
                f"{variant} seed={seed}: "
                f"bad pred shape={pred.shape}"
            )

        if true.shape != (EXPECTED_N, H):
            raise RuntimeError(
                f"{variant} seed={seed}: "
                f"bad true shape={true.shape}"
            )

        if not np.allclose(
            true,
            canonical_true,
            atol=1e-6,
            rtol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"{variant} seed={seed}: "
                "true.npy differs from canonical true"
            )

        if not np.isfinite(pred).all():
            raise RuntimeError(
                f"{variant} seed={seed}: "
                "pred contains NaN/Inf"
            )

        for aid, info in groups.items():

            station_pred = np.asarray(
                pred[info["rows"]],
                dtype=np.float32,
            )

            if station_pred.shape != (360, H):
                raise RuntimeError(
                    f"{variant} seed={seed} "
                    f"{info['city']}/{info['station']} "
                    f"shape={station_pred.shape}"
                )

            dest = (
                OUT /
                export_name /
                info["city"] /
                info["station"]
            )

            np.save(
                dest /
                f"pred_seed{seed}.npy",
                station_pred,
                allow_pickle=False,
            )

        count += 1

        source_manifest.append({
            "variant": variant,
            "export_name": export_name,
            "seed": seed,
            "source_run": str(run_dir),
        })

        print(
            f"[{count:02d}/35] "
            f"{variant} seed={seed} COMPLETE"
        )

# ============================================================
# Manifest：只描述原始数组，不写评价指标
# ============================================================

manifest = {
    "protocol": "revision-v2",
    "experiment_group": "adapter_time_ablation",
    "source_root": str(SOURCE),
    "output_root": str(OUT),
    "variant_mapping": VARIANTS,
    "seeds": SEEDS,
    "n_variants": len(VARIANTS),
    "n_seeds": len(SEEDS),
    "n_runs": count,
    "n_cities": len(cities),
    "n_stations": len(groups),
    "n_test_samples": EXPECTED_N,
    "samples_per_station": 360,
    "forecast_horizon": H,
    "station_array_shape": [360, H],
    "contains_metrics": False,
    "layout": (
        "<ablation>/<city>/<station>/"
        "true.npy + pred_seedXXXX.npy"
    ),
    "sources": source_manifest,
}

(
    OUT /
    "EXPORT_MANIFEST.json"
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
print("Runs     :", count)
print("Variants :", len(VARIANTS))
print("Stations :", len(groups))

#!/usr/bin/env python3

from pathlib import Path
import json
import re
import numpy as np


ROOT = Path(__file__).resolve().parents[3]

TSFM_ROOT = ROOT / "experiments/results/13_tsfm_necessity_v2"
ADAPTER_ROOT = ROOT / "experiments/results/10_adapter_time_ablation_v2"
GLOBAL_ARCHIVE = ROOT / "experiments/results/02_global_baselines/00_global11_v2_array_archive"
INDEX_ROOT = ROOT / "experiments/data_v2/indices"

OUT = ROOT / "experiments/exports/tsfm_necessity_raw_by_station_v2"

SEEDS = [2026, 2027, 2028, 2029, 2030]

EXPECTED_N = 106200
EXPECTED_H = 48
EXPECTED_STATIONS = 295


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


def bad_path(path):
    s = str(path).lower()

    return any(x in s for x in [
        "smoke",
        "backup",
        "/logs/",
        "intervention",
        "00_smoke",
    ])


def valid_full_run(run_dir):
    pred_path = run_dir / "pred.npy"
    true_path = run_dir / "true.npy"

    if not pred_path.exists():
        return False

    if not true_path.exists():
        return False

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
        return False

    return (
        pred.shape == (EXPECTED_N, EXPECTED_H)
        and
        true.shape == (EXPECTED_N, EXPECTED_H)
    )


def find_necessity_run(prefix, seed):
    """
    在 TSFM necessity 中精确找 N0 / N1 / N3。
    seed 必须是完整目录组件 seedXXXX。
    """

    candidates = []

    for pred_path in TSFM_ROOT.rglob("pred.npy"):

        if bad_path(pred_path):
            continue

        run_dir = pred_path.parent

        if run_dir.name != f"seed{seed}":
            continue

        parts_lower = [
            x.lower()
            for x in run_dir.parts
        ]

        matched = any(
            x.startswith(prefix.lower())
            for x in parts_lower
        )

        if not matched:
            continue

        if valid_full_run(run_dir):
            candidates.append(run_dir)

    candidates = sorted(set(candidates))

    if len(candidates) != 1:
        print()
        print(
            f"ERROR prefix={prefix} seed={seed}: "
            f"candidates={len(candidates)}"
        )

        for x in candidates:
            print(" ", x)

        raise RuntimeError(
            "Expected exactly one formal TSFM necessity run."
        )

    return candidates[0]


def find_p4_run(seed):
    """
    自动寻找 Adapter/Time ablation 中的 P4 主模型。
    """

    candidates = []

    for pred_path in ADAPTER_ROOT.rglob("pred.npy"):

        if bad_path(pred_path):
            continue

        run_dir = pred_path.parent

        if run_dir.name != f"seed{seed}":
            continue

        parts_lower = [
            x.lower()
            for x in run_dir.parts
        ]

        matched = any(
            x.startswith("p4_")
            for x in parts_lower
        )

        if not matched:
            continue

        if valid_full_run(run_dir):
            candidates.append(run_dir)

    candidates = sorted(set(candidates))

    if len(candidates) != 1:
        print()
        print(
            f"ERROR P4 seed={seed}: "
            f"candidates={len(candidates)}"
        )

        for x in candidates:
            print(" ", x)

        raise RuntimeError(
            "Expected exactly one P4 formal run."
        )

    return candidates[0]


def find_tsmixer_dir():
    """
    找 Global archive 里的 TSMixer。
    """

    hits = []

    for p in GLOBAL_ARCHIVE.iterdir():

        if not p.is_dir():
            continue

        if p.name.lower() == "tsmixer":
            hits.append(p)

    if len(hits) != 1:
        print("TSMixer candidates:")

        for x in hits:
            print(" ", x)

        raise RuntimeError(
            f"Expected one TSMixer archive dir, got {len(hits)}"
        )

    return hits[0]


# ============================================================
# Load revision-v2 metadata/index
# ============================================================

meta_path = INDEX_ROOT / "meta.json"
test_index_path = INDEX_ROOT / "test.npz"

if not meta_path.exists():
    raise FileNotFoundError(meta_path)

if not test_index_path.exists():
    raise FileNotFoundError(test_index_path)

meta = json.loads(
    meta_path.read_text(
        encoding="utf-8"
    )
)

with np.load(
    test_index_path,
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
            f"{len(rows)} samples, expected=360"
        )

    cids = np.unique(
        city_id[rows]
    )

    if len(cids) != 1:
        raise RuntimeError(
            f"array_id={aid}: multiple cities={cids}"
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
# Find TSMixer
# ============================================================

TSMIXER_DIR = find_tsmixer_dir()

canonical_path = (
    TSMIXER_DIR
    / "seed2026"
    / "true.npy"
)

if not canonical_path.exists():
    raise FileNotFoundError(canonical_path)


canonical_true = np.load(
    canonical_path,
    mmap_mode="r",
    allow_pickle=False,
)


if canonical_true.shape != (
    EXPECTED_N,
    EXPECTED_H,
):
    raise RuntimeError(
        f"canonical true shape={canonical_true.shape}"
    )


print("=" * 80)
print("TSFM NECESSITY RAW EXPORT")
print("=" * 80)
print("TSFM root :", TSFM_ROOT)
print("P4 root   :", ADAPTER_ROOT)
print("TSMixer   :", TSMIXER_DIR)
print("Seeds     :", SEEDS)
print("Cities    :", len(cities))
print("Stations  :", len(groups))
print("N         :", EXPECTED_N)


# ============================================================
# Preflight all 25 runs
# ============================================================

methods = {
    "N0_No_Backbone": {},
    "N1_Persistence": {},
    "N3_Random_Timer": {},
    "N4_Pretrained_Timer_Full_Adapter": {},
    "TSMixer": {},
}


print()
print("===== PREFLIGHT =====")


for seed in SEEDS:

    methods["N0_No_Backbone"][seed] = find_necessity_run(
        "n0_",
        seed,
    )

    print(
        f"[FOUND] N0 seed={seed}"
    )


for seed in SEEDS:

    methods["N1_Persistence"][seed] = find_necessity_run(
        "n1_",
        seed,
    )

    print(
        f"[FOUND] N1 seed={seed}"
    )


for seed in SEEDS:

    methods["N3_Random_Timer"][seed] = find_necessity_run(
        "n3_",
        seed,
    )

    print(
        f"[FOUND] N3 seed={seed}"
    )


for seed in SEEDS:

    methods["N4_Pretrained_Timer_Full_Adapter"][seed] = find_p4_run(
        seed
    )

    print(
        f"[FOUND] N4/P4 seed={seed}"
    )


for seed in SEEDS:

    run_dir = (
        TSMIXER_DIR
        / f"seed{seed}"
    )

    if not valid_full_run(run_dir):
        raise RuntimeError(
            f"Invalid TSMixer seed={seed}: {run_dir}"
        )

    methods["TSMixer"][seed] = run_dir

    print(
        f"[FOUND] TSMixer seed={seed}"
    )


total_runs = sum(
    len(x)
    for x in methods.values()
)


if total_runs != 25:
    raise RuntimeError(
        f"Formal runs={total_runs}, expected=25"
    )


print()
print("Preflight PASS: 5 methods × 5 seeds = 25 runs")


# ============================================================
# Export
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


run_count = 0
source_manifest = []


for method_name, runs in methods.items():

    print()
    print("=" * 80)
    print(method_name)
    print("=" * 80)

    # 每个方法、每个站点只保存一份 true
    for aid, info in groups.items():

        dest = (
            OUT
            / method_name
            / info["city"]
            / info["station"]
        )

        dest.mkdir(
            parents=True,
            exist_ok=True,
        )

        station_true = np.asarray(
            canonical_true[
                info["rows"]
            ],
            dtype=np.float32,
        )

        if station_true.shape != (
            360,
            EXPECTED_H,
        ):
            raise RuntimeError(
                f"Bad station true shape: "
                f"{method_name} "
                f"{info['city']}/{info['station']} "
                f"{station_true.shape}"
            )

        np.save(
            dest / "true.npy",
            station_true,
            allow_pickle=False,
        )


    for seed in SEEDS:

        run_dir = runs[seed]

        pred_path = run_dir / "pred.npy"
        true_path = run_dir / "true.npy"

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


        if pred.shape != (
            EXPECTED_N,
            EXPECTED_H,
        ):
            raise RuntimeError(
                f"{method_name} seed={seed}: "
                f"pred shape={pred.shape}"
            )


        if true.shape != (
            EXPECTED_N,
            EXPECTED_H,
        ):
            raise RuntimeError(
                f"{method_name} seed={seed}: "
                f"true shape={true.shape}"
            )


        if not np.allclose(
            true,
            canonical_true,
            rtol=0.0,
            atol=1e-6,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"{method_name} seed={seed}: "
                "true differs from canonical target"
            )


        if not np.isfinite(pred).all():
            raise RuntimeError(
                f"{method_name} seed={seed}: "
                "pred contains NaN/Inf"
            )


        for aid, info in groups.items():

            station_pred = np.asarray(
                pred[
                    info["rows"]
                ],
                dtype=np.float32,
            )

            if station_pred.shape != (
                360,
                EXPECTED_H,
            ):
                raise RuntimeError(
                    f"{method_name} seed={seed} "
                    f"{info['city']}/{info['station']}: "
                    f"{station_pred.shape}"
                )

            dest = (
                OUT
                / method_name
                / info["city"]
                / info["station"]
            )

            np.save(
                dest
                / f"pred_seed{seed}.npy",
                station_pred,
                allow_pickle=False,
            )


        run_count += 1

        source_manifest.append({
            "method": method_name,
            "seed": seed,
            "source_run": str(run_dir),
        })

        print(
            f"[{run_count:02d}/25] "
            f"{method_name} "
            f"seed={seed} COMPLETE"
        )


# ============================================================
# Manifest
# ============================================================

manifest = {
    "protocol": "revision-v2",
    "experiment_group": "tsfm_necessity",
    "output_root": str(OUT),
    "methods": list(methods.keys()),
    "seeds": SEEDS,
    "n_methods": len(methods),
    "n_seeds": len(SEEDS),
    "n_runs": run_count,
    "n_cities": len(cities),
    "n_stations": len(groups),
    "n_test_samples": EXPECTED_N,
    "samples_per_station": 360,
    "forecast_horizon": EXPECTED_H,
    "station_array_shape": [360, EXPECTED_H],
    "contains_metrics": False,
    "notes": {
        "N0": "No backbone / direct model",
        "N1": "Persistence backbone + full adapter",
        "N3": "Randomly initialized Timer + full adapter",
        "N4": "Pretrained frozen Timer + full adapter; reused from P4",
        "TSMixer": "Global supervised reference; reused from Global-11 archive",
    },
    "layout": (
        "<method>/<city>/<station>/"
        "true.npy + pred_seedXXXX.npy"
    ),
    "sources": source_manifest,
}


(
    OUT
    / "EXPORT_MANIFEST.json"
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
print("Methods  :", len(methods))
print("Stations :", len(groups))

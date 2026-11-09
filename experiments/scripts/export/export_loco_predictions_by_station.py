#!/usr/bin/env python3

from pathlib import Path
import json
import re
import numpy as np


ROOT = Path(__file__).resolve().parents[3]

LOCO_ROOT = (
    ROOT /
    "experiments/results/14_loco_22cities_v2"
)

INDEX_ROOT = (
    ROOT /
    "experiments/data_v2/indices"
)

OUT = (
    ROOT /
    "experiments/exports/"
    "loco_unseen_city_raw_by_station_v2"
)

SEEDS = [2026, 2027, 2028, 2029, 2030]

EXPECTED_N = 106200
EXPECTED_H = 48
EXPECTED_CITIES = 22
EXPECTED_STATIONS = 295


def safe_name(x):
    return re.sub(
        r'[<>:"/\\|?*]',
        "_",
        str(x).strip()
    )


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
                v = item.get(key)
                if v is not None and str(v).strip():
                    return safe_name(v)

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
        "tmp",
    ])


def load_pair(run_dir):
    pred_path = run_dir / "pred.npy"
    true_path = run_dir / "true.npy"

    if not pred_path.exists() or not true_path.exists():
        return None

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
        return None

    if pred.ndim != 2 or true.ndim != 2:
        return None

    if pred.shape[1] != EXPECTED_H:
        return None

    if true.shape != pred.shape:
        return None

    return pred, true


# ============================================================
# 1. Official revision-v2 test identity
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


cities_raw = [str(x) for x in meta["cities"]]
cities = [safe_name(x) for x in cities_raw]

if len(cities) != EXPECTED_CITIES:
    raise RuntimeError(
        f"cities={len(cities)}, expected={EXPECTED_CITIES}"
    )


# city -> full revision-v2 rows
city_rows = {}

# aid -> station identity
stations = {}

# city -> station aids
city_station_aids = {}


for cid, city in enumerate(cities):

    rows = np.flatnonzero(
        city_id == cid
    )

    city_rows[city] = rows

    aids = sorted(
        set(
            int(x)
            for x in array_id[rows]
        )
    )

    city_station_aids[city] = aids

    for aid in aids:

        station_rows = np.flatnonzero(
            array_id == aid
        )

        this_city = np.unique(
            city_id[station_rows]
        )

        if len(this_city) != 1 or int(this_city[0]) != cid:
            raise RuntimeError(
                f"array_id={aid}: city mapping error"
            )

        if len(station_rows) != 360:
            raise RuntimeError(
                f"array_id={aid}: "
                f"samples={len(station_rows)}, expected=360"
            )

        # positions inside held-out city array
        pos_map = {
            int(full_row): pos
            for pos, full_row in enumerate(rows)
        }

        local_positions = np.asarray(
            [
                pos_map[int(r)]
                for r in station_rows
            ],
            dtype=np.int64,
        )

        stations[aid] = {
            "city": city,
            "station": station_name(meta, aid),
            "full_rows": station_rows,
            "local_positions": local_positions,
        }


if len(stations) != EXPECTED_STATIONS:
    raise RuntimeError(
        f"stations={len(stations)}, expected={EXPECTED_STATIONS}"
    )


print("=" * 80)
print("LOCO UNSEEN-CITY RAW EXPORT")
print("=" * 80)
print("Source   :", LOCO_ROOT)
print("Cities   :", len(cities))
print("Stations :", len(stations))
print("Seeds    :", SEEDS)


# ============================================================
# 2. Discover LOCO full runs
# ============================================================

full_runs = {
    city: {}
    for city in cities
}


for pred_path in LOCO_ROOT.rglob("pred.npy"):

    if bad_path(pred_path):
        continue

    run_dir = pred_path.parent

    # 必须包含精确 seedXXXX 路径组件
    seed = None

    for component in run_dir.parts:
        m = re.fullmatch(
            r"seed(\d+)",
            component.lower(),
        )

        if m:
            value = int(m.group(1))
            if value in SEEDS:
                seed = value
                break

    if seed is None:
        continue

    pair = load_pair(run_dir)

    if pair is None:
        continue

    pred, true = pair

    # LOCO run只测试一个held-out city，
    # 因此用行数匹配城市，再结合路径城市名确认。
    possible = []

    run_text = str(run_dir).lower()

    for city in cities:

        n_city = len(city_rows[city])

        if pred.shape != (n_city, EXPECTED_H):
            continue

        # 优先要求city名出现在路径中
        if city.lower() in run_text:
            possible.append(city)

    if len(possible) != 1:
        continue

    city = possible[0]

    if seed in full_runs[city]:
        raise RuntimeError(
            f"Duplicate LOCO full run: "
            f"city={city} seed={seed}\n"
            f"OLD={full_runs[city][seed]}\n"
            f"NEW={run_dir}"
        )

    full_runs[city][seed] = run_dir


print()
print("===== FULL LOCO PREFLIGHT =====")

missing = []

for city in cities:

    found = sorted(
        full_runs[city].keys()
    )

    print(
        f"{city}: seeds={found}"
    )

    for seed in SEEDS:
        if seed not in full_runs[city]:
            missing.append(
                (city, seed)
            )


if missing:
    print()
    print("Missing LOCO full runs:")
    for x in missing:
        print(" ", x)

    raise RuntimeError(
        f"Missing {len(missing)} LOCO full runs"
    )


n_full = sum(
    len(x)
    for x in full_runs.values()
)

if n_full != 110:
    raise RuntimeError(
        f"full runs={n_full}, expected=110"
    )


print()
print("Full LOCO preflight PASS: 22 × 5 = 110")


# ============================================================
# 3. Discover deterministic references
# ============================================================

REF_ROOT = (
    LOCO_ROOT /
    "03_reference"
)


def find_reference(city, ref_name):

    ref_dir = (
        REF_ROOT /
        ref_name /
        city
    )

    if not ref_dir.exists():
        raise FileNotFoundError(
            ref_dir
        )

    candidates = []

    for pred_path in ref_dir.rglob(
        "pred.npy"
    ):

        run_dir = pred_path.parent

        pair = load_pair(run_dir)

        if pair is None:
            continue

        pred, true = pair

        expected_n = len(
            city_rows[city]
        )

        if pred.shape == (
            expected_n,
            EXPECTED_H,
        ):
            candidates.append(
                run_dir
            )

    candidates = sorted(
        set(candidates)
    )

    if len(candidates) != 1:
        print()
        print(
            f"Reference {ref_name}/{city}: "
            f"candidates={len(candidates)}"
        )

        for x in candidates:
            print(" ", x)

        raise RuntimeError(
            "Expected exactly one reference run."
        )

    return candidates[0]


references = {
    "Persistence": {},
    "Timer_Source_Scaled": {},
}


print()
print("===== REFERENCE PREFLIGHT =====")


for city in cities:

    references[
        "Persistence"
    ][city] = find_reference(
        city,
        "persistence",
    )

    references[
        "Timer_Source_Scaled"
    ][city] = find_reference(
        city,
        "timer_source_scaled",
    )

    print(
        f"[FOUND] {city}: "
        f"Persistence + Timer_Source_Scaled"
    )


print()
print(
    "Reference preflight PASS: "
    "22 × 2 = 44"
)


# ============================================================
# 4. Export LOCO Full
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

manifest_sources = []


for city in cities:

    rows = city_rows[city]

    aids = city_station_aids[
        city
    ]

    print()
    print("=" * 80)
    print("LOCO_Full:", city)
    print("=" * 80)

    # first seed supplies canonical held-out true
    first_run = full_runs[
        city
    ][
        SEEDS[0]
    ]

    _, city_true = load_pair(
        first_run
    )

    city_true = np.asarray(
        city_true,
        dtype=np.float32,
    )


    # Validate all seeds and export
    for seed in SEEDS:

        run_dir = full_runs[
            city
        ][
            seed
        ]

        pair = load_pair(
            run_dir
        )

        if pair is None:
            raise RuntimeError(
                f"Invalid pair: {run_dir}"
            )

        pred, true = pair

        pred = np.asarray(
            pred,
            dtype=np.float32,
        )

        true = np.asarray(
            true,
            dtype=np.float32,
        )

        if not np.allclose(
            true,
            city_true,
            rtol=0.0,
            atol=1e-6,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"LOCO true mismatch: "
                f"{city} seed={seed}"
            )

        if not np.isfinite(
            pred
        ).all():
            raise RuntimeError(
                f"Non-finite pred: "
                f"{city} seed={seed}"
            )

        for aid in aids:

            info = stations[
                aid
            ]

            pos = info[
                "local_positions"
            ]

            dest = (
                OUT /
                "LOCO_Full" /
                city /
                info["station"]
            )

            dest.mkdir(
                parents=True,
                exist_ok=True,
            )

            if seed == SEEDS[0]:

                np.save(
                    dest / "true.npy",
                    np.asarray(
                        city_true[pos],
                        dtype=np.float32,
                    ),
                    allow_pickle=False,
                )

            np.save(
                dest /
                f"pred_seed{seed}.npy",
                np.asarray(
                    pred[pos],
                    dtype=np.float32,
                ),
                allow_pickle=False,
            )

        manifest_sources.append({
            "method": "LOCO_Full",
            "city": city,
            "seed": seed,
            "source_run": str(run_dir),
        })

        print(
            f"  seed={seed} COMPLETE"
        )


# ============================================================
# 5. Export references
# ============================================================

for method_name, city_map in references.items():

    print()
    print("=" * 80)
    print(method_name)
    print("=" * 80)

    for city in cities:

        run_dir = city_map[
            city
        ]

        pair = load_pair(
            run_dir
        )

        if pair is None:
            raise RuntimeError(
                f"Bad reference: {run_dir}"
            )

        pred, true = pair

        pred = np.asarray(
            pred,
            dtype=np.float32,
        )

        true = np.asarray(
            true,
            dtype=np.float32,
        )

        if not np.isfinite(
            pred
        ).all():
            raise RuntimeError(
                f"Non-finite reference pred: "
                f"{method_name}/{city}"
            )

        aids = city_station_aids[
            city
        ]

        for aid in aids:

            info = stations[
                aid
            ]

            pos = info[
                "local_positions"
            ]

            dest = (
                OUT /
                method_name /
                city /
                info["station"]
            )

            dest.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                dest / "true.npy",
                np.asarray(
                    true[pos],
                    dtype=np.float32,
                ),
                allow_pickle=False,
            )

            np.save(
                dest / "pred.npy",
                np.asarray(
                    pred[pos],
                    dtype=np.float32,
                ),
                allow_pickle=False,
            )

        manifest_sources.append({
            "method": method_name,
            "city": city,
            "source_run": str(run_dir),
        })

        print(
            f"  {city} COMPLETE"
        )


# ============================================================
# 6. Manifest
# ============================================================

manifest = {
    "protocol": "revision-v2",
    "experiment_group": "LOCO_unseen_city",
    "source_root": str(LOCO_ROOT),
    "output_root": str(OUT),

    "methods": [
        "LOCO_Full",
        "Persistence",
        "Timer_Source_Scaled",
    ],

    "full_loco": {
        "n_cities": 22,
        "n_seeds": 5,
        "n_runs": 110,
        "seeds": SEEDS,
    },

    "references": {
        "Persistence": (
            "Deterministic persistence reference."
        ),
        "Timer_Source_Scaled": (
            "Frozen Timer reference scaled "
            "using source-city statistics only."
        ),
    },

    "n_stations": EXPECTED_STATIONS,
    "forecast_horizon": EXPECTED_H,
    "samples_per_station": 360,

    "contains_metrics": False,

    "layout": {
        "LOCO_Full": (
            "LOCO_Full/<city>/<station>/"
            "true.npy + pred_seedXXXX.npy"
        ),
        "references": (
            "<reference>/<city>/<station>/"
            "true.npy + pred.npy"
        ),
    },

    "sources": manifest_sources,
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
print("Output       :", OUT)
print("LOCO runs    :", n_full)
print("Cities       :", len(cities))
print("Stations     :", len(stations))
print("References   : 44 city-level evaluations")

from pathlib import Path
import argparse
import csv
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "datasets"
OUT_ROOT = ROOT / "processed"
ARRAY_ROOT = OUT_ROOT / "stage1_arrays"
INDEX_ROOT = OUT_ROOT / "stage1_indices"
STATS_PATH = OUT_ROOT / "stage1_city_var_stats.json"
SUMMARY_PATH = OUT_ROOT / "stage1_index_summary.csv"

VARIABLES = [
    "PM2.5", "PM10", "SO2", "NO2", "O3", "CO",
    "temperature", "relative_humidity"
]

L = 96
H = 48
TOTAL_LEN = L + H

TRAIN_START = pd.Timestamp("2023-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2024-10-31 23:00:00")

VAL_START = pd.Timestamp("2024-11-01 00:00:00")
VAL_END = pd.Timestamp("2024-12-31 23:00:00")

TRAIN_ALL_START = pd.Timestamp("2023-01-01 00:00:00")
TRAIN_ALL_END = pd.Timestamp("2024-12-31 23:00:00")

TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2025-12-31 23:00:00")


def read_csv_smart(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def open_writer(path: Path):
    ensure_parent(path)
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "city",
            "station_file",
            "station_id",
            "array_path",
            "var_name",
            "var_idx",
            "start_idx",
            "input_start",
            "input_end",
            "target_start",
            "target_end",
        ],
    )
    writer.writeheader()
    return f, writer


def in_range(ts_start, ts_end, split_start, split_end):
    return (ts_start >= split_start) and (ts_end <= split_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stride", type=int, default=12)
    parser.add_argument("--val_stride", type=int, default=24)
    parser.add_argument("--train_all_stride", type=int, default=12)
    parser.add_argument("--test_stride", type=int, default=24)
    parser.add_argument("--min_station_files", type=int, default=10)
    args = parser.parse_args()

    ARRAY_ROOT.mkdir(parents=True, exist_ok=True)
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)

    writers = {}
    files = {}

    split_paths = {
        "train": INDEX_ROOT / "stage1_index_train.csv",
        "val": INDEX_ROOT / "stage1_index_val.csv",
        "train_all": INDEX_ROOT / "stage1_index_train_all.csv",
        "test": INDEX_ROOT / "stage1_index_test.csv",
    }

    for split, path in split_paths.items():
        f, w = open_writer(path)
        files[split] = f
        writers[split] = w

    split_counts = {split: 0 for split in split_paths}
    split_city_counts = {}

    # For city-variable statistics, use 2023-2024 only, not 2025.
    stats_acc = {}
    summary_rows = []

    cities = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
    cities = [c for c in cities if len(list(c.glob("*.csv"))) >= args.min_station_files]

    print(f"Found {len(cities)} cities with >= {args.min_station_files} stations.")

    for city_dir in cities:
        city = city_dir.name
        csv_files = sorted(city_dir.glob("*.csv"))
        print(f"\n==== city={city}, stations={len(csv_files)} ====")

        stats_acc[city] = {
            v: {"sum": 0.0, "sumsq": 0.0, "count": 0} for v in VARIABLES
        }

        for station_i, fp in enumerate(csv_files, start=1):
            station_file = fp.name
            station_id = fp.stem

            df = read_csv_smart(fp)
            if "time" not in df.columns:
                raise ValueError(f"{fp} missing time column")

            time = pd.to_datetime(
                df["time"].astype(str),
                format="%Y%m%d %H:%M",
                errors="coerce",
            )

            df = df.copy()
            df["time_parsed"] = time
            df = df.dropna(subset=["time_parsed"]).sort_values("time_parsed").reset_index(drop=True)

            for col in VARIABLES:
                if col not in df.columns:
                    raise ValueError(f"{fp} missing column: {col}")

            times = df["time_parsed"].to_numpy()
            values = np.stack(
                [pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32) for col in VARIABLES],
                axis=1,
            )

            # Save compact station array.
            city_array_dir = ARRAY_ROOT / city
            city_array_dir.mkdir(parents=True, exist_ok=True)
            array_path = city_array_dir / f"{station_id}.npz"

            if not array_path.exists():
                np.savez_compressed(
                    array_path,
                    values=values,
                    time_ns=df["time_parsed"].astype("int64").to_numpy(),
                    variables=np.array(VARIABLES, dtype=object),
                    city=city,
                    station_id=station_id,
                    station_file=station_file,
                )

            rel_array_path = str(array_path.relative_to(ROOT))

            # Accumulate stats on 2023-2024.
            time_pd = pd.Series(df["time_parsed"])
            stat_mask = (time_pd >= TRAIN_ALL_START) & (time_pd <= TRAIN_ALL_END)
            stat_values = values[stat_mask.to_numpy()]

            for vi, var in enumerate(VARIABLES):
                vv = stat_values[:, vi]
                finite = np.isfinite(vv)
                vv = vv[finite]
                if vv.size > 0:
                    stats_acc[city][var]["sum"] += float(vv.sum())
                    stats_acc[city][var]["sumsq"] += float((vv * vv).sum())
                    stats_acc[city][var]["count"] += int(vv.size)

            n = len(df)
            if n < TOTAL_LEN:
                continue

            # For each variable, build valid window mask.
            for vi, var in enumerate(VARIABLES):
                series = values[:, vi]
                finite = np.isfinite(series).astype(np.int32)

                # valid window for input+target length 144
                rolling = np.convolve(finite, np.ones(TOTAL_LEN, dtype=np.int32), mode="valid")
                valid_starts = np.where(rolling == TOTAL_LEN)[0]

                for start_idx in valid_starts:
                    input_start = pd.Timestamp(times[start_idx])
                    input_end = pd.Timestamp(times[start_idx + L - 1])
                    target_start = pd.Timestamp(times[start_idx + L])
                    target_end = pd.Timestamp(times[start_idx + TOTAL_LEN - 1])

                    row = {
                        "city": city,
                        "station_file": station_file,
                        "station_id": station_id,
                        "array_path": rel_array_path,
                        "var_name": var,
                        "var_idx": vi,
                        "start_idx": int(start_idx),
                        "input_start": input_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "input_end": input_end.strftime("%Y-%m-%d %H:%M:%S"),
                        "target_start": target_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "target_end": target_end.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # train: selected subset for early stopping.
                    if (
                        in_range(input_start, target_end, TRAIN_START, TRAIN_END)
                        and start_idx % args.train_stride == 0
                    ):
                        writers["train"].writerow(row)
                        split_counts["train"] += 1
                        split_city_counts[("train", city)] = split_city_counts.get(("train", city), 0) + 1

                    # val
                    if (
                        in_range(input_start, target_end, VAL_START, VAL_END)
                        and start_idx % args.val_stride == 0
                    ):
                        writers["val"].writerow(row)
                        split_counts["val"] += 1
                        split_city_counts[("val", city)] = split_city_counts.get(("val", city), 0) + 1

                    # train_all: final training set using 2023-2024.
                    if (
                        in_range(input_start, target_end, TRAIN_ALL_START, TRAIN_ALL_END)
                        and start_idx % args.train_all_stride == 0
                    ):
                        writers["train_all"].writerow(row)
                        split_counts["train_all"] += 1
                        split_city_counts[("train_all", city)] = split_city_counts.get(("train_all", city), 0) + 1

                    # test: 2025 only.
                    if (
                        in_range(input_start, target_end, TEST_START, TEST_END)
                        and start_idx % args.test_stride == 0
                    ):
                        writers["test"].writerow(row)
                        split_counts["test"] += 1
                        split_city_counts[("test", city)] = split_city_counts.get(("test", city), 0) + 1

            print(f"  [{station_i:03d}/{len(csv_files):03d}] {station_file} done")

    for f in files.values():
        f.close()

    # Finalize stats.
    stats = {}
    for city, var_dict in stats_acc.items():
        stats[city] = {}
        for var, acc in var_dict.items():
            count = acc["count"]
            if count <= 1:
                mean = 0.0
                std = 1.0
            else:
                mean = acc["sum"] / count
                var_val = acc["sumsq"] / count - mean * mean
                std = float(np.sqrt(max(var_val, 1e-8)))
                if not np.isfinite(std) or std < 1e-6:
                    std = 1.0
            stats[city][var] = {
                "mean": float(mean),
                "std": float(std),
                "count": int(count),
            }

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "variables": VARIABLES,
                "history_length": L,
                "prediction_length": H,
                "stats_range": ["2023-01-01 00:00:00", "2024-12-31 23:00:00"],
                "city_variable_stats": stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save summary.
    for (split, city), count in sorted(split_city_counts.items()):
        summary_rows.append({"split": split, "city": city, "n_samples": count})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("\n==== Finished ====")
    print("Index paths:")
    for split, path in split_paths.items():
        print(f"  {split:10s}: {path} | n={split_counts[split]:,}")
    print("Stats:", STATS_PATH)
    print("Summary:", SUMMARY_PATH)

    print("\nSplit counts:")
    for split, count in split_counts.items():
        print(f"  {split:10s}: {count:,}")


if __name__ == "__main__":
    main()

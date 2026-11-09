#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[0]
RESULT = ROOT / "experiments/results/15_reviewer45_v2"
INDEX_ROOT = ROOT / "experiments/data_v2/indices"
OUT = ROOT / "experiments/exports/reviewer45_raw_by_station_v2"

N = 106200
H = 48
Q = 5
SEEDS = [2026, 2027, 2028, 2029, 2030]

VARIANTS = {
    "q_quantile_full": "01_quantile/q_quantile_full",
    "n2_climatology": "02_necessity/n2_climatology",
    "n3_linear_ar": "02_necessity/n3_linear_ar",
    "a0_affine_only": "03_adaptation/a0_affine_only",
    "a1_lora_r8": "03_adaptation/a1_lora_r8",
    "a2_full_finetune": "03_adaptation/a2_full_finetune",
    "r1_lagged_corr": "04_relation/r1_lagged_corr",
    "r2_mutual_information": "04_relation/r2_mutual_information",
    "r3_learned_attention": "04_relation/r3_learned_attention",
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def status_complete(run):
    p = run / "run_status.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("state") == "complete"
    except Exception:
        return False

def unique_file(run, names, expected_shape):
    for name in names:
        hits = [p for p in run.rglob(name) if p.is_file()]
        good = []
        for p in hits:
            try:
                a = np.load(p, mmap_mode="r", allow_pickle=False)
                if tuple(a.shape) == tuple(expected_shape):
                    good.append(p)
            except Exception:
                pass
        if len(good) == 1:
            return good[0]
        if len(good) > 1:
            raise RuntimeError(
                f"Ambiguous {name} under {run}: {good}"
            )

    # fallback：按 shape + 文件名语义找
    candidates = []
    for p in run.rglob("*.npy"):
        low = p.name.lower()
        if "quant" in low:
            continue
        try:
            a = np.load(p, mmap_mode="r", allow_pickle=False)
        except Exception:
            continue
        if tuple(a.shape) != tuple(expected_shape):
            continue
        candidates.append(p)

    return None

def find_point_pred(run):
    p = unique_file(
        run,
        ["pred.npy", "pred_mean.npy", "pred_q50.npy", "test_pred.npy"],
        (N, H),
    )
    if p is not None:
        return p

    hits = []
    for p in run.rglob("*.npy"):
        low = p.name.lower()
        if "pred" not in low:
            continue
        if "quant" in low or "true" in low or "target" in low:
            continue
        try:
            a = np.load(p, mmap_mode="r", allow_pickle=False)
            if a.shape == (N, H):
                hits.append(p)
        except Exception:
            pass

    if len(hits) != 1:
        raise RuntimeError(
            f"Cannot uniquely resolve point pred in {run}; candidates={hits}"
        )
    return hits[0]

def find_true(run):
    p = unique_file(
        run,
        ["true.npy", "test_true.npy", "target.npy", "y_true.npy"],
        (N, H),
    )
    if p is not None:
        return p

    hits = []
    for p in run.rglob("*.npy"):
        low = p.name.lower()
        if not ("true" in low or "target" in low):
            continue
        try:
            a = np.load(p, mmap_mode="r", allow_pickle=False)
            if a.shape == (N, H):
                hits.append(p)
        except Exception:
            pass

    if len(hits) != 1:
        raise RuntimeError(
            f"Cannot uniquely resolve true array in {run}; candidates={hits}"
        )
    return hits[0]

def find_quantile_tensor(run, kind):
    hits = []

    for p in run.rglob("*.npy"):
        low = p.name.lower()

        if "quant" not in low:
            continue

        if kind == "raw":
            if "rearr" in low or "sorted" in low or "monot" in low:
                continue
            if "raw" not in low:
                continue

        if kind == "rearranged":
            if not (
                "rearr" in low
                or "sorted" in low
                or "monot" in low
            ):
                continue

        try:
            a = np.load(p, mmap_mode="r", allow_pickle=False)
        except Exception:
            continue

        if a.shape == (N, H, Q):
            hits.append(p)

    if len(hits) == 1:
        return hits[0]

    if len(hits) == 0:
        raise RuntimeError(
            f"Missing {kind} quantile tensor in {run}"
        )

    raise RuntimeError(
        f"Ambiguous {kind} quantile tensors in {run}: {hits}"
    )

# ------------------------------------------------------------
# 1. 最终 archive audit
# ------------------------------------------------------------

archive_manifest = RESULT / "06_summary/archive_manifest.json"

if not archive_manifest.exists():
    raise RuntimeError(
        "06_summary/archive_manifest.json missing. "
        "Run Reviewer45 summarize first."
    )

audit = json.loads(archive_manifest.read_text(encoding="utf-8"))

print("===== REVIEWER45 FINAL AUDIT =====")
print(json.dumps(audit, indent=2, ensure_ascii=False))

if int(audit.get("completed_runs", -1)) != 45:
    raise RuntimeError(
        f"Reviewer45 not 45/45: {audit.get('completed_runs')}"
    )

if int(audit.get("audit_ok_runs", -1)) != 45:
    raise RuntimeError(
        f"Reviewer45 audit not 45/45: {audit.get('audit_ok_runs')}"
    )

if int(audit.get("issues", -1)) != 0:
    raise RuntimeError(
        f"Reviewer45 has audit issues: {audit.get('issues')}"
    )

# ------------------------------------------------------------
# 2. revision-v2 index
# ------------------------------------------------------------

z = np.load(INDEX_ROOT / "test.npz", allow_pickle=False)
array_id = np.asarray(z["array_id"], dtype=np.int64)

if array_id.shape != (N,):
    raise RuntimeError(f"Unexpected array_id shape: {array_id.shape}")

meta = json.loads(
    (INDEX_ROOT / "meta.json").read_text(encoding="utf-8")
)

array_paths = meta["array_paths"]

unique_aids = sorted(np.unique(array_id).tolist())

if len(unique_aids) != 295:
    raise RuntimeError(
        f"Expected 295 stations, got {len(unique_aids)}"
    )

stations = []

for aid in unique_aids:
    idx = np.flatnonzero(array_id == aid)

    if len(idx) != 360:
        raise RuntimeError(
            f"array_id={aid}: expected 360 test samples, got {len(idx)}"
        )

    p = Path(str(array_paths[aid]))
    city = p.parent.name
    station = p.stem

    if not city or not station:
        raise RuntimeError(
            f"Cannot resolve city/station from array_path={p}"
        )

    stations.append((aid, city, station, idx))

print(f"Stations = {len(stations)}")
print(f"Samples  = {sum(len(x[3]) for x in stations)}")

# ------------------------------------------------------------
# 3. 清理旧的同名导出目录
# ------------------------------------------------------------

if OUT.exists():
    print("Removing previous export:", OUT)
    shutil.rmtree(OUT)

OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 4. 逐 variant / seed 导出
# ------------------------------------------------------------

canonical_true_sha = None
canonical_true_source = None

pred_count = 0
true_count = 0
raw_q_count = 0
rearr_q_count = 0

source_manifest = []

for vi, (variant, rel) in enumerate(VARIANTS.items(), 1):

    print()
    print("=" * 80)
    print(f"[{vi}/9] {variant}")
    print("=" * 80)

    for seed in SEEDS:

        run = RESULT / rel / f"seed{seed}"

        if not status_complete(run):
            raise RuntimeError(
                f"Incomplete formal run: {run}"
            )

        pred_path = find_point_pred(run)
        true_path = find_true(run)

        pred = np.load(
            pred_path, mmap_mode="r", allow_pickle=False
        )
        true = np.load(
            true_path, mmap_mode="r", allow_pickle=False
        )

        if pred.shape != (N, H):
            raise RuntimeError(
                f"{variant} seed{seed}: bad pred shape {pred.shape}"
            )

        if true.shape != (N, H):
            raise RuntimeError(
                f"{variant} seed{seed}: bad true shape {true.shape}"
            )

        true_sha = sha256(true_path)

        if canonical_true_sha is None:
            canonical_true_sha = true_sha
            canonical_true_source = str(true_path)
            print("Canonical true:", true_path)

        elif true_sha != canonical_true_sha:
            raise RuntimeError(
                f"{variant} seed{seed}: true.npy SHA256 differs "
                f"from canonical true."
            )

        raw_q = None
        rearr_q = None
        raw_q_path = None
        rearr_q_path = None

        if variant == "q_quantile_full":
            raw_q_path = find_quantile_tensor(run, "raw")
            rearr_q_path = find_quantile_tensor(run, "rearranged")

            raw_q = np.load(
                raw_q_path, mmap_mode="r", allow_pickle=False
            )
            rearr_q = np.load(
                rearr_q_path, mmap_mode="r", allow_pickle=False
            )

            if raw_q.shape != (N, H, Q):
                raise RuntimeError(
                    f"Bad raw quantile shape: {raw_q.shape}"
                )

            if rearr_q.shape != (N, H, Q):
                raise RuntimeError(
                    f"Bad rearranged quantile shape: {rearr_q.shape}"
                )

            # raw q50 应等于正式 point prediction
            if not np.allclose(
                raw_q[..., 2],
                pred,
                rtol=0,
                atol=0,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"{variant} seed{seed}: raw q50 != pred"
                )

        print(
            f"  seed={seed} "
            f"pred={pred_path.name} "
            f"true={true_path.name}"
        )

        if raw_q_path is not None:
            print(
                f"           rawQ={raw_q_path.name} "
                f"rearrQ={rearr_q_path.name}"
            )

        for aid, city, station, idx in stations:

            dst = OUT / variant / city / station
            dst.mkdir(parents=True, exist_ok=True)

            true_dst = dst / "true.npy"

            if seed == SEEDS[0]:
                np.save(
                    true_dst,
                    np.asarray(true[idx], dtype=np.float32),
                    allow_pickle=False,
                )
                true_count += 1

            np.save(
                dst / f"pred_seed{seed}.npy",
                np.asarray(pred[idx], dtype=np.float32),
                allow_pickle=False,
            )
            pred_count += 1

            if variant == "q_quantile_full":
                np.save(
                    dst / f"raw_quantiles_seed{seed}.npy",
                    np.asarray(raw_q[idx], dtype=np.float32),
                    allow_pickle=False,
                )
                raw_q_count += 1

                np.save(
                    dst / f"rearranged_quantiles_seed{seed}.npy",
                    np.asarray(rearr_q[idx], dtype=np.float32),
                    allow_pickle=False,
                )
                rearr_q_count += 1

        source_manifest.append({
            "variant": variant,
            "seed": seed,
            "run_dir": str(run),
            "pred_source": str(pred_path),
            "true_source": str(true_path),
            "raw_quantile_source": (
                str(raw_q_path) if raw_q_path else None
            ),
            "rearranged_quantile_source": (
                str(rearr_q_path) if rearr_q_path else None
            ),
        })

# ------------------------------------------------------------
# 5. 数量检查
# ------------------------------------------------------------

expected_pred = 9 * 5 * 295
expected_true = 9 * 295
expected_q = 5 * 295

print()
print("===== EXPORT COUNTS =====")
print("pred                =", pred_count, "/", expected_pred)
print("true                =", true_count, "/", expected_true)
print("raw_quantiles       =", raw_q_count, "/", expected_q)
print("rearranged_quantiles=", rearr_q_count, "/", expected_q)

if pred_count != expected_pred:
    raise RuntimeError("Prediction count mismatch")

if true_count != expected_true:
    raise RuntimeError("True count mismatch")

if raw_q_count != expected_q:
    raise RuntimeError("Raw quantile count mismatch")

if rearr_q_count != expected_q:
    raise RuntimeError("Rearranged quantile count mismatch")

# ------------------------------------------------------------
# 6. 再抽查所有输出 shape
# ------------------------------------------------------------

bad = []

for p in OUT.rglob("*.npy"):
    a = np.load(p, mmap_mode="r", allow_pickle=False)

    if "quantiles_" in p.name:
        expected = (360, 48, 5)
    else:
        expected = (360, 48)

    if tuple(a.shape) != expected:
        bad.append((str(p), tuple(a.shape), expected))

if bad:
    raise RuntimeError(
        "Bad exported shapes:\n" +
        "\n".join(map(str, bad[:20]))
    )

manifest = {
    "state": "complete",
    "protocol": "revision-v2",
    "experiment": "Reviewer45",
    "variants": list(VARIANTS),
    "seeds": SEEDS,
    "cities": 22,
    "stations": 295,
    "samples_per_station": 360,
    "forecast_horizon": 48,
    "formal_runs": 45,
    "point_prediction_files": pred_count,
    "true_files": true_count,
    "raw_quantile_files": raw_q_count,
    "rearranged_quantile_files": rearr_q_count,
    "canonical_true_sha256": canonical_true_sha,
    "canonical_true_source": canonical_true_source,
    "quantile_levels": [0.05, 0.10, 0.50, 0.90, 0.95],
    "source_runs": source_manifest,
}

(OUT / "EXPORT_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print()
print("=" * 80)
print("REVIEWER45 RAW EXPORT COMPLETE")
print("=" * 80)
print("Output:", OUT)
print("Formal runs : 45")
print("Variants    : 9")
print("Stations    : 295")
print("Pred files  :", pred_count)
print("True files  :", true_count)
print("Raw Q files :", raw_q_count)
print("Rearr Q     :", rearr_q_count)

"""Resume frozen independent fits with a lower resource-only batch cap.

The original training files, protocol and result paths remain unchanged.
Dry runs inspect CPU-side hashes and completion only; they never query CUDA.
An actual run requires the previous training process to have stopped.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import torch

import run_robustness as runner


ROOT = runner.ROOT
OUT = runner.OUT
DEFAULT_REASON = (
    "Resource-only reduction from at most 64 to at most 32 independent fits: "
    "the Harvey fold 3 batch showed unexpected throughput collapse at about "
    "7674/8151 MiB dedicated GPU memory, 225 MHz SM clock, 100% reported "
    "utilization and about 11 W. Memory/driver pressure is suspected, not established. "
    "No forecast result or model ranking determines this scheduling change."
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_cpu(cap):
    """Validate frozen inputs without touching any CUDA API or original file."""
    path = OUT / "neural_protocol.json"
    if not path.is_file():
        raise FileNotFoundError("An existing frozen neural protocol is required")
    frozen = read_json(path)
    maximum = int(frozen["maximum_independent_batch_size"])
    if not 1 <= cap <= maximum:
        raise ValueError(f"Resource cap must be between 1 and frozen maximum {maximum}")
    if frozen["torch"] != torch.__version__ or frozen["python"] != platform.python_version():
        raise RuntimeError("Python/PyTorch differ from the frozen execution environment")
    checked = {}
    for field in ("source_sha256", "protocol_sha256"):
        for relative, expected in frozen[field].items():
            actual = file_hash(ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"Frozen hash mismatch: {relative}")
            checked[relative] = actual
    for name, expected in frozen["data_sha256"].items():
        actual = file_hash(runner.base.DATA_DIR / name)
        if actual != expected:
            raise RuntimeError(f"Frozen processed-data hash mismatch: {name}")
        checked[f"processed_data/{name}"] = actual
    return frozen, {"neural_protocol_sha256": file_hash(path),
                    "verified_hashes": checked,
                    "recovery_scheduler_sha256": file_hash(Path(__file__)),
                    "torch": torch.__version__, "python": platform.python_version()}


def group_cases(stage):
    """Reproduce the frozen runner's case order exactly."""
    stages = ("stage1", "stage2", "synthetic") if stage == "all" else (stage,)
    for current in stages:
        if current in ("stage1", "stage2"):
            for event in runner.base.EVENTS:
                for fold, frame, split, _, _ in runner.base.load_event_folds(event):
                    for budget in ((260, 1040) if current == "stage1" else (1040,)):
                        cases = []
                        for retention in ((100,) if current == "stage1" else (100, 50, 25)):
                            for noise in ((0,) if current == "stage1" else (0, 5, 10)):
                                for seed in ((11, 22, 33) if current == "stage1" else range(11, 111, 11)):
                                    cases.append(runner.empirical_case(event, fold, frame, split,
                                                 seed, budget, retention, noise))
                        yield f"{current}_{event}_fold{fold}_b{budget}", cases
        else:
            paths = sorted((OUT / "synthetic/datasets").glob("*.npz"))
            if len(paths) != 26:
                raise ValueError(f"Expected 26 frozen synthetic datasets, found {len(paths)}")
            manifest = read_json(OUT / "synthetic/dataset_manifest.json")
            expected = {item["dataset"]: item["sha256"] for item in manifest}
            if set(expected) != {path.stem for path in paths}:
                raise ValueError("Synthetic data filenames differ from frozen manifest")
            for path in paths:
                if file_hash(path) != expected[path.stem]:
                    raise RuntimeError(f"Frozen synthetic data hash mismatch: {path.name}")
            yield "synthetic", [runner.synthetic_case(path, seed)
                                for path in paths for seed in (11, 22, 33)]


def completion(case, model):
    """JSON is written last by the frozen runner and marks a completed fit."""
    stem = case.stem(model)
    path = stem.with_suffix(".json")
    if not path.exists():
        # An interrupted export may leave files without its completion marker.
        # Repeating the same frozen case/seed is permitted and is recorded.
        orphans = [str(stem.with_suffix(ext).relative_to(ROOT))
                   for ext in (".csv", ".pt", ".trace.csv", ".latent.npz")
                   if stem.with_suffix(ext).exists()]
        return False, orphans
    record = read_json(path)
    required = [".csv", ".pt", ".trace.csv"]
    if case.family == "synthetic_neural" and not model.startswith("Data-"):
        required.append(".latent.npz")
    missing = [ext for ext in required if not stem.with_suffix(ext).is_file()]
    if missing:
        raise RuntimeError(f"Completed JSON lacks artifacts {missing}: {path}")
    run = record["run"]
    expected = {"event": case.event, "fold": case.fold, "model": model,
                "seed": case.seed, "epochs_completed": case.budget,
                "parameter_count": 18619}
    for key, value in expected.items():
        if run[key] != value:
            raise RuntimeError(f"Completed record {key} mismatch: {path}")
    if record["input_sha256"] != case.input_sha256:
        raise RuntimeError(f"Completed record input hash mismatch: {path}")
    return True, []


def enumerate_plan(stage, chosen_model, cap):
    rows = []
    unique_all, unique_done, unique_orphans = set(), set(), set()
    for name, cases in group_cases(stage):
        for model in ((chosen_model,) if chosen_model else runner.MODELS):
            done = pending = 0
            for case in cases:
                key = str(case.stem(model).relative_to(ROOT))
                unique_all.add(key)
                complete, orphans = completion(case, model)
                unique_orphans.update(orphans)
                if complete:
                    done += 1
                    unique_done.add(key)
                else:
                    pending += 1
            rows.append({"group": name, "model": model, "complete": done,
                         "pending": pending, "maximum_new_batch_size": cap})
    return {"groups": rows, "unique_expected_fits": len(unique_all),
            "unique_completed_fits": len(unique_done),
            "unique_pending_fits": len(unique_all - unique_done),
            "orphan_artifacts_without_completion_json": sorted(unique_orphans),
            "counts_note": "Stage 1 and stage 2 share clean 1040-update fits; unique counts remove that overlap."}


def verify_cuda_environment(frozen):
    # This function is deliberately never called from --dry-run.
    if frozen["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frozen CUDA device is unavailable")
    actual = torch.cuda.get_device_name()
    if actual != frozen["gpu_name"]:
        raise RuntimeError(f"GPU changed: {actual}")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return torch.device("cuda")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stage1", "stage2", "synthetic", "all"), default="all")
    parser.add_argument("--model", choices=runner.MODELS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reason", default=DEFAULT_REASON)
    args = parser.parse_args()
    frozen, verification = verify_frozen_cpu(args.batch_size)
    plan = enumerate_plan(args.stage, args.model, args.batch_size)
    timestamp = datetime.now(timezone.utc)
    identifier = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    record = {"execution_id": identifier, "created_utc": timestamp.isoformat(),
        "dry_run": args.dry_run, "reason": args.reason,
        "stage": args.stage, "model_filter": args.model,
        "frozen_maximum_independent_batch_size": frozen["maximum_independent_batch_size"],
        "actual_scheduling_cap": args.batch_size, "environment": verification,
        "initial_coverage": plan, "executed_batches": [],
        "runtime_scope": "Unchanged frozen train_batch timing: synchronized batch wall time divided by actual batch size; not isolated model latency.",
        "numerical_scope": "Grouping independent fits changes resource use and can change floating-point ordering. It does not change data, initial seeds, per-fit objectives, optimizer budgets or selection rules; bitwise-identical weights are not promised.",
        "cuda_environment_verification": "not queried in dry run" if args.dry_run else "pending"}
    folder = OUT / "recovery"
    folder.mkdir(exist_ok=True)
    path = folder / f"{'dry_run' if args.dry_run else 'execution'}_{identifier}.json"
    runner.dump(path, record)
    print(json.dumps({key: plan[key] for key in ("unique_expected_fits", "unique_completed_fits", "unique_pending_fits")}), flush=True)
    print(f"Resource schedule cap={args.batch_size}; original maximum={frozen['maximum_independent_batch_size']}; record={path}", flush=True)
    if args.dry_run:
        print("Dry run passed. CUDA was not queried or initialized.", flush=True)
        return
    device = verify_cuda_environment(frozen)
    record["cuda_environment_verification"] = "passed"
    runner.dump(path, record)
    counter = 0
    for group, cases in group_cases(args.stage):
        for model in ((args.model,) if args.model else runner.MODELS):
            todo = [case for case in cases if not completion(case, model)[0]]
            for start in range(0, len(todo), args.batch_size):
                # Re-check at dispatch to protect completed fits if another
                # process accidentally wrote results after enumeration.
                batch = todo[start:start + args.batch_size]
                if any(completion(case, model)[0] for case in batch):
                    raise RuntimeError("Results changed after scheduling; stop the other writer before resuming")
                verify_frozen_cpu(args.batch_size)
                batch_id = f"recovery_{identifier}_{group}_{model}_batch{counter}"
                entry = {"batch_id": batch_id, "group": group, "model": model,
                    "actual_batch_size": len(batch), "started_utc": datetime.now(timezone.utc).isoformat(),
                    "case_stems": [str(case.stem(model).relative_to(ROOT)) for case in batch],
                    "status": "running"}
                record["executed_batches"].append(entry)
                runner.dump(path, record)
                elapsed = runner.train_batch(batch, model, device, batch_id)
                if not all(completion(case, model)[0] for case in batch):
                    raise RuntimeError("A returned training batch lacks completed artifacts")
                entry.update(status="complete", batch_runtime_seconds=elapsed,
                             completed_utc=datetime.now(timezone.utc).isoformat())
                runner.dump(path, record)
                counter += 1
    record["status"] = "complete"
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    record["final_coverage"] = enumerate_plan(args.stage, args.model, args.batch_size)
    runner.dump(path, record)
    print("Requested remaining fits complete.", flush=True)


if __name__ == "__main__":
    main()

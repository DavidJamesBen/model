"""Build the four event-level activity series used by the baseline study.

The script deliberately keeps *activity fluxes* separate from latent model
stocks.  It performs no smoothing or spline interpolation.  The resulting
CSV files contain consecutive, chronologically ordered bins and are the only
labels consumed by ``run_multievent_baselines.py``.

Data mappings
-------------
* Harvey: the author-supplied hourly table; ``newcount_of_spread`` is the
  rumor channel and ``newcount`` is the debunking channel, following the
  supplied notebooks.  These semantics still require author verification.
* Ottawa/Sydney: RumourEval 2019 Twitter stance annotations.  ``support``
  (including the source claim) maps to rumor activity and ``deny`` maps to
  debunking activity; ``query`` and ``comment`` are excluded.
* CSDC-Rumor: a heterogeneous COVID-19 rumor ecology, not one conversation.
  Rumor activity is the number of rumor-source posts plus forwarding records.
  Corrective activity is the number of dated fact-check records whose verdict
  is not ``确实如此`` (confirmed true).  This is an explicit proxy because the
  CSDC interaction files do not contain support/deny stance labels.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RUMOUREVAL = (
    ROOT
    / "data"
    / "raw"
    / "rumoureval2019"
    / "training"
    / "rumoureval-2019-training-data"
)
CSDC = ROOT / "data" / "raw" / "csdc" / "extracted" / "rumor"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_series(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame[["timestamp", "rumor_activity", "debunk_activity"]].copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"{name}: duplicated timestamps")
    if (frame[["rumor_activity", "debunk_activity"]] < 0).any().any():
        raise ValueError(f"{name}: negative activity")
    path = OUT / f"{name}.csv"
    frame.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bins": int(len(frame)),
        "start": frame["timestamp"].iloc[0].isoformat(),
        "end": frame["timestamp"].iloc[-1].isoformat(),
        "rumor_total": float(frame["rumor_activity"].sum()),
        "debunk_total": float(frame["debunk_activity"].sum()),
        "nonzero_rumor_bins": int((frame["rumor_activity"] > 0).sum()),
        "nonzero_debunk_bins": int((frame["debunk_activity"] > 0).sum()),
    }


def prepare_harvey() -> dict[str, Any]:
    source = ROOT / "Figure 9(1).xlsx"
    frame = pd.read_excel(source, sheet_name="debunk")
    required = {"All_the_time", "newcount", "newcount_of_spread"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Harvey workbook is missing {sorted(missing)}")
    out = pd.DataFrame(
        {
            # Excel serial timestamps contain sub-second floating-point drift.
            "timestamp": pd.to_datetime(frame["All_the_time"], utc=True).dt.round("h"),
            "rumor_activity": pd.to_numeric(
                frame["newcount_of_spread"], errors="raise"
            ),
            "debunk_activity": pd.to_numeric(frame["newcount"], errors="raise"),
        }
    )
    stats = write_series("harvey", out)
    stats.update(
        {
            "bin_width": "1 hour (author-supplied aggregation)",
            "source": source.name,
            "source_sha256": sha256(source),
            "mapping": (
                "newcount_of_spread -> rumor activity; newcount -> debunking "
                "activity, following the supplied notebooks"
            ),
            "caveat": (
                "Processed author-supplied table; tweet identifiers, labeling "
                "rules, and the derivation of noninteger counts are unavailable."
            ),
        }
    )
    return stats


def load_json(path: Path) -> Any:
    # The workspace name is long enough for nested RumourEval thread paths to
    # exceed the legacy Win32 MAX_PATH limit.  The extended-length prefix keeps
    # the source archive layout intact.
    resolved = str(path.resolve())
    if len(resolved) >= 248 and not resolved.startswith("\\\\?\\"):
        resolved = "\\\\?\\" + resolved
    with open(resolved, "r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_rumoureval_event(event: str, output_name: str) -> dict[str, Any]:
    key = load_json(RUMOUREVAL / "train-key.json")["subtaskaenglish"]
    event_dir = RUMOUREVAL / "twitter-english" / event
    records: list[tuple[datetime, str, str]] = []
    missing_labels = 0
    for thread in sorted(path for path in event_dir.iterdir() if path.is_dir()):
        json_paths = list((thread / "source-tweet").glob("*.json"))
        json_paths.extend(sorted((thread / "replies").glob("*.json")))
        for path in json_paths:
            tweet = load_json(path)
            tweet_id = str(tweet.get("id_str") or tweet.get("id") or path.stem)
            label = key.get(tweet_id)
            if label is None:
                missing_labels += 1
                continue
            if label not in {"support", "deny"}:
                continue
            when = parsedate_to_datetime(tweet["created_at"]).astimezone()
            # parsedate_to_datetime preserves the +0000 offset in these files.
            records.append((when, label, tweet_id))
    if not records:
        raise ValueError(f"No labeled records found for {event}")

    raw = pd.DataFrame(records, columns=["timestamp", "stance", "tweet_id"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    # Breaking-news conversations are concentrated in the first few hours.
    # Fifteen-minute bins preserve enough chronological bins for held-out
    # validation while retaining the original (unsmoothed) count observations.
    raw["time_bin"] = raw["timestamp"].dt.floor("15min")

    # Use the complete annotated event interval, from the first to the last
    # support/deny record.  No event-specific tail is added or removed.
    first_bin = raw["time_bin"].min()
    last_bin = raw["time_bin"].max()
    time_bins = pd.date_range(first_bin, last_bin, freq="15min", tz="UTC")
    in_window = raw[raw["time_bin"].isin(time_bins)]
    counts = (
        in_window.groupby(["time_bin", "stance"]).size().unstack(fill_value=0)
        .reindex(time_bins, fill_value=0)
        .rename_axis("timestamp")
        .reset_index()
    )
    counts["rumor_activity"] = counts.get("support", 0)
    counts["debunk_activity"] = counts.get("deny", 0)
    stats = write_series(output_name, counts)
    stance_counts = Counter(raw["stance"])
    stats.update(
        {
            "bin_width": "15 minutes",
            "source": "RumourEval 2019 training data (Twitter English)",
            "source_sha256": sha256(
                ROOT
                / "data"
                / "raw"
                / "rumoureval2019"
                / "rumoureval2019.tar.bz2"
            ),
            "mapping": "support -> rumor activity; deny -> debunking activity",
            "excluded_labels": "query and comment",
            "full_labeled_support": int(stance_counts["support"]),
            "full_labeled_deny": int(stance_counts["deny"]),
            "outside_observed_window_support_or_deny": int(len(raw) - len(in_window)),
            "records_without_stance_key": int(missing_labels),
            "license": "CC BY 4.0 annotations; tweet content remains subject to Twitter rights",
        }
    )
    return stats


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


DATE_RE = re.compile(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})日\s+(?P<h>\d{1,2}):(?P<min>\d{2})")


def parse_csdc_interaction_date(value: str, year: int = 2020) -> datetime | None:
    match = DATE_RE.search(value or "")
    if not match:
        return None
    parts = {key: int(number) for key, number in match.groupdict().items()}
    return datetime(year, parts["m"], parts["d"], parts["h"], parts["min"])


def prepare_csdc() -> dict[str, Any]:
    rumor_times: list[datetime] = []
    malformed_dates = 0

    for path in sorted((CSDC / "rumor_weibo").glob("*.json")):
        for item in read_json_or_jsonl(path):
            value = item.get("publishTime") or item.get("publish_time")
            when = pd.to_datetime(value, errors="coerce")
            if pd.isna(when):
                # Filename dates are preferable to silently dropping a source.
                when = pd.to_datetime(path.name[:10], errors="coerce")
            if pd.isna(when):
                malformed_dates += 1
            else:
                rumor_times.append(when.to_pydatetime())

    interaction_counts = Counter()
    for path in sorted((CSDC / "rumor_forward_comment").glob("*.json")):
        year_match = re.match(r"(\d{4})-", path.name)
        year = int(year_match.group(1)) if year_match else 2020
        for item in read_json_or_jsonl(path):
            interaction_counts[str(item.get("comment_or_forward", "unknown"))] += 1
            if item.get("comment_or_forward") != "forward":
                continue
            when = parse_csdc_interaction_date(str(item.get("date", "")), year)
            if when is None:
                malformed_dates += 1
            else:
                rumor_times.append(when)

    fact_items = read_json_or_jsonl(CSDC / "fact.json")
    corrective_times: list[datetime] = []
    confirmed_true = 0
    verdict_counts = Counter()
    for item in fact_items:
        verdict = str(item.get("explain", "unknown")).strip()
        verdict_counts[verdict] += 1
        if verdict == "确实如此":
            confirmed_true += 1
            continue
        when = pd.to_datetime(item.get("date"), errors="coerce")
        if pd.isna(when):
            malformed_dates += 1
        else:
            corrective_times.append(when.to_pydatetime())

    if not rumor_times or not corrective_times:
        raise ValueError("CSDC proxy channels could not be constructed")
    first_day = min(min(rumor_times), min(corrective_times)).date()
    last_day = max(max(rumor_times), max(corrective_times)).date()
    days = pd.date_range(first_day, last_day, freq="D", tz="UTC")
    rumor = pd.Series(0.0, index=days)
    debunk = pd.Series(0.0, index=days)
    for when in rumor_times:
        key = pd.Timestamp(when, tz="UTC").floor("D")
        if key in rumor.index:
            rumor.loc[key] += 1
    for when in corrective_times:
        key = pd.Timestamp(when, tz="UTC").floor("D")
        if key in debunk.index:
            debunk.loc[key] += 1
    out = pd.DataFrame(
        {
            "timestamp": days,
            "rumor_activity": rumor.to_numpy(),
            "debunk_activity": debunk.to_numpy(),
        }
    )
    stats = write_series("csdc", out)
    stats.update(
        {
            "bin_width": "1 day",
            "source": "THUNLP CSDC-Rumor download",
            "source_sha256": sha256(ROOT / "data" / "raw" / "csdc" / "rumor.zip"),
            "mapping": (
                "rumor source posts + forwarding records -> rumor activity; "
                "non-confirmed-true fact-check records -> corrective activity"
            ),
            "interaction_record_counts": dict(sorted(interaction_counts.items())),
            "fact_verdict_counts": dict(sorted(verdict_counts.items())),
            "confirmed_true_fact_records_excluded": int(confirmed_true),
            "malformed_or_missing_dates": int(malformed_dates),
            "caveat": (
                "Aggregated multi-rumor ecology and proxy channels; CSDC does "
                "not supply support/deny stance labels for interactions."
            ),
        }
    )
    return stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "harvey": prepare_harvey(),
        "csdc": prepare_csdc(),
        "ottawa": prepare_rumoureval_event("ottawashooting", "ottawa"),
        "sydney": prepare_rumoureval_event("sydneysiege", "sydney"),
    }
    metadata_path = OUT / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

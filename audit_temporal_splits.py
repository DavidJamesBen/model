"""Print activity support for candidate chronological evaluation windows."""

from __future__ import annotations

from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

import prepare_multievent_data as prep


def rumoureval_records(event: str) -> pd.DataFrame:
    key = prep.load_json(prep.RUMOUREVAL / "train-key.json")["subtaskaenglish"]
    records: list[tuple[pd.Timestamp, str]] = []
    event_dir = prep.RUMOUREVAL / "twitter-english" / event
    for thread in sorted(path for path in event_dir.iterdir() if path.is_dir()):
        paths = list((thread / "source-tweet").glob("*.json"))
        paths.extend(sorted((thread / "replies").glob("*.json")))
        for path in paths:
            tweet = prep.load_json(path)
            tweet_id = str(tweet.get("id_str") or tweet.get("id") or path.stem)
            label = key.get(tweet_id)
            if label in {"support", "deny"}:
                timestamp = pd.Timestamp(parsedate_to_datetime(tweet["created_at"]))
                records.append((timestamp.tz_convert("UTC"), label))
    return pd.DataFrame(records, columns=["timestamp", "stance"]).sort_values("timestamp")


def main() -> None:
    for event in ("ottawashooting", "sydneysiege"):
        raw = rumoureval_records(event)
        print(f"\n{event}: {len(raw)} labeled records")
        for frequency in ("15min", "30min", "1h"):
            start = raw["timestamp"].min().floor(frequency)
            end = raw["timestamp"].max().floor(frequency)
            index = pd.date_range(start, end, freq=frequency, tz="UTC")
            counts = (
                raw.assign(bin=raw["timestamp"].dt.floor(frequency))
                .groupby(["bin", "stance"]).size().unstack(fill_value=0)
                .reindex(index, fill_value=0)
            )
            for column in ("support", "deny"):
                if column not in counts:
                    counts[column] = 0
            cumulative = counts[["support", "deny"]].sum(axis=1).cumsum()
            prefix = int(np.searchsorted(cumulative.to_numpy(), 0.99 * cumulative.iloc[-1]) + 1)
            active = counts.iloc[:prefix]
            train_end = int(np.floor(0.60 * prefix))
            validation_end = int(np.floor(0.80 * prefix))
            validation = active.iloc[train_end:validation_end]
            test = active.iloc[validation_end:]
            print(
                frequency,
                "bins/split=", (prefix, train_end, validation_end - train_end, prefix - validation_end),
                "validation=", validation[["support", "deny"]].sum().to_dict(),
                "test=", test[["support", "deny"]].sum().to_dict(),
                "test nonzero=", (test[["support", "deny"]] > 0).sum().to_dict(),
            )


if __name__ == "__main__":
    main()

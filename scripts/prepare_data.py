"""Combine the four supplied, 14-column UCI processed tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
COLUMNS = FEATURES + ["target_raw"]
SOURCES = {
    "processed.cleveland.data": "cleveland",
    "processed.hungarian.data": "hungary",
    "processed.switzerland.data": "switzerland",
    "processed.va.data": "va_long_beach",
}


def prepare_data(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for filename, site in SOURCES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected UCI source table: {path}")
        frame = pd.read_csv(path, header=None, names=COLUMNS, na_values=["?", "-9", "-9.0"])
        for col in COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["site"] = site
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["target_raw"])
    data["target"] = (data.pop("target_raw") > 0).astype("int64")
    return data[FEATURES + ["site", "target"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("heart+disease"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/heart_disease.csv"))
    args = parser.parse_args()
    data = prepare_data(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Wrote {len(data)} rows to {args.output}; target counts: {data.target.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()

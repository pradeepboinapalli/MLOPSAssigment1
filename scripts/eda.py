"""Create reproducible EDA summary and assignment-ready plots."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_mLOps.schema import FEATURES, TARGET


def make_eda(data_path: Path, output_dir: Path) -> dict:
    data = pd.read_csv(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows": int(len(data)), "features": len(FEATURES),
        "target_counts": {str(int(k)): int(v) for k, v in data[TARGET].value_counts().sort_index().items()},
        "missing_by_feature": {k: int(v) for k, v in data[FEATURES].isna().sum().items()},
        "site_counts": {str(k): int(v) for k, v in data.site.value_counts().items()},
    }
    (output_dir / "eda_summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    sns.set_theme(style="whitegrid", palette="deep")
    data[FEATURES].hist(figsize=(13, 10), bins=20, edgecolor="white")
    plt.suptitle("UCI Heart Disease: feature distributions", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_histograms.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[FEATURES + [TARGET]].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Numeric-coded feature and target correlations")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    plt.figure(figsize=(7, 4))
    sns.countplot(data=data, x=TARGET)
    plt.title("Binary target class balance")
    plt.xlabel("Heart disease present (1)")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(make_eda(Path("data/processed/heart_disease.csv"), Path("artifacts/eda")), indent=2))

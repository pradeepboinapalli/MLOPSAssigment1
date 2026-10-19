import pandas as pd

from heart_mLOps.schema import FEATURES, TARGET


def test_cleaned_data_schema_and_target():
    data = pd.read_csv("data/processed/heart_disease.csv")
    assert data[FEATURES].shape[1] == 13
    assert set(data[TARGET].dropna().unique()) <= {0, 1}
    assert set(data.site.unique()) == {"cleveland", "hungary", "switzerland", "va_long_beach"}
    assert data[TARGET].notna().all()


def test_source_sites_and_missing_values_are_retained_for_imputation():
    data = pd.read_csv("data/processed/heart_disease.csv")
    assert len(data) >= 900
    assert data[FEATURES].isna().sum().sum() > 0

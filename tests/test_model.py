import numpy as np
import pandas as pd

from heart_mLOps.model import candidate_models
from heart_mLOps.schema import FEATURES


def test_preprocessing_and_candidates_fit_with_missing_values():
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6],
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 1, 0, 3],
        [56, 1, 2, np.nan, 236, 0, 0, 178, 0, 0.8, 1, 0, 3],
        [57, 0, 4, 120, 354, 0, 0, 163, 1, 0.6, 1, np.nan, 3],
    ]
    X = pd.DataFrame(rows, columns=FEATURES)
    y = pd.Series([1, 0, 0, 1])
    for estimator in candidate_models().values():
        estimator.fit(X, y)
        assert estimator.predict(X).shape == (4,)
        assert np.isfinite(estimator.predict_proba(X)).all()

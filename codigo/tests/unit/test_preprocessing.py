import numpy as np
import pytest

from road_accidents.config import PipelineError
from tests.fixtures.run_factory import model_frame


def test_shared_preprocessing_is_fitted_only_on_training_and_persists(tmp_path):
    from road_accidents.features.preprocessing import SharedPreprocessor

    training, _ = model_frame()
    future, _ = model_frame(2024)
    future["road_type"] = "validation-only"
    future["accidents_lag_52w"] = 1000.0
    processor = SharedPreprocessor().fit(training)
    before = processor.state_hash
    trained = processor.transform(training)
    evaluated = processor.transform(future)
    assert trained.shape == evaluated.shape
    assert np.isfinite(evaluated).all()
    assert processor.state_hash == before
    assert not any("validation-only" in name for name in processor.feature_names)
    path = tmp_path / "preprocessor.pkl"
    processor.save(path)
    loaded = SharedPreprocessor.load(path)
    np.testing.assert_allclose(loaded.transform(future), evaluated)
    assert any("accidents_lag_52w" in name for name in processor.feature_names)


def test_fitting_on_validation_or_test_is_rejected():
    from road_accidents.features.preprocessing import SharedPreprocessor

    for year in (2024, 2025, 2026):
        frame, _ = model_frame(year)
        with pytest.raises(PipelineError):
            SharedPreprocessor().fit(frame)

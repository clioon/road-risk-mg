import numpy as np
import pytest

from tests.fixtures.run_factory import model_frame


@pytest.mark.parametrize("family", ["random_forest", "xgboost", "mlp"])
def test_tabular_model_predicts_and_roundtrips_on_same_inputs(tmp_path, family):
    from road_accidents.features.preprocessing import SharedPreprocessor
    from road_accidents.models.baselines import fit_model, load_model, save_model, score_model

    training, labels = model_frame()
    validation, _ = model_frame(2024)
    processor = SharedPreprocessor().fit(training)
    parameters = (
        {"n_estimators": 4, "max_depth": 2}
        if family != "mlp"
        else {"epochs": 2, "hidden_channels": 8, "batch_size": 16}
    )
    model = fit_model(family, processor.transform(training), labels, parameters, 101)
    scores = score_model(model, processor.transform(validation))
    assert scores.shape == labels.shape
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    path = tmp_path / (family + ".model")
    save_model(model, path, family)
    reloaded = load_model(path, family)
    from road_accidents.evaluation.metrics import classification_metrics

    for year in (2025, 2026):
        test_features, test_labels = model_frame(year)
        test_scores = score_model(reloaded, processor.transform(test_features))
        assert test_scores.shape == test_labels.shape
        assert all(
            result["observation_count"] == len(test_labels)
            for result in classification_metrics(test_labels, test_scores, 0.5).values()
        )
    np.testing.assert_allclose(
        score_model(reloaded, processor.transform(validation)), scores, atol=1e-7
    )

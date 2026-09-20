from oracle_builder.training.train import model_summary_path


def test_stratified_model_summary_path_is_owned_by_dimension(tmp_path):
    config = {"classification": {"stratification": {"_active_dimension": 64}}}
    assert model_summary_path(tmp_path, config) == tmp_path / "model" / "strata" / "64" / "model_summary.txt"


def test_single_model_summary_path_is_unchanged(tmp_path):
    assert model_summary_path(tmp_path, {}) == tmp_path / "model" / "model_summary.txt"

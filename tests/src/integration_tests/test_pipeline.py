"""Integration tests for the sim-integration forecasting pipeline.#-#
#-#
These tests use the local tests/test.parquet placeholder in the repo's tests/ directory.#-#
"""#-#
#-#
from __future__ import annotations#-#
#-#
import math#-#
from pathlib import Path#-#
#-#
import numpy as np#-#
import pytest#-#
#-#
from sim_integration.dataloader import ForecastingDataset, load_parquet#-#
from sim_integration.evaluator import compute_metrics#-#
from sim_integration.methods.base import ForecastMethod#-#
from sim_integration.methods.const_vel import ConstantVelocityMethod#-#
from sim_integration.runner import Pipeline#-#
#-#
# Resolve repo root test.parquet relative to this file.#-#
_REPO_ROOT = Path(__file__).resolve().parents[3]#-#
_TEST_PARQUET = _REPO_ROOT / "tests" / "test.parquet"#-#
#-#
# Use a small sample to keep tests fast.#-#
_MAX_SAMPLES = 50#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Fixtures#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
@pytest.fixture(scope="module")#-#
def dataset() -> ForecastingDataset:#-#
    return load_parquet(_TEST_PARQUET, max_samples=_MAX_SAMPLES)#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Dataloader tests#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
class TestDataloader:#-#
    def test_load_parquet_shapes(self, dataset: ForecastingDataset):#-#
        assert dataset.obs.shape == (_MAX_SAMPLES, 20, 2)#-#
        assert dataset.gt.shape == (_MAX_SAMPLES, 30, 2)#-#
        assert len(dataset.seq_ids) == _MAX_SAMPLES#-#
        assert len(dataset.track_ids) == _MAX_SAMPLES#-#
#-#
    def test_load_parquet_dtypes(self, dataset: ForecastingDataset):#-#
        assert dataset.obs.dtype == np.float64#-#
        assert dataset.gt.dtype == np.float64#-#
#-#
    def test_load_parquet_file_not_found(self):#-#
        with pytest.raises(FileNotFoundError):#-#
            load_parquet("/nonexistent/path.parquet")#-#
#-#
    def test_load_parquet_max_samples(self):#-#
        ds5 = load_parquet(_TEST_PARQUET, max_samples=5)#-#
        assert ds5.obs.shape[0] == 5#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Method interface tests#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
class TestConstantVelocity:#-#
    def test_implements_interface(self):#-#
        m = ConstantVelocityMethod()#-#
        assert isinstance(m, ForecastMethod)#-#
        assert m.name == "const_vel"#-#
#-#
    def test_forecast_shape(self, dataset: ForecastingDataset):#-#
        m = ConstantVelocityMethod()#-#
        forecasts = m.forecast(dataset)#-#
#-#
        assert len(forecasts) == _MAX_SAMPLES#-#
        for sid in dataset.seq_ids:#-#
            preds = forecasts[sid]#-#
            assert len(preds) >= 1#-#
            assert preds[0].shape == (30, 2)#-#
#-#
    def test_forecast_is_finite(self, dataset: ForecastingDataset):#-#
        m = ConstantVelocityMethod()#-#
        forecasts = m.forecast(dataset)#-#
        for preds in forecasts.values():#-#
            for p in preds:#-#
                assert np.all(np.isfinite(p))#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Evaluator tests#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
class TestEvaluator:#-#
    def test_perfect_prediction(self):#-#
        gt = {"a": np.ones((30, 2))}#-#
        forecasts = {"a": [np.ones((30, 2))]}#-#
        m = compute_metrics(forecasts, gt, method_name="perfect")#-#
        assert m.ade == pytest.approx(0.0)#-#
        assert m.fde == pytest.approx(0.0)#-#
        assert m.miss_rate == pytest.approx(0.0)#-#
#-#
    def test_nonzero_error(self, dataset: ForecastingDataset):#-#
        m = ConstantVelocityMethod()#-#
        forecasts = m.forecast(dataset)#-#
        gt_dict = {sid: dataset.gt[i] for i, sid in enumerate(dataset.seq_ids)}#-#
        metrics = compute_metrics(forecasts, gt_dict, method_name="const_vel")#-#
        assert metrics.num_sequences == _MAX_SAMPLES#-#
        assert metrics.ade >= 0.0#-#
        assert metrics.fde >= 0.0#-#
        assert 0.0 <= metrics.miss_rate <= 1.0#-#
#-#
    def test_empty_returns_nan(self):#-#
        m = compute_metrics({}, {}, method_name="empty")#-#
        assert m.num_sequences == 0#-#
        assert math.isnan(m.ade)#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Pipeline (end-to-end) tests#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
class TestPipeline:#-#
    def test_run_const_vel(self):#-#
        pipe = Pipeline(#-#
            parquet=_TEST_PARQUET,#-#
            methods=["const_vel"],#-#
            max_samples=_MAX_SAMPLES,#-#
        )#-#
        results = pipe.run()#-#
        assert len(results) == 1#-#
        r = results[0]#-#
        assert r.method == "const_vel"#-#
        assert r.num_sequences == _MAX_SAMPLES#-#
        assert r.ade > 0#-#
        assert r.fde > 0#-#
#-#
    def test_run_custom_method(self, dataset: ForecastingDataset):#-#
        """Verify that a user-defined ForecastMethod integrates cleanly."""#-#
#-#
        class ZeroMethod(ForecastMethod):#-#
            @property#-#
            def name(self) -> str:#-#
                return "zero"#-#
#-#
            def forecast(self, ds):#-#
                return {sid: [np.zeros((ds.gt.shape[1], 2))] for sid in ds.seq_ids}#-#
#-#
        pipe = Pipeline(#-#
            parquet=_TEST_PARQUET,#-#
            methods=[ZeroMethod()],#-#
            max_samples=_MAX_SAMPLES,#-#
        )#-#
        results = pipe.run(dataset=dataset)#-#
        assert len(results) == 1#-#
        assert results[0].method == "zero"#-#
#-#
    def test_multiple_methods(self):#-#
        pipe = Pipeline(#-#
            parquet=_TEST_PARQUET,#-#
            methods=["const_vel", "const_vel"],#-#
            max_samples=10,#-#
        )#-#
        results = pipe.run()#-#
        assert len(results) == 2#-#
#-#
    def test_unknown_method_raises(self):#-#
        with pytest.raises(KeyError, match="Unknown method"):#-#
            Pipeline(parquet=_TEST_PARQUET, methods=["nonexistent"])#-#

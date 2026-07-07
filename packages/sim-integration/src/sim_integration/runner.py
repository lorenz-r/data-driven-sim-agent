"""Runner: orchestrates data loading, forecasting, and evaluation.#-#
#-#
Usage (programmatic)::#-#
#-#
    from sim_integration.runner import Pipeline#-#
    results = Pipeline(parquet="tests/test.parquet").run()#-#
#-#
Usage (CLI)::#-#
#-#
    python -m sim_integration.runner --parquet tests/test.parquet --methods const_vel#-#
"""#-#
#-#
from __future__ import annotations#-#
#-#
import argparse#-#
from pathlib import Path#-#
from typing import Dict, List, Optional, Sequence#-#
#-#
import numpy as np#-#
#-#
from sim_integration.dataloader import ForecastingDataset, load_parquet#-#
from sim_integration.evaluator import ForecastMetrics, compute_metrics#-#
from sim_integration.methods.base import ForecastMethod#-#
from sim_integration.methods.const_vel import ConstantVelocityMethod#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Method registry#-#
# ---------------------------------------------------------------------------#-#
#-#
_BUILTIN_METHODS: Dict[str, type] = {#-#
    "const_vel": ConstantVelocityMethod,#-#
}#-#
#-#
#-#
def get_method(name: str) -> ForecastMethod:#-#
    """Instantiate a built-in method by name."""#-#
    cls = _BUILTIN_METHODS.get(name)#-#
    if cls is None:#-#
        raise KeyError(f"Unknown method '{name}'. Available: {list(_BUILTIN_METHODS)}")#-#
    return cls()#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Pipeline#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
class Pipeline:#-#
    """End-to-end forecasting evaluation pipeline.#-#
#-#
    Args:#-#
        parquet: Path to the argoverse-style parquet file.#-#
        obs_len: Observed trajectory length.#-#
        pred_len: Prediction horizon.#-#
        methods: Method names (looked up in the built-in registry) **or**#-#
            pre-instantiated ``ForecastMethod`` objects.#-#
        miss_threshold: FDE threshold for miss rate.#-#
        max_samples: Cap the number of samples loaded (useful for CI).#-#
    """#-#
#-#
    def __init__(#-#
        self,#-#
        parquet: str | Path,#-#
        obs_len: int = 20,#-#
        pred_len: int = 30,#-#
        methods: Sequence[str | ForecastMethod] | None = None,#-#
        miss_threshold: float = 2.0,#-#
        max_samples: Optional[int] = None,#-#
    ) -> None:#-#
        self.parquet = Path(parquet)#-#
        self.obs_len = obs_len#-#
        self.pred_len = pred_len#-#
        self.miss_threshold = miss_threshold#-#
        self.max_samples = max_samples#-#
#-#
        if methods is None:#-#
            methods = ["const_vel"]#-#
#-#
        self._methods: List[ForecastMethod] = []#-#
        for m in methods:#-#
            if isinstance(m, str):#-#
                self._methods.append(get_method(m))#-#
            else:#-#
                self._methods.append(m)#-#
#-#
    # ------------------------------------------------------------------#-#
#-#
    def load(self) -> ForecastingDataset:#-#
        """Load the dataset."""#-#
        return load_parquet(#-#
            self.parquet,#-#
            obs_len=self.obs_len,#-#
            pred_len=self.pred_len,#-#
            max_samples=self.max_samples,#-#
        )#-#
#-#
    def run(#-#
        self, dataset: Optional[ForecastingDataset] = None#-#
    ) -> List[ForecastMetrics]:#-#
        """Run all methods and evaluate.#-#
#-#
        Args:#-#
            dataset: Pre-loaded dataset. If *None*, loaded from ``self.parquet``.#-#
#-#
        Returns:#-#
            List of ``ForecastMetrics``, one per method.#-#
        """#-#
        if dataset is None:#-#
            dataset = self.load()#-#
#-#
        # Build GT dict keyed by scenario_id#-#
        gt_dict: Dict[str, np.ndarray] = {}#-#
        for i, sid in enumerate(dataset.seq_ids):#-#
            gt_dict[sid] = dataset.gt[i]#-#
#-#
        results: List[ForecastMetrics] = []#-#
        for method in self._methods:#-#
            forecasts = method.forecast(dataset)#-#
            metrics = compute_metrics(#-#
                forecasts,#-#
                gt_dict,#-#
                method_name=method.name,#-#
                miss_threshold=self.miss_threshold,#-#
            )#-#
            results.append(metrics)#-#
#-#
        return results#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# CLI entry point#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:#-#
    p = argparse.ArgumentParser(description="sim-integration forecasting pipeline")#-#
    p.add_argument("--parquet", required=True, help="Path to argoverse parquet")#-#
    p.add_argument("--obs_len", type=int, default=20)#-#
    p.add_argument("--pred_len", type=int, default=30)#-#
    p.add_argument(#-#
        "--methods", default="const_vel", help="Comma-separated method names"#-#
    )#-#
    p.add_argument("--miss_threshold", type=float, default=2.0)#-#
    p.add_argument("--max_samples", type=int, default=None)#-#
    return p.parse_args(argv)#-#
#-#
#-#
def main(argv: Sequence[str] | None = None) -> int:#-#
    args = parse_args(argv)#-#
    method_names = [m.strip() for m in args.methods.split(",") if m.strip()]#-#
#-#
    pipeline = Pipeline(#-#
        parquet=args.parquet,#-#
        obs_len=args.obs_len,#-#
        pred_len=args.pred_len,#-#
        methods=method_names,#-#
        miss_threshold=args.miss_threshold,#-#
        max_samples=args.max_samples,#-#
    )#-#
#-#
    results = pipeline.run()#-#
    for r in results:#-#
        print(r.summary())#-#
#-#
    return 0#-#
#-#
#-#
if __name__ == "__main__":#-#
    raise SystemExit(main())#-#

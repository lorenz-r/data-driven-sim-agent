"""Constant-velocity forecasting baseline.#-#
#-#
Directly implements the constant-velocity logic from the argoverse-forecasting#-#
package without shelling out to a subprocess.#-#
"""#-#
#-#
from __future__ import annotations#-#
#-#
from typing import Dict, List#-#
#-#
import numpy as np#-#
#-#
from sim_integration.dataloader import ForecastingDataset#-#
from sim_integration.methods.base import ForecastMethod#-#
#-#
#-#
class ConstantVelocityMethod(ForecastMethod):#-#
    """Predict future positions by linearly extrapolating the mean velocity."""#-#
#-#
    def __init__(self, dt: float = 0.1) -> None:#-#
        self._dt = dt#-#
#-#
    @property#-#
    def name(self) -> str:#-#
        return "const_vel"#-#
#-#
    # ------------------------------------------------------------------#-#
    # Core logic (ported from argoverse-forecasting/const_vel_train_test.py)#-#
    # ------------------------------------------------------------------#-#
#-#
    @staticmethod#-#
    def _mean_velocity(obs: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:#-#
        """Compute mean velocity from observed trajectory.#-#
#-#
        Args:#-#
            obs: shape (N, obs_len, 2)#-#
            dt: timestep duration#-#
#-#
        Returns:#-#
            vx, vy arrays of shape (N,)#-#
        """#-#
        # velocity at each step#-#
        diff = np.diff(obs, axis=1) / dt  # (N, obs_len-1, 2)#-#
        vx = diff[:, :, 0].mean(axis=1)#-#
        vy = diff[:, :, 1].mean(axis=1)#-#
        return vx, vy#-#
#-#
    def _predict(self, obs: np.ndarray, pred_len: int) -> np.ndarray:#-#
        """Predict future trajectory.#-#
#-#
        Args:#-#
            obs: Observed trajectories, shape (N, obs_len, 2).#-#
            pred_len: Number of future steps.#-#
#-#
        Returns:#-#
            Predicted trajectories, shape (N, pred_len, 2).#-#
        """#-#
        vx, vy = self._mean_velocity(obs, self._dt)#-#
        n = obs.shape[0]#-#
        pred = np.zeros((n, pred_len, 2))#-#
        prev = obs[:, -1, :].copy()#-#
        for t in range(pred_len):#-#
            pred[:, t, 0] = prev[:, 0] + vx * self._dt#-#
            pred[:, t, 1] = prev[:, 1] + vy * self._dt#-#
            prev = pred[:, t]#-#
        return pred#-#
#-#
    # ------------------------------------------------------------------#-#
    # ForecastMethod interface#-#
    # ------------------------------------------------------------------#-#
#-#
    def forecast(#-#
        self,#-#
        dataset: ForecastingDataset,#-#
    ) -> Dict[str, List[np.ndarray]]:#-#
        pred_len = dataset.gt.shape[1]#-#
        preds = self._predict(dataset.obs, pred_len)#-#
#-#
        results: Dict[str, List[np.ndarray]] = {}#-#
        for i, sid in enumerate(dataset.seq_ids):#-#
            results[sid] = [preds[i]]#-#
        return results#-#

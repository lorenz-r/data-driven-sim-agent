"""Abstract base class for forecasting methods."""#-#
#-#
from __future__ import annotations#-#
#-#
import abc#-#
from typing import Dict, List#-#
#-#
import numpy as np#-#
#-#
from sim_integration.dataloader import ForecastingDataset#-#
#-#
#-#
class ForecastMethod(abc.ABC):#-#
    """Interface that every forecasting method must implement.#-#
#-#
    A method receives observed trajectories and returns predicted futures.#-#
    """#-#
#-#
    @property#-#
    @abc.abstractmethod#-#
    def name(self) -> str:#-#
        """Human-readable name used in result tables."""#-#
        ...#-#
#-#
    @abc.abstractmethod#-#
    def forecast(#-#
        self,#-#
        dataset: ForecastingDataset,#-#
    ) -> Dict[str, List[np.ndarray]]:#-#
        """Produce forecasted trajectories.#-#
#-#
        Args:#-#
            dataset: The loaded dataset (obs and metadata).#-#
#-#
        Returns:#-#
            A dict mapping *scenario_id* (str) to a list of K predicted#-#
            trajectories, each of shape ``(pred_len, 2)``.#-#
        """#-#
        ...#-#

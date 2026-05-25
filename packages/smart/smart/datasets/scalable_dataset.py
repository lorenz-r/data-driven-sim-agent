import os
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from torch_geometric.data import Dataset
from smart.utils.log import Logging
import numpy as np
from .preprocess import TokenProcessor


def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


class MultiDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        raw_dir: Optional[List[str]] = None,
        processed_dir: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        dim: int = 3,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        predict_unseen_agents: bool = False,
        vector_repr: bool = True,
        cluster: bool = False,
        processor=None,
        use_intention=False,
        token_size=512,
    ) -> None:
        self.logger = Logging().log(level="DEBUG")
        self.root = root
        self.well_done = [0]
        if split not in ("train", "val", "test"):
            raise ValueError(f"{split} is not a valid split")
        self.split = split
        self.training = split == "train"
        self.logger.debug("Starting loading dataset")
        self._raw_file_names = []
        self._raw_paths = []
        self._raw_file_dataset = []
        if raw_dir is not None:
            # Accept a single str/Path as well as a list.
            if isinstance(raw_dir, (str, Path)):
                raw_dir = [raw_dir]
            self._raw_dir = raw_dir
            for raw_path in self._raw_dir:
                p = Path(raw_path).expanduser().resolve()
                file_list = sorted(
                    f for f in os.listdir(p)
                    if f.endswith(".pkl")
                )
                self._raw_file_names.extend(file_list)
                self._raw_paths.extend([str(p / f) for f in file_list])
                self._raw_file_dataset.extend(["waymo"] * len(file_list))
        if self.root is not None:
            split_datainfo = os.path.join(root, "split_datainfo.pkl")
            with open(split_datainfo, "rb+") as f:
                split_datainfo = pickle.load(f)
            if split == "test":
                split = "val"
            self._processed_file_names = split_datainfo[split]
        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self._num_samples = (
            len(self._processed_file_names) - 1
            if processed_dir is not None
            else len(self._raw_file_names)
        )
        self.logger.debug(
            "The number of {} dataset is ".format(split) + str(self._num_samples)
        )
        self.token_processor = TokenProcessor(2048)
        super(MultiDataset, self).__init__(
            root=root, transform=transform, pre_transform=None, pre_filter=None
        )

    @property
    def raw_dir(self) -> List[str]:
        return self._raw_dir

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def len(self) -> int:
        return self._num_samples

    def generate_ref_token(self):
        pass

    def get(self, idx: int):
        with open(self.raw_paths[idx], "rb") as handle:
            data = pickle.load(handle)

        data = self.token_processor.preprocess(data)

        return data

    def __getitem__(self, idx):
        file_path = self.raw_paths[idx]

        try:
            data = self.get(idx)

            if data is None:
                return None

            if self.transform is not None:
                data = self.transform(data)

            return data

        except Exception as e:
            self.logger.error(
                f"[{self.split}] Failed processing sample: {file_path} | "
                f"Index: {idx} | "
                f"Error: {type(e).__name__}: {e}"
            )

            return None
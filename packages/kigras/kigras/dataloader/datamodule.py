from pathlib import Path
from typing import Literal, Optional

import pytorch_lightning as L
from torch.utils.data import DataLoader

from kigras.dataloader.dataset import ArgoDataset, collate_fn


class ArgoDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        shuffle: bool = True,
        mode: Literal["raw", "proc"] = "proc",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.persistent_workers = persistent_workers and num_workers > 0

    def _make_dataset(self, path: Optional[str]) -> ArgoDataset:
        if path is None:
            raise ValueError("Invalid Path: None")
        if self.hparams.mode == "proc":
            return ArgoDataset(processed_path=Path(path))
        return ArgoDataset(raw_path=Path(path))

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_dataset(self.hparams.train_dir)
            self.val_dataset = self._make_dataset(self.hparams.val_dir)
        if stage == "validate" or stage is None:
            self.val_dataset = self._make_dataset(self.hparams.val_dir)
        if stage in ("test", None):
            self.test_dataset = self._make_dataset(self.hparams.test_dir)
        if stage in ("predict", None):
            self.predict_dataset = self._make_dataset(self.hparams.test_dir)

    def _create_dataloader(
        self, dataset, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self):
        return self._create_dataloader(
            self.train_dataset,
            self.hparams.train_batch_size,
            shuffle=self.hparams.shuffle,
        )

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, self.hparams.val_batch_size)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, self.hparams.test_batch_size)

    def predict_dataloader(self):
        return self._create_dataloader(
            self.predict_dataset, self.hparams.test_batch_size
        )

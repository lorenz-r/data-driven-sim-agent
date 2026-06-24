from pathlib import Path
from typing import Optional

import pytorch_lightning as L
from torch.utils.data import DataLoader

from kigras.dataloader.dataset import ArgoDataset, collate_fn


class ArgoDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = "",
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        shuffle: bool = True,
        train_raw_dir: Optional[str] = None,
        val_raw_dir: Optional[str] = None,
        test_raw_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.persistent_workers = persistent_workers and num_workers > 0

        root_path = Path(root) if root else None
        self.train_raw_dir = train_raw_dir or (
            str(root_path / "train") if root_path else ""
        )
        self.val_raw_dir = val_raw_dir or (str(root_path / "val") if root_path else "")
        self.test_raw_dir = test_raw_dir or (
            str(root_path / "test") if root_path else ""
        )

    def _make_config(self, raw_dir: str) -> dict:
        if raw_dir.endswith(".npy"):
            return {"preprocess": True, "preprocess_path": raw_dir}
        return {"root": raw_dir, "preprocess": False}

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = ArgoDataset(
                self._make_config(self.train_raw_dir),
                train=True,
            )
            self.val_dataset = ArgoDataset(
                self._make_config(self.val_raw_dir),
                train=False,
            )
        elif stage == "validate":
            self.val_dataset = ArgoDataset(
                self._make_config(self.val_raw_dir),
                train=False,
            )
        if stage in ("test", None):
            self.test_dataset = ArgoDataset(
                self._make_config(self.test_raw_dir),
                train=False,
            )
        if stage in ("predict", None):
            self.predict_dataset = ArgoDataset(
                self._make_config(self.test_raw_dir),
                train=False,
            )

    def _create_dataloader(
        self, dataset, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        """Helper method to keep dataloader instantiation DRY."""
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
            batch_size=self.hparams.train_batch_size,
            shuffle=self.hparams.shuffle,
        )

    def val_dataloader(self):
        return self._create_dataloader(
            self.val_dataset, batch_size=self.hparams.val_batch_size
        )

    def test_dataloader(self):
        return self._create_dataloader(
            self.test_dataset, batch_size=self.hparams.test_batch_size
        )

    def predict_dataloader(self):
        return self._create_dataloader(
            self.predict_dataset, batch_size=self.hparams.test_batch_size
        )

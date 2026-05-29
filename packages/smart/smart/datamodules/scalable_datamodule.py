from typing import Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from smart.datasets.scalable_dataset import MultiDataset
from smart.transforms import WaymoTargetBuilder
from torch_geometric.data import Batch


class MultiDataModule(pl.LightningDataModule):
    transforms = {
        "WaymoTargetBuilder": WaymoTargetBuilder,
    }

    dataset = {
        "scalable": MultiDataset,
    }

    def __init__(
        self,
        root: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_raw_dir: Optional[str] = None,
        val_raw_dir: Optional[str] = None,
        test_raw_dir: Optional[str] = None,
        train_processed_dir: Optional[str] = None,
        val_processed_dir: Optional[str] = None,
        test_processed_dir: Optional[str] = None,
        transform: Optional[str] = None,
        dataset: Optional[str] = None,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        processor="ntp",
        use_intention=False,
        token_size=512,
        agent_token_path=None,
        map_token_path=None,
        build_agent_tokens_if_missing=False,
        agent_token_build_max_samples=1_000_000,
        agent_token_build_seed=42,
        agent_token_build_kdisks_max_attempts=1_000_000,
        token_processor_backend: str = "rust",
        **kwargs,
    ) -> None:
        super(MultiDataModule, self).__init__()
        self.root = root
        self.dataset_class = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.processor = processor
        self.use_intention = use_intention
        self.token_size = token_size
        self.agent_token_path = agent_token_path
        self.map_token_path = map_token_path
        self.build_agent_tokens_if_missing = build_agent_tokens_if_missing
        self.agent_token_build_max_samples = agent_token_build_max_samples
        self.agent_token_build_seed = agent_token_build_seed
        self.agent_token_build_kdisks_max_attempts = (
            agent_token_build_kdisks_max_attempts
        )
        self.token_processor_backend = token_processor_backend
        assert transform is not None

        train_transform = MultiDataModule.transforms[transform](
            num_historical_steps, num_future_steps, "train"
        )
        val_transform = MultiDataModule.transforms[transform](
            num_historical_steps, num_future_steps, "val"
        )
        test_transform = MultiDataModule.transforms[transform](
            num_historical_steps, num_future_steps
        )

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: Optional[str] = None) -> None:
        assert self.dataset_class is not None
        self.train_dataset = MultiDataModule.dataset[self.dataset_class](
            self.root,
            "train",
            processed_dir=self.train_processed_dir,
            raw_dir=self.train_raw_dir,
            processor=self.processor,
            transform=self.train_transform,
            token_size=self.token_size,
            agent_token_path=self.agent_token_path,
            map_token_path=self.map_token_path,
            build_agent_tokens_if_missing=self.build_agent_tokens_if_missing,
            agent_token_build_max_samples=self.agent_token_build_max_samples,
            agent_token_build_seed=self.agent_token_build_seed,
            agent_token_build_kdisks_max_attempts=(
                self.agent_token_build_kdisks_max_attempts
            ),
            token_processor_backend=self.token_processor_backend,
        )
        self.val_dataset = MultiDataModule.dataset[self.dataset_class](
            None,
            "val",
            processed_dir=self.val_processed_dir,
            raw_dir=self.val_raw_dir,
            processor=self.processor,
            transform=self.val_transform,
            token_size=self.token_size,
            agent_token_path=self.agent_token_path,
            map_token_path=self.map_token_path,
            token_processor_backend=self.token_processor_backend,
        )
        self.test_dataset = MultiDataModule.dataset[self.dataset_class](
            None,
            "test",
            processed_dir=self.test_processed_dir,
            raw_dir=self.test_raw_dir,
            processor=self.processor,
            transform=self.test_transform,
            token_size=self.token_size,
            agent_token_path=self.agent_token_path,
            map_token_path=self.map_token_path,
            token_processor_backend=self.token_processor_backend,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=safe_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=safe_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=safe_collate,
        )


def safe_collate(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    return Batch.from_data_list(batch)

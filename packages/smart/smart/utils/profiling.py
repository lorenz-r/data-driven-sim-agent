from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Optional
import pstats

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.profilers import SimpleProfiler
import torch
from torch.profiler import ProfilerActivity
import wandb


PROFILE_CHOICES = ("off", "simple", "advanced", "pytorch")


def add_profiler_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="off",
        help=(
            "Enable training profiling. 'simple' writes a compact timing log, "
            "'advanced' writes cProfile details, and 'pytorch' writes a Chrome/"
            "TensorBoard trace."
        ),
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="profiles",
        help="Directory where profiler outputs are written.",
    )
    parser.add_argument(
        "--profile_name",
        type=str,
        default=None,
        help=(
            "Profiler output base name. Defaults to smart_profile_trial-<trial_number>."
        ),
    )
    parser.add_argument(
        "--no_profile_wandb",
        action="store_true",
        help="Do not upload profiler outputs to the active wandb run.",
    )
    parser.add_argument(
        "--keep_profile_action_files",
        action="store_true",
        help=(
            "Keep per-action .prof files from the advanced profiler. By default "
            "they are merged into one SnakeViz-friendly .prof file and removed."
        ),
    )


class TrainingProfiler:
    def __init__(
        self,
        profile: str,
        profile_dir: Path,
        profile_name: str,
        upload_to_wandb: bool = True,
        keep_action_files: bool = False,
    ) -> None:
        if profile not in PROFILE_CHOICES:
            raise ValueError(f"Unknown profiler: {profile}")

        self.profile = profile
        self.profile_dir = profile_dir
        self.profile_name = self._normalize_profile_name(profile_name)
        self.upload_to_wandb = upload_to_wandb
        self.keep_action_files = keep_action_files
        self._files_before: dict[Path, tuple[int, int]] = {}

    @classmethod
    def from_args(cls, args: Namespace) -> "TrainingProfiler":
        profile_name = args.profile_name or f"smart_profile_trial-{args.trial_number}"
        return cls(
            profile=args.profile,
            profile_dir=Path(args.profile_dir),
            profile_name=profile_name,
            upload_to_wandb=not args.no_profile_wandb,
            keep_action_files=args.keep_profile_action_files,
        )

    @property
    def enabled(self) -> bool:
        return self.profile != "off"

    def build(self):
        if not self.enabled:
            return None

        self._files_before = self._snapshot_files()
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        if self.profile == "simple":
            return SimpleProfiler(
                dirpath=str(self.profile_dir),
                filename=self.profile_name,
            )
        if self.profile == "advanced":
            return AdvancedProfiler(
                dirpath=str(self.profile_dir),
                filename=self.profile_name,
                dump_stats=True,
            )
        if self.profile == "pytorch":
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            return PyTorchProfiler(
                dirpath=str(self.profile_dir),
                filename=self.profile_name,
                export_to_chrome=True,
                group_by_input_shapes=True,
                profile_memory=True,
                record_shapes=True,
                row_limit=100,
                sort_by_key="cuda_time_total",
                with_stack=True,
                activities=activities,
            )

        raise ValueError(f"Unknown profiler: {self.profile}")

    def finalize(
        self,
        wandb_logger: Optional[WandbLogger],
        is_global_zero: bool,
    ) -> list[Path]:
        if not self.enabled or not is_global_zero:
            return []

        profile_files = self._new_or_changed_files()
        if self.profile == "advanced":
            profile_files = self._merge_stats(profile_files)

        if profile_files:
            self._print_outputs(profile_files)
            if self.upload_to_wandb:
                self._log_to_wandb(wandb_logger, profile_files)
        else:
            print(f"No profiler files found in {self.profile_dir}")

        return profile_files

    def _snapshot_files(self) -> dict[Path, tuple[int, int]]:
        if not self.profile_dir.exists():
            return {}
        return {
            path: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in self.profile_dir.rglob("*")
            if path.is_file()
        }

    def _new_or_changed_files(self) -> list[Path]:
        if not self.profile_dir.exists():
            return []
        return sorted(
            path
            for path in self.profile_dir.rglob("*")
            if path.is_file()
            and path.stat().st_size > 0
            and self._files_before.get(path)
            != (path.stat().st_mtime_ns, path.stat().st_size)
        )

    @staticmethod
    def _normalize_profile_name(profile_name: str) -> str:
        while profile_name.endswith(".txt"):
            profile_name = profile_name[: -len(".txt")]
        return profile_name

    def _merge_stats(self, profile_files: list[Path]) -> list[Path]:
        merged_path = self.profile_dir / f"{self.profile_name}.merged.prof"
        prof_files = [
            path
            for path in profile_files
            if path.suffix == ".prof" and path != merged_path
        ]
        if not prof_files:
            return profile_files

        stats = pstats.Stats(str(prof_files[0]))
        for path in prof_files[1:]:
            stats.add(str(path))
        stats.dump_stats(str(merged_path))

        if not self.keep_action_files:
            for path in prof_files:
                path.unlink(missing_ok=True)

        retained_files = [
            path
            for path in profile_files
            if path.suffix != ".prof" or self.keep_action_files
        ]
        return sorted({*retained_files, merged_path})

    @staticmethod
    def _print_outputs(profile_files: list[Path]) -> None:
        print("\n-- Profiling outputs ---------------------")
        for path in profile_files:
            print(f"  {path}")
        print("------------------------------------------\n")

    @staticmethod
    def _log_to_wandb(
        wandb_logger: Optional[WandbLogger],
        profile_files: list[Path],
    ) -> None:
        if wandb_logger is None or not profile_files:
            return

        artifact = wandb.Artifact("training-profile", type="profile")
        for path in profile_files:
            artifact.add_file(str(path), name=path.name)
            wandb_logger.experiment.save(str(path), policy="now")
        wandb_logger.experiment.log_artifact(artifact)

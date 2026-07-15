"""Shared, model-agnostic helpers for rollout ``.npz`` artifacts.

SMART and KiGRAS-HA use the same structural batch layout: a top-level
manifest maps scenario ids to prefixed arrays in one ``.npz`` archive.  This
module deliberately depends only on NumPy and the standard library so command
line tools can discover and read either producer's artifacts without importing
model code.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np


_BATCH_FORMAT_PREFIXES = (
    "smart_batch_rollouts",
    "kigras_batch_rollouts",
)
_BATCH_MEMBER_KEY = re.compile(r"^scenario_\d+_.+")


class RolloutNpzError(ValueError):
    """Raised when a rollout archive does not satisfy its structural contract."""


@dataclass(frozen=True)
class ScenarioRef:
    """A reference to one scenario in a standalone or batch archive."""

    npz_path: Path
    scenario_key: str | None
    scenario_id: str
    batch_metadata: dict[str, Any] | None = None

    @property
    def source_path(self) -> Path:
        """Alias useful to callers that do not care about the container type."""

        return self.npz_path

    @property
    def input_label(self) -> str:
        """Human-readable source label including the batch member when present."""

        if self.scenario_key is None:
            return str(self.npz_path)
        return f"{self.npz_path}::{self.scenario_id}"


class ScenarioNpzView(Mapping[str, np.ndarray]):
    """Expose one batch member as an unprefixed read-only mapping.

    The underlying :class:`numpy.lib.npyio.NpzFile` must stay open for the
    lifetime of this view.  Use :func:`open_scenario` in normal call sites.
    """

    def __init__(self, data: np.lib.npyio.NpzFile, scenario_key: str):
        self._data = data
        self._prefix = f"{scenario_key}_"
        fields_key = f"{scenario_key}_fields"
        if fields_key not in data:
            raise RolloutNpzError(
                f"Batch scenario {scenario_key!r} is missing field manifest "
                f"{fields_key!r}"
            )
        fields = decode_string_array(data[fields_key], field_name=fields_key)
        _raise_for_duplicates(fields, label=f"fields in {fields_key!r}")
        missing = [field for field in fields if f"{self._prefix}{field}" not in data]
        if missing:
            raise RolloutNpzError(
                f"Batch scenario {scenario_key!r} lists missing field(s): "
                f"{', '.join(missing)}"
            )
        self._fields = tuple(fields)
        self._field_set = frozenset(fields)

    @property
    def files(self) -> list[str]:
        """Return fields in the order recorded by the batch writer."""

        return list(self._fields)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._field_set

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._field_set:
            raise KeyError(key)
        return np.asarray(self._data[f"{self._prefix}{key}"])

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)


def require_array(data: Mapping[str, Any], key: str) -> np.ndarray:
    """Return a required archive field as an array with a useful error."""

    if key not in data:
        files = getattr(data, "files", list(data))
        raise KeyError(f"Missing required key {key!r} in {files}")
    return np.asarray(data[key])


def optional_array(data: Mapping[str, Any], key: str) -> np.ndarray | None:
    """Return an optional archive field as an array."""

    if key not in data:
        return None
    return np.asarray(data[key])


def scalar_string(value: Any, field_name: str = "value") -> str:
    """Decode one NumPy scalar (including byte strings) as text."""

    array = np.asarray(value)
    if array.size != 1:
        raise RolloutNpzError(
            f"{field_name} must contain exactly one value, got shape {array.shape}"
        )
    scalar = array.reshape(()).item()
    if isinstance(scalar, bytes):
        try:
            return scalar.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RolloutNpzError(f"{field_name} is not valid UTF-8") from exc
    return str(scalar)


def decode_string_array(value: Any, field_name: str = "value") -> list[str]:
    """Decode a one-dimensional NumPy string/byte-string array."""

    array = np.asarray(value)
    if array.ndim != 1:
        raise RolloutNpzError(
            f"{field_name} must be a one-dimensional string array, "
            f"got shape {array.shape}"
        )
    decoded: list[str] = []
    for index, item in enumerate(array.tolist()):
        if isinstance(item, bytes):
            try:
                decoded.append(item.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise RolloutNpzError(
                    f"{field_name}[{index}] is not valid UTF-8"
                ) from exc
        else:
            decoded.append(str(item))
    return decoded


def load_inference_metadata(
    data: Mapping[str, Any], key: str = "inference_metadata_json"
) -> dict[str, Any] | None:
    """Decode optional JSON inference metadata from an archive mapping."""

    encoded = optional_array(data, key)
    if encoded is None:
        return None
    text = scalar_string(encoded, field_name=key)
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RolloutNpzError(f"{key} does not contain valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise RolloutNpzError(f"{key} must decode to a JSON object")
    return cast(dict[str, Any], metadata)


def ensure_rollout_positions(pred_traj: Any) -> np.ndarray:
    """Normalize positions to ``(rollouts, agents, steps, 2)``."""

    positions = np.asarray(pred_traj)
    if positions.ndim == 3:
        positions = positions[np.newaxis, ...]
    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise RolloutNpzError(
            "pred_traj must have shape (agents, steps, 2) or "
            f"(rollouts, agents, steps, 2), got {positions.shape}"
        )
    return positions


def ensure_rollout_headings(pred_head: Any | None) -> np.ndarray | None:
    """Normalize optional headings to ``(rollouts, agents, steps)``."""

    if pred_head is None:
        return None
    headings = np.asarray(pred_head)
    if headings.ndim == 2:
        headings = headings[np.newaxis, ...]
    if headings.ndim != 3:
        raise RolloutNpzError(
            "pred_head must have shape (agents, steps) or "
            f"(rollouts, agents, steps), got {headings.shape}"
        )
    return headings


def ensure_rollout_valid_mask(pred_valid_mask: Any) -> np.ndarray:
    """Normalize a validity mask to bool ``(rollouts, agents, steps)``."""

    valid_mask = np.asarray(pred_valid_mask, dtype=bool)
    if valid_mask.ndim == 2:
        valid_mask = valid_mask[np.newaxis, ...]
    if valid_mask.ndim != 3:
        raise RolloutNpzError(
            "pred_valid_mask must have shape (agents, steps) or "
            f"(rollouts, agents, steps), got {valid_mask.shape}"
        )
    return valid_mask


def _raise_for_duplicates(values: Sequence[str], *, label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise RolloutNpzError(
            f"Duplicate {label}: {', '.join(repr(value) for value in duplicates)}"
        )


def _batch_format_version(data: np.lib.npyio.NpzFile) -> str | None:
    if "format_version" not in data:
        return None
    return scalar_string(data["format_version"], field_name="format_version")


def _is_supported_batch_version(version: str) -> bool:
    return version.startswith(_BATCH_FORMAT_PREFIXES)


def _looks_like_batch(data: np.lib.npyio.NpzFile) -> tuple[bool, str | None]:
    version = _batch_format_version(data) if "format_version" in data else None
    tagged = version is not None and _is_supported_batch_version(version)
    has_batch_member = any(_BATCH_MEMBER_KEY.fullmatch(key) for key in data.files)
    # Any producer may implement the structural contract by supplying a
    # scenario_keys manifest.  Known producer tags matter only when recognizing
    # an interrupted archive whose manifest was never written. SMART and
    # KiGRAS-HA stream scenario_XXXXXX_* members before writing that final
    # manifest, so recognize those members as an interrupted batch too.
    return "scenario_keys" in data or tagged or has_batch_member, version


def _validated_batch_refs(
    data: np.lib.npyio.NpzFile, npz_path: Path, version: str | None
) -> list[ScenarioRef]:
    if "scenario_keys" not in data:
        raise RolloutNpzError(
            f"Incomplete batch archive {npz_path}: missing 'scenario_keys' manifest"
        )

    scenario_keys = decode_string_array(
        data["scenario_keys"], field_name="scenario_keys"
    )
    if not scenario_keys:
        raise RolloutNpzError(f"No scenarios found in batch archive {npz_path}")
    if any(not key for key in scenario_keys):
        raise RolloutNpzError(f"Batch archive {npz_path} contains an empty scenario key")
    _raise_for_duplicates(scenario_keys, label="scenario keys")

    if "scenario_ids" in data:
        scenario_ids = decode_string_array(
            data["scenario_ids"], field_name="scenario_ids"
        )
    elif version is None:
        # Preserve compatibility with the untagged structural batches accepted
        # by the original evaluator.
        scenario_ids = list(scenario_keys)
    else:
        raise RolloutNpzError(
            f"Incomplete batch archive {npz_path}: missing 'scenario_ids' manifest"
        )
    if len(scenario_ids) != len(scenario_keys):
        raise RolloutNpzError(
            f"Batch manifest length mismatch in {npz_path}: "
            f"{len(scenario_keys)} scenario_keys but {len(scenario_ids)} scenario_ids"
        )
    if any(not scenario_id for scenario_id in scenario_ids):
        raise RolloutNpzError(f"Batch archive {npz_path} contains an empty scenario id")
    _raise_for_duplicates(scenario_ids, label="scenario ids")

    if "num_scenarios" in data:
        count_array = np.asarray(data["num_scenarios"])
        if count_array.size != 1:
            raise RolloutNpzError(
                f"num_scenarios must be scalar in {npz_path}, "
                f"got shape {count_array.shape}"
            )
        count_value = count_array.reshape(()).item()
        if isinstance(count_value, (bool, np.bool_)) or not isinstance(
            count_value, (int, np.integer)
        ):
            raise RolloutNpzError(
                f"num_scenarios must be an integer in {npz_path}, "
                f"got {count_value!r}"
            )
        if int(count_value) != len(scenario_keys):
            raise RolloutNpzError(
                f"Batch manifest count mismatch in {npz_path}: "
                f"num_scenarios={count_value}, manifest has {len(scenario_keys)}"
            )

    batch_metadata = load_inference_metadata(data)
    for scenario_key, scenario_id in zip(scenario_keys, scenario_ids, strict=True):
        view = ScenarioNpzView(data, scenario_key)
        if "scenario_id" in view:
            stored_id = scalar_string(
                view["scenario_id"], field_name=f"{scenario_key}_scenario_id"
            )
            if stored_id != scenario_id:
                raise RolloutNpzError(
                    f"Batch scenario id mismatch for {scenario_key!r} in {npz_path}: "
                    f"manifest has {scenario_id!r}, scenario stores {stored_id!r}"
                )

    return [
        ScenarioRef(
            npz_path=npz_path,
            scenario_key=scenario_key,
            scenario_id=scenario_id,
            batch_metadata=batch_metadata,
        )
        for scenario_key, scenario_id in zip(scenario_keys, scenario_ids, strict=True)
    ]


def _standalone_ref(data: np.lib.npyio.NpzFile, npz_path: Path) -> ScenarioRef:
    scenario_id = (
        scalar_string(data["scenario_id"], field_name="scenario_id")
        if "scenario_id" in data
        else npz_path.stem
    )
    if not scenario_id:
        raise RolloutNpzError(f"Standalone archive {npz_path} has an empty scenario id")
    return ScenarioRef(
        npz_path=npz_path,
        scenario_key=None,
        scenario_id=scenario_id,
        batch_metadata=None,
    )


def _inspect_archive(npz_path: Path) -> tuple[bool, list[ScenarioRef]]:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            is_batch, version = _looks_like_batch(data)
            if is_batch:
                return True, _validated_batch_refs(data, npz_path, version)
            return False, [_standalone_ref(data, npz_path)]
    except RolloutNpzError:
        raise
    except Exception as exc:
        raise RolloutNpzError(f"Could not read rollout archive {npz_path}: {exc}") from exc


def discover_scenarios(input_path: str | Path) -> list[ScenarioRef]:
    """Discover deterministic scenario references from a file or directory.

    For a directory, complete batch archives are ordered first by relative file
    path and retain their manifest order.  Standalone scenarios already covered
    by a batch are ignored; any other duplicate scenario id is ambiguous and
    rejected.
    """

    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Rollout input does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise RolloutNpzError(f"Rollout input must be a .npz file: {path}")
        _, refs = _inspect_archive(path)
        return refs
    if not path.is_dir():
        raise RolloutNpzError(f"Rollout input is neither a file nor directory: {path}")

    npz_paths = sorted(
        (candidate for candidate in path.rglob("*.npz") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    )
    if not npz_paths:
        raise FileNotFoundError(f"No .npz rollout archives found under {path}")

    batch_refs: list[ScenarioRef] = []
    standalone_refs: list[ScenarioRef] = []
    for npz_path in npz_paths:
        is_batch, refs = _inspect_archive(npz_path)
        (batch_refs if is_batch else standalone_refs).extend(refs)

    _raise_for_duplicates(
        [ref.scenario_id for ref in batch_refs], label="batch scenario ids"
    )
    batch_ids = {ref.scenario_id for ref in batch_refs}
    uncovered_standalone = [
        ref for ref in standalone_refs if ref.scenario_id not in batch_ids
    ]
    _raise_for_duplicates(
        [ref.scenario_id for ref in uncovered_standalone],
        label="standalone scenario ids",
    )
    refs = [*batch_refs, *uncovered_standalone]
    if not refs:
        raise RolloutNpzError(f"No scenarios found under {path}")
    return refs


@contextmanager
def open_scenario(ref: ScenarioRef) -> Iterator[Mapping[str, np.ndarray]]:
    """Open a scenario reference and close its containing archive afterward."""

    with np.load(ref.npz_path, allow_pickle=False) as data:
        if ref.scenario_key is None:
            yield cast(Mapping[str, np.ndarray], data)
        else:
            yield ScenarioNpzView(data, ref.scenario_key)


def read_scenario_list(path: str | Path) -> list[str]:
    """Read ordered, unique scenario ids from a UTF-8 line-oriented file."""

    scenario_ids: list[str] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            scenario_id = line.strip()
            if not scenario_id or scenario_id.startswith("#"):
                continue
            if scenario_id not in seen:
                scenario_ids.append(scenario_id)
                seen.add(scenario_id)
    return scenario_ids


def select_scenarios(
    refs: Sequence[ScenarioRef],
    *,
    num_scenarios: int | None = None,
    scenario_ids: Sequence[str] | None = None,
) -> tuple[list[ScenarioRef], list[str]]:
    """Apply one ordered selector and return selected refs plus missing ids."""

    if num_scenarios is not None and scenario_ids is not None:
        raise RolloutNpzError(
            "num_scenarios and scenario_ids are mutually exclusive selectors"
        )
    if num_scenarios is not None:
        if (
            isinstance(num_scenarios, bool)
            or not isinstance(num_scenarios, (int, np.integer))
            or num_scenarios <= 0
        ):
            raise RolloutNpzError("num_scenarios must be a positive integer")
        return list(refs[: int(num_scenarios)]), []
    if scenario_ids is None:
        return list(refs), []

    by_id: dict[str, ScenarioRef] = {}
    for ref in refs:
        if ref.scenario_id in by_id:
            raise RolloutNpzError(
                f"Duplicate scenario id {ref.scenario_id!r} cannot be selected "
                "unambiguously"
            )
        by_id[ref.scenario_id] = ref

    selected: list[ScenarioRef] = []
    missing: list[str] = []
    seen: set[str] = set()
    for scenario_id in scenario_ids:
        if scenario_id in seen:
            continue
        seen.add(scenario_id)
        matching_ref = by_id.get(scenario_id)
        if matching_ref is None:
            missing.append(scenario_id)
        else:
            selected.append(matching_ref)
    return selected, missing


__all__ = [
    "RolloutNpzError",
    "ScenarioNpzView",
    "ScenarioRef",
    "decode_string_array",
    "discover_scenarios",
    "ensure_rollout_headings",
    "ensure_rollout_positions",
    "ensure_rollout_valid_mask",
    "load_inference_metadata",
    "open_scenario",
    "optional_array",
    "read_scenario_list",
    "require_array",
    "scalar_string",
    "select_scenarios",
]

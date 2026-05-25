from pathlib import Path
from typing import Optional


SMART_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
TOKEN_DIR = SMART_PACKAGE_ROOT / "smart" / "tokens"
DEFAULT_MAP_TOKEN_FILENAME = "map_traj_token5.pkl"


def default_agent_token_path(token_size: int) -> Path:
    return TOKEN_DIR / f"cluster_frame_5_{token_size}.pkl"


def default_map_token_path() -> Path:
    return TOKEN_DIR / DEFAULT_MAP_TOKEN_FILENAME


def resolve_token_path(path: Optional[str], default_path: Path) -> Path:
    if not path:
        return default_path

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    candidates = [
        Path.cwd() / candidate,
        SMART_PACKAGE_ROOT / candidate,
        REPO_ROOT / candidate,
    ]
    for resolved in candidates:
        if resolved.exists():
            return resolved

    if candidate.parts and candidate.parts[0] == "packages":
        return REPO_ROOT / candidate
    return Path.cwd() / candidate


def missing_token_error(path: Path, label: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{label} token file does not exist: {path}\n"
        "Set the token path in the SMART config, or build tokens explicitly with "
        "`uv run python packages/smart/scripts/build_agent_tokens.py ...`."
    )

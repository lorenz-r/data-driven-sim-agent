from __future__ import annotations

from smart.utils.token_paths import (
    default_agent_token_path,
    default_map_token_path,
    resolve_token_path,
)

from .preprocess import TokenProcessor as PythonTokenProcessor


class RustTokenProcessor:
    def __init__(
        self,
        token_size: int,
        agent_token_path=None,
        map_token_path=None,
        training: bool = False,
    ) -> None:
        import smart_preprocess

        agent_token_path = str(
            resolve_token_path(agent_token_path, default_agent_token_path(token_size))
        )
        map_token_traj_path = str(
            resolve_token_path(map_token_path, default_map_token_path())
        )
        self._processor = smart_preprocess.TokenProcessor(
            agent_token_path,
            map_token_traj_path,
            training=training,
        )

    def preprocess(self, data):
        return self._processor.preprocess(data)


def build_token_processor(
    token_size: int,
    backend: str = "rust",
    training: bool = False,
    logger=None,
    agent_token_path=None,
    map_token_path=None,
):
    normalized_backend = backend.lower()
    if normalized_backend in {"python", "legacy"}:
        return PythonTokenProcessor(
            token_size,
            agent_token_path=agent_token_path,
            map_token_path=map_token_path,
            training=training,
        )
    if normalized_backend not in {"rust", "auto"}:
        raise ValueError(
            f"{backend} is not a valid token processor backend. "
            "Expected 'rust', 'python', or 'auto'."
        )

    try:
        return RustTokenProcessor(
            token_size,
            agent_token_path=agent_token_path,
            map_token_path=map_token_path,
            training=training,
        )
    except Exception as exc:
        if normalized_backend == "auto":
            if logger is not None:
                logger.warning(
                    "Rust token processor is unavailable; falling back to Python "
                    f"backend. Error: {type(exc).__name__}: {exc}"
                )
            return PythonTokenProcessor(
                token_size,
                agent_token_path=agent_token_path,
                map_token_path=map_token_path,
                training=training,
            )
        raise

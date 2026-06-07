#!/usr/bin/env python
import argparse
from pathlib import Path

from smart.utils.agent_token_builder import build_agent_tokens
from smart.utils.token_paths import default_agent_token_path, resolve_token_path
from smart.utils.trajtok_agent_token_builder import build_trajtok_agent_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SMART agent motion tokens from SMART-format AV2 pickle data."
    )
    parser.add_argument(
        "--tokenizer",
        choices=["smart", "trajtok"],
        default="smart",
        help="Select between smart and trajtok",
    )
    parser.add_argument(
        "--raw-dir",
        action="append",
        required=True,
        help="Directory containing SMART-format .pkl files. Can be passed multiple times.",
    )
    parser.add_argument("--token-size", type=int, default=2048)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output .pkl path. Defaults to SMART's cluster_frame_5_<token-size>.pkl.",
    )
    parser.add_argument("--max-samples-per-type", type=int, default=1_000_000)
    parser.add_argument("--kdisks-max-attempts", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing token file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_token_path(args.output, default_agent_token_path(args.token_size))

    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Token file already exists: {output}\n"
            "Pass --overwrite if you intentionally want to replace it."
        )
    if args.tokenizer == "smart":
        build_agent_tokens(
            raw_dirs=[Path(raw_dir) for raw_dir in args.raw_dir],
            output_path=output,
            token_size=args.token_size,
            max_samples_per_type=args.max_samples_per_type,
            seed=args.seed,
            kdisks_max_attempts=args.kdisks_max_attempts,
        )
        print(f"Wrote SMART agent tokens to {output}")
    elif args.tokenizer == "trajtok":
        build_trajtok_agent_tokens(
            raw_dirs=[Path(raw_dir) for raw_dir in args.raw_dir],
            output_path=output,
            shift=5,
            max_file_nums=None,
            max_traj_nums=12_000_000,
            max_workers=16,
            use_traj_cache=False,
            traj_cache_path=None,
            flip_trajs=True,
        )

    else:
        raise ValueError(f"Unknown tokenizer: {args.tokenizer}")


if __name__ == "__main__":
    main()

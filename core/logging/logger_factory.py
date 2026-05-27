from dotenv import load_dotenv

from pytorch_lightning.loggers import (
    WandbLogger,
)

import os


def create_logger(config):

    # --------------------------------------------------------
    # Disabled Logging
    # --------------------------------------------------------

    if not config["logging"]["wandb_logging"]:
        return None

    # --------------------------------------------------------
    # Load .env
    # --------------------------------------------------------

    load_dotenv()

    api_key = os.getenv("WANDB_API_KEY")

    if api_key is None:
        print("WARNING: WANDB_API_KEY not found.")

        return None

    # --------------------------------------------------------
    # Optional Offline Mode
    # --------------------------------------------------------

    if config["logging"].get(
        "offline",
        False,
    ):
        os.environ["WANDB_MODE"] = "offline"

    # --------------------------------------------------------
    # Logger
    # --------------------------------------------------------

    logger = WandbLogger(
        project=config["logging"]["project_name"],
        name=config["logging"].get(
            "run_name",
            None,
        ),
        save_dir=config["output"]["output_dir"],
        log_model=config["logging"].get("log_model_artifacts", False),
    )

    # --------------------------------------------------------
    # Config Logging
    # --------------------------------------------------------

    # Configs may still be overvwritten by SMART later, so we log the config at the end of the training loop instead of here. TODO: Make sure that the top level configuration counts for the whole thing --> we want to remove potential later overwrites to keep everything central.
    # logger.experiment.config.update(
    #    flatten_config(config)
    # )

    return logger


def flatten_config(
    d,
    parent_key="",
    sep=".",
):

    items = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(
                flatten_config(
                    v,
                    new_key,
                    sep=sep,
                ).items()
            )

        else:
            items.append((new_key, v))

    return dict(items)

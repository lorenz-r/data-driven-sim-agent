def split_filename(
    split: str, size: str = "", grid_power: float = 1.0, num_bins: int | None = None
) -> str:
    """Name of a preprocessed split file, tagged with the action grid.

    The ``_b<num_bins>`` and ``_p<grid_power>`` suffixes (e.g.
    ``train_split_b127_p1.0.npy``) make the per-axis bin count and the bin warp
    explicit, so runs with a different grid never share a file and training only
    ever loads tokens produced on the same grid the model decodes with.
    """
    prefix = f"{size}_" if size else ""
    bins = f"_b{int(num_bins)}" if num_bins is not None else ""
    return f"{prefix}{split}_split{bins}_p{float(grid_power)}.npy"

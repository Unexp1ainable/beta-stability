import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import marimo as mo
    import pandas as pd

    split_root = Path("data/splits")
    split_names = ("train", "val", "test")
    dataset_names = sorted(
        path.name for path in split_root.iterdir() if path.is_dir()
    )
    return dataset_names, mo, pd, plt, split_names, split_root


@app.cell
def _(dataset_names, mo):
    dataset_selector = mo.ui.dropdown(
        options=dataset_names,
        value=dataset_names[0] if dataset_names else None,
        label="Dataset",
    )
    mo.vstack(
        [
            mo.md("# Dataset split analysis"),
            mo.md(
                "Choose a dataset to inspect the train, validation, and test "
                "splits in `data/splits`."
            ),
            dataset_selector,
        ]
    )
    return (dataset_selector,)


@app.cell
def _(dataset_selector, pd, split_names, split_root):
    if dataset_selector.value is None:
        raise ValueError(f"No datasets found in {split_root}")

    dataset_dir = split_root / dataset_selector.value
    splits = {}
    for _split_name in split_names:
        split_path = dataset_dir / f"{_split_name}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        _split = pd.read_csv(split_path)
        required_columns = {"uuid", "identity"}
        missing_columns = required_columns - set(_split.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{split_path} is missing required columns: {missing}")
        if _split["identity"].isna().any():
            raise ValueError(f"{split_path} contains missing identity values")
        splits[_split_name] = _split
    return (splits,)


@app.cell
def _(pd, splits):
    all_rows = pd.concat(splits.values(), ignore_index=True)
    stats = []
    identity_counts_by_split = {}

    for _split_name, _split in splits.items():
        _identity_counts = _split["identity"].value_counts()
        identity_counts_by_split[_split_name] = _identity_counts
        stats.append(
            {
                "Split": _split_name,
                "Entries": len(_split),
                "Unique identities": _identity_counts.size,
                "Average entries per identity": _identity_counts.mean(),
                "Identities with one entry": (_identity_counts == 1).sum(),
                "Minimum entries per identity": _identity_counts.min(),
                "Median entries per identity": _identity_counts.median(),
                "Maximum entries per identity": _identity_counts.max(),
                "Std. dev. entries per identity": _identity_counts.std(ddof=0),
            }
        )

    all_identity_counts = all_rows["identity"].value_counts()
    stats.append(
        {
            "Split": "All split rows",
            "Entries": len(all_rows),
            "Unique identities": all_identity_counts.size,
            "Average entries per identity": all_identity_counts.mean(),
            "Identities with one entry": (all_identity_counts == 1).sum(),
            "Minimum entries per identity": all_identity_counts.min(),
            "Median entries per identity": all_identity_counts.median(),
            "Maximum entries per identity": all_identity_counts.max(),
            "Std. dev. entries per identity": all_identity_counts.std(ddof=0),
        }
    )

    summary = pd.DataFrame(stats).set_index("Split")
    split_size_summary = summary[["Entries", "Unique identities"]].copy()
    split_size_summary["Entry proportion"] = (
        split_size_summary["Entries"] / len(all_rows)
    )
    split_size_summary["Identity proportion"] = (
        split_size_summary["Unique identities"] / all_identity_counts.size
    )
    return identity_counts_by_split, split_size_summary, summary


@app.cell
def _(identity_counts_by_split, plt):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for axis, (_split_name, _identity_counts) in zip(
        axes, identity_counts_by_split.items()
    ):
        max_count = int(_identity_counts.max())
        bins = range(1, max_count + 2)
        axis.hist(_identity_counts, bins=bins, align="left", rwidth=0.85)
        axis.set_title(_split_name.title())
        axis.set_xlabel("Entries per identity")
        axis.set_ylabel("Number of identities")
        axis.set_xticks(range(1, max_count + 1))

    axes[-1].set_visible(False)
    figure.suptitle("Distribution of entries per identity")
    histogram = figure
    histogram
    return


@app.cell
def _(identity_counts_by_split, mo, pd):
    overlap_rows = []
    for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
        left_ids = set(identity_counts_by_split[left_split].index)
        right_ids = set(identity_counts_by_split[right_split].index)
        overlap_rows.append(
            {
                "Split pair": f"{left_split} / {right_split}",
                "Shared identities": len(left_ids & right_ids),
                "Only in first split": len(left_ids - right_ids),
                "Only in second split": len(right_ids - left_ids),
            }
        )

    overlap_table = pd.DataFrame(overlap_rows).set_index("Split pair")
    mo.vstack(
        [
            mo.md("### Identity overlap between splits"),
            mo.ui.table(overlap_table.reset_index()),
        ]
    )
    return


@app.cell
def _(dataset_selector, mo, split_size_summary, summary):
    mo.vstack(
        [
            mo.md(f"## {dataset_selector.value}"),
            mo.md("### Split sizes"),
            mo.ui.table(
                split_size_summary
            ),
            mo.md("### Identity-count statistics"),
            mo.ui.table(
                summary
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

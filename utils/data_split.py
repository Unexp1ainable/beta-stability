# I have stolen this from somewhere
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import optimize

from utils.statss import intersect_stats
from utils.tools import apply_filters, print_div


def split_classes_objective(r0, w, class_counts, train_ratio, unseen_ratio):
    """
    Calculate the split objective for a given class distribution.

    Parameters:
    - r0: Initial ratio for selecting classes.
    - w: Weight for the partial class count.
    - class_counts: List of class counts.
    - train_ratio: Target ratio for training data.
    - unseen_ratio: Ratio of unseen data.

    Returns:
    - Split objective value.
    """
    r1 = 1 - unseen_ratio * (1 - r0)
    i0 = int(r0 * len(class_counts))
    i1 = int(r1 * len(class_counts))
    train_full = np.sum(class_counts[:i0], initial=0)
    train_part = w * np.sum(class_counts[i0:i1], initial=0)
    full = np.sum(class_counts)
    return np.abs(train_ratio * full - (train_full + train_part))


def split_df(df, train_ratio=0.7, unseen_ratio=0.5, is_val=True, stratify_col='name', print_key='name_viewpoint', verbose=False, min_num=2):
    """
    Splits a DataFrame into training, testing (and optionally validation) sets based on specified ratios and stratification column.

    Parameters:
    df (DataFrame): The pandas DataFrame to be split.
    train_ratio (float, optional): The proportion of the dataset to include in the train split (between 0 and 1). Defaults to 0.7.
    unseen_ratio (float, optional): The proportion of unique classes to be unseen in the test (and validation) sets (between 0 and 1). Defaults to 0.5.
    is_val (bool, optional): If True, split the test set further into test and validation sets. Defaults to True.
    stratify_col (str, optional): The column on which to stratify the splits. Defaults to 'name'.
    print_key (str, optional): Key used for printing statistics if verbose is True. Defaults to 'name_viewpoint'.
    verbose (bool, optional): If True, prints additional information about the splits. Defaults to False.

    Returns:
    tuple: Depending on 'is_val', returns a tuple of (train_df, test_df) or (train_df, test_df, val_df).

    """

    # Assertions to check validity of ratio inputs
    assert (train_ratio > 0 and train_ratio < 1), "train_ratio must be between 0 and 1."
    assert (unseen_ratio >= 0 and unseen_ratio <= 1), "unseen_ratio must be between 0 and 1."

    if verbose:
        print("Filtering...")
    print(f"Before filtering: {len(df)} annotations")
    # Apply filters based on the stratify column
    df = apply_filters(df, stratify_col, None, min_num)
    print(f"After filtering: {len(df)} annotations")
    
    # Get class counts and sort them
    # class_counts = df[stratify_col].value_counts().sort_values(ascending=False)
    class_counts = df[stratify_col].value_counts()
    sorted_classes = class_counts.index.tolist()

    # Optimize the split
    res = optimize.minimize(
        lambda x: split_classes_objective(x[0], x[1], np.array(class_counts), train_ratio, unseen_ratio), 
        [0.5, 0.5], bounds=scipy.optimize.Bounds(lb=0, ub=1, keep_feasible=True), tol=1e-12, method='Nelder-Mead')
    
    # Calculate indices for the splits
    r0 = res.x[0]
    r1 = 1 - unseen_ratio * (1 - r0)
    i0 = int(r0 * len(class_counts))
    i1 = int(r1 * len(class_counts))
    w = res.x[1]

    # Define the ratios for the split
    ratios = np.zeros(len(sorted_classes))
    ratios[:i0] = 1
    ratios[i0:i1] = w

    # Perform the stratified split
    dfa_train, dfa_test = stratified_split(df, sorted_classes, ratios, stratify_col)
    # print(len(dfa_train) + len(dfa_test))
    # print(len(df))
    # If validation set is not required, return train and test sets
    if not is_val:
        if verbose:
            print('Calculating stats for combined subsets')
            intersect_stats(dfa_train, dfa_test, None, key=print_key)

        return dfa_train, dfa_test

    # Split the test set further into test and validation sets
    dfa_test_val = dfa_test
    test_names, val_names = dfa_test_val[stratify_col].unique()[::2], dfa_test_val[stratify_col].unique()[1::2]
    dfa_test = dfa_test_val[dfa_test_val[stratify_col].isin(test_names)]
    dfa_val = dfa_test_val[dfa_test_val[stratify_col].isin(val_names)]

    # Print statistics if verbose
    if verbose:
        print_div()
        print('Calculating stats for combined subsets')
        intersect_stats(dfa_train, dfa_test, dfa_val, key=print_key)

    # Return the datasets
    return dfa_train, dfa_test, dfa_val



def stratified_split(df, classes, ratios, class_col, shuffle=True):
    """
    Perform a stratified split of a DataFrame into training and test sets based on specified classes and ratios.

    Parameters:
    - df: DataFrame to be split.
    - classes: List of unique classes used for stratification.
    - ratios: List of ratios for each class in the split.
    - class_col: Name of the column containing class labels.
    - shuffle: Boolean to control whether to shuffle the indices (default: True).

    Returns:
    - Two DataFrames: the training set and the test set.
    """
    train_indices = np.zeros(0, np.int64)
    for c, ratio in zip(classes, ratios):
        indices = np.array((df[df[class_col] == c]).index)
        if shuffle:
            np.random.shuffle(indices)
        n = int(len(indices) * ratio)
        train_indices = np.append(train_indices, indices[:n])
    train_df = df.loc[train_indices]
    test_df = df.drop(train_indices)
    return train_df, test_df



if __name__ == "__main__":
    import os, sys
    from pathlib import Path
    sys.path.append(Path(__file__).parents[1].as_posix())
    from data_utils import load_raw
    datasets = [
    "beluga",
    "forestelephants",
    "giraffe",
    "GZCD",
    "lion",
    "plainszebra",
    "whaleshark",
    ]

    embeddings = {}
    identities = {}
    uuids = {}
    for dataset_name in datasets:
        es, us, ids = load_raw(dataset_name)
        embeddings[dataset_name] = es
        identities[dataset_name] = ids
        uuids[dataset_name] = us

    for dataset_name in datasets:
        df = pd.DataFrame({
            "uuid": list(uuids[dataset_name]),
            "identity": list(identities[dataset_name]),
        })

        print(f"\n\nSplitting dataset: {dataset_name}")
        train_df, test_df, val_df = split_df(df, stratify_col="identity", min_num=2)
        print("\nSplit summary")
        print(f"  Annotations: total={len(df):,}, train={len(train_df):,}, "
            f"test={len(test_df):,}, validation={len(val_df):,}")
        print(f"  Annotation ratios: train={len(train_df) / len(df):.2%}, "
            f"test={len(test_df) / len(df):.2%}, validation={len(val_df) / len(df):.2%}")
        train_ids = set(train_df.identity)
        test_ids = set(test_df.identity)
        val_ids = set(val_df.identity)
        print(f"  Identities: train={len(train_ids):,}, test={len(test_ids):,}, "
            f"validation={len(val_ids):,}")
        print("  Identity overlap (expected to be zero for unseen splits):")
        print(f"    train ∩ test:       {len(train_ids & test_ids):,}")
        print(f"    train ∩ validation: {len(train_ids & val_ids):,}")
        print(f"    test ∩ validation:  {len(test_ids & val_ids):,}")

        out_path = Path(f"data/splits/{dataset_name}")
        out_path.mkdir(exist_ok=True, parents=True)
        train_df.to_csv(out_path / "train.csv", index=False)
        test_df.to_csv(out_path / "test.csv", index=False)
        val_df.to_csv(out_path / "val.csv", index=False)



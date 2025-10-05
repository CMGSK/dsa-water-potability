import os
import pandas as pd
import numpy as np
from collections import Counter

from repr import benchmark
from util import *
from water import WaterSource


def knn_predict(x_train, y_train, x_test, k=5, dist_approach=None, dap=None, bias_threshold=None):
    """
    K Nearest Neighbours. Calculate distance from test point to all training points, find K nearest neighbours,
    most common label among K neighbours wins.

    Parameters
    ----------
    x_train : DataFrame
    y_train : DataFrame
    y_test : DataFrame
    k : int
        number of neighbours referenced
    dist_approach : str (`euc`, `min`, `man`, `che`, `cos`)
        define the distance calculation approach
    dap : float
        define the p parameter for minkowski approach
    bias_threshold : double (range between 1 and 0)
        define the percentage of bias applied to the voting process by diverting
        the vote to the negative side if that certain ammount of neighbours are 
        negative, even with possitive majority.

    Returns
    -------
    `0` | `1`
        (based on water potability)
    """
    # First step. Calculate all distances
    print(f"'{dist_approach}' distance calculations approach")
    distances = []
    for ix, tr in enumerate(x_train):
        dist = calculate_distances(x_test, tr, dist_approach, dap)
        distances.append((dist, y_train[ix]))  # Stores a tuple (distance, label)

    # Second step. Sort by distance, take K closest
    distances.sort(key=lambda x: x[0])  # Sorts by the first element in the tuple (dist)
    kn = distances[:k]

    labels = [label for _, label in kn]  # Extract labels from k nearest disregarding distance
    if bias_threshold:
        # Third step. Biased votation
        ctr = Counter(labels)
        non_potable = ctr.get(0, 0)  # Get the non-potable results
        if (non_potable / k) >= bias_threshold:
            return 0
        else:
            return ctr.most_common(1)[0][0]
    else:
        # Third step. Vote
        return Counter(labels).most_common(1)[0][0]  # Extract most frequent label


def naive_approach(dist_approach=None, dap=None, **kwargs):
    """
    The naive approach is defined by calling to this function without any argument. Additional arguments can be passed to
    tune up the algorithm in several ways.

    Parameters
    ----------
    dist_approach : str (`euc`, `min`, `man`, `che`, `cos`)
        define the distance calculation approach
    dap : float
        define the p parameter for minkowski approach
    only_load_model : bool
        skip the benchmarking, load and retrieve the data types to use for a prediction
    imputation : bool
        define if we perform a value imputation or a drop for the NaN values
    scaling : str (`rob`, `ubs`, `std`)
        define the scaling method for our parameters
    """
    # Load model
    ds = pd.read_csv('../water_potability.csv')

    if 'imputation' in kwargs:
        if kwargs['imputation']:
            ds = missing_val_imputation_by_mean(ds)
        else: 
            ds = ds.dropna()
    else: 
        ds = ds.dropna()

    # Separate features and targets (into numpy arr)
    x = ds.drop('Potability', axis=1).values  # Drops Potability Column
    y = ds['Potability'].values

    # Define a random seed
    np.random.seed(27)

    # Randomly split for testing data
    idx = np.random.permutation(len(x))
    split_idx = int(0.9 * len(x))
    train, test = idx[:split_idx], idx[split_idx:]

    # X stands for features, Y stands for targets
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]

    # Parameter normalization
    if 'scaling' in kwargs:
        x_train_sc, x_test_sc = implement_scaling_method(x_train, x_test, approach=kwargs['scaling'])
    else:
        # KNN measures distance, features with larger scales can dominate over others
        mean = x_train.mean(axis=0)  # Mean of each feature col
        std = x_train.std(axis=0)  # Std deviation of each col

        x_train_sc = (x_train - mean) / std
        x_test_sc = (x_test - mean) / std

    if (only_load_model := kwargs['only_load_model']) and not only_load_model:
        k_benchmark(x_train_sc, y_train, x_test_sc, y_test, dist_approach, dap)

    return (x_train, x_train_sc, x_test, x_test_sc, y_train, y_test)


def main():
    x_train, x_train_sc, x_test, x_test_sc, y_train, y_test = naive_approach(dist_approach='min', dap=1.5, imputation=True, scaling='uvs', only_load_model=True)
    water = WaterSource(4.668101687405915,193.68173547507868,47580.99160333534,7.166638935482532,359.94857436696,526.4241709223593,13.894418518194527,66.68769478539706,4.4358209095098).mount()
    print(f'Water source potability: {bool(knn_predict(x_train, y_train, water, 33, 'cos'))}')


if __name__ == "__main__":
    # Eliminated from the dataset for out-of-code testing
    # 4.668101687405915,193.68173547507868,47580.99160333534,7.166638935482532,359.94857436696,526.4241709223593,13.894418518194527,66.68769478539706,4.4358209095098
    # Should return: 1
    main()



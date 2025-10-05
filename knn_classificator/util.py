from collections import Counter
import pandas as pd
import numpy as np

from main import knn_predict
from repr import benchmark

def missing_val_imputation_by_mean(data: pd.DataFrame):
    """
    This is NOT the correct approach to imputation, since reduces natural variability and ignores relationships.
    To address this, imputation method should be another regression model such as KNN or MICE.

    For training purposes, let's just keep it simple and add some noise to the data so we also learn about
    how inputing noise manually.
    """
    data_copy = data.copy()  # We do not want to modify the original since python passes values by ref

    for column in data_copy.columns:
        if data_copy[column].isnull().any():
            mean = data_copy[column].mean()  # Get the mean
            stdd = data_copy[column].std()  # Get the std deviation

            # Find all NaN in the current column
            nan_idx = data_copy[column][data_copy[column].isnull()].index

            # Normal distribution, scaled by standar deviation, and multiplied by a custom noise factor
            # In some cases you might want to clamp the value to a min/max range with `.clip()`
            noise = np.random.normal(loc=0, scale=(stdd*0.05), size=len(nan_idx))
            data_copy.loc[nan_idx, column] = mean + noise

    return data_copy
    

def calculate_distances(a, b, approach=None, p=2):
    match approach:
        case 'minkowski' | 'min':
            return np.sum(np.abs(a-b)**p)**(1/p)
            
        case 'manhattan' | 'man':
            return np.sum(np.abs(a-b))

        case 'chebyshev' | 'che':
            return np.max(np.abs(a-b))
            
        case 'consine' | 'cos':
            dot = np.dot(a, b)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 1.0
            return 1 - (dot / (na * nb))

        case 'euclidean' | 'euc' | _ :
            return np.sqrt(np.sum((a - b) ** 2))


def implement_scaling_method(x_train, x_test, approach=None):
    match approach:
        case 'robust' | 'rob':
            print("Robust scaling method")
            iqr = np.percentile(x_train, 75, axis=0) - np.percentile(x_train, 25, axis=0)
            median = np.median(x_train, axis=0)
            x_train_sc, x_test_sc = (((x_train - median) / iqr), (x_test - median) / iqr)

            return (x_train_sc, x_test_sc)

        case 'unit_vector' | 'uvs':
            print("L2 unit vector scaling method")
            l2n = np.linalg.norm(x_train, axis=1, keepdims=True)
            l2n[l2n == 0] = 1  # Prevent division by zero errors
            x_train_sc = x_train - l2n
            l2n = np.linalg.norm(x_test, axis=1, keepdims=True)
            l2n[l2n == 0] = 1
            x_test_sc = x_test - l2n

            return (x_train_sc, x_test_sc)

        case 'standarization' | 'std' | _ :
            print("Standarization scaling method")
            mean = x_train.mean(axis=0)  
            std = x_train.std(axis=0)  
            x_train_sc = (x_train - mean) / std
            x_test_sc = (x_test - mean) / std

            return (x_train_sc, x_test_sc)

def knn_predict_cost_sensitive_bias(x_train, y_train, x_test, k=5, dist_approach=None, dap=None, threshold=0.4):
    """
    K Nearest Neighbours. Calculate distance from test point to all training points, find K nearest neighbours,
    most common class among K neighbours wins where K is a 5 fallback, user-inputed number of neighbours.
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

    # Third step. Count occurrences.
    labels = [label for _, label in kn]  # Extract labels from k nearest disregarding distance
    count = Counter(labels)
    non_potable = count.get(0, 0)

    # Fourth step. Biased votation.
    # If non-potable count exceeds a threshold, divert the vote.
    if (non_potable / k) > threshold:
        return 0
    else:
        return count.most_common(1)[0][0]


    return vote


def k_benchmark(x_train_sc, y_train, x_test_sc, y_test, dist_approach=None, dap=None):
    acc = []
    # From 5 up to 10% dataset with 4 unit jumps to avoid ties in votation
    k_values = [n for n in range(5, x_train_sc*0.1, 4)]
    for k in k_values:
        predictions = [knn_predict(x_train_sc, y_train, single_x_val, k, dist_approach, dap) for single_x_val in x_test_sc]
        accuracy = np.mean(predictions == y_test)  # Correct predictions
        acc.append(accuracy)
        print(f'k={k}: Accuracy = {accuracy:.7f}')
    benchmark(k_values, acc)
    argmax = np.argmax(acc)
    print(f'Top accuracy: {acc[argmax]:.3f} with K={k_values[argmax]}')
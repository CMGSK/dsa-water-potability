import pandas as pd
import numpy as np
from repr import benchmark
from collections import Counter

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



# Calculate straight line distance between two points in an n-dimentional space by sqrt(sum((a - b)^2))
def calculate_distances(a, b, approach=None, p=2):
    match approach:
        case 'euclidean' | 'euc':
            return np.sqrt(np.sum((a - b) ** 2))

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

        case _:  # Fallback to euclidean
            return np.sqrt(np.sum((a - b) ** 2))


def implement_scaling_method(x_train, x_test, approach=None):
    def robust_scaler_fit(x_train):
        median = np.median(x_train, axis=0)
        q1 = np.percentile(x_train, 25, axis=0)
        q3 = np.percentile(x_train, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1e-7  # Prevent division by zero errors
        return median, iqr

    match approach:
        case 'robust' | 'rob':
            median, iqr = robust_scaler_fit(x_train)
            x_train_sc, x_test_sc = (((x_train - median) / iqr), (x_test - median) / iqr)
            return (x_train_sc, x_test_sc)

        case 'unit_vector' | 'uvs':
            l2n = np.linalg.norm(x_train, axis=1, keepdims=True)
            l2n[l2n == 0] = 1  # Prevent division by zero errors
            x_train_sc = x_train - l2n
            l2n = np.linalg.norm(x_test, axis=1, keepdims=True)
            l2n[l2n == 0] = 1
            x_test_sc = x_test - l2n
            return (x_train_sc, x_test_sc)

        case 'standarization' | 'std' | _ :
            mean = x_train.mean(axis=0)  
            std = x_train.std(axis=0)  
            x_train_sc = (x_train - mean) / std
            x_test_sc = (x_test - mean) / std

            return (x_train_sc, x_test_sc)



def knn_predict(x_train, y_train, x_test, k=5, dist_approach=None, dap=None):
    """
    K Nearest Neighbours. Calculate distance from test point to all training points, find K nearest neighbours,
    most common class among K neighbours wins where K is a 5 fallback, user-inputed number of neighbours.
    """
    # First step. Calculate all distances
    distances = []
    for ix, tr in enumerate(x_train):
        dist = calculate_distances(x_test, tr, dist_approach, dap)
        distances.append((dist, y_train[ix]))  # Stores a tuple (distance, label)

    # Second step. Sort by distance, take K closest
    distances.sort(key=lambda x: x[0])  # Sorts by the first element in the tuple (dist)
    kn = distances[:k]

    # Third step. Vote.
    labels = [label for _, label in kn]  # Extract labels from k nearest disregarding distance
    vote = Counter(labels).most_common(1)[0][0]  # Extract most frequent label

    return vote

def k_benchmark(x_train_sc, y_train, x_test_sc, y_test, dist_approach=None, dap=None):
    acc = []
    k_values = [n for n in range(3, 100, 3)]
    for k in k_values:
        predictions = [knn_predict(x_train_sc, y_train, single_x_val, k, dist_approach, dap) for single_x_val in x_test_sc]
        accuracy = np.mean(predictions == y_test)  # Correct predictions
        acc.append(accuracy)
        print(f'k={k}: Accuracy = {accuracy:.7f}')
    benchmark(k_values, acc)
    argmax = np.argmax(acc)
    print(f'Top accuracy: {acc[argmax]:.3f} with K={k_values[argmax]}')

def naive_approach(dist_approach=None, dap=None, **kwargs):
    """
    The naive approach is defined by calling to this function without any argument. Additional arguments can be passed to
    tune up the algorithm in several ways.

    Parameters
    ----------
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

class WaterSource:
    ph: float
    hardness: float
    solids: float
    chloramines: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float

    def __init__(self, ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
        self.ph = ph
        self.hardness = hardness
        self.solids =  solids
        self.chloramines = chloramines
        self.sulfate = sulfate
        self.conductivity = conductivity
        self.organic_carbon = organic_carbon
        self.trihalomethanes = trihalomethanes
        self.turbidity = turbidity

    
    def mount(self):
        data = {
            'ph': self.ph,
            'Hardness': self.hardness,
            'Solids': self.solids,
            'Chloramines': self.chloramines,
            'Sulfate': self.sulfate,
            'Conductivity': self.conductivity,
            'Organic_carbon': self.organic_carbon,
            'Trihalomethanes': self.trihalomethanes,
            'Turbidity': self.turbidity
        }

        return pd.Series(data).values.reshape(1, -1)  # Single shape 1 row N col




def main():
    x_train, x_train_sc, x_test, x_test_sc, y_train, y_test = naive_approach(dist_approach='min', dap=1.5, imputation=True, scaling='uvs', only_load_model=True)
    water = WaterSource(4.668101687405915,193.68173547507868,47580.99160333534,7.166638935482532,359.94857436696,526.4241709223593,13.894418518194527,66.68769478539706,4.4358209095098).mount()
    print(f'Water source potability: {bool(knn_predict(x_train, y_train, water, 33, 'cos'))}')


    # x_train, x_train_sc, x_test, x_test_sc, y_train, y_test = naive_approach()
    # k_benchmark(x_train_sc, y_train, x_test_sc, y_test)
    # knn_predict()
    # ds = pd.read_csv('../water_potability.csv')
    # ds = missing_val_imputation_by_mean(ds)


if __name__ == "__main__":
    # Eliminated from the dataset for out-of-code testing
    # 4.668101687405915,193.68173547507868,47580.99160333534,7.166638935482532,359.94857436696,526.4241709223593,13.894418518194527,66.68769478539706,4.4358209095098
    # Should return: 1
    main()



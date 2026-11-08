import numpy as np

def test_train_split(X, y, test_size=0.3, seed=None):
    X = np.array(X)
    y = np.array(y)

    # Set the random seed for reproducibility
    np.random.seed(seed)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_count = int(len(X) * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

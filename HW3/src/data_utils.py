import numpy as np
from sklearn.model_selection import train_test_split

def _to_np(x, dtype=None):
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr

def load_federated(train_path, test_path):
    train_arr = np.load(train_path, allow_pickle=True)
    test_arr  = np.load(test_path,  allow_pickle=True)
    return train_arr, test_arr

def client_train_val_split(client_dict, val_size=0.2, seed=42):
    # Coerce lists -> ndarray, normalize to [0,1]
    X = _to_np(client_dict['images'], dtype=np.float32)
    if X.size and X.max() > 1.0:
        X = X / 255.0
    y = _to_np(client_dict['labels']).reshape(-1)

    # Stratified split can fail if a client has too-few samples per class
    try:
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=val_size, random_state=seed, stratify=y
        )
    except ValueError:
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=val_size, random_state=seed, stratify=None
        )
    return (Xtr, ytr), (Xval, yval)

def flatten_images(X):
    X = _to_np(X, dtype=np.float32)
    if X.ndim == 3:           # [N, 28, 28]
        return X.reshape((X.shape[0], -1))
    if X.ndim == 2:           # [N, 784] already flat
        return X
    if X.ndim == 1:           # [784] -> [1, 784]
        return X.reshape((1, -1))
    raise ValueError(f"Unexpected image shape {X.shape}")

def load_test_global(test_arr):
    d = test_arr[0]
    X = _to_np(d['images'], dtype=np.float32)
    if X.size and X.max() > 1.0:
        X = X / 255.0
    X = flatten_images(X)
    y = _to_np(d['labels']).reshape(-1)
    return X, y

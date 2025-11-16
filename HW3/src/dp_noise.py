# dp_noise.py
import numpy as np

def add_laplace_noise(X, b: float):
    """
    Add element-wise Laplace(0, b) noise to images.

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, 28, 28) or (n_samples, 784).
        Values are expected in [0, 1].
    b : float
        Laplace scale parameter. If b <= 0, X is returned unchanged.

    Returns
    -------
    np.ndarray
        Noisy images, clipped back to [0, 1].
    """
    if b <= 0:
        return X

    eps = np.random.laplace(loc=0.0, scale=b, size=X.shape)
    X_noisy = X + eps
    # Keep images in valid range
    X_noisy = np.clip(X_noisy, 0.0, 1.0)
    return X_noisy
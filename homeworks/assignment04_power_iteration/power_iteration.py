import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after num_steps steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    b_k = np.random.rand(data.shape[1])
    
    for _ in range(num_steps):
        b_k1 = np.dot(data, b_k)  # matrix-vector multiplication
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

    eigenvector = b_k
    eigenvalue = np.dot(b_k.T, np.dot(data, b_k))

    return float(eigenvalue), eigenvector
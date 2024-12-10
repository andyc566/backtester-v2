import numpy as np

# a bit different here... a weighted r squared... 

def r2_score(y_true, y_pred, sample_weight):
    """Custom R2 metric."""
    numerator = np.sum(sample_weight * (y_true - y_pred) ** 2)
    denominator = np.sum(sample_weight * y_true**2)
    return 1 - numerator / (denominator + 1e-38)

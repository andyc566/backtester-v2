import pandas as pd
import numpy as np


def predict(models, X_test):
    """Predict using an ensemble of models."""
    predictions = np.mean([model.predict(X_test) for model in models], axis=0)
    return predictions

from sklearn.datasets import load_wine
import pandas as pd

def load_data(return_X_y=False):
    data = load_wine()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    if return_X_y:
        return X, y

    df = X.copy()
    df["target"] = y
    return df
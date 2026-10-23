"""
diamond_features.py
-------------------
Shared feature definitions for the Diamond Price Prediction System.

This file is imported by BOTH the Jupyter notebook (training) and the Streamlit
app (prediction), so the raw inputs and engineered features are guaranteed to be
computed in exactly the same way. The saved model file (.joblib) refers to the
function `add_features` in this module, so this file must stay next to app.py.
"""
import numpy as np
import pandas as pd

# The 9 raw inputs a user provides (this is the exact input schema of the model)
NUMERIC_RAW = ["carat", "depth", "table", "x", "y", "z"]
CATEGORICAL = ["cut", "color", "clarity"]
RAW_FEATURES = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]

# Ordered grades (worst -> best). Used for dropdown order and one-hot categories.
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]            # D = best (colourless)
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]  # IF = best

# Features computed from the raw inputs
ENGINEERED = ["volume", "log_carat", "log_volume", "xy_ratio"]
NUMERIC_MODEL = NUMERIC_RAW + ENGINEERED


def add_features(X):
    """Take a DataFrame with the 9 raw inputs and add the engineered features.

    volume     = x * y * z                 (approximate size of the stone, mm^3)
    log_carat  = ln(carat)                 (price grows roughly exponentially with weight)
    log_volume = ln(volume)
    xy_ratio   = x / y                     (length-to-width ratio, a shape indicator)
    """
    X = pd.DataFrame(X).copy()
    missing = [c for c in RAW_FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    for col in ["carat", "x", "y", "z"]:
        if (X[col] <= 0).any():
            raise ValueError(f"'{col}' must be strictly positive.")
    X["volume"] = X["x"] * X["y"] * X["z"]
    X["log_carat"] = np.log(X["carat"])
    X["log_volume"] = np.log(X["volume"])
    X["xy_ratio"] = X["x"] / X["y"]
    return X[CATEGORICAL + NUMERIC_MODEL]

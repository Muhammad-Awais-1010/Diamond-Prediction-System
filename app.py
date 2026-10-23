"""
Diamond Price Prediction System - Streamlit web app
---------------------------------------------------
Loads the trained scikit-learn pipeline (diamond_price_model.joblib) and predicts the
price (USD) of a diamond from its quality grades and measurements.

Files needed next to this script:
    diamond_price_model.joblib   the trained pipeline
    model_info.json              training ranges / metrics written by the notebook
    diamond_features.py          shared feature-engineering code used during training
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from diamond_features import RAW_FEATURES  # also makes the saved pipeline importable (add_features)

APP_DIR = Path(__file__).resolve().parent
MODEL_FILE = APP_DIR / "diamond_price_model.joblib"
INFO_FILE = APP_DIR / "model_info.json"


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_info():
    return json.loads(INFO_FILE.read_text())


def validate_inputs(v, info):
    """Return (errors, warnings) for one diamond. Errors block the prediction."""
    errors, warnings = [], []
    for col in ["carat", "x", "y", "z", "depth", "table"]:
        lo, hi = info["ranges"][col]["min"], info["ranges"][col]["max"]
        if v[col] is None or not np.isfinite(v[col]) or v[col] <= 0:
            errors.append(f"'{col}' must be a positive number.")
        elif not (lo <= v[col] <= hi):
            errors.append(f"'{col}' = {v[col]} is outside the range the model was trained on ({lo:g} to {hi:g}).")
    for col, allowed in info["categories"].items():
        if v[col] not in allowed:
            errors.append(f"'{col}' must be one of: {', '.join(allowed)}.")
    if errors:
        return errors, warnings

    # Weight vs size must be physically consistent (same rule used to clean the training data)
    dev = np.log(v["carat"] / (v["x"] * v["y"] * v["z"])) - info["consistency"]["median_ln_carat_per_volume"]
    if abs(dev) > info["consistency"]["threshold"]:
        warnings.append("The carat weight does not match the length x width x depth you entered (a typical round diamond of "
                        f"these dimensions weighs about {np.exp(info['consistency']['median_ln_carat_per_volume']) * v['x'] * v['y'] * v['z']:.2f} ct). "
                        "Please double-check the numbers; the estimate may be unreliable.")
    if not (0.8 <= v["x"] / v["y"] <= 1.25):
        warnings.append("Length and width differ a lot. The model was trained almost entirely on round-shaped diamonds, so the estimate may be unreliable.")
    implied_depth = 200 * v["z"] / (v["x"] + v["y"])
    if abs(implied_depth - v["depth"]) > 2:
        warnings.append(f"Depth % ({v['depth']:.1f}) differs from 2z/(x+y) = {implied_depth:.1f}. Check your measurements.")
    for col in ["carat", "depth", "table"]:
        r = info["ranges"][col]
        if not (r["p01"] <= v[col] <= r["p99"]):
            warnings.append(f"'{col}' is unusual for the training data (typical range {r['p01']:g} to {r['p99']:g}); the estimate is less reliable.")
    return errors, warnings


def predict_price(model, v):
    row = pd.DataFrame([{c: v[c] for c in RAW_FEATURES}])[RAW_FEATURES]
    return float(model.predict(row)[0])


# ------------------------------------------------------------------ page
st.set_page_config(page_title="Diamond Price Prediction System", page_icon="💎", layout="centered")

st.title("💎 Diamond Price Prediction System")
st.write("Estimate the price of a diamond using its physical measurements and quality characteristics "
         "with a trained machine learning model.")

if not MODEL_FILE.exists() or not INFO_FILE.exists():
    st.error("Model files not found. Make sure 'diamond_price_model.joblib' and 'model_info.json' are in the same folder as app.py.")
    st.stop()

model, info = load_model(), load_info()
R = info["ranges"]
cats = info["categories"]

with st.sidebar:
    st.header("About this model")
    st.write(f"**Algorithm:** {info['best_model']}")
    st.write(f"**Trained on:** {info['n_training_rows']:,} diamonds")
    st.write(f"**Currency:** US dollars (USD)")
    st.write(f"**10-fold cross-validation error:** typical absolute error of about "
             f"**${info['cv_mean_mae']:,.0f}** (MAE), RMSE ${info['cv_mean_rmse']:,.0f}, R² {info['cv_mean_r2']:.3f}.")
    st.caption("These figures describe average accuracy on historical data. Individual estimates can be off by more, "
               "especially for rare, very large or unusually proportioned stones.")

st.subheader("Diamond details")
left, right = st.columns(2)

with left:
    st.markdown("**Quality grades**")
    cut = st.selectbox("Cut", cats["cut"], index=cats["cut"].index("Ideal"),
                       help="Cut quality, from Fair (lowest) to Ideal (highest).")
    color = st.selectbox("Color", cats["color"], index=cats["color"].index("G"),
                         help="Colour grade from J (most tinted) to D (colourless, best).")
    clarity = st.selectbox("Clarity", cats["clarity"], index=cats["clarity"].index("VS2"),
                           help="Clarity grade from I1 (most inclusions) to IF (internally flawless).")
    carat = st.number_input("Carat (weight)", min_value=float(R["carat"]["min"]), max_value=float(R["carat"]["max"]),
                            value=1.20, step=0.01, format="%.2f", help="Weight of the diamond in carats.")

with right:
    st.markdown("**Proportions and dimensions**")
    depth = st.number_input("Depth (%)", min_value=float(R["depth"]["min"]), max_value=float(R["depth"]["max"]),
                            value=61.5, step=0.1, format="%.1f", help="Total depth percentage = 2 x depth / (length + width) x 100.")
    table = st.number_input("Table (%)", min_value=float(R["table"]["min"]), max_value=float(R["table"]["max"]),
                            value=57.0, step=0.1, format="%.1f", help="Width of the top flat facet relative to the widest point.")
    x = st.number_input("Length (mm)", min_value=float(R["x"]["min"]), max_value=float(R["x"]["max"]),
                        value=6.80, step=0.01, format="%.2f")
    y = st.number_input("Width (mm)", min_value=float(R["y"]["min"]), max_value=float(R["y"]["max"]),
                        value=6.70, step=0.01, format="%.2f")
    z = st.number_input("Depth (mm)", min_value=float(R["z"]["min"]), max_value=float(R["z"]["max"]),
                        value=4.10, step=0.01, format="%.2f")

st.caption("The values above are just starting values for the form, not a real stone. Change them to describe your diamond.")

if st.button("Predict Diamond Price", type="primary", use_container_width=True):
    values = {"carat": carat, "cut": cut, "color": color, "clarity": clarity,
              "depth": depth, "table": table, "x": x, "y": y, "z": z}
    errors, warnings = validate_inputs(values, info)
    if errors:
        for e in errors:
            st.error(e)
    else:
        for w in warnings:
            st.warning(w)
        try:
            with st.spinner("Estimating price..."):
                price = predict_price(model, values)
            if not np.isfinite(price) or price <= 0 or price > 2 * info["price_range"]["max"]:
                st.error("The model returned an unreliable value for these inputs. Please check them and try again.")
            else:
                st.success(f"### Estimated Diamond Price: ${price:,.0f} USD")
                st.caption(f"Model: {info['best_model']}. In 10-fold cross-validation its typical absolute error was about "
                           f"${info['cv_mean_mae']:,.0f}.")
        except Exception as exc:  # never show a raw traceback to the user
            st.error(f"Could not compute a prediction: {exc}")

st.divider()
st.caption("**Disclaimer:** this is an *estimate* produced by a machine learning model trained on a historical dataset of diamond prices "
           "(US dollars). It is not an appraisal or a guaranteed market price. Actual retail or resale prices depend on market conditions, "
           "certification, seller, and other factors the model does not see.")

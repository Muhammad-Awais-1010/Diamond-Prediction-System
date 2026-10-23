# Diamond Price Prediction System (Streamlit app)

Estimate the price (in US dollars) of a diamond from its quality grades and measurements, using a machine-learning model trained on a historical diamond dataset.

## Files in this folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit web app (the "main file" to deploy) |
| `diamond_price_model.joblib` | The trained scikit-learn pipeline (feature engineering + pre-processing + model) |
| `model_info.json` | Training ranges, categories and cross-validation metrics used by the app (so the app never needs the training data) |
| `diamond_features.py` | Feature-engineering code shared with the training notebook. **Required**: the saved model refers to it |
| `requirements.txt` | Pinned library versions (the versions the model was trained with) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload all files in this folder to the root of a public GitHub repository and deploy `app.py` on Streamlit Community Cloud (Python 3.12). See `DEPLOYMENT_GUIDE.md` in the main project for step-by-step instructions.

## Important

* The prediction is an **estimate** based on historical data, not an appraisal or a market quote.
* The model was trained almost entirely on round brilliant diamonds of up to about 5 carats; inputs far outside that are rejected or flagged.
* `scikit-learn` must be the pinned version in `requirements.txt`, otherwise the saved model may not load.

# Customer-Segmentation-wth-Streamlit-Using-Machine-Learning-K-Means-Clustering

This repository contains a customer segmentation analysis and a Streamlit app to predict customer cluster membership.

Files
- [Cust_seg_Model.ipynb](Cust_seg_Model.ipynb) — Jupyter notebook that loads the dataset, preprocesses features, fits a scaler and KMeans model, runs PCA and visualizations. Key notebook symbols: [`Cust_seg_Model.kmeans`](Cust_seg_Model.ipynb) and [`Cust_seg_Model.scaler`](Cust_seg_Model.ipynb).
- [segmentation.py](segmentation.py) — Streamlit app that loads saved model artifacts and exposes an input form. Key symbols: [`segmentation.input_data`](segmentation.py) and [`segmentation.kmeans`](segmentation.py).
- [customer_segmentation.csv](customer_segmentation.csv) — primary dataset used for analysis and model training.
- [german_credit_data.csv](german_credit_data.csv) — additional dataset included in the workspace.
- Saved artifacts produced by the notebook (after training): `kmeans_model.pkl` and `scaler.pkl` (ensure these files exist in the repo root before running the app).

Quick start

1. Install requirements:
```Python
pip install -r requirements.txt
```
Recommended packages: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib

2. Train and save model (via notebook):
- Open and run [Cust_seg_Model.ipynb](Cust_seg_Model.ipynb). The notebook:
  - reads [customer_segmentation.csv](customer_segmentation.csv)
  - constructs features stored in the `features` list
  - fits a `StandardScaler` (`Cust_seg_Model.scaler`) and a KMeans model (`Cust_seg_Model.kmeans`)
  - saves artifacts with joblib (expected names: `kmeans_model.pkl` and `scaler.pkl`)

3. Run the Streamlit app:
```python
streamlit run segmentation.py
```
The app loads `Kmeans_model.pkl` and `scaler.pkl` (see [segmentation.py](segmentation.py)) and exposes an input form for:
- Age
- Income
- Total Spending
- Number of Web Purchases
- Number of Store Purchases
- Number of Web Visits Per Month
- Recency (days since last purchase)

The app builds [`segmentation.input_data`](segmentation.py), scales it with the loaded scaler, and predicts the cluster with the loaded KMeans (`segmentation.kmeans`).

Usage example
- Open the Streamlit URL shown in the terminal after running the app.
- Fill the form values and click "Predict Segment".
- The app returns: Predicted Segment: Cluster X

Notes & troubleshooting
- File name consistency: the notebook saves `kmeans_model.pkl` (lowercase) while `segmentation.py` attempts to load `Kmeans_model.pkl` (capital K). Ensure the filenames match exactly.
- Confirm `scaler.pkl` and the KMeans pickle are present in the repo root before starting the Streamlit app.
- If you re-train the models in [Cust_seg_Model.ipynb](Cust_seg_Model.ipynb), re-run the save cell to overwrite the pickle files.
- The notebook exposes `scaler.feature_names_in_` after loading the scaler for debugging; this can help verify feature order at inference time.

References (workspace links)
- Notebook: [Cust_seg_Model.ipynb](Cust_seg_Model.ipynb)
- Streamlit app: [segmentation.py](segmentation.py)
- Dataset: [customer_segmentation.csv](customer_segmentation.csv)
- Additional dataset: [german_credit_data.csv](german_credit_data.csv)
- Notebook symbols: [`Cust_seg_Model.kmeans`](Cust_seg_Model.ipynb), [`Cust_seg_Model.scaler`](Cust_seg_Model.ipynb)
- App symbols: [`segmentation.input_data`](segmentation.py), [`segmentation.kmeans`](segmentation.py)

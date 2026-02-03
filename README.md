# Smart Bundle Recommendation (Telecom)

This project builds a clustering-based recommendation model for telecom bundles using the Kaggle **Churn in Telecoms** dataset. It includes:

- Data cleaning + feature engineering notebook
- K-Means clustering and bundle profiling
- Streamlit web UI for recommendations

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download the dataset from Kaggle and place the CSV at:

```
data/telecom_churn.csv
```

Dataset link (Kaggle):
```
https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset
```

3. Run the notebook to train and save the model:

```
jupyter notebook notebooks/01_bundle_recommender.ipynb
```

4. Run the Streamlit app:

```
streamlit run app/app.py
```

## Notes
- The dataset does not include real mobile data usage. We create a **data proxy** feature from evening and night usage. This is documented as a limitation.
- The model uses **unsupervised clustering (K-Means)** to group similar users and map clusters to bundle tiers.

## Project Structure

See `PROJECT_STRUCTURE.md`.

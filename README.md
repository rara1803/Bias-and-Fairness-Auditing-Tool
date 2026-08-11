# Ethical and Bias Aware AI Auditing Tool

An interactive Streamlit application that helps both technical and non-technical users detect discriminative practices, hidden biases, and fairness violations in machine learning models — without requiring deep technical expertise.

## Overview

Most auditing tools are either too technical for non-expert users or only cover narrow slices of fairness. This tool integrates three modules into a single guided workflow:

- **Data Analysis Module** – loads and previews datasets, with optional metadata mapping for readable column names and value labels.
- **Fairness Evaluation Module** – calculates Demographic Parity and Equal Opportunity across user-selected sensitive attributes (e.g. gender, race, age).
- **Bias Reporting Module** – combines statistical fairness checks, SHAP global feature importance, LIME local explanations, and counterfactual analysis into a plain-language risk assessment with actionable recommendations.

The result is a dashboard that surfaces both *whether* a model is biased and *why*, so users can make informed decisions before deployment.

## Features

- Upload datasets in CSV, Excel, or JSON format
- Optional metadata file for human-readable column names and category mappings
- Select sensitive attributes and a target/label column through a simple UI
- Upload a pre-trained scikit-learn pipeline (`.pkl` / `.joblib`)
- **Fairness metrics**: Demographic Parity and Equal Opportunity, with automatic flagging of disparities over 10%
- **SHAP summary plot**: global feature importance, with an explicit check for whether sensitive attributes are top drivers of predictions
- **LIME explanation**: local, per-record interpretability
- **Counterfactual analysis**: the smallest numeric change needed to flip a prediction
- **Automated risk assessment**: a final High Risk / Low Risk verdict with recommendations

## Tech Stack

- [Streamlit](https://streamlit.io/) – web interface
- [scikit-learn](https://scikit-learn.org/) – model pipeline handling
- [SHAP](https://shap.readthedocs.io/) – global explainability
- [LIME](https://github.com/marcotcr/lime) – local explainability
- pandas / numpy – data handling

## Project Structure

```
.
├── lib/
│   ├── app.py              # Streamlit UI and workflow
│   ├── audit_model.py      # Fairness metrics, SHAP, LIME, counterfactual logic
│   ├── utils.py             # Data loading and preprocessing helpers
│   └── train_model.py       # Script to train a sample model pipeline
├── sample data/              # Example dataset(s) for trying out the tool
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>/lib
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn shap lime matplotlib joblib openpyxl
```

### 3. (Optional) Train a sample model
```bash
python train_model.py --data_path "../sample data/your_dataset.csv" --target_col "your_target_column"
```

### 4. Run the app
```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`) in your browser.

## Usage

1. Upload a dataset (CSV, Excel, or JSON).
2. Optionally upload a metadata file to rename columns or map coded values to labels.
3. Select one or more sensitive attributes and the target/label column.
4. Upload a trained model file (`.pkl` or `.joblib`).
5. Click **Run Fairness Audit** to generate fairness metrics, SHAP and LIME explanations, a counterfactual example, and a final risk assessment.

## Limitations & Future Work

- Currently supports binary classification pipelines.
- Fairness metrics are limited to Demographic Parity and Equal Opportunity; additional metrics (e.g. Equalized Odds, Predictive Parity) could be added.
- SHAP's KernelExplainer fallback for non-tree models can be slow on larger datasets.

## License

This project is open source and available for educational and research use.

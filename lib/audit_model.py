import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def load_resources(model_path, data_path, target_col):
    """Loads model and data, returns prepared X, y, and pipeline"""
    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    return pipe, X, y


def calculate_fairness_metrics(pipe, X, y, sensitive_cols):
    """Calculates Demographic Parity and Equal Opportunity"""
    y_pred = pipe.predict(X)
    metrics_data = []

    for s_col in sensitive_cols:
        if s_col not in X.columns:
            continue

        groups = X[s_col].unique()
        for g in groups:
            mask = (X[s_col] == g)
            if mask.sum() == 0:
                continue

            group_pred = y_pred[mask]
            group_y = y[mask]

            # 1. Demographic Parity (Approval Rate)
            pos_rate = np.mean(group_pred == 1)

            # 2. Equal Opportunity (True Positive Rate)
            if np.sum(group_y == 1) > 0:
                tpr = np.sum((group_pred == 1) & (group_y == 1)) / np.sum(group_y == 1)
            else:
                tpr = 0.0

            metrics_data.append({
                "Attribute": s_col,
                "Group": str(g),
                "Count": int(mask.sum()),
                "Demographic Parity (Approval Rate)": pos_rate,
                "Equal Opportunity (True Positive Rate)": tpr
            })

    return pd.DataFrame(metrics_data)


def generate_shap_plot(pipe, X):
    """Generates SHAP summary plot"""
    clf = pipe.named_steps.get("clf", pipe)
    preproc = pipe.named_steps.get("preproc", None)

    # Transform data for SHAP
    X_trans = preproc.transform(X) if preproc else X.values

    # Use small sample for speed
    X_sample = X_trans[:50]

    # Select Explainer
    try:
        if "RandomForest" in str(type(clf)) or "XGB" in str(type(clf)):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            background = shap.kmeans(X_trans, 10)
            explainer = shap.KernelExplainer(clf.predict, background)
            shap_values = explainer.shap_values(X_sample)

        feat_names = get_feature_names(preproc, X)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_sample, feature_names=feat_names, show=False, plot_type="bar")
        plt.tight_layout()

        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(feat_names, vals)), columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        top_3 = feature_importance.head(3)['col_name'].tolist()

        return fig, top_3
    except Exception as e:
        print(f"SHAP Error: {e}")
        return plt.figure(), ["Error calculating features"]


def generate_lime_html(pipe, X, instance_index=0):
    """Generates LIME explanation handling categorical strings safely"""
    try:
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_indices = [X.columns.get_loc(c) for c in cat_cols]

        X_encoded = X.copy()
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            X_encoded[c] = le.fit_transform(X_encoded[c].astype(str))
            encoders[c] = le

        data_array = X_encoded.values

        def predict_fn(x_numpy):
            df_temp = pd.DataFrame(x_numpy, columns=X.columns)

            for c in cat_cols:
                df_temp[c] = df_temp[c].round().astype(int)
                valid_max = len(encoders[c].classes_) - 1
                df_temp[c] = df_temp[c].clip(0, valid_max)
                df_temp[c] = encoders[c].inverse_transform(df_temp[c])

            return pipe.predict_proba(df_temp)

        # 4. Run LIME
        explainer = LimeTabularExplainer(
            training_data=data_array,
            feature_names=X.columns.tolist(),
            categorical_features=cat_indices,
            class_names=["Rejected", "Approved"],
            mode='classification'
        )

        exp = explainer.explain_instance(
            data_row=data_array[instance_index],
            predict_fn=predict_fn,
            num_features=5
        )
        return exp.as_html()

    except Exception as e:
        return f"<div>Error generating LIME: {str(e)}</div>"


def generate_counterfactual(pipe, X, instance_index=0):
    """
    Finds the smallest numeric change needed to flip the prediction.
    """
    try:
        target_row = X.iloc[instance_index].copy()

        input_df = pd.DataFrame([target_row], columns=X.columns)
        original_pred = pipe.predict(input_df)[0]

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

        for col in numeric_cols:
            step = (X[col].max() - X[col].min()) / 10.0
            if step == 0: step = 1

            for i in range(1, 11):
                temp_row = target_row.copy()
                new_val = temp_row[col] + (step * i)
                temp_row[col] = new_val

                test_df = pd.DataFrame([temp_row], columns=X.columns)
                new_pred = pipe.predict(test_df)[0]

                if new_pred != original_pred:
                    change_direction = "increase"
                    return f"Counterfactual found: If you {change_direction} '{col}' to {new_val:.2f}, the prediction flips from {original_pred} to {new_pred}."

            for i in range(1, 11):
                temp_row = target_row.copy()
                new_val = temp_row[col] - (step * i)
                temp_row[col] = new_val

                test_df = pd.DataFrame([temp_row], columns=X.columns)
                new_pred = pipe.predict(test_df)[0]

                if new_pred != original_pred:
                    change_direction = "decrease"
                    return f"Counterfactual found: If you {change_direction} '{col}' to {new_val:.2f}, the model changes its prediction from '{original_pred}' to '{new_pred}'."

        return "No simple numeric change was able to flip this prediction."

    except Exception as e:
        return f"Could not generate counterfactual: {e}"


def get_feature_names(preproc, X):
    try:
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        num_cols = X.select_dtypes(exclude=['object', 'category']).columns
        new_cats = preproc.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
        return list(num_cols) + list(new_cats)
    except:
        return X.columns.tolist()
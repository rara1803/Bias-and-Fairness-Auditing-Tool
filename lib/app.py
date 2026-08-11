import streamlit as st
import pandas as pd
import json
import os
import streamlit.components.v1 as components
from audit_model import load_resources, calculate_fairness_metrics, generate_shap_plot, generate_lime_html, \
    generate_counterfactual

st.set_page_config(page_title="Ethical and Bias Awareness Tool", layout="wide")

def load_any_file(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file)
    elif ext == ".xlsx":
        return pd.read_excel(file)
    elif ext == ".json":
        return pd.read_json(file)
    else:
        return pd.read_csv(file, header=None, sep=r"\s+")


def load_metadata_file(file):
    if file.name.endswith(".json"):
        return json.load(file)
    elif file.name.endswith(".csv"):
        df_meta = pd.read_csv(file)
        return {"columns": df_meta.to_dict(orient="records")}
    return None


def apply_metadata(df, metadata):
    df_new = df.copy()
    if metadata is None or "columns" not in metadata: return df_new
    for col_meta in metadata["columns"]:
        idx = col_meta.get("index")
        name = col_meta.get("name")
        mapping = col_meta.get("mapping", None)
        if idx is not None and 0 <= idx < len(df_new.columns):
            if name: df_new.columns.values[idx] = str(name)
            if mapping: df_new.iloc[:, idx] = df_new.iloc[:, idx].map(mapping).fillna(df_new.iloc[:, idx])
    return df_new


# UI
st.title("Ethical and Bias Awareness Tool")
st.markdown("A guided tool to help you detect bias in machine learning models.")

st.header("1) Upload dataset")
dataset_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json", "data"])

if dataset_file is not None:
    df = load_any_file(dataset_file)
    st.success(f"Dataset loaded with shape {df.shape}")

    st.header("2) Upload optional metadata")
    st.markdown("Metadata allows readable column names and value mappings. Optional.")
    metadata_file = st.file_uploader("Upload metadata file", type=["json", "csv"])
    metadata = load_metadata_file(metadata_file) if metadata_file else None
    df_mapped = apply_metadata(df, metadata)

    st.header("3) Dataset preview")
    st.dataframe(df_mapped.head(10))

    st.header("4) Select columns for analysis")

    with st.expander("Definitions", expanded=True):
        st.markdown("""
        **Sensitive Attributes:** Features that might contain unfairness (e.g., Gender, Race, Age).
        **Label Column:** The outcome the model predicts (e.g., Approved, Rejected).
        """)

    col_names = df_mapped.columns.tolist()
    c1, c2 = st.columns(2)
    sensitive_cols = c1.multiselect("Select sensitive attribute(s)", col_names)
    label_col = c2.selectbox("Select label column (Target)", col_names)

    st.header("5) Upload Model")
    model_file = st.file_uploader("Upload Model File (.pkl or .joblib)", type=["pkl", "joblib"])

    st.header("6) Run Audit")

    if st.button("Run Fairness Audit"):
        if model_file is None:
            st.error("Please upload a model file before running the audit.")
        else:
            os.makedirs("temp", exist_ok=True)
            data_path = "temp/data.csv"
            model_path = "temp/model.joblib"
            df_mapped.to_csv(data_path, index=False)
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())

            st.divider()
            st.header("Audit Results")

            with st.spinner("Analyzing model behavior..."):
                try:
                    pipe, X, y = load_resources(model_path, data_path, label_col)

                    bias_issues = []

                    st.subheader("A. Fairness Metrics")
                    st.write("Evaluating Demographic Parity and Equal Opportunity.")

                    if sensitive_cols:
                        metrics_df = calculate_fairness_metrics(pipe, X, y, sensitive_cols)

                        display_df = metrics_df.copy()
                        display_df['Demographic Parity (Approval Rate)'] = display_df[
                            'Demographic Parity (Approval Rate)'].apply(lambda x: f"{x:.1%}")
                        display_df['Equal Opportunity (True Positive Rate)'] = display_df[
                            'Equal Opportunity (True Positive Rate)'].apply(lambda x: f"{x:.1%}")

                        st.table(display_df)

                        st.info("Interpretation:")
                        for s in sensitive_cols:
                            sub = metrics_df[metrics_df['Attribute'] == s]
                            if len(sub) >= 2:
                                min_rate = sub['Demographic Parity (Approval Rate)'].min()
                                max_rate = sub['Demographic Parity (Approval Rate)'].max()
                                diff = max_rate - min_rate

                                if diff > 0.10:
                                    msg = f"Bias Detected in {s} ({diff:.1%} disparity)"
                                    bias_issues.append(msg)
                                    st.error(
                                        f"{msg}: There is a significant difference in approval rates between groups.")
                                else:
                                    st.success(f"No major bias in {s}: Approval rates are within 10% of each other.")
                    else:
                        st.warning("No sensitive attributes selected.")

                    st.subheader("B. Global Explanation (SHAP)")
                    st.write("Top factors driving model decisions:")

                    with st.spinner("Calculating feature importance..."):
                        fig, top_features = generate_shap_plot(pipe, X)
                        st.pyplot(fig)
                        st.info(f"The model relies most heavily on: {', '.join(top_features)}.")

                        for s in sensitive_cols:
                            if s in top_features:
                                bias_issues.append(f"Sensitive attribute '{s}' is a top 3 driver of decisions")

                    st.subheader("C. Local Explanation (LIME)")
                    st.write("Detailed view of the first record in the dataset:")
                    with st.spinner("Generating LIME explanation..."):
                        lime_html = generate_lime_html(pipe, X, 0)
                        components.html(lime_html, height=400, scrolling=True)

                    st.subheader("D. Counterfactual Explanation")
                    st.write("What is the smallest change needed to flip the prediction for the first record?")

                    with st.spinner("Searching for counterfactuals..."):
                        cf_text = generate_counterfactual(pipe, X, 0)
                        st.success(f"Result: {cf_text}")

                    st.divider()
                    st.subheader("E. Final Risk Assessment")

                    if len(bias_issues) > 0:
                        st.error("HIGH RISK DETECTED")
                        st.write("The audit detected the following potential fairness issues:")
                        for issue in bias_issues:
                            st.write(f"- {issue}")
                        st.write(
                            "Recommendation: Do not deploy this model without further investigation and mitigation.")
                    else:
                        st.success("LOW RISK")
                        st.write(
                            "No major statistical disparities or sensitive feature dependencies were detected during this audit.")
                        st.write(
                            "Recommendation: The model passes standard fairness checks, but human oversight is still recommended.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a dataset to start.")
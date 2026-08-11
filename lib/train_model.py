# lib/train_model.py
import argparse
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from utils import load_any_file, make_basic_preprocessor, save_model, get_feature_names_from_preprocessor


def main(data_path, target_col, model_path="models/model.joblib"):
    print(f"Loading dataset from {data_path}...")
    df = load_any_file(data_path)

    df.columns = df.columns.astype(str).str.strip()

    if target_col not in df.columns:
        # Check if user passed an index (integer) instead of a name
        try:
            target_idx = int(target_col)
            target_col_name = df.columns[target_idx]
            print(f"Using column index {target_idx} -> '{target_col_name}' as target.")
            target_col = target_col_name
        except ValueError:
            raise ValueError(f"Target column '{target_col}' not found in dataset. Columns are: {df.columns.tolist()}")

    print(f"Target is: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == 'object':
        unique_vals = y.unique()
        y = y.astype('category').cat.codes
        print(f"Converted target labels {unique_vals} to integers.")

    print("Preparing preprocessor...")
    # This uses the utils function (Make sure you updated utils.py as I showed in the previous step!)
    preprocessor, numeric_cols, categorical_cols = make_basic_preprocessor(X)

    print("Building pipeline...")
    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline([("preproc", preprocessor), ("clf", clf)])

    print("Splitting and training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    save_model(pipe, model_path)
    print(f"Saved trained model pipeline to {model_path}")

    try:
        feature_names = get_feature_names_from_preprocessor(
            pipe.named_steps['preproc'],
            numeric_cols,
            categorical_cols
        )
    except:
        feature_names = X.columns.tolist()

    feature_file = os.path.join(os.path.dirname(model_path), "feature_names.txt")
    with open(feature_file, "w", encoding="utf-8") as f:
        feature_names = [str(f) for f in feature_names]
        f.write("\n".join(feature_names))

    print("Feature names saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--target_col", type=str, required=True, help="Name of the target column")
    parser.add_argument("--model_path", type=str, default="models/model.joblib", help="Where to save the model")

    args = parser.parse_args()

    main(args.data_path, args.target_col, args.model_path)
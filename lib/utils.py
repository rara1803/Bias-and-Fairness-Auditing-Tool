# lib/utils.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_any_file(path):
    """Load dataset from any supported file type"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".xlsx":
        return pd.read_excel(path)
    if ext == ".json":
        return pd.read_json(path)
    # Fallback for .data or others
    return pd.read_csv(path, header=None, delim_whitespace=True)


def save_model(model, path):
    """Save the model to a file"""
    joblib.dump(model, path)


def make_basic_preprocessor(X):
    """
    Creates a ColumnTransformer that scales numbers and one-hot encodes categories.
    Returns: preprocessor, numeric_cols, categorical_cols
    """
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ])

    return preprocessor, numeric_cols, categorical_cols


def get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols):
    """
    Attempts to extract feature names after OneHotEncoding.
    """
    output_features = []

    # Numeric features usually stay the same
    output_features.extend(numeric_cols)

    if hasattr(preprocessor, 'named_transformers_'):
        try:
            cat_encoder = preprocessor.named_transformers_['cat']['encoder']
            cat_features = cat_encoder.get_feature_names_out(categorical_cols)
            output_features.extend(cat_features)
        except Exception:
            output_features.extend(categorical_cols)

    return output_features
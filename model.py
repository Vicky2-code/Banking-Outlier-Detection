import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def detect_outliers(df, eps=1.8, min_samples=5):
    """
    Detect outliers in Kaggle Credit Card Fraud dataset using DBSCAN.
    Outliers are transactions DBSCAN labels as -1.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset (with or without 'Class').
    eps : float, optional
        DBSCAN epsilon value (default=1.8).
    min_samples : int, optional
        Minimum samples for DBSCAN clustering (default=5).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with added 'cluster' and 'is_outlier' columns.
    """

    # Select only numeric columns (ignore non-numeric safely)
    features = df.select_dtypes(include="number").copy()

    # Drop 'Class' (ground truth, not used for training)
    if "Class" in features.columns:
        features = features.drop(columns=["Class"])

    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    clusters = db.fit_predict(scaled)

    # Add results to dataframe
    df = df.copy()  # avoid modifying original
    df["cluster"] = clusters
    df["is_outlier"] = (clusters == -1)

    return df

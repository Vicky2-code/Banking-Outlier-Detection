# model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def detect_outliers(df):
    """
    Detect outliers in Kaggle Credit Card Fraud dataset using DBSCAN.
    Outliers are transactions DBSCAN labels as -1.
    """
    # Drop 'Class' (ground truth, we won't use for unsupervised detection)
    features = df.drop(columns=["Class"], errors="ignore")

    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Run DBSCAN
    db = DBSCAN(eps=1.8, min_samples=5).fit(scaled)
    clusters = db.labels_

    # Add results to dataframe
    df['cluster'] = clusters
    df['is_outlier'] = (clusters == -1)

    return df

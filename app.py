# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from model import detect_outliers
from sklearn.metrics import classification_report

st.set_page_config(page_title="Banking Outlier Detection", layout="wide")

st.title("💳 Banking Transaction Outlier Detection (DBSCAN)")
st.write("This app detects unusual transactions from the **Kaggle Credit Card Fraud Dataset** using DBSCAN clustering.")

# Upload CSV file
uploaded = st.file_uploader("📂 Upload Kaggle Credit Card Dataset (creditcard.csv)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # Run DBSCAN model
    df = detect_outliers(df)

    st.subheader("📈 Outlier Detection Results")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Let user pick sample size for visualization
        sample_size = st.slider("Select sample size for scatter plot:", 1000, 20000, 5000, step=1000)
        df_sample = df.sample(sample_size, random_state=42)

        # Scatter plot (sampled for performance)
        fig = px.scatter(
            df_sample,
            x="Time", y="Amount",
            color=df_sample['is_outlier'].map({True: "Outlier", False: "Normal"}),
            title=f"Transaction Outliers (Sampled {sample_size} Points)",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Total Transactions", len(df))
        st.metric("Outliers Detected", int(df['is_outlier'].sum()))

    st.subheader("🚨 Flagged Transactions (Outliers)")
    st.dataframe(df[df['is_outlier'] == True].head(50))  # show top 50 outliers

    # Download option
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results",
        data=csv,
        file_name="outlier_results.csv",
        mime="text/csv"
    )

    # Evaluation (compare with actual fraud labels, if Class column exists)
    if "Class" in df.columns:
        st.subheader("📊 Model Evaluation (vs Ground Truth)")
        y_true = df["Class"]
        y_pred = df["is_outlier"].astype(int)
        report = classification_report(y_true, y_pred, target_names=["Normal", "Fraud"], output_dict=True)
        eval_df = pd.DataFrame(report).transpose()
        st.dataframe(eval_df.style.background_gradient(cmap="Blues"))

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="eDNA AI Predictor", layout="wide")

st.title("🌊 eDNA Analysis with AI")
st.write("Upload your sequence dataset (CSV with `sequence_id` and `sequence`) to get AI predictions!")

@st.cache_resource
def load_model():
    try:
        model = joblib.load("trained_dna_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Trained model or vectorizer not found. Please train your model first.")
        return None, None

model, vectorizer = load_model()

# --- File uploader ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None and model is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Uploaded Data")
    st.dataframe(df)

    if "sequence" not in df.columns:
        st.error("CSV must contain a 'sequence' column.")
    else:
        st.subheader("🔬 AI Predictions")

        # Transform sequences to k-mer features using saved vectorizer
        X = vectorizer.transform(df["sequence"])
        predictions = model.predict(X)

        df["Predicted Cluster"] = predictions
        st.dataframe(df)

        # --- Summary ---
        st.subheader("📊 Summary of Predicted Clusters")
        summary = df["Predicted Cluster"].value_counts()
        st.bar_chart(summary)

        st.success("✅ Predictions generated using trained AI model!")

else:
    if model is None:
        st.info("Please train the model first by running train_model.py")
    else:
        st.info("Please upload a CSV file to begin.")

from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Leaf Species Classifier")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "leaf_model.pkl", "rb"))


@st.cache_resource
def load_scaler():
    return pickle.load(open(BASE / "leaf_scaler.pkl", "rb"))


@st.cache_resource
def load_meta():
    return pickle.load(open(BASE / "leaf_meta.pkl", "rb"))


model = load_model()
scaler = load_scaler()
meta = load_meta()
classes = meta["classes"]
feature_cols = meta["feature_cols"]
samples = meta["sample_leaves"]

st.title("Leaf Species Classifier")
st.write(
    "A Logistic Regression model (trained on the Kaggle Leaf Classification dataset) predicts a plant species "
    "from 192 pre-extracted numeric measurements of a leaf (margin, shape, and texture descriptors), not from "
    "the leaf image itself."
)

tab_sample, tab_csv = st.tabs(["Try a training leaf", "Upload feature CSV"])

with tab_sample:
    species = st.selectbox("Pick a species (a real leaf from the training data)", samples["species"])
    noise = st.slider("Add random noise to the measurements", 0.0, 2.0, 0.0, 0.1, help="0 = the original measurements. Higher values simulate a noisier or partial measurement.")

    if st.button("Classify this leaf"):
        row = samples.loc[samples["species"] == species, feature_cols].values[0]
        if noise > 0:
            row = row + np.random.default_rng().normal(0, noise * row.std(), row.shape)
        X = scaler.transform(row.reshape(1, -1))
        proba = pd.Series(model.predict_proba(X)[0], index=classes).sort_values(ascending=False)

        st.success(f"Predicted species: **{proba.index[0]}** ({proba.iloc[0]:.1%})")
        st.write(f"True species: **{species}**" + ("  ✅ correct" if proba.index[0] == species else "  ❌ incorrect"))
        st.bar_chart(proba.head(5))

with tab_csv:
    st.write(f"Upload a CSV with one row and the {len(feature_cols)} feature columns (margin1-64, shape1-64, texture1-64).")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            st.error(f"Missing {len(missing)} feature columns, for example: {missing[:3]}")
        else:
            X = scaler.transform(data[feature_cols].values)
            proba = model.predict_proba(X)
            for i in range(len(data)):
                top = pd.Series(proba[i], index=classes).sort_values(ascending=False)
                st.write(f"Row {i}: **{top.index[0]}** ({top.iloc[0]:.1%})")

st.caption(
    "Model: Logistic Regression on standardized features (validation log loss ≈ 0.057 on the Kaggle data, with "
    "99 species and only 10 training leaves per species). The 'training leaf' tab picks a real leaf from the "
    "training set, so a noise level of 0 should almost always be classified correctly."
)

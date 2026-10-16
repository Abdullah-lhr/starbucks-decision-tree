
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Starbucks AI", page_icon="☕")

st.title("☕ Starbucks Beverage Predictor")
st.write("Predict a beverage category using a trained Decision Tree.")

MODEL_PATH = "starbucks_decision_tree.pkl"

@st.cache_resource
def load_saved_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found in the GitHub repository.")
    st.stop()

saved = load_saved_model()

model = saved["model"]
encoder = saved["label_encoder"]
features = saved["features"]

st.subheader("Enter Beverage Information")

with st.form("beverage_form"):
    values = {}

    for feature in features:
        if feature == "Beverage_prep":
            values[feature] = st.text_input(
                "Beverage Preparation",
                value="Whole Milk"
            )
        else:
            values[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=0.0,
                step=1.0
            )

    submitted = st.form_submit_button("Predict Beverage Category")

if submitted:
    input_data = pd.DataFrame([values], columns=features)

    try:
        prediction = model.predict(input_data)[0]
        category = encoder.inverse_transform([prediction])[0]

        st.success(f"Predicted category: {category}")

        probabilities = model.predict_proba(input_data)[0]
        classes = model.named_steps["tree"].classes_

        results = pd.DataFrame({
            "Category": encoder.inverse_transform(classes),
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.subheader("Prediction Probabilities")
        st.bar_chart(results.set_index("Category"))

    except Exception as error:
        st.error(f"Prediction error: {error}")

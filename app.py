import re
import joblib
import streamlit as st

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# ---------------------------------
# Text cleaning function
# ---------------------------------
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\brt\b", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------
# Load fitted pipeline (NO CACHE!)
# ---------------------------------
def load_pipeline():
    return joblib.load("sentiment_pipeline.pkl")

model = load_pipeline()

# ---------------------------------
# UI
# ---------------------------------
st.title("💬 Twitter Sentiment Analysis")
st.write("Model: **TF-IDF + Logistic Regression** (Sentiment140)")

st.markdown("---")

user_input = st.text_area(
    "Enter a tweet:",
    value="I really love this movie!",
    height=120
)

show_cleaned = st.checkbox("Show cleaned text", value=True)
show_proba = st.checkbox("Show probabilities", value=True)

if st.button("Predict Sentiment ✅"):
    cleaned_text = clean_tweet(user_input)

    if show_cleaned:
        st.subheader("🧹 Cleaned Text")
        st.code(cleaned_text if cleaned_text else "(empty after cleaning)")

    if cleaned_text.strip() == "":
        st.warning("Text is empty after cleaning. Please enter a tweet.")
        st.stop()

    prediction = int(model.predict([cleaned_text])[0])

    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😡")

    if show_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned_text])[0]
        st.subheader("📊 Prediction Probabilities")
        st.write({
            "Negative (0)": float(proba[0]),
            "Positive (1)": float(proba[1])
        })

st.markdown("---")
st.caption(
    "This Streamlit app uses a fitted TF-IDF + Logistic Regression pipeline "
    "trained on the Sentiment140 dataset."
)

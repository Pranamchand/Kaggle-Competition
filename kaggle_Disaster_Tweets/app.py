import streamlit as st
import pickle
import pandas as pd
import numpy as np
from src.config import MODEL_PATH, VECTORISED_PATH
from src.data_loader import clean_data

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Disaster Tweet Detector",
    page_icon="🚨",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------


model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORISED_PATH, "rb"))

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

.stApp{
    background:#0E1117;
    color:white;
}

.main-title{
    font-size:48px;
    font-weight:700;
    text-align:center;
    background: linear-gradient(90deg,#ff4b4b,#ff914d);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:#BBBBBB;
    font-size:18px;
    margin-bottom:30px;
}

.result-box{
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:25px;
    font-weight:bold;
}

.real{
    background:#162d1b;
    border:2px solid #27ae60;
    color:#2ecc71;
}

.fake{
    background:#311414;
    border:2px solid #ff4b4b;
    color:#ff4b4b;
}

.stButton>button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    background:linear-gradient(90deg,#ff4b4b,#ff914d);
    color:white;
    font-size:18px;
    font-weight:bold;
}

textarea{
    font-size:18px !important;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------

st.markdown("<h1 class='main-title'>🚨 Disaster Tweet Detector</h1>", unsafe_allow_html=True)

st.markdown(
    "<p class='subtitle'>Predict whether a tweet is about a real disaster using Machine Learning.</p>",
    unsafe_allow_html=True
)

# ----------------------------
# Input
# ----------------------------

tweet = st.text_area(
    "Enter Tweet",
    height=180,
    placeholder="Example: Massive earthquake hits California..."
)

# ----------------------------
# Predict
# ----------------------------

if st.button("🔍 Predict"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        tweet = clean_data(pd.DataFrame({"text": [tweet]}))["text_only"].iloc[0]

        vector = vectorizer.transform([tweet])

        prediction = model.predict(vector)[0]

        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(vector))*100
        else:
            confidence = None

        st.write("")

        if prediction == 1:

            st.markdown("""
            <div class="result-box real">
            ✅ REAL DISASTER
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-box fake">
            ❌ NOT A DISASTER
            </div>
            """, unsafe_allow_html=True)

        if confidence is not None:

            st.progress(int(confidence))

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

# ----------------------------
# Examples
# ----------------------------

st.divider()

st.subheader("Example Tweets")

col1, col2 = st.columns(2)

with col1:
    st.info("🔥 Forest fire spreading rapidly across the hills.")

with col2:
    st.info("😂 My phone battery died again lol.")

st.divider()

st.caption(
    "Built with ❤️ using Streamlit • Scikit-learn • TF-IDF"
)
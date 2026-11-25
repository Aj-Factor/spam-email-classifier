from pathlib import Path
import subprocess
import sys
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "spam_classifier.joblib"

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Enter a message to estimate whether it is spam or genuine (ham).")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        result = subprocess.run([sys.executable, str(ROOT / "train_model.py")],
                                cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "Training failed.")
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    message = st.text_area("Message to check", height=150,
        placeholder="Congratulations! You have won a free prize. Claim now.")
    if st.button("Check message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = model.predict([message])[0]
            probs = model.predict_proba([message])[0]
            confidence = float(probs[list(model.classes_).index(prediction)])
            if prediction == "spam":
                st.error("Likely SPAM")
            else:
                st.success("Likely GENUINE (HAM)")
            st.metric("Model confidence", f"{confidence:.1%}")
            st.caption("This is an estimate, not a guarantee. Review suspicious messages carefully.")
    with st.expander("Try sample messages"):
        st.code("Congratulations! You have won a free prize. Call now to claim your reward.")
        st.code("Hi, are we still meeting for lunch tomorrow?")
except Exception as exc:
    st.error("The application could not load the model.")
    st.exception(exc)
    st.info("Check dependencies and confirm data/sms_spam.csv exists.")

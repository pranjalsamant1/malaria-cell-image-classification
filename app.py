from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Malaria Cell Classifier",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models") / "malaria_cnn.keras"
IMG_SIZE = 128  # matches your CLI predictor
APP_TITLE = "Malaria Cell Image Classifier"
APP_SUBTITLE = "CNN-based binary classification on blood smear cell images"

# -----------------------------
# Styling (professional look)
# -----------------------------
CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(1200px 800px at 20% 10%, rgba(125,211,252,0.20), transparent 60%),
                radial-gradient(1000px 700px at 80% 20%, rgba(167,139,250,0.16), transparent 55%),
                linear-gradient(180deg, #0b1220 0%, #070b14 100%);
    color: rgba(255,255,255,0.92);
}
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px; }

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.35);
}

.pill {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.06);
    margin-right: 8px;
    margin-bottom: 8px;
}

h1, h2, h3 { letter-spacing: -0.02em; }

.stButton > button {
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    background: rgba(255,255,255,0.08) !important;
}
.stButton > button:hover {
    border: 1px solid rgba(125,211,252,0.6) !important;
    background: rgba(125,211,252,0.12) !important;
}

section[data-testid="stFileUploaderDropzone"]{
    border-radius: 18px;
    border: 1px dashed rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.04);
}
</style>
"""


@st.cache_resource
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH.as_posix()} — expected models/malaria_cnn.keras"
        )
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model: tf.keras.Model, img: Image.Image) -> tuple[str, float, float]:
    x = preprocess(img)

    # sigmoid output: P(Uninfected) (matches your CLI predictor)
    p_uninfected = float(model.predict(x, verbose=0)[0][0])

    if p_uninfected >= 0.5:
        pred = "Uninfected"
        conf = p_uninfected
    else:
        pred = "Parasitized"
        conf = 1.0 - p_uninfected

    return pred, conf, p_uninfected


def badge_row():
    st.markdown(
        """
        <span class="pill">TensorFlow</span>
        <span class="pill">CNN</span>
        <span class="pill">Computer Vision</span>
        <span class="pill">Medical Imaging</span>
        <span class="pill">Streamlit App</span>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🧠 Model")
        st.write(f"**Path:** `{MODEL_PATH.as_posix()}`")
        st.write(f"**Input:** `{IMG_SIZE}×{IMG_SIZE}` RGB")
        st.write("**Output:** Sigmoid `P(Uninfected)`")
        st.divider()
        st.markdown("### ✅ How to use")
        st.write("1) Upload a PNG/JPG image")
        st.write("2) Click **Run Prediction**")
        st.write("3) Review prediction + confidence")
        st.divider()
        st.markdown("### ℹ️ Notes")
        st.write("- If results feel flipped, it’s just label mapping (easy fix).")
        st.write("- Best results come from images similar to the training dataset.")

    # Header
    colA, colB = st.columns([0.72, 0.28], vertical_alignment="center")
    with colA:
        st.markdown(f"# {APP_TITLE}")
        st.markdown(
            f"<span style='color: rgba(255,255,255,0.72); font-size: 16px;'>{APP_SUBTITLE}</span>",
            unsafe_allow_html=True,
        )
        badge_row()
    with colB:
        st.markdown(
            """
            <div class="card">
              <div style="font-size: 14px; color: rgba(255,255,255,0.70);">Pipeline</div>
              <div style="margin-top: 6px; font-size: 14px;">Upload → Preprocess → CNN → Prediction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    left, right = st.columns([0.52, 0.48], gap="large")

    # Upload panel
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📤 Upload")
        uploaded = st.file_uploader(
            "Upload a cell image (PNG / JPG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )

        if uploaded is None:
            st.info("Upload an image to begin.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        try:
            img = Image.open(io.BytesIO(uploaded.read()))
        except Exception:
            st.error("Could not read that file as an image.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        st.markdown("#### Preview")
        st.image(img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction panel
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔎 Prediction")

        try:
            model = load_model()
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        run = st.button("Run Prediction", type="primary")

        if not run:
            st.caption("Click **Run Prediction** to perform inference.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        with st.spinner("Running inference..."):
            pred, conf, p_uninfected = predict(model, img)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", pred)
        m2.metric("Confidence", f"{conf*100:.2f}%")
        m3.metric("P(Uninfected)", f"{p_uninfected*100:.2f}%")

        # Probabilities
        st.markdown("#### Probabilities")
        st.progress(
            min(max(p_uninfected, 0.0), 1.0),
            text=f"Uninfected: {p_uninfected*100:.2f}%",
        )
        st.progress(
            min(max(1.0 - p_uninfected, 0.0), 1.0),
            text=f"Parasitized: {(1.0-p_uninfected)*100:.2f}%",
        )

        st.caption("Matches your CLI predictor: sigmoid output is **P(Uninfected)**.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown(
        "<div style='text-align:center; color: rgba(255,255,255,0.55); font-size: 12px;'>"
        "Built by Pranjal Samant · Malaria CNN Classifier"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

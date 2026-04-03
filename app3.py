import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import time
import os
from ultralytics import YOLO
import tensorflow as tf

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Drone & Bird Detection",
                   layout="wide",
                   initial_sidebar_state="expanded")

# -------------------- HELPERS --------------------
@st.cache_resource
def load_resnet(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_yolo(path):
    return YOLO(path)

def pil_to_cv2(img_pil):
    arr = np.array(img_pil.convert('RGB'))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def annotate_and_get_bytes(img_np, results):
    annotated = results[0].plot()

    if annotated.dtype != np.uint8:
        annotated = (annotated * 255).astype(np.uint8)

    pil_img = Image.fromarray(annotated)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf, pil_img


# -------------------- MODEL PATHS (RELATIVE) --------------------
# These work locally AND on Streamlit Cloud
resnet_path = "models/transfer_ResNet50V2_model_01.h5"
yolo_path   = "models/best.pt"

st.sidebar.title("⚙️ Controls")
show_labels = st.sidebar.checkbox("Show model info", value=True)
st.sidebar.markdown("---")

# -------------------- LOAD MODELS --------------------
with st.spinner("Loading models..."):
    resnet_model = load_resnet(resnet_path) if os.path.exists(resnet_path) else None
    yolo_model = load_yolo(yolo_path) if os.path.exists(yolo_path) else None

if show_labels:
    cols = st.columns(2)
    with cols[0]:
        st.subheader("ResNet Model")
        if resnet_model:
            st.write("Loaded successfully ✔️")
            st.write("Output:", resnet_model.output_shape)
        else:
            st.error("ResNet model NOT FOUND. Check 'models/' folder!")

    with cols[1]:
        st.subheader("YOLO Model")
        if yolo_model:
            st.write("Loaded successfully ✔️")
        else:
            st.error("YOLO model NOT FOUND. Check 'models/' folder!")

st.write("---")

# -------------------- UPLOAD --------------------
st.header("🖼️ Input Image")
colA, colB = st.columns([1, 2])

with colA:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

with colB:
    st.subheader("Preview")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
    else:
        st.info("Upload an image to begin.")

st.write("---")

# -------------------- PREDICTIONS --------------------
st.header("🔎 Prediction")

if uploaded:
    uploaded.seek(0)
    img = Image.open(uploaded).convert("RGB")
    img_cv2 = pil_to_cv2(img)

    # ---------------- Classification (ResNet)
    if resnet_model:
        t0 = time.time()

        arr = cv2.resize(img_cv2, (224, 224))
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        preds = resnet_model.predict(arr)
        prob_drone = float(preds[0][0])
        prob_bird = 1 - prob_drone

        t1 = time.time()

        cols = st.columns(2)
        cols[0].metric("Predicted", "DRONE" if prob_drone >= 0.5 else "BIRD")
        cols[1].metric("Confidence", f"{max(prob_drone, prob_bird)*100:.2f}%")

        st.caption(f"ResNet inference time: {(t1 - t0)*1000:.0f} ms")
    else:
        st.error("ResNet model not loaded.")

    st.write("---")

    # ---------------- YOLO Detection
    if yolo_model:
        t0 = time.time()
        results = yolo_model.predict(img_cv2, conf=0.25, imgsz=640)
        t1 = time.time()

        buf, annotated_img = annotate_and_get_bytes(img_cv2, results)

        st.image(annotated_img, caption="YOLO Detection", use_container_width=True)

        try:
            n = len(results[0].boxes)
        except:
            n = 0

        st.write(f"Detections: {n}")
        st.download_button("Download detection image",
                           data=buf,
                           file_name="detections.png",
                           mime="image/png")

        st.caption(f"YOLO inference time: {(t1 - t0)*1000:.0f} ms")
    else:
        st.error("YOLO model not loaded.")
else:
    st.info("Upload an image to run predictions.")

st.write("---")


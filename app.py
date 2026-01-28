# ================================
# ENVIRONMENT (MUST BE FIRST)
# ================================
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================================
# IMPORTS
# ================================
import gradio as gr
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# ================================
# CONSTANTS (MUST MATCH TRAINING)
# ================================
IMG_SIZE = (128, 128)
MAX_LEN = 50
LABELS = ["Critical", "High", "Medium", "Low"]

# ================================
# LOAD MODEL (KERAS 3 SAFE)
# ================================
model = tf.keras.models.load_model(
    "fusion_model_keras3.keras",
    compile=False
)
print("✅ Fusion model loaded")

# ================================
# LOAD TOKENIZER
# ================================
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("✅ Tokenizer loaded")

# ================================
# IMAGE PREPROCESS
# ================================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ================================
# TEXT PREPROCESS
# ================================
def preprocess_text(text: str):
    if text is None:
        text = ""
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAX_LEN
    )
    return padded

# ================================
# PREDICTION FUNCTION
# ================================
def predict_ticket(image, text):
    if image is None:
        return {
            "Critical": 0.0,
            "High": 0.0,
            "Medium": 0.0,
            "Low": 0.0,
        }

    img = preprocess_image(image)
    txt = preprocess_text(text)

    probs = model.predict([img, txt], verbose=0)[0]

    return {
        "Critical": float(probs[0]),
        "High": float(probs[1]),
        "Medium": float(probs[2]),
        "Low": float(probs[3]),
    }

# ================================
# GRADIO UI
# ================================
interface = gr.Interface(
    fn=predict_ticket,
    inputs=[
        gr.Image(type="pil", label="📤 Upload Ticket Screenshot"),
        gr.Textbox(
            lines=4,
            placeholder="Describe the issue (recommended)",
            label="✍️ Ticket Description"
        )
    ],
    outputs=gr.Label(num_top_classes=4, label="🚨 Predicted Severity"),
    title="🎫 Ticket Severity Classification",
    description=(
        "CNN + NLP **Fusion Model** for ticket urgency detection.\n\n"
        "**Classes:** Critical | High | Medium | Low"
    )
)

interface.launch()
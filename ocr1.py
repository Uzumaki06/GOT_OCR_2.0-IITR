import streamlit as st
import os
import json
import base64
import uuid
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import io
import shutil
import torch

model_name = "srimanth-d/GOT_CPU"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval()

UPLOAD_FOLDER = "./uploads"
RESULTS_FOLDER = "./results"

# Ensure directories exist
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Function to convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to run the GOT model
def run_GOT(image, got_mode, fine_grained_mode="", ocr_color="", ocr_box=""):
    unique_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.png")
    result_path = os.path.join(RESULTS_FOLDER, f"{unique_id}.html")

    image.save(image_path)  # Save image to disk

    try:
        if got_mode == "plain texts OCR":
            res = model.chat(tokenizer, image_path, ocr_type='ocr')
            return res, None
        elif got_mode == "format texts OCR":
            res = model.chat(tokenizer, image_path, ocr_type='format', render=True, save_render_file=result_path)
        elif got_mode == "plain multi-crop OCR":
            res = model.chat_crop(tokenizer, image_path, ocr_type='ocr')
            return res, None
        elif got_mode == "format multi-crop OCR":
            res = model.chat_crop(tokenizer, image_path, ocr_type='format', render=True, save_render_file=result_path)
        elif got_mode == "plain fine-grained OCR":
            res = model.chat(tokenizer, image_path, ocr_type='ocr', ocr_box=ocr_box, ocr_color=ocr_color)
            return res, None
        elif got_mode == "format fine-grained OCR":
            res = model.chat(tokenizer, image_path, ocr_type='format', ocr_box=ocr_box, ocr_color=ocr_color, render=True, save_render_file=result_path)

        if "format" in got_mode and os.path.exists(result_path):
            with open(result_path, 'r') as f:
                html_content = f.read()
            return res, html_content
        else:
            return res, None

    except Exception as e:
        return f"Error: {str(e)}", None

# Streamlit UI elements
def main():
    st.title("OCR Multilingual (GOT OCR 2.0)")
    st.markdown("### Upload an image for OCR")

    # File uploader for the image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Select task
    task_option = st.selectbox(
        "Select OCR Mode",
        ["plain texts OCR", "format texts OCR", "plain multi-crop OCR", "format multi-crop OCR", "plain fine-grained OCR", "format fine-grained OCR"]
    )
    
    # Process if image is uploaded
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Submit button
        if st.button("Run OCR"):
            st.write("Running OCR...")
            result_text, result_html = run_GOT(image, task_option)
            st.write(result_text)
            if result_html:
                st.markdown(result_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

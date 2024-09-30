import streamlit as st
from transformers import AutoModel, AutoTokenizer
import os
import base64
import io
import uuid
import shutil
from pathlib import Path
import time
import tempfile

model_name = "srimanth-d/GOT_CPU"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval()

UPLOAD_FOLDER = "./uploads"
RESULTS_FOLDER = "./results"

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Cleanup function for removing old files
def cleanup_old_files():
    current_time = time.time()
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        for file_path in Path(folder).glob('*'):
            if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                file_path.unlink()

# Function to search and highlight keywords in text
def search_in_text(text, keywords):
    """Searches for keywords within the text and highlights matches."""
    if not keywords:
        return text
    highlighted_text = text
    for keyword in keywords.split():
        highlighted_text = highlighted_text.replace(keyword, f"<mark>{keyword}</mark>")
    return highlighted_text

# OCR processing function
def run_GOT(image, got_mode, fine_grained_mode="", ocr_color="", ocr_box=""):
    unique_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.png")
    result_path = os.path.join(RESULTS_FOLDER, f"{unique_id}.html")

    shutil.copy(image, image_path)

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
        res_markdown = res

        if "format" in got_mode and os.path.exists(result_path):
            with open(result_path, 'r') as f:
                html_content = f.read()
            encoded_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            iframe_src = f"data:text/html;base64,{encoded_html}"
            iframe = f'<iframe src="{iframe_src}" width="100%" height="600px"></iframe>'
            download_link = f'<a href="data:text/html;base64,{encoded_html}" download="result_{unique_id}.html">Download Full Result</a>'
            return res_markdown, f"{download_link}<br>{iframe}"
        else:
            return res_markdown, None
    except Exception as e:
        return f"Error: {str(e)}", None
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# Streamlit interface
st.title("GOT OCR 2.0 Model")

st.markdown("""
Upload your image below and select your preferred mode. Note that more characters may increase wait times.
- **Plain Texts OCR & Format Texts OCR:** Use these modes for basic image-level OCR. Format Text OCR is preferred for better results.
- **Plain Multi-Crop OCR & Format Multi-Crop OCR:** Ideal for images with complex content, offering higher-quality results.
- **Plain Fine-Grained OCR & Format Fine-Grained OCR:** These modes allow you to specify fine-grained regions on the image for more flexible OCR. Regions can be defined by coordinates or colors (red, blue, green, black or white).
""")

uploaded_image = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
got_mode = st.selectbox("Choose OCR mode", [
    "plain texts OCR",
    "format texts OCR",
    "plain multi-crop OCR",
    "format multi-crop OCR",
    "plain fine-grained OCR",
    "format fine-grained OCR"
])

if "fine-grained" in got_mode:
    ocr_box = st.text_input("Input OCR box [x1,y1,x2,y2]")
    ocr_color = st.selectbox("Choose OCR color", ["red", "green", "blue", "black", "white"])
else:
    ocr_box = ""
    ocr_color = ""

# Maintain state for OCR result
if 'ocr_result' not in st.session_state:
    st.session_state.ocr_result = None
if 'html_result' not in st.session_state:
    st.session_state.html_result = None

if st.button("Run OCR"):
    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(uploaded_image.read())
            ocr_result, html_result = run_GOT(temp.name, got_mode, ocr_box=ocr_box, ocr_color=ocr_color)
            st.session_state.ocr_result = ocr_result
            st.session_state.html_result = html_result
            st.text_area("OCR Result", ocr_result)
    else:
        st.warning("Please upload an image.")

# Display the OCR result if it has been set
if st.session_state.ocr_result:
    st.text_area("OCR Result", st.session_state.ocr_result,key="display_area")

    # Keyword search functionality
    keywords = st.text_input("Enter keywords for highlighting",key="keyword_input")
    if keywords:
        highlighted_text = search_in_text(st.session_state.ocr_result, keywords)
        st.markdown(highlighted_text, unsafe_allow_html=True)

    if st.session_state.html_result:
        st.markdown(st.session_state.html_result, unsafe_allow_html=True)

if __name__ == "__main__":
    cleanup_old_files()

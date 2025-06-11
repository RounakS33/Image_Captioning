import streamlit as st
from PIL import Image
from src.utils import load_model, preprocess_image, generate_captions

st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("Image Captioning System")
st.write("Upload an image and get comprehensive captions.")


@st.cache_resource
def get_model():
    return load_model("models/model.pth")


model = get_model()

uploaded_file = st.file_uploader("Browse", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image",
             use_column_width=False, width=300)
    with st.spinner("Generating captions..."):
        image_tensor = preprocess_image(image)
        captions = generate_captions(
            model, image_tensor, beam_width=10, num_captions=10)
    st.markdown("Captions:")
    for i, cap in enumerate(captions, 1):
        st.write(f"{i}. {cap}")

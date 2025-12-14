import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("AI Intern – Text to Image Generator")

@st.cache_resource
def load_model():
    return StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )

pipe = load_model()

prompt = st.text_input("Describe the image")

if prompt:
    image = pipe(prompt).images[0]
    st.image(image)

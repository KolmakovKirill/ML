import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

@st.cache_resource
def load_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_pipeline()

st.title("🎨 Генерация изображений по описанию (Stable Diffusion)")

prompt = st.text_input("Опиши изображение (на английском)", "a fantasy landscape with castles and dragons")

steps = st.slider("Число шагов генерации", 10, 100, 50)
seed = st.number_input("Случайное зерно (seed)", value=42)

if st.button("Сгенерировать"):
    generator = torch.manual_seed(seed)
    with st.spinner("Генерируем изображение..."):
        image = pipe(prompt, num_inference_steps=steps, generator=generator).images[0]
        st.image(image, caption="Сгенерированное изображение", use_column_width=True)

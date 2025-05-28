import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas


st.title("🖍️ Распознавание рукописных цифр")

# Загрузка модели
@st.cache_resource
def load_digit_model():
    return load_model("model.h5")

model = load_digit_model()

# Канва для рисования
st.markdown("Нарисуйте цифру:")
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Обработка изображения
    image = Image.fromarray((canvas_result.image_data[:, :, 0] * 255).astype(np.uint8))
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    img_array = np.array(image).reshape(1, 28, 28) / 255.0

    st.image(image, caption="Предобработанное изображение", width=150)

    if st.button("Предсказать"):
        prediction = model.predict(img_array)
        st.subheader(f"Предсказание: {np.argmax(prediction)}")
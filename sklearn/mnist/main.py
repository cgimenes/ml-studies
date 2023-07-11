import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from joblib import load

model = load("model.joblib")

st.set_page_config(
    page_title="Number recognition",
    page_icon="🤖",
    # layout="wide",
    # initial_sidebar_state="expanded",
    menu_items={
        "About": """
          ### Exploration code
          https://github.com/cgimenes/ml-studies/sklearn/mnist/mnist.ipynb

          ### Model file generation
          https://github.com/cgimenes/ml-studies/sklearn/mnist/generation.py
          """,
    },
)

st.title("Number recognition using Multilayer Perceptron")
st.text("by Marcelo Gimenes")

col1, col2 = st.columns([3, 1])

with col1:
    canvas_result = st_canvas(
        stroke_width=60,
        stroke_color="#fff",
        background_color="#000",
        update_streamlit=True,
        height=500,
        width=500,
        drawing_mode="freedraw",
        display_toolbar=True,
    )

with col2:
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data.astype("uint8")
        im = Image.fromarray(img_data, mode="RGBA")
        im = im.convert("L")
        im.thumbnail((28, 28), Image.LANCZOS)

        number = np.asarray(im).reshape(-1)
        prediction = model.predict([number])[0]

        st.header(f"Predicted number: {prediction}")

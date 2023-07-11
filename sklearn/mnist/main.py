import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from joblib import load

import urllib.request
import tarfile
from pathlib import Path

model_path = Path("model.joblib")
if not model_path.is_file():
    url = "https://github.com/cgimenes/ml-studies/raw/master/sklearn/mnist/model.joblib"
    urllib.request.urlretrieve(url, model_path)

model = load(model_path)

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
st.markdown("by [Marcelo Gimenes de Oliveira](https://github.com/cgimenes)")

col1, col2 = st.columns([3, 1])

with col1:
    canvas_result = st_canvas(
        stroke_width=50,
        stroke_color="#fff",
        background_color="#000",
        update_streamlit=True,
        height=28 * 17,
        width=28 * 17,
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

        st.markdown("### Predicted number")
        st.markdown(f"# {prediction}")

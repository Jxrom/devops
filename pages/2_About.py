import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps

st.set_page_config(page_title="Classification Page", page_icon="🔎")

@st.cache
def load_model():
    model = tf.keras.models.load_model("assets/car_bike_classifier.h5", compile=False)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
    return model

def import_and_predict(image_data, model):
    size = (75, 75)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_reshape = gray[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

st.write("""
# Car-Bike Detection
""")

uploaded_images = st.file_uploader("Choose up to 5 Car or Bike photos from your computer", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if not uploaded_images:
    st.text("Please upload image files")
else:
    model = load_model()
    class_names = ["Bike", "Car"]
    num_images = min(len(uploaded_images), 5)

    for i in range(num_images):
        image = Image.open(uploaded_images[i])
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        string = "OUTPUT : " + class_names[np.argmax(prediction)]
        st.success(string)

# Discussion of best architectures for classification in deep learning
st.write("""
## Best Architectures for Classification in Deep Learning
""")
# Create a table with architecture names, descriptions, and hyperlinks
architectures = {
    "Architecture": ["Convolutional Neural Networks (CNNs)", "Recurrent Neural Networks (RNNs)", "Transformer-based Models", "Support Vector Machines (SVMs)", "Dense Neural Networks", "Ensemble Methods", "Custom Architectures", "AutoML and NAS", "Pre-trained Models", "Hybrid Architectures"],
    "Description": [
        "Effective for image classification and object recognition tasks.",
        "Ideal for sequential data like NLP and time series analysis.",
        "Revolutionized NLP and applied to various domains.",
        "Effective for structured data classification.",
        "Basic building blocks for deep learning models.",
        "Combine multiple models to improve performance.",
        "Designed to address unique challenges in specific problems.",
        "Automated tools to find optimal architectures.",
        "Leverage pre-trained models for specific tasks.",
        "Combine different architectures for hybrid models."
    ],
    "Learn More": [
        "[Learn More](https://en.wikipedia.org/wiki/Convolutional_neural_network)",
        "[Learn More](https://en.wikipedia.org/wiki/Recurrent_neural_network)",
        "[Learn More](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))",
        "[Learn More](https://en.wikipedia.org/wiki/Support_vector_machine)",
        "[Learn More](https://en.wikipedia.org/wiki/Artificial_neural_network)",
        "[Learn More](https://en.wikipedia.org/wiki/Ensemble_learning)",
        "[Learn More](https://en.wikipedia.org/wiki/Artificial_neural_network#Customization)",
        "[Learn More](https://en.wikipedia.org/wiki/Automated_machine_learning)",
        "[Learn More](https://huggingface.co/transformers/)",
        "[Learn More](https://en.wikipedia.org/wiki/Ensemble_learning)"
    ]
}

# Create a DataFrame and display it as a table
import pandas as pd
df = pd.DataFrame(architectures)
st.table(df)

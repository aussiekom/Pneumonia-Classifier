import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


set_background('bg6.jpg')

# set title
st.title('Pneumonia classification')

# Page title
st.markdown("""
- This app allows you to predict the presence or absence of pneumonia by chest Xray.
- App built in `Python (with using tensorflow)` + `Streamlit` by Evgeniia Komarova [LinkedIn](https://www.linkedin.com/in/evgeniia-komarova-523139235/) [GitHub](https://github.com/aussiekom)

""")

# set header
# st.header('Please upload a chest X-ray image')

# Sidebar
with st.sidebar.header('Please upload a chest X-ray image'):
    file = st.sidebar.file_uploader("Upload your image", type=['jpeg', 'jpg', 'png'])

# upload file
# file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('pneumonia_classifier.h5')

# load class names
with open('labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pickle

INPUT_SHAPE = 256

plt.figure(figsize=(12, 8))
img_main = st.file_uploader('Upload a PNG image', type=['png', 'jpg', 'jpeg'])
st.divider()

if img_main is not None:
    img = Image.open(img_main)
    model = pickle.load(open('model.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))
    pred = model.predict(np.expand_dims(img, 0), verbose=False)
    cls_index = np.argmax(pred)
    cls_name = classes[cls_index]

    st.devider()
    st.image(img)
    st.header({cls_name})

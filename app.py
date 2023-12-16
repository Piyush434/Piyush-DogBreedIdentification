import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
INPUT_SHAPE = 256

plt.figure(figsize=(12, 8))
img_main = st.file_uploader('Upload a PNG image', type=['png', 'jpg', 'jpeg'])
st.divider()

img = Image.open(img_main)
img = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE))
pred = model.predict(np.expand_dims(img, 0), verbose=False)
cls_index = np.argmax(pred)
cls_name = classes[cls_index]
plt.subplot(230 + i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'True Class: {samples.loc[[index], "breed"].values[0]} \n Predicted Class: {cls_name}')
    
plt.tight_layout()
plt.show()
import streamlit as st
import os
import cv2
import numpy as np

from PIL import Image

from src.model.sort_images import SortImageModel

def alone_app(img, show_bounding_box):
    with open(os.path.join("temporary_files",img.name),"wb") as f:
        f.write(img.getbuffer())
    img_path = os.path.join("temporary_files",img.name)
    model = SortImageModel()
    prediction = model.predict(img_path)
    with open(img_path, 'rb') as fd:
        image = Image.open(fd)
        if prediction[0] == 1:
            st.error('Некачественное изображение', icon="🚨")
            st.image(image)
        elif prediction[1] == 1:
            st.warning('Пустое изображение', icon="⚠️")
            st.image(image)
        else:
            st.success('Изображение с животным')
            result = model.object_detector(image)
            st.image(result.render(labels=False))
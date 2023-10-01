import streamlit as st
import os
from PIL import Image
from torchvision import transforms
from torchvision import datasets
import torch
from torch import nn
import io

from app.packet_processing import packet_app
from app.showcase import alone_app

def app():
    st.markdown('# FotolovushkaAI')
    st.markdown('## Обработка одного снимка')
    uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        # Process demo case for one image
        # show_boundaries = False
        # if st.checkbox('Показать животных на снимке'):
        #     show_boundaries = True
        alone_app(uploaded_file, True)

    container = st.container()
    container.markdown("## Пакетная обработка")
    container.markdown("В данном блоке вы можете загрузить снимки из папки и автоматически их обработать.")
    container.warning("Внимание: поддерживаются только следующие форматы - jpg, jpeg, png, JPG, JPEG, PNG", icon="⚠️")
    load_path = container.text_input("Укажите абсолютный путь к папке, в которой содержатся снимки в формате")
    save_path = container.text_input("Укажите абсолютный путь к папке, в которую вы хотите сложить отсортированные снимки")
    button_clicked = container.button("Подтвердить")
    save_csv = False
    if container.checkbox('Сохранить результаты в .csv файл'):
        save_csv = True
    if button_clicked:
        if os.path.isdir(load_path) and os.path.isdir(save_path):
            result_dataframe = packet_app(load_path, save_path)
            if save_csv:
                result_dataframe = result_dataframe.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label='Скачать .csv файл',
                    data=result_dataframe,
                    file_name='submission.csv'
                )
        else:
            st.error('Указан некорректный путь', icon="🚨")

if __name__ == '__main__':
    app()
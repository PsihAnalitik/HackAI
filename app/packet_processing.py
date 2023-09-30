import streamlit as st
import os
import glob
import shutil
import pandas as pd
import logging

log = logging.getLogger(__name__)

from src.model.sort_images import SortImageModel

def packet_app(src_path: str, dst_path: str):
    if os.path.isdir(f'{dst_path}/broken_images'):
         shutil.rmtree(f'{dst_path}/broken_images')
    os.makedirs(f'{dst_path}/broken_images')

    if os.path.isdir(f'{dst_path}/empty_images'):
         shutil.rmtree(f'{dst_path}/empty_images')
    os.makedirs(f'{dst_path}/empty_images')

    if os.path.isdir(f'{dst_path}/animal_images'):
         shutil.rmtree(f'{dst_path}/animal_images')
    os.makedirs(f'{dst_path}/animal_images')

    cl0_cnt = 0
    cl1_cnt = 0
    cl2_cnt = 0

    image_extensions = [".png", ".jpg", ".jpeg", '.JPG', '.JPEG', '.PNG']
    load_images = []
    for ext in image_extensions:
          load_images += glob.glob(f'{src_path}/**/*{ext}', recursive=True)

    st.title("Обработка запроса")
    st.markdown(f'Обнаружено {len(load_images)} снимков')
    model = SortImageModel()
    progress_bar = st.progress(0, text="Подождите пока ваш запрос обработается...")
    predict_data = []
    process = 0
    for ix, img_path in enumerate(load_images):
         row = [f'{"/".join(img_path.split("/")[-3:])}'] + model.predict(img_path)
         predict_data.append(row)
         if row[1] == 1:
              cl0_cnt += 1
              shutil.copy(img_path, dst_path+'/broken_images')
         elif row[2] == 1:
              cl1_cnt += 1
              shutil.copy(img_path, dst_path+'/empty_images')
         else:
              cl2_cnt += 1
              shutil.copy(img_path, dst_path+'/animal_images')
         progress_bar.progress(process, text="Подождите пока ваш запрос обработается...")
         process += 1/len(load_images)
    progress_bar.empty()

    predict_data = pd.DataFrame(data=predict_data, columns=['filename', 'broken', 'empty', 'animal'])

    st.success(f"Путь к результатам: {os.getcwd() + '/res'}. Всего {cl0_cnt} испорченных снимков, {cl1_cnt} пустых снимков, {cl2_cnt} снимков с животными")
    # TODO: мб отчетик с тем скольлко животных обнаружено
    return predict_data
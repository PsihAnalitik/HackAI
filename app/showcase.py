import streamlit as st
import os

def alone_app(path):
    if path is not None:
        transform = transforms.ToPILImage()
        image = preprocess_image(path)
        image = transform(image)
        st.image(image, caption=f"Image", use_column_width=True)
        claster = Algorithm(image) # Our classifier solution
        folder_name = '/res/' + str(claster) 
        filename = f"{claster}"
        st.success(f"picture class: {claster}")
        file_name = path.name
        file_path = os.path.join(folder_name, file_name)
        file_path = f"./{file_path}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        image.save(file_path)
        st.success(f"Путь к результатам: {os.getcwd() + '/res'}")
    st.write("Укажите правильный путь к файлу!")
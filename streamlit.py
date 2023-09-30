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

#@st.cache(allow_output_mutation=True)
def get_image_path(image_folder, image_name):
    return os.path.join(image_folder, image_name)

def select_dataset_folder():
    # Получаем список папок в текущем рабочем каталоге
    folders = [f for f in os.listdir('.') if os.path.isdir(f)]


    folders.insert(0, "Все папки")

    selected_folder = st.sidebar.selectbox("Выберите папку с датасетом", folders)

    return selected_folder

def preprocess_image(uploaded_file):
    #return uploaded_file
    image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)
    return image

def save_image_locally(folder_name, uploaded_file, filename):
    image_folder = f"./{folder_name}"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image = preprocess_image(uploaded_file)

    image_path = get_image_path(image_folder, filename)
    save_image(image, image_path)

def save_image(image, image_path):
    transforms.ToPILImage()(image).save(image_path)
    #image.save(image_path)

def Algorithm(img):
    '''weights = torch.load('./my_mnist_model.pt', map_location=torch.device('cpu'))

    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
    model.load_state_dict(weights)'''
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    #tensor_img = transform(img).view(1, -1)

    tensor_img = transform(img)
    model = torch.load('./my_mnist_model.pt')
    model.eval()
    tensor_img = tensor_img.view(tensor_img.size(0), -1) 
    with torch.no_grad():
        logps = model(tensor_img)
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    return str(probab.index(max(probab)))

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")

def main():
    uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        # Process demo case for one image
        show_boundaries = False
        if st.checkbox('Показать животных на снимке'):
            show_boundaries = True
        alone_app(uploaded_file, show_boundaries)

    container = st.container()
    container.markdown("## Классификатор изображений")
    container.markdown("Внимание: поддерживаются только следующие форматы - jpg, jpeg, png")
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
    main()
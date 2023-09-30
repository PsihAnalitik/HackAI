import streamlit as st
import os
from PIL import Image
from torchvision import transforms
from torchvision import datasets
import torch
from torch import nn
import io



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


def app(folder_path):
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
    num_images = sum(1 for f in os.listdir(folder_path) if f.lower().endswith(tuple(image_extensions)) and os.path.isfile(os.path.join(folder_path, f)))

    st.title("Сохранение результатов работы модели")
    count = 0
    progress_bar = st.progress(count)
    for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            image = preprocess_image(img_path)
            label = filename
            if image is not None:
                    transform = transforms.ToPILImage()
                    image = transform(image)
                    #st.image(image, caption=f"Saved_img{iter}", use_column_width=True)
                    folder_name = '/res/' + str(Algorithm(image)) # Our classifier solution
                    filename = f"{label}"
                    #st.success(f"picture folder: {folder_name}, {label}")
                    save_image_locally(folder_name, folder_path+'/'+filename, filename)
                    count +=1
                    if (count) / num_images <=1:
                        progress_bar.progress((count) / num_images)

    st.success(f"Количество изображений: {count}. Путь к результатам: {os.getcwd() + '/res'}")
    return

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

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")

def main():
    SAVE_FOLDER = [f"./{i}" for i in range(10)]

    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    #df = 'mnist_data_test/images'
    df = None
    uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        alone_app(uploaded_file)

    container = st.container()
    container.markdown("## Классификатор изображений")
    path = container.text_input("Абсолютный путь к папке датасета")
    button_clicked = container.button("Указать путь")
    if button_clicked:
        df = path
        st.success(f"{path} and {df}" )
        if len(df) >0:
                app(df)
        else:
                 st.success("Укажите путь к папке", )

    if st.button("Очистить папку"):
        for folder in SAVE_FOLDER:
            clear_folder(folder)
        st.success("Папка успешно очищена")

if __name__ == '__main__':
    main()
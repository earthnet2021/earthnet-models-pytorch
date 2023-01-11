import io
import os
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
import wget


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    return model


def load_labels():
    labels_path = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    labels_file = os.path.basename(labels_path)
    if not os.path.exists(labels_file):
        wget.download(labels_path)
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict(model, categories, image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        st.write(categories[top5_catid[i]], top5_prob[i].item())


def main():
    st.title('Pretrained model demo')
    model = load_model()
    categories = load_labels()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, categories, image)


if __name__ == '__main__':
    main()
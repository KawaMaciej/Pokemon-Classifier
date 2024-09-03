import streamlit as st
from PIL import Image
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from pokemon_classifier import predict_streamlit, api_info
import json


with open('class_names.json', 'r') as f:
    class_names = json.load(f)

model = models.efficientnet_b0(pretrained=True)

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 150)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((360, 360)),
    transforms.RandomAffine(degrees=0, shear=20, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


st.set_page_config(
    page_title="Pokémon Classifier",
    page_icon=":sparkles:",
    layout="centered",
)


st.sidebar.title("About")
st.sidebar.write("""
    **Pokémon Classifier** is a machine learning application that identifies Pokémon species
    from images. Upload a Pokémon image, and the app will classify it and provide details
    such as base experience, height, weight, and abilities.
""")

st.sidebar.title("How to Use")
st.sidebar.write("""
    1. Upload a clear image of a Pokémon.
    2. Wait for the model to classify the image.
    3. View the classification results and Pokémon details.
""")

st.title("✨ Pokémon Classifier ✨")

uploaded_file = st.file_uploader("Choose a JPG/PNG image of a Pokémon", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image', use_column_width=False, width=400,)
    predictions = predict_streamlit(image, model, train_transform, class_names)
    

    pokemon_info = api_info(predictions.lower())
    st.markdown(f"## **Pokémon Name**: {pokemon_info[0].capitalize()}")
    
    st.markdown(f"""
    <div class="info-section">
        <strong>Base Experience:</strong> {pokemon_info[1]}<br>
        <strong>Height:</strong> {pokemon_info[2] / 10} m<br>
        <strong>Weight:</strong> {pokemon_info[3] / 10} kg<br>
        <strong>Abilities:</strong> {pokemon_info[4].capitalize()}
    </div>
    """, unsafe_allow_html=True)
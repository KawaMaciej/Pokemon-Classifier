# Pokémon Classifier

Welcome to the Pokémon Classifier project! This repository contains a deep learning model designed to classify Pokémon species based on images. The project includes a Streamlit web application for real-time predictions and a Jupyter notebook used for training the model.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [File Structure](#file-structure)


## Overview

The Pokémon Classifier is a machine learning project aimed at identifying different Pokémon species from images. The model is trained using a efficientnet_b0 and achieves high accuracy on the test dataset. The project showcases the power of deep learning in image classification tasks.


## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/KawaMaciej/Pokemon-Classifier.git
    cd pokemon-classifier
    ```

2. **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset:**

    Make sure you have the Pokémon dataset. You can download it from this repo.

5. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```
### Docker Installation

If you prefer to run the app using Docker, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/KawaMaciej/Pokemon-Classifier.git
    cd pokemon-classifier
    ```

2. **Build the Docker image using the existing `Dockerfile`:**

    ```bash
    docker build -t pokemon-classifier .
    ```

3. **Run the Docker container:**

    ```bash
    docker run -p 8501:8501 pokemon-classifier
    ```

4. **Access the app:**

    Open your web browser and go to `http://localhost:8501` to use the Pokémon Classifier app.

This method leverages Docker to create a containerized environment, ensuring that the app runs consistently across different systems without needing to manage Python environments or dependencies manually.
## Usage

Once the Streamlit app is running, you can upload an image of a Pokémon, and the model will predict its species. The app provides a simple and intuitive interface for users to interact with the classifier.

## Model Training

The model is trained in a Jupyter notebook. To train the model yourself, follow these steps:

1. **Navigate to the `notebooks` directory:**

    ```bash
    cd notebooks
    ```

2. **Open the training notebook:**

    Use Jupyter Notebook or JupyterLab to open `trainning_notebook.ipynb`.

    ```bash
    jupyter notebook trainning_notebook.ipynb
    ```

3. **Run the cells in the notebook:**

    Follow the instructions in the notebook to preprocess the data, define the model architecture, and train the model.

## File Structure

The repository is organized as follows:

```bash
pokemon-classifier/
├── app.py                     # Streamlit app
├── Dockerfile
├── class_names                   
├── notebooks/
│   └── training_notebook.ipynb  # Jupyter notebook for model training
├── PokemonData/
│   └── folders with 150 pokemon
├── model.pth
├── pokemon_classifier
├── requirements.txt            # Python dependencies\
├── requirements_for_docker.txt # Docker dependencies\
├── README.md                   # Project documentation
```

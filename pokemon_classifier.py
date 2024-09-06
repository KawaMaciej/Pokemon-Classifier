
import os 
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, models
import torch 
import os
from typing import Tuple, List, Callable, Optional
import torch.optim.lr_scheduler as lr_scheduler

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_val_loss: float, epochs_no_improve: int, path: str) -> None:
    """Saves a checkpoint of the model, optimizer, epoch, best_val_loss, and epochs_no_improve at the specified path."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve
    }
    torch.save(state, path)
    print(f"Checkpoint saved at epoch {epoch+1}")

def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: str) -> Tuple[int, float, int]:
    """Loads a checkpoint from the given path, restoring the model, optimizer, epoch, best validation loss, and number of epochs since improvement. If no checkpoint is found, starts from scratch."""
    if os.path.isfile(path):
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        print(f"Checkpoint loaded: start from epoch {epoch+1}")
        return epoch, best_val_loss, epochs_no_improve
    else:
        print(f"No checkpoint found at {path}, starting from scratch.")
        return 0, float('inf'), 0

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, class_names: List[str]) -> Tuple[float, float]:
    """
    Validates a model on the given validation set.
    Args:
        model (nn.Module): The model to be validated.
        val_loader (DataLoader): The DataLoader for the validation set.
        criterion (nn.Module): The loss function to use.
        class_names (List[str]): List of class names for the dataset.
    Returns:
        A tuple containing the validation loss and validation accuracy.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    wrong_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    wrong_predictions.append((inputs[i].cpu(), predicted[i].cpu(), labels[i].cpu()))

    for i, (img, pred, label) in enumerate(wrong_predictions[:1]):
        img = img.permute(1, 2, 0) 
        img = img.numpy()

        plt.figure(figsize=(2, 2))
        plt.imshow(img)
        plt.title(f'Predicted: {class_names[pred]}, Actual: {class_names[label]}')
        plt.axis('off')
        plt.show()

    val_accuracy = 100 * correct / total
    model.train()
    return val_loss / len(val_loader), val_accuracy

def train(model: nn.Module, train_loader: DataLoader, 
    val_loader: DataLoader, criterion: nn.Module, 
    optimizer: optim.Optimizer,
    num_epochs: int, patience: int, checkpoint_path: str, class_names: List[str], 
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> None:
    """
    Trains a model on the given train_loader and validates it on the given val_loader.
    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The DataLoader for the training set.
        val_loader (DataLoader): The DataLoader for the validation set.
        criterion (nn.Module): The loss function to use.
        optimizer (optim.Optimizer): The optimizer to use.
        num_epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait before stopping training if the validation loss does not improve.
        checkpoint_path (str): The path to save the model checkpoints.
        class_names (List[str]): List of class names for the dataset.
    Returns:
        None
    """
    start_epoch, best_val_loss, epochs_no_improve = load_checkpoint(model, optimizer, checkpoint_path)

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            scheduler.step()
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total

            val_loss, val_accuracy = validate(model, val_loader, criterion, class_names)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience and train_accuracy > 90 and val_accuracy > 80:
                print(f'Early stopping after {epoch+1} epochs')
                break

            save_checkpoint(model, optimizer, epoch, best_val_loss, epochs_no_improve, checkpoint_path)

    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint...")
        save_checkpoint(model, optimizer, epoch, best_val_loss, epochs_no_improve, checkpoint_path)
        print("Checkpoint saved. You can resume training later.")

    print('Finished Training')

def predict_image(image_path: str, model: nn.Module, transform: Callable, labels: List[str]) -> str:
    """
    Predicts the class of a given image using the given model and transform.
    Args:
        image_path (str): The path to the image to be predicted.
        model (nn.Module): The model to be used for prediction.
        val_transform (Callable): The transform to be used for validation.
        labels (List[str]): The list of class labels.
    Returns:
        The predicted class of the given image.
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        image = image.to('cuda')
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return labels[predicted]
def api_info(pokemon: str) -> Tuple:
    """
    Fetches information about the given pokemon from PokeAPI and prints it to the console.
    
    Args:
        pokemon (str): The name of the pokemon to fetch information about.
    
    Returns:
        A tuple containing the name, base experience, height, weight, and abilities of the pokemon.
    """
    url = f'https://pokeapi.co/api/v2/pokemon/{pokemon}'
    r = requests.get(url)
    name = r.json()['name']
    base_experience = r.json()['base_experience']
    height = r.json()['height']
    weight = r.json()['weight']
    print("Name: ", r.json()['name'])
    print("Base Experience: ", r.json()['base_experience'])
    print("Height: ", r.json()['height'] / 10, 'm')
    print("Weight: ", r.json()['weight'] / 10, 'kg')
    abilities = r.json()['abilities']
    ability_names = [ability['ability']['name'] for ability in abilities]
    ability_names = ', '.join(ability_names)
    print("Abilities:", ability_names)
 
    return name, base_experience, height, weight, ability_names


def get_info(image_path: str, model: nn.Module, val_transform: Callable, labels: List[str]) -> None:
    """
    Predicts the class of the given image and fetches information about it from PokeAPI.
    Args:
        image_path (str): The path to the image to be predicted.
        model (nn.Module): The model to be used for prediction.
        val_transform (Callable): The transform to be used for validation.
        labels (List[str]): The list of class labels.
    Returns:
        None
    """
    predicted = predict_image(image_path, model, val_transform, labels)

    pokemon = predicted.lower()
    api_info(pokemon)

    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off') 
    plt.show()

def predict_streamlit(image, model, transform, class_names):
    """
    Predicts the class of the given image using the given model and transform.

    Args:
        image (PIL.Image): The image to be predicted.
        model (nn.Module): The model to be used for prediction.
        transform (Callable): The transformation to be applied to the image.
        class_names (List[str]): The list of class labels.

    Returns:
        str: The predicted class of the given image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  

    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    _, predicted_idx = torch.max(output, 1)
    
    predicted_label = class_names[predicted_idx.item()]

    return predicted_label

def classifier_trainer(directory: str, checkpoint_path: str, train_transform: Callable, 
                               train_split: float = 0.8, 
                               num_epochs: int = 100, 
                               learning_rate: float = 0.001, 
                               patience: int = 25, 
                               criterion: nn.Module = nn.CrossEntropyLoss(),
                               batch_size: int = 32,
                               model: nn.Module = models.efficientnet_b0(pretrained=True),
                               out: str = 'model',
                               weight_decay: float = 0.001) -> str:
    """
    Trains a model to classify Pokémon images in a given directory, saving checkpoints at the specified path.

    Args:
        directory (str): The path to the directory containing Pokémon images, organized by class in subdirectories.
        checkpoint_path (str): The path where model checkpoints will be saved.
        train_transform (Callable): The transformation to be applied to the training data.
        train_split (float, optional): The proportion of the data to be used for training, with the remainder used for validation. Defaults to 0.8.
        num_epochs (int, optional): The number of epochs to train the model. Defaults to 100.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        patience (int, optional): The number of epochs to wait for improvement in validation loss before stopping training early. Defaults to 25.
        criterion (nn.Module, optional): The loss function used for training. Defaults to `CrossEntropyLoss`.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        model (nn.Module, optional): The neural network model to be trained. Defaults to `EfficientNet-B0` with pre-trained weights.
        out (str, optional): Directory name or prefix for saving the trained model. Defaults to 'model'.
        weight_decay (float, optional): The weight decay (L2 penalty) applied to the optimizer. Defaults to 0.001.
    Returns:
        str: A message indicating that training is complete, including any relevant details about model saving and early stopping.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.20)


    dataset = datasets.ImageFolder(directory)

    class_names=dataset.classes
        
    train_dataset = datasets.ImageFolder(directory, transform=train_transform)
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, checkpoint_path, class_names, scheduler)
    torch.save(model.state_dict(), f'models/{out}.pth')


    return "Training Complete"    
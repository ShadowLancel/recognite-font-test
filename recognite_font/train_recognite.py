﻿import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import os
import argparse
from font_model import FontClassifier
from tqdm import tqdm
from sklearn.metrics import classification_report

def load_images(path):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader

def train_model(num_classes, path_train_imgs, path_test_imgs, num_epochs, save_folder):
    # Загрузка данных
    train_loader = load_images(path_train_imgs)
    test_loader = load_images(path_test_imgs)

    model = FontClassifier(num_classes)

    # Обучение модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
	model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss / len(train_loader)}")

    # Cохранение модели
    path_to_save = os.path.join(save_folder, 'model_demo.pth')

    # Проверка существования пути и создание директории, если не существует
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(model.state_dict(), path_to_save)
    print(f"Model saved successfully at {path_to_save}")

    # Тестирование модели
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Font Recognition Model")
    parser.add_argument("--num_classes", type=int, default=15, help="Number of classes")
    parser.add_argument("--train_images_folder", type=str, default=r"C:\Users\ShadowLancel\source\repos\character_generator_1\character_generator_1\output\train_imgs", help="Path to the train images folder")
    parser.add_argument("--test_images_folder", type=str, default=r"C:\Users\ShadowLancel\source\repos\character_generator_1\character_generator_1\output\test_imgs", help="Path to the test images folder")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--save_folder", type=str, default="models", help="Folder to save the trained model")

    args = parser.parse_args()
    train_model(args.num_classes, args.train_images_folder, args.test_images_folder, args.num_epochs, args.save_folder)
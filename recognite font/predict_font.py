#coding=utf-8
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse

# Использование GPU при наличии такой возможности
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FontClassifier(nn.Module):
    def __init__(self):
        super(FontClassifier, self).__init__()
        self.feature_extractor = models.resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 15)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

def predict_font(model_path, class_names_path, image_path):
    # Загрузка модели
    model = FontClassifier()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Загрузка словаря из текстового файла
    class_names = {}
    with open(class_names_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                class_number, font_name = map(str.strip, line.split(':'))
                class_names[int(class_number)] = font_name

    # Загрузка изображения для предсказания
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Получение предсказания
    with torch.no_grad():
        output = model(image)

    predicted_probabilities = nn.functional.softmax(output, dim=1)[0]
    predicted_class = torch.argmax(predicted_probabilities).item()
    predicted_probability = predicted_probabilities[predicted_class].item()

    # Вывод результатов
    print(f"Predicted Class: {predicted_class}")
    print(f"Predicted Font: {class_names.get(predicted_class, 'Unknown')}")
    print(f"Probability: {predicted_probability*100}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Font using trained model")
    parser.add_argument("--model_path", type=str, default=r"D:\models font recognition\model_demo.pth", help="Path to the folder with model_demo.pth")
    parser.add_argument("--class_names_path", type=str, default='class_names.txt', help="Path to the class names file")
    parser.add_argument("--image_path", type=str, default=r"C:\Users\ShadowLancel\source\repos\character_generator_1\character_generator_1\output\test_imgs\class_5\image_4005.png", help="Path to the image for font recognition")
    args = parser.parse_args()
    predict_font(args.model_path, args.class_names_path, args.image_path)
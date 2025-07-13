import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import argparse
from font_model import FontClassifier

# Использование GPU при наличии такой возможности
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_font(model_path, class_names_path, image_path):
    # Загрузка модели
    model = FontClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True))
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
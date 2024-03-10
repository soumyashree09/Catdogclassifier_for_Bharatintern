from torchvision import models, transforms
from PIL import Image
import torch

# Function to load and preprocess an image
def preprocess_image(img_path):
    # Load the image
    img = Image.open(img_path)

    # Resize and normalize the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # Add a batch dimension

    return img_tensor

# Function to classify an image using a pre-trained model
def classify_image(img_tensor, model, labels):
    # Make a prediction
    with torch.no_grad():
        output = model(img_tensor)
    
    # Get the predicted label
    _, predicted_class = torch.max(output, 1)
    prediction = labels[predicted_class.item()]

    return prediction

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Load labels used by the pre-trained model
with open("imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f]

# Paths to your cat and dog images
cat_image_path = "C:\Users\swomy\Desktop\catDogSpecifires\cat image.jpg"
dog_image_path = "C:\Users\swomy\Desktop\catDogSpecifires\dog image.jpeg"

# Classify cat image
cat_img_tensor = preprocess_image(cat_image_path)
cat_prediction = classify_image(cat_img_tensor, model, labels)

# Classify dog image
dog_img_tensor = preprocess_image(dog_image_path)
dog_prediction = classify_image(dog_img_tensor, model, labels)

# Display predictions
print(f"Prediction for {cat_image_path}: {cat_prediction}")
print(f"Prediction for {dog_image_path}: {dog_prediction}")
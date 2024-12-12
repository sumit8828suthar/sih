from fastapi import APIRouter, HTTPException, Depends,File , UploadFile
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json

def serialize_index(index): return list(index)
def predict(image_path, model, transform, device):
    # Load and preprocess the image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Ensure 3 channels
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.cpu().numpy()
router = APIRouter()

@router.get("/getModelOutputs/")
async def getModelOutputs(query_params: str,file: UploadFile = File(...)):
    
    try:
        
        # Define the path to your saved model
        model_path = r"resnet50_best_model.pth"  # Update the path to match the actual structure

        # Load the saved ResNet-50 model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = 3  # Update this to match your number of classes
        model = models.resnet50()  # Initialize the ResNet-50 architecture
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Update the final layer
        model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights from the specified path
        model = model.to(device)
        model.eval()

        # Define the preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Function to run inference on a single image
        

        # Map predicted class indices to labels
        class_labels = {0: 'bird', 1: 'drone+bird', 2: 'drone'}  # Update this dictionary based on your training labels

        # Example usage
        file_content = await file.read()
        with open(file.filename, "wb") as f:
            f.write(file_content)
        predicted_class, probabilities = predict(file.filename, model, transform, device)
        print(f"Predicted Class: {class_labels[predicted_class]}")
        print(f"Class Probabilities: {probabilities}")        

        model_results = {
            "predicted_class": class_labels[predicted_class],
            "class_probabilities": probabilities.tolist()
        }    
        response = {
            "statusCode": 200,
            "body": json.dumps(model_results, default=serialize_index)
            
        }

        return response

    except Exception as e:
        
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
    
    
   
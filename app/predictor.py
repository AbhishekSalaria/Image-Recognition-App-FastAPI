import io

from torchvision import models
from torchvision.models.densenet import DenseNet121_Weights
import torchvision.transforms as transforms
from PIL import Image
import json

model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
model.eval()

imagenet_class_index = json.load(open('imagenet_class_index.json'))

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

def get_result(image_file):
    bytes = image_file.file.read()
    id,name = get_prediction(image_bytes=bytes)

    return {
        "Response": "Success",
        "Predictions":{
            "id": id,
            "name": name
        }
    }
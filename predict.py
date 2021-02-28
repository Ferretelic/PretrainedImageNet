import json
import os

import torch
import torchvision
from PIL import Image

def transform_image(image):
  transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
  return transform(image)

def load_image(image_path):
  image = Image.open(image_path).convert("RGB")
  return torch.unsqueeze(transform_image(image), 0)

def get_index2label():
  with open("./label_data.json") as f:
    class_index = json.load(f)

  index2label = [class_index[str(k)][1] for k in range(len(class_index))]
  return index2label

def predict_label(image_name):
  device = torch.device("cuda")
  image = load_image(os.path.join("images", image_name)).to(device)
  model = torchvision.models.resnet152(pretrained=True).to(device).eval()
  output = torch.topk(model(image), 5).indices[0]
  index2label = get_index2label()
  labels = [index2label[label.item()] for label in output]
  return labels

image_name = "./sample2.jpg"
print(predict_label(image_name))

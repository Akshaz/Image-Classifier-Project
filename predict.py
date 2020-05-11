import torch
import json
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch import optim
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Example with non-optional arguments')

parser.add_argument('image_path', action="store")
parser.add_argument('checkpoint', action="store")
parser.add_argument('--category_names', action="store" , default = 'cat_to_name.json')
parser.add_argument('--top_k', action="store", type=int, default = 1)
parser.add_argument('--gpu', action='store_true', default=False)

args = vars(parser.parse_args())

with open(args['category_names'], 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint():
    checkpoint = torch.load(args['checkpoint'])
    models_ = {
    "alexnet": models.alexnet(pretrained=True),
    "vgg16": models.vgg16(pretrained=True),
    "densenet": models.densenet121(pretrained=True),
    }
    model = models_[f"{checkpoint['model_name']}"]
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

model, optimizer = load_checkpoint()

def process_image(path):
    img = Image.open(path)
    width, height = img.size
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img/255
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    img = img[np.newaxis,:]
    image = torch.from_numpy(img)
    image = image.float()
    return image

if args['gpu']:
    model.to('cuda');

def predict(image_path, model, topk=5):
    img_tensor = process_image(image_path).view(1, 3, 224, 224)
    if args['gpu']:
        img_tensor = img_tensor.cuda()
    with torch.no_grad():
        model.eval()
        out = model.forward(img_tensor)
        ps = torch.exp(out)
        top_k, top_class = ps.topk(topk, dim=1)
        top_classes = [
            cat_to_name[f'{class_}'] for class_ in top_class.cpu().numpy()[0]
        ]
        top_p = torch.exp(top_k).cpu().numpy()[0]
        return top_p, top_classes
top_p, top_classes = predict(args['image_path'], model, args['top_k'])
print("Predictions: ")
for ii, (i, j) in enumerate(zip(top_p, top_classes)):
    print(f"{ii+1}. {j} : {i * 100}%")
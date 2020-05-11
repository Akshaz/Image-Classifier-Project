import argparse, torch, json, os
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image

parser = argparse.ArgumentParser(description='All the arguments mentioned in project requirements.')

parser.add_argument('data_directory', action="store", default = '/home/workspace/ImageClassifier/flowers')
parser.add_argument('--arch', action="store", default = "vgg16")
parser.add_argument('--save_dir', action="store", default=os.getcwd())
parser.add_argument('--learning_rate', action="store", type=float, default = 0.001)
parser.add_argument('--hidden_units', action="store", type=int, default = 1024)
parser.add_argument('-epochs', action="store", type=int, default = 10)
parser.add_argument('--gpu', action='store_true', default=False)

args = vars(parser.parse_args())

def save_checkpoint(path, model, optimizer, dataset, name):
    checkpoint = {
            'class_to_idx': dataset.class_to_idx,
            'epochs': model.epochs,
    }
    checkpoint['model_name'] = name
    checkpoint['classifier'] = model.classifier
    checkpoint['state_dict'] = model.state_dict()
    checkpoint['optimizer'] = optimizer
    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, f"{args['save_dir']}/checkpoint.pth")

data_dir = args['data_directory']

transforms = [
transforms.Compose([
    transforms.RandomRotation(25),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(), 
    transforms.CenterCrop(size=224),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
transforms.Compose([
    transforms.Resize(255), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
]

train_data = datasets.ImageFolder(data_dir + '/train', transform=transforms[0])
validation_data = datasets.ImageFolder(data_dir + '/valid', transform=transforms[1])
test_data = datasets.ImageFolder(data_dir + '/test', transform=transforms[1])

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


models_ = {
    "alexnet": models.alexnet(pretrained=True),
    "vgg16": models.vgg16(pretrained=True),
    "densenet": models.densenet121(pretrained=True),
}
 
model = models_[f"{args['arch']}"]

for param in model.parameters():
    param.requires_grad = False


for i in model.classifier:
    if hasattr(i, 'in_features'):
        n_inputs = i.in_features
        break
        
model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(n_inputs, args['hidden_units']), 
            nn.ReLU(),
            nn.Linear(args['hidden_units'], 102)
)

optimizer = optim.Adam(model.classifier.parameters(), lr=args['learning_rate'])
    
criterion = nn.CrossEntropyLoss()

if(args['gpu']):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

model.to(device);

try:
    print(f'Model has been trained for: {model.epochs} epochs.\n')
except:
    model.epochs = 0
    print(f'Starting Training from Scratch.\n')

for epoch in range(args['epochs']):
    model.train()
    running_loss = 0
    for images, labels in trainloader:
        if args['gpu']:
            images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        accuracy = 0
        validationloss = 0
        model.epochs += 1
        with torch.no_grad():
            model.eval()
            
            for images, labels in validationloader:
                if args['gpu']:
                    images, labels = images.cuda(), labels.cuda()
                logps = model(images)
                loss = criterion(logps, labels)
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                validationloss += loss.item()
        print("Epoch: {}/{}.. ".format(epoch+1, args['epochs']),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Validation Loss: {:.3f}.. ".format(validationloss/len(validationloader)),
              "Accuracy: {:.3f}".format(accuracy/len(validationloader)))
            
save_checkpoint(args['save_dir'], model, optimizer, train_data, args['arch'])


model.eval()
test_loss = 0
accuracy = 0
with torch.no_grad():
    for images, labels in testloader:
        if args['gpu']:
            images, labels = images.to(device), labels.to(device)
        logps = model(images)
        test_loss += criterion(logps, labels)        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
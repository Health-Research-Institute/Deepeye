#Train eye diseases recognition system

import os
import cv2
import pandas as pd
import pandas as pd
import torch
aa = torch.cuda.is_available()
print(aa)

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import functional
from torchvision.io import read_image


#Create dataset from STARE images
class CustomImageDataset:
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        transform = transforms.Grayscale()
        image = transform(image)
        transform = transforms.Resize((121,140))
        image = transform(image)
        image = functional.convert_image_dtype(image,torch.float32)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


eyeTrainDataset = CustomImageDataset("../Images/STARE/label14train.csv", "../Images/STARE")
print("eyeTrainDataset length is:", eyeTrainDataset.__len__())
eyeTestDataset = CustomImageDataset("../Images/STARE/label14test.csv", "../Images/STARE")
print("eyeTestDataset length is:", eyeTestDataset.__len__())

##We pass the Dataset as an argument to DataLoader.
##  This wraps an iterable over our dataset, and supports automatic batching, sampling, 
## shuffling and multiprocess data loading. Here we define a batch size of 64, 
## i.e. each element in the dataloader iterable will return a batch of 64 features and labels.
batch_size = 64

# Create data loaders
train_dataloader = DataLoader(eyeTrainDataset, batch_size=batch_size)
test_dataloader = DataLoader(eyeTestDataset, batch_size=batch_size)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape} {X.dtype}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

import time
seconds = time.time()
print("Time in seconds since the epoch:", seconds)

#Creating Model 
# To define a neural network in PyTorch, we create a class that inherits from nn.Module. 
# We define the layers of the network in the __init__ function and specify how data 
# will pass through the network in the forward function. 
# To accelerate operations in the neural network, we move it to the GPU if available. 
device = "cuda" if torch.cuda.is_available() else "cpu" #DO NOT WORK CURRENTLY

print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(121*140, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#To train a model, we need a loss function and an optimizer.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
# and backpropagates the prediction error to adjust the model’s parameters.
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#We also check the model’s performance against the test dataset to ensure it is learning.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#The training process is conducted over several iterations (epochs). 
# During each epoch, the model learns parameters to make better predictions.
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#Saving Models
torch.save(model.state_dict(), "eyeModel.pth")
print("Saved PyTorch Model State to eyeModel.pth")

#Loading Models
model = NeuralNetwork()
model.load_state_dict(torch.load("eyeModel.pth"))

#This model can now be used to make predictions.
classes = [
    "Heathy",
    "Age Related Macular Degeneration",
]

model.eval()
for i in range(0,eyeTestDataset.__len__()):
    x, y = eyeTestDataset[i][0], eyeTestDataset[i][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


secondsS = time.time()
print("Time in seconds since timer start:", secondsS-seconds)

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# Define the transformation pipeline for data preprocessing
transform = transforms.Compose([
    transforms.Resize((200,200)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225])
])

# Specify the paths to your train and test datasets
train_root_dir = r'D:\my_pro\train'
test_root_dir = r'D:\my_pro\test'

# Load the datasets using ImageFolder
train_dataset = datasets.ImageFolder(root=train_root_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_root_dir, transform=transform)

# Create data loaders for batching and shuffling
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define your CNN model architecture
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1,padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=1),

    nn.Conv2d(32,64 , kernel_size=3, stride=1, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    

    nn.AdaptiveAvgPool2d(output_size=(7, 7)),
    nn.Flatten(),

    nn.Linear(128*7*7, 256),  # Adjust the output size of the linear layer
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid()    # Output size matches the number of classes (apple, tomato)
)

# Define the loss function (CrossEntropyLoss for multi-class classification)
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer (SGD with momentum)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 30
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_dataloader:
        #print(labels.shape,labels.dtype)
        # Forward pass
        outputs = model(inputs)
        output_tensor = outputs.squeeze()

       # Convert the tensor to float32 dtype
        outputs = output_tensor.type(torch.float32)
        #print(outputs.dtype,outputs)
        # Calculate the loss
        loss = criterion(outputs, labels.float())
        
        # Zero gradients, backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute training statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 0)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    # Calculate accuracy and average loss for the epoch
    epoch_loss = running_loss / len(train_dataloader)
    epoch_accuracy = (correct_predictions / total_predictions) * 100.0

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")


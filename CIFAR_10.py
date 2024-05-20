import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from copy import deepcopy

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and split the CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Define the CNN architecture
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.fc1 = torch.nn.Linear(64*8*8, 64)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Train with Adam optimizer, batch size of 32, and epochs from 1 to 100
optimizer_class = torch.optim.Adam
batch_size = 32
epochs = 100

model = Net().to(device)  # move model to GPU
criterion = torch.nn.CrossEntropyLoss()
optimizer = optimizer_class(model.parameters())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

train_losses = []
train_accs = []
test_accs = []
test_precisions = []

best_test_acc = 0
best_model_wts = None

for epoch in range(epochs):

    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    train_losses.append(running_loss/len(trainloader))
    train_accs.append(correct/total)

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # move data to GPU
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_acc = correct/total
    test_accs.append(test_acc)
    test_precision = precision_score(all_labels, all_predictions, average='macro')
    test_precisions.append(test_precision)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_wts = deepcopy(model.state_dict())

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}, Train Acc: {train_accs[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}, Test Precision: {test_precisions[-1]:.4f}')

# Plot the loss and accuracy curves
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(train_losses)
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Training Loss')

axs[0, 1].plot(train_accs)
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Training Accuracy')

axs[1, 0].plot(test_accs)
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Test Accuracy')

axs[1, 1].plot(test_precisions)
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Test Precision')

plt.tight_layout()
plt.show()

print(f'Best test accuracy: {best_test_acc:.4f}')
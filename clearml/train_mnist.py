"""
ClearML Experiment Tracking Demo
---------------------------------
Complete ML training pipeline demonstrating ClearML's auto-magical experiment tracking.

This demo trains a CNN on MNIST dataset and shows:
1. Automatic hyperparameter logging
2. Real-time metric tracking
3. Model versioning
4. Git integration
5. Experiment comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from clearml import Task

task = Task.init(
    project_name='MNIST Classification',
    task_name='CNN Training Demo',
    tags=['demo', 'pytorch', 'mnist']
)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
args = parser.parse_args()

print(f"\n{'='*50}")
print("Training Configuration:")
print(f"{'='*50}")
print(f"Batch Size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Hidden Size: {args.hidden_size}")
print(f"Dropout: {args.dropout}")
print(f"{'='*50}\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}\n")

class MNISTNet(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTNet(args.hidden_size, args.dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

print("Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")

def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    print(f'\nEpoch {epoch} Summary - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n')
    
    from clearml import Logger
    logger = Logger.current_logger()
    logger.report_scalar("train", "loss", iteration=epoch, value=epoch_loss)
    logger.report_scalar("train", "accuracy", iteration=epoch, value=epoch_acc)
    
    return epoch_loss, epoch_acc

def test_epoch(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    print(f'Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%\n')
    
    from clearml import Logger
    logger = Logger.current_logger()
    logger.report_scalar("test", "loss", iteration=epoch, value=test_loss)
    logger.report_scalar("test", "accuracy", iteration=epoch, value=test_acc)
    
    return test_loss, test_acc

print(f"{'='*50}")
print("Starting Training")
print(f"{'='*50}\n")

best_acc = 0.0

for epoch in range(1, args.epochs + 1):
    print(f"--- Epoch {epoch}/{args.epochs} ---")
    train_loss, train_acc = train_epoch(epoch)
    test_loss, test_acc = test_epoch(epoch)
    
    if test_acc > best_acc:
        best_acc = test_acc
        print(f"New best accuracy: {best_acc:.2f}% - Saving model...")
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"{'='*50}\n")

print(f"{'='*50}")
print("Training Complete!")
print(f"{'='*50}")
print(f"Best Test Accuracy: {best_acc:.2f}%")
print(f"\nModel saved to: best_model.pth")
print(f"\nView results in ClearML Web UI:")
print(f"https://app.clear.ml")
print(f"{'='*50}\n")

print("\n✅ All metrics, models, and artifacts automatically logged to ClearML!")
print("✅ Check the ClearML dashboard to:")
print("   - Compare this experiment with others")
print("   - View training curves")
print("   - Download the model")
print("   - Clone/reproduce this experiment")
print("   - See Git commit info and code changes\n")

# api {
#   web_server: https://app.clear.ml/
#   api_server: https://api.clear.ml
#   files_server: https://files.clear.ml
#   credentials {
#     "access_key" = "S02RA2W1DLGGT0C60E15EPL9LJBWYW"
#     "secret_key" = "aWeyFAFuezkA0CARIiznlji5nJZ3Yt2aZCuS2voVaIL0hli-hQ0t1ALeU4UamqFOuOg"
#   }
# }
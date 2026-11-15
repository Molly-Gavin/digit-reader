# PyTorch itself
import torch
# Toolbox for building models and measuring mistakes
from torch import nn
# Tool that serves images to the model in small groups
from torch.utils.data import DataLoader
# Tools for loading the digit images
from torchvision import datasets, transforms
# Our model from model.py
from model import DigitReader, DigitReaderCNN

# Use the Mac's graphics chip ("mps") if it's available, otherwise the regular processor ("cpu")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Training on:", device)

# Load the training and test images
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Serve the training images in batches of 64
# shuffle=True mixes up the order each round so the model can't memorize the sequence
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# Test images don't need shuffling; bigger batches make testing faster
test_loader = DataLoader(test_data, batch_size=1000)

# Create the model and move it onto the graphics chip
model = DigitReaderCNN().to(device)

# The loss function measures how wrong the model's guesses are
# A big number means very wrong; a small number means close to right
loss_fn = nn.CrossEntropyLoss()

# The optimizer is the part that actually nudges the dials
# lr (learning rate) controls how big each nudge is
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# One full pass through all 60,000 training images
def train_one_round():
    # Put the model in training mode
    model.train()
    # Go through the training images one batch of 64 at a time
    for images, labels in train_loader:
        # Move this batch onto the graphics chip
        images, labels = images.to(device), labels.to(device)
        # Step 1: the model makes its guesses
        scores = model(images)
        # Step 2: measure how wrong those guesses were
        loss = loss_fn(scores, labels)
        # Step 3: clear out the nudge directions from the last batch
        optimizer.zero_grad()
        # Step 4: work backward to find which way each dial should move
        loss.backward()
        # Step 5: nudge every dial a tiny bit in that direction
        optimizer.step()
    # Return the loss from the last batch so we can watch it shrink
    return loss.item()


# Quiz the model on the 10,000 test images it has never studied
def test():
    # Put the model in testing mode
    model.eval()
    # Start a count of correct answers
    correct = 0
    # no_grad() tells PyTorch not to track calculations, since we aren't learning here
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # Get the model's scores for this batch
            scores = model(images)
            # For each image, pick the digit with the highest score
            guesses = scores.argmax(dim=1)
            # Count how many guesses match the real answers
            correct += (guesses == labels).sum().item()
    # Turn the count into a percentage
    return correct / len(test_data) * 100


# Check the score before any training, as a starting point
print(f"Before training: {test():.1f}% correct")

# Train for 5 rounds (each full pass through the data is called an epoch)
for round_number in range(1, 6):
    loss = train_one_round()
    accuracy = test()
    print(f"Round {round_number}: loss {loss:.3f}, test accuracy {accuracy:.1f}%")

# Save the trained dial settings to a file so we can use them later without retraining
torch.save(model.state_dict(), "digit_reader_cnn.pt")
print("Saved the trained dials to digit_reader_cnn.pt")
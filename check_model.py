# Tools for loading the digit images
from torchvision import datasets, transforms
# Our model from model.py
from model import DigitReader

# Load the test images (already downloaded, so this is quick)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Create a brand new model; its dials start at random settings
model = DigitReader()

# Count every dial in the model
# parameters() gives each group of dials; numel() counts how many are in a group
total = sum(p.numel() for p in model.parameters())
print("Number of dials:", total)

# Get the first test image and its correct answer
image, label = test_data[0]

# Give the image to the model
# unsqueeze(0) wraps it as a batch of 1, because models expect a stack of images
scores = model(image.unsqueeze(0))

# Show the 10 scores (one for each digit, 0 through 9)
print("Scores:", scores)
# argmax() finds the position of the highest score; that position is the guess
print("Model guesses:", scores.argmax().item())
# The real answer, for comparison
print("Real answer:", label)
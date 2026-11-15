# PyTorch itself
import torch
# Tools for loading the digit images
from torchvision import datasets, transforms
# Tool for drawing pictures
import matplotlib.pyplot as plt
# Our model from model.py
from model import DigitReader, DigitReaderCNN
# Load the test images
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Create a fresh model (random dials)
model = DigitReaderCNN()
# Replace the random dials with the trained ones we saved
# map_location="cpu" loads them onto the regular processor, which is fine for this
model.load_state_dict(torch.load("digit_reader_cnn.pt", map_location="cpu"))
# Put the model in testing mode
model.eval()

# An empty list to collect every wrong answer
mistakes = []

# Don't track calculations, since we aren't learning
with torch.no_grad():
    # Go through all 10,000 test images one at a time
    for i in range(len(test_data)):
        image, label = test_data[i]
        # Get the model's guess for this one image
        guess = model(image.unsqueeze(0)).argmax().item()
        # If the guess is wrong, save the image, the guess, and the real answer
        if guess != label:
            mistakes.append((image, guess, label))

# Report the total number of mistakes
print("Mistakes:", len(mistakes), "out of", len(test_data))

# Make a grid of 3 rows and 4 columns
fig, axes = plt.subplots(3, 4, figsize=(7, 6))

# Show the first 12 mistakes; zip pairs each grid spot with one mistake
for ax, (image, guess, label) in zip(axes.flat, mistakes):
    ax.imshow(image.squeeze(), cmap="gray")
    # Show what the model guessed and what the answer really was
    ax.set_title(f"Guess {guess}, Real {label}")
    ax.axis("off")

# Space the pictures out and open the window
plt.tight_layout()
plt.show()
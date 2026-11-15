# Tools for downloading the digit images and converting them to numbers
from torchvision import datasets, transforms
# Tool for drawing pictures and charts
import matplotlib.pyplot as plt

# Download the 60,000 training images into the data/ folder
# train=True means "give me the training set"
# ToTensor() converts each picture into a grid of numbers from 0 to 1
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())

# Download the 10,000 test images (train=False means "give me the test set")
# The model never studies these; we use them later as a quiz
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# Print how many images are in each set
print("Training images:", len(train_data))
print("Test images:", len(test_data))

# Each item in the dataset is a pair: the image and its correct answer
image, label = train_data[0]

# Shape [1, 28, 28] means 1 color channel (gray) and 28 by 28 pixels
print("Shape of one image:", image.shape)
# The correct digit for this image
print("Label:", label)
# 0.0 is black, 1.0 is white
print("Smallest pixel value:", image.min().item())
print("Largest pixel value:", image.max().item())

# Make a grid of 3 rows and 4 columns for showing pictures
fig, axes = plt.subplots(3, 4, figsize=(6, 5))

# Loop through the 12 spots in the grid, numbering them 0 to 11
for i, ax in enumerate(axes.flat):
    # Get image number i and its label
    image, label = train_data[i]
    # squeeze() removes the color channel so matplotlib gets a plain 28 by 28 grid
    ax.imshow(image.squeeze(), cmap="gray")
    # Put the correct answer above the picture
    ax.set_title(f"Label: {label}")
    # Hide the axis numbers so only the digit shows
    ax.axis("off")

# Space the pictures so the titles don't overlap
plt.tight_layout()
# Open the window (the program waits here until you close it)
plt.show()
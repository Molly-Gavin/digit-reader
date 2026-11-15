from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Download the MNIST digits into the data/ folder
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# How many images are there?
print("Training images:", len(train_data))
print("Test images:", len(test_data))

# Look closely at one image
image, label = train_data[0]
print("Shape of one image:", image.shape)
print("Label:", label)
print("Smallest pixel value:", image.min().item())
print("Largest pixel value:", image.max().item())

# Show the first 12 images with their labels
fig, axes = plt.subplots(3, 4, figsize=(6, 5))
for i, ax in enumerate(axes.flat):
    image, label = train_data[i]
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.tight_layout()
plt.show()
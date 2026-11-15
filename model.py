# nn is PyTorch's toolbox for building neural networks (models)
from torch import nn


# MODEL 1: the simple model we already trained (97.7%)
class DigitReader(nn.Module):

    # This runs once, when we create the model
    def __init__(self):
        # Set up the parts PyTorch's model class needs behind the scenes
        super().__init__()

        # Sequential means "run these steps in order, passing each result to the next"
        self.layers = nn.Sequential(
            # Turn the 28 by 28 grid into one long list of 784 numbers
            nn.Flatten(),
            # Mix the 784 pixels together in 128 different ways
            nn.Linear(784, 128),
            # Keep positive numbers and change negatives to 0
            nn.ReLU(),
            # Turn the 128 pattern detectors into 10 scores, one per digit
            nn.Linear(128, 10),
        )

    # This runs every time we give the model an image
    def forward(self, x):
        return self.layers(x)


# MODEL 2: a convolutional neural network (CNN) that looks for shapes
class DigitReaderCNN(nn.Module):

    # This runs once, when we create the model
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # Slide 16 small 3 by 3 windows across the image
            # Each window learns to spot one simple shape, like an edge or a corner
            # padding=1 adds a 1 pixel border so the image stays 28 by 28
            # Result: 16 "shape maps," each 28 by 28
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # Keep positive numbers and change negatives to 0
            nn.ReLU(),
            # Shrink each map by half, keeping the strongest value in each 2 by 2 square
            # This helps the model care less about exactly where a shape is
            # Result: 16 maps, each 14 by 14
            nn.MaxPool2d(2),

            # Slide 32 new windows across those 16 maps
            # These combine simple shapes into bigger ones, like loops and curves
            # Result: 32 maps, each 14 by 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Shrink by half again
            # Result: 32 maps, each 7 by 7
            nn.MaxPool2d(2),

            # NOW we flatten: 32 maps × 7 × 7 = 1,568 numbers
            # By this point, the numbers describe shapes instead of single pixels
            nn.Flatten(),
            # Same idea as the simple model from here on
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            # 10 scores, one per digit
            nn.Linear(128, 10),
        )

    # This runs every time we give the model an image
    def forward(self, x):
        return self.layers(x)
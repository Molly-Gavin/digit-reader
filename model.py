# nn is PyTorch's toolbox for building neural networks (models)
from torch import nn

# Our model is a class, built on top of PyTorch's basic model class (nn.Module)
class DigitReader(nn.Module):

    # This runs once, when we create the model
    def __init__(self):
        # Set up the parts PyTorch's model class needs behind the scenes
        super().__init__()

        # Sequential means "run these steps in order, passing each result to the next"
        self.layers = nn.Sequential(
            # Step 1: turn the 28 by 28 grid into one long list of 784 numbers
            nn.Flatten(),
            # Step 2: mix the 784 pixels together in 128 different ways
            # Each of the 128 can learn to notice a pattern, like a loop or a line
            # Most of the model's dials live here
            nn.Linear(784, 128),
            # Step 3: keep positive numbers and change negatives to 0
            # Without this, the model could only learn very simple patterns
            nn.ReLU(),
            # Step 4: turn the 128 pattern detectors into 10 scores, one per digit
            # The highest score is the model's guess
            nn.Linear(128, 10),
        )

    # This runs every time we give the model an image
    def forward(self, x):
        # Send the image through all the steps above and return the 10 scores
        return self.layers(x)
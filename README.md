# Digit Reader

A machine learning project that reads handwritten digits (0 through 9) using PyTorch.

I built this as a practice project to learn how machine learning models work before building a car identification model for a larger app. It compares two kinds of models trained on the same data.

## Results

Both models were trained for 5 rounds on 60,000 images and tested on 10,000 images they had never seen.

| Model | Mistakes | Accuracy |
|---|---|---|
| Simple model | 232 | 97.7% |
| CNN | 101 | 99.0% |

The CNN cut the number of mistakes by more than half.

## Why the CNN does better

The simple model flattens each 28 by 28 image into one row of 784 numbers before doing anything else. After that, it no longer knows which pixels are next to each other. It learns which pixel positions tend to be bright for each digit, so a digit that is thicker or shifted over can confuse it.

The CNN (convolutional neural network) slides small windows across the image to find shapes like edges, curves, and loops. It only flattens the image at the end. Because it learns shapes instead of exact pixel positions, it handles different handwriting styles much better.

Looking at the mistakes showed the difference clearly. The simple model missed some digits that are easy for a person to read. Most of the CNN's remaining mistakes are digits that a person might also misread, such as a 4 with a closed top that looks like a 9.

## What I learned

- A model is code plus a large set of numbers (weights) that start random and get adjusted during training
- Training repeats the same loop: guess, measure how wrong the guess was, work backward, and adjust the weights
- Test data has to be kept separate from training data, or there is no way to know if the model learned or memorized
- Reviewing the mistakes tells you more than the accuracy number alone
- When most remaining mistakes are ones a person would also make, better data usually helps more than a better model

## Project files

| File | What it does |
|---|---|
| `explore.py` | Downloads the dataset and shows sample images |
| `model.py` | Defines both models |
| `check_model.py` | Shows that an untrained model guesses randomly |
| `train.py` | Trains a model and reports test accuracy after each round |
| `mistakes.py` | Shows images the trained model got wrong |

## How to run it

```bash
git clone https://github.com/Molly-Gavin/digit-reader.git
cd digit-reader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python mistakes.py
```

The dataset downloads automatically the first time you run a script. `train.py` is currently set to train the CNN. To train the simple model instead, change `DigitReaderCNN()` to `DigitReader()` and update the file name in the save line.

## Built with

- Python
- PyTorch and torchvision
- Matplotlib
- MNIST dataset
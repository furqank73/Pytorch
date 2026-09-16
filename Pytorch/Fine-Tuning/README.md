# Transfer Learning with VGG16 on Fashion-MNIST

This project fine-tunes a pretrained VGG16 image-classification model to recognize the 10 Fashion-MNIST classes.

The notebook is available here:

- [transfer_learning_fashion_mnist_pytorch_.ipynb](transfer_learning_fashion_mnist_pytorch_.ipynb)

## What Is the Goal?

Fashion-MNIST contains small grayscale images of clothing items such as shirts, shoes, and bags. Each image is `28 x 28` pixels and has a label from `0` to `9`.

The notebook uses **transfer learning**:

1. Start with VGG16, a model that has already learned general visual patterns.
2. Freeze its convolutional feature extractor.
3. Replace its original ImageNet classifier with a classifier for 10 Fashion-MNIST classes.
4. Train the new classifier on the Fashion-MNIST images.
5. Measure accuracy and use the model to make predictions.

## How the Process Works

```text
CSV file
   |
   v
Separate pixels and labels
   |
   v
Split into training and test data
   |
   v
Convert 28 x 28 grayscale images to 3-channel images
   |
   v
Resize to 224 x 224 and normalize
   |
   v
Create PyTorch datasets and data loaders
   |
   v
Load pretrained VGG16
   |
   v
Freeze features and replace classifier
   |
   v
Train, evaluate, save, and predict
```

### 1. Load and inspect the data

Pandas reads the CSV file. The first column contains the label, and the other 784 columns contain the flattened pixels.

The notebook displays sample images so we can check that the data was loaded correctly.

### 2. Split the data

The examples are divided into training and test sets:

- Training data teaches the model.
- Test data checks performance on images the model did not see during training.

### 3. Prepare the images

VGG16 was designed for ImageNet images, not 28 x 28 grayscale images. The custom dataset therefore:

- Reshapes 784 pixels into a `28 x 28` image.
- Converts pixel values to `uint8`.
- Copies the grayscale channel three times to create RGB data.
- Converts the array to a PIL image.
- Resizes and center-crops it to `224 x 224`.
- Converts it to a tensor and applies ImageNet normalization.

The final image tensor has this shape:

```text
[3, 224, 224]
```

A batch of 32 images has this shape:

```text
[32, 3, 224, 224]
```

### 4. Build datasets and batches

`CustomDataset` tells PyTorch how to retrieve one image and its label. `DataLoader` groups these examples into batches for efficient training.

Training batches are shuffled. Test batches are not shuffled because their order does not need to change during evaluation.

### 5. Apply transfer learning

VGG16's convolutional layers are used as a feature extractor. Their parameters are frozen, so training does not change them.

The original VGG16 classifier predicts 1,000 ImageNet classes. It is replaced with a classifier that produces 10 output scores, one for each Fashion-MNIST class.

### 6. Train the model

For every batch, the notebook:

1. Sends images and labels to the selected device.
2. Runs a forward pass through VGG16.
3. Calculates classification loss with `CrossEntropyLoss`.
4. Computes gradients with backpropagation.
5. Updates the trainable classifier weights with AdamW.

This repeats for 10 epochs.

### 7. Evaluate and make predictions

The notebook calculates accuracy on both the training and test sets. Comparing these values can reveal overfitting.

It then predicts individual test images and displays the actual and predicted labels.

## Requirements

Install the main dependencies in the Python environment used by VS Code:

```bash
pip install torch torchvision pandas scikit-learn matplotlib pillow
```

A CUDA-enabled PyTorch installation can be used when an NVIDIA GPU is available. Otherwise, the notebook automatically uses the CPU.

## Run Locally

1. Open `transfer_learning_fashion_mnist_pytorch_.ipynb` in VS Code.
1. Select a Python kernel with the required packages installed.
1. Make sure `fashion-mnist_test.csv` is in this folder.
1. Update the CSV loading cell from the Colab path:

```python
df = pd.read_csv("/content/sample_data/mnist_test.csv")
```

to the local file path:

```python
df = pd.read_csv("fashion-mnist_test.csv")
```

1. Run the cells from top to bottom.

## Important Notes

- The notebook currently uses only the available test CSV and splits it into training and test portions for demonstration.
- For a proper experiment, use the separate Fashion-MNIST training file for training and keep the test file untouched for final evaluation.
- The saved model weights are written to `vgg16_mnist.pth` after training.
- The pretrained VGG16 weights may need to be downloaded the first time the model cell runs.

## Fashion-MNIST Labels

| Label | Class |
| ---: | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

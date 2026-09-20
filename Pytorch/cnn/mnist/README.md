# Fashion-MNIST CNN Classification

This project uses PyTorch to teach a computer how to recognize clothing images. Each Fashion-MNIST image is a small, grayscale `28 x 28` picture, and the model predicts which of 10 clothing categories it belongs to.

The notebook currently loads `fashion-mnist_train.csv`, uses 80% of those examples for training, and holds out the remaining 20% for evaluation. Pixel values are normalized from `0-255` to `0-1` and reshaped from 784 numbers into `1 x 28 x 28` image tensors.

## How It Works

The process follows these steps:

1. **Load the data:** Read the image pixels and their labels from the CSV file.
2. **Prepare the images:** Scale pixel values to `0-1`, then reshape each row into a grayscale image.
3. **Learn visual features:** Convolution layers look for simple patterns such as edges and gradually combine them into more useful shapes.
4. **Reduce the image size:** Max pooling keeps the strongest features while reducing the amount of information the model must process.
5. **Make a prediction:** Fully connected layers use the learned features to choose one of the 10 clothing categories.
6. **Measure performance:** The model is evaluated by comparing its predicted class with the correct label.

## Model Explanation

The model is a convolutional neural network (CNN). A CNN is useful for images because it considers nearby pixels together instead of treating every pixel as unrelated data.

- **First convolutional block:** Creates 32 feature maps to detect basic patterns such as edges and lines.
- **Second convolutional block:** Creates 64 feature maps to detect more complex patterns by combining the earlier features.
- **ReLU:** Keeps positive signals and adds non-linearity, allowing the network to learn more than simple linear relationships.
- **Batch normalization:** Helps keep the values flowing through the network stable during training.
- **Max pooling:** Reduces each feature map from `28 x 28` to `14 x 14`, and then to `7 x 7`, while keeping strong signals.
- **Flattening:** Converts the `64 x 7 x 7` feature maps into a single vector of 3,136 values.
- **Dense classifier:** Reduces those 3,136 values to 128, then 64, and finally 10 output scores.
- **Dropout:** Randomly ignores 40% of some neurons during training so the model is less likely to memorize the training images.

The output contains one score for each clothing class. The class with the highest score becomes the model's prediction.

## Training

The network is trained for five epochs using:

- **Cross-entropy loss:** Measures how far the prediction is from the correct class.
- **Stochastic Gradient Descent (SGD):** Adjusts the model's weights to reduce the loss.
- **Learning rate:** `0.001`
- **Batch size:** `32`

After training, the notebook calculates accuracy on the training split and the held-out evaluation split.

## Files

- `mnistdataset.ipynb` - Data preparation, CNN definition, training, and evaluation workflow
- `fashion-mnist_train.csv` - Fashion-MNIST training data
- `fashion-mnist_test.csv` - Additional Fashion-MNIST test data included in the folder; it is not currently loaded by the notebook
- `fashion_mnist_cnn.pth` - Saved CNN model weights

## Fashion-MNIST Classes

The labels represent: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

# Facial Expression Recognition with PyTorch

This project uses a Convolutional Neural Network (CNN) to classify human facial expressions into 7 categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The notebook in this folder trains and evaluates a model that learns to recognize expressions from facial images. It uses PyTorch and the `torchvision` library for data loading, image preprocessing, and model training.

---

## Project Goal

The main purpose is to build a model that can look at a 48x48 RGB face image and predict the emotion shown in that face. This is a classic image classification task, and CNNs are well-suited for it because they detect important visual patterns such as:

- eye shape and position
- mouth curvature
- eyebrow movement
- overall face structure

---

## Dataset

The notebook extracts a dataset zip file and loads it using `ImageFolder` from `torchvision.datasets`.

The dataset is organized in folders like this:

```text
/extracted_data/
    train/
        Angry/
        Disgust/
        Fear/
        Happy/
        Neutral/
        Sad/
        Surprise/
    val/
    test/
```

Each folder contains images belonging to one emotion class.

---

## Data Preprocessing

Before training, the images are resized and prepared for the model:

```python
train_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
```

### Why this is important

- `Resize((48, 48))`: standardizes image size
- `RandomHorizontalFlip()`: helps the model learn more robustly by flipping images
- `RandomRotation(10)`: adds small variations to improve generalization
- `ToTensor()`: converts images from PIL array format to PyTorch tensors

The validation and test datasets use a simpler transform without augmentation so the evaluation is consistent.

---

## Data Loaders

The notebook creates PyTorch `DataLoader` objects:

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

These loaders feed batches of images into the model during training and evaluation.

- `batch_size=32`: processes 32 images at a time
- `shuffle=True` for training: makes the model learn without order bias
- `shuffle=False` for validation/test: keeps evaluation stable and reproducible

---

## CNN Model Architecture

The model is defined as a custom class named `EmotionCNN`.

### Model structure

```python
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7)
        )
```

### What each part does

- `Conv2d`: learns edges, textures, and facial features
- `BatchNorm2d`: stabilizes training and speeds up learning
- `ReLU`: introduces non-linearity
- `MaxPool2d`: reduces spatial size and keeps important features
- `Flatten`: converts 2D feature maps into 1D vectors
- `Linear`: maps features into 7 emotion classes
- `Dropout`: prevents overfitting

The final output has 7 neurons, one for each emotion class.

---

## Training Process

Training is performed using:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### Loss function

`CrossEntropyLoss` is used because this is a multi-class classification problem. It compares the model's predicted probabilities with the true emotion label.

### Optimizer

`Adam` is used as the optimization algorithm. It updates the network weights in a stable and efficient manner.

### Training loop

For each epoch:

1. Set the model to training mode
2. Load a batch of images and labels
3. Clear old gradients
4. Feed the batch into the model
5. Compute the loss
6. Backpropagate the error
7. Update the weights
8. Track training accuracy

```python
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Validation and Accuracy

After training, the model is evaluated on the validation set.

```python
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
```

The script measures validation accuracy using:

```python
val_acc = 100 * correct / total
```

This tells us how often the model predicts the correct emotion on unseen validation images.

---

## Saving the Model

The trained model parameters are saved to a file:

```python
torch.save(model.state_dict(), "emotion_cnn.pth")
```

This allows the model to be reloaded later for inference without retraining.

---

## Prediction on a New Image

The notebook also shows how to test the trained model on a single image.

### Steps

1. Open a sample face image using PIL
2. Resize it to `48x48`
3. Convert it to a tensor
4. Add a batch dimension
5. Send it to the GPU/CPU
6. Run the model
7. Use `torch.argmax` to get the predicted class

```python
output = model(image_tensor)
predicted_class = torch.argmax(output, dim=1).item()
print(class_names[predicted_class])
```

The `class_names` list comes from the dataset folder names, so it maps the numeric prediction back to a human-readable emotion.

---

## Interpretation of Results

The model output is a vector of 7 values, one per emotion class. The highest value indicates the class the model believes is most likely.

Example:

```python
probabilities = F.softmax(output, dim=1)
confidence, predicted_class = torch.max(probabilities, dim=1)
```

This gives both:

- the predicted emotion
- the confidence score for that prediction

---

## Notes for Running the Notebook

### Environment

This project requires:

- Python
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- PIL (Pillow)
- scikit-learn
- pandas

### Important path note

The notebook currently uses Google Colab-style paths such as:

```python
zip_file_path = '/content/archive (9).zip'
extract_dir = '/content/extracted_data'
```

If you run it locally, replace these with your own local dataset path.

---

## Summary

This project demonstrates a complete PyTorch workflow for image classification:

- load image data
- transform and augment it
- build a CNN model
- train on labeled emotions
- validate performance
- save the trained model
- predict emotion on new images

It is a practical example of how deep learning can be applied to real-world computer vision tasks such as facial expression recognition.

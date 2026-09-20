# Notebook Summary (TensorFlow + PyTorch)

This notebook teaches **how image datasets are loaded, processed, augmented, and converted into batches for CNN training**. 

---

# Part 1: TensorFlow / Keras Workflow

### 1. Download Dataset

* Download agricultural and non-agricultural image dataset.
* Extract `.tar` file.
* Dataset structure:

```text
images_dataSAT/
├── class_0_non_agri/
└── class_1_agri/
```



---

### 2. Collect Image Paths

Instead of loading all images into RAM:

```text
Store paths first
↓
Load images when needed
```

Example:

```python
all_image_paths
all_labels
```

Labels:

```text
Non-agri → 0
Agri     → 1
```



---

### 3. Shuffle Dataset

```python
zip(paths, labels)
shuffle()
```

Important:

```text
Image and label stay together
```



---

### 4. Custom Generator

Generator loads images batch-by-batch:

```text
Paths
 ↓
Load 8 images
 ↓
Return batch
```

Inside generator:

```python
load_img()
img_to_array()
/255.0
yield
```

Image resize:

```text
Original
 ↓
64×64
```

Normalization:

```text
0-255
 ↓
0-1
```



---

### 5. Keras Dataset Loader

Instead of writing generators manually:

```python
image_dataset_from_directory()
```

Keras automatically:

* Finds images
* Creates labels
* Creates batches

Split:

```text
80% Training
20% Validation
```



---

### 6. Data Augmentation

Creates new image variations:

```text
Original Image
 ↓
Flip
 ↓
Rotate
```

Used layers:

```python
RandomFlip()
RandomRotation()
```

Benefit:

```text
More robust CNN
Less overfitting
```



---

### 7. Cache and Prefetch

Cache:

```python
train_ds.cache()
```

Stores processed data in memory.

Prefetch:

```python
train_ds.prefetch()
```

While model trains on current batch:

```text
CPU prepares next batch
```

Result:

```text
Faster training
```



---

# Part 2: PyTorch Workflow

### 1. Custom Dataset

PyTorch datasets need:

```python
__init__()
__len__()
__getitem__()
```

Flow:

```text
Dataset
 ↓
One Image + Label
 ↓
DataLoader
 ↓
Batch
```



---

### 2. CustomDataset Class

Responsibilities:

```text
Store image paths
Store labels
Load image when requested
Apply transforms
Return image,label
```

Label mapping:

```text
Non-agri → 0
Agri → 1
```



---

### 3. Transformations

Pipeline:

```text
Image
 ↓
Resize
 ↓
Flip
 ↓
Rotate
 ↓
Tensor
 ↓
Normalize
```

Code:

```python
Resize()
RandomHorizontalFlip()
RandomVerticalFlip()
RandomRotation()
ToTensor()
Normalize()
```



---

### 4. ImageFolder

Instead of creating a custom dataset:

```python
datasets.ImageFolder()
```

Automatically:

```text
Reads folders
Creates labels
Loads images
```

Folder structure must be:

```text
root/
├── class1/
└── class2/
```



---

### 5. DataLoader

Converts dataset into batches.

```python
DataLoader()
```

Example:

```text
Dataset
 ↓
DataLoader
 ↓
8 Images
 +
8 Labels
```



---

### 6. Tensor Shapes

Very important interview question.

TensorFlow:

```text
[B, H, W, C]
```

Example:

```text
[8, 64, 64, 3]
```

PyTorch:

```text
[B, C, H, W]
```

Example:

```text
[8, 3, 64, 64]
```



---

# Custom Dataset vs ImageFolder

| Custom Dataset        | ImageFolder             |
| --------------------- | ----------------------- |
| More flexible         | Easier                  |
| More code             | Less code               |
| Best for complex data | Best for normal folders |
| Manual labels         | Automatic labels        |



# Final 5 Things to Remember

1. **Dataset Structure**

```text
root/
 ├── class1/
 └── class2/
```

2. **Image + Label Pairing**

```text
Image → Label
```

3. **Batching**

```text
Single Images
 ↓
Batch
```

4. **Tensor Shapes**

```text
TensorFlow → [B,H,W,C]
PyTorch    → [B,C,H,W]
```

5. **CustomDataset vs ImageFolder**

```text
CustomDataset = Flexible
ImageFolder   = Easy
```

These 5 concepts are the core lessons of the entire notebook. 

# RNN-Based Question Answering System

This project is a simple PyTorch NLP example that teaches how to build a question-answering model using a Recurrent Neural Network (RNN). The notebook reads a small dataset of question-answer pairs, converts the text into numbers, trains a neural network, and predicts a short answer for a new question.

---

## Project Goal

The goal is to show the basic flow of a text-based deep learning model:

- convert text into tokens
- build a vocabulary
- map words to integer IDs
- use embeddings to represent words numerically
- feed the sequence into an RNN
- predict the answer word

This is a simplified QA system, not a full chatbot or deep reading comprehension model.

---

## Dataset

The dataset is stored in:

- `general_knowledge_qa.csv`

It contains columns such as:

- `question`
- `answer`

Example rows:

```text
What is the capital of France? | Paris
What is the largest planet? | Jupiter
Who discovered gravity? | Newton
```

The notebook uses these pairs to train the model.

---

## Workflow

### 1. Tokenization

The text is converted into words using a simple tokenizer:

```python
def tokenize(text):
    text = text.lower()
    text = text.replace("?", "")
    text = text.replace("'", "")
    return text.split()
```

This turns a sentence like:

```text
What is the capital of France?
```

into:

```python
["what", "is", "the", "capital", "of", "france"]
```

---

### 2. Vocabulary Building

The notebook creates a vocabulary dictionary where each unique token gets a number.

```python
vocab = {'<UNK>': 0}
```

For example:

```python
{
    '<UNK>': 0,
    'what': 1,
    'is': 2,
    'capital': 3,
    'france': 4,
    ...
}
```

This allows the model to work with integer IDs instead of raw text.

---

### 3. Text to Indices

Each word in a sentence is replaced by its vocabulary ID.

```python
def text_to_indices(text, vocab):
    indexed_text = []
    for token in tokenize(text):
        if token in vocab:
            indexed_text.append(vocab[token])
        else:
            indexed_text.append(vocab['<UNK>'])
    return indexed_text
```

Example:

```python
text_to_indices("How many days are in a week?", vocab)
```

returns something like:

```python
[14, 25, 7, 12, 3, 9]
```

---

### 4. Custom Dataset

A custom `Dataset` is created so each question-answer pair becomes a PyTorch training sample.

```python
class qadataset(Dataset):
    def __getitem__(self, index):
        question = text_to_indices(...)
        answer = text_to_indices(...)
        return torch.tensor(question), torch.tensor(answer)
```

This prepares the data for the data loader.

---

### 5. DataLoader

The notebook wraps the dataset in a `DataLoader`:

```python
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
```

This allows batches of question-answer pairs to be fed into the model during training.

---

## RNN Model Architecture

The model is a simple recurrent neural network built with PyTorch.

```python
class simplernn(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=50)
        self.rnn = nn.RNN(50, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)
```

### What each layer does

- `nn.Embedding`: converts word IDs into dense vector representations
- `nn.RNN`: processes the word sequence and keeps memory of past words
- `nn.Linear`: transforms the final hidden state into output scores for every vocabulary word

The network learns to choose the most likely answer word from the vocabulary.

---

## Training Process

The model is trained using:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

During each epoch:

1. the question is passed through the model
2. the output is compared to the true answer
3. loss is calculated
4. gradients are computed with backpropagation
5. weights are updated

```python
for epoch in range(epochs):
    for question, answer in dataloader:
        optimizer.zero_grad()
        output = model(question)
        loss = criterion(output, answer[:, 0])
        loss.backward()
        optimizer.step()
```

---

## Prediction

After training, the model can answer a new question:

```python
predict(model, "capital of japan")
```

It converts the text into indices, sends it to the model, and chooses the word with the highest probability.

---

## Important Note

This is a simplified QA model. It is designed to predict a single answer word, such as:

```text
Question: What is the capital of France?
Answer: Paris
```

It does not generate long, natural-sentence answers like a modern transformer or chatbot. For that, you would need more advanced architectures like:

- LSTM
- Seq2Seq models
- attention-based networks
- BERT or transformer models

---

## What You Learn from This Project

This notebook demonstrates the fundamentals of NLP with PyTorch:

- tokenization
- vocabulary creation
- integer encoding
- embeddings
- recurrent neural networks
- model training with PyTorch
- prediction from text input

---

## Summary

This project is a beginner-friendly introduction to building a text-based deep learning model in PyTorch. It shows how a question can be transformed into numbers, processed by an RNN, and used to predict an answer word.

It is a great starting point for learning how NLP models work before moving to more advanced architectures like LSTM, attention, and transformers.

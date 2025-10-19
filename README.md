# 🧠 Next Word Predictor using PyTorch RNN

This project implements a **Next Word Prediction** model trained on a custom text dataset using a **Recurrent Neural Network (RNN)** built with PyTorch.  
The goal is to predict the most likely next word in a given sequence, demonstrating language modeling and sequence prediction concepts.

---

## 🚀 Features
- Preprocesses and tokenizes raw text data  
- Builds a custom vocabulary dynamically  
- Implements a PyTorch-based Recurrent Neural Network (RNN)  
- Trains the model using cross-entropy loss  
- Predicts the next possible word(s) given a sentence  
- Supports GPU (CUDA) acceleration for faster training  
- Provides clean and reusable modular code structure  

---

## 📂 Dataset
The dataset (`next_word_predictor.txt`) contains plain English text used to train the model for sequence-based next-word prediction.  
Each line or paragraph contributes to building the word vocabulary and learning contextual patterns.

**Example Data Snippet:**
```
I love to eat apples
I love to play cricket
I love to read books
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ashsus09/Next-Word-Predictor-PyTorch-RNN.git
cd next-word-predictor
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Training Script
```bash
python train.py
```

**requirements.txt**
```
torch
numpy
pandas
matplotlib
```

---

## 🧠 Model Architecture

| Layer | Description |
|-------|--------------|
| **Embedding Layer** | Converts words into dense vector representations |
| **RNN Layer** | Learns sequential relationships between words |
| **Fully Connected Layer** | Outputs probabilities for each word in the vocabulary |

---

## 📊 Training Summary

| Epoch | Loss | Observation |
|--------|-------|-------------|
| 1 | 7.10 | Model begins random predictions |
| 10 | 3.65 | Learns basic structure |
| 20 | 2.22 | Recognizes frequent word patterns |
| 30 | 1.68 | Predicts contextually accurate next words |

---

## 🤖 Prediction Examples

**Input → Output Predictions**
```
"the sun was shining" → ['brightly', 'through', 'over']
"deep in the forest" → ['darkness', 'silence', 'trees']
"quantum computing is" → ['revolutionizing', 'changing', 'advancing']
"she walked into the" → ['room', 'house', 'city']
```

---

## 🧩 Model Evaluation

- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam (lr=0.001)  
- **Gradient Clipping:** max_norm = 5 (to prevent exploding gradients)  
- **Training Device:** CPU / GPU (Auto-detected)

---

## 🔮 Future Improvements

- Upgrade model to **LSTM or GRU** for improved long-term dependency learning  
- Integrate pre-trained embeddings like **GloVe** or **Word2Vec**  
- Build a **Streamlit web interface** for interactive word prediction  
- Expand training dataset for richer vocabulary and better generalization  
- Compare with Transformer-based models (e.g., GPT-like architecture)

---

## 🧰 Tech Stack

**Language:** Python  
**Framework:** PyTorch  
**Tools:** NumPy, Pandas, Matplotlib  
**Model:** Recurrent Neural Network (RNN)

---

## 👨‍💻 Author

**Aastha**  
GitHub: @ashsus09

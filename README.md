# 🧠 TamilEcho @ DravidianLangTech 2026  
## Hybrid XLM-RoBERTa with Sarcasm-Aware Feature Fusion for Political Multiclass Sentiment Analysis in Tamil X (Twitter) Comments

![GitHub repo](https://img.shields.io/badge/GitHub-TamilEcho-blue?logo=github)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Transformers](https://img.shields.io/badge/🤗_Transformers-HuggingFace-yellow?logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?logo=scikit-learn)
![DravidianLangTech](https://img.shields.io/badge/DravidianLangTech-2026-ff69b4)

🚀 **System submitted to the DravidianLangTech 2026 Shared Task**

---

# 📌 Overview

This repository contains the implementation of **TamilEcho**, our system for **Political Multiclass Sentiment Analysis in Tamil X (Twitter) comments**.

Our approach combines:

- **XLM-RoBERTa contextual embeddings**
- **TF-IDF lexical features**
- **Emoji-based sarcasm features**

These features are **fused together in a hybrid neural architecture** to improve sentiment classification performance.

### 🔹 Key Highlights

- Hybrid deep learning architecture  
- Contextual embeddings using **XLM-RoBERTa**  
- Lexical representation using **TF-IDF + SVD**  
- Sarcasm-aware **emoji feature extraction**  
- Class imbalance handled using **weighted loss**

---

# 🏛 Methodology

## ✨ Preprocessing

Several preprocessing steps are applied to enhance tweet representation.

### 🔹 Hashtag Expansion

Political hashtags are expanded using a **knowledge base dictionary**.

Example:

```
#DMK → Dravida Munnetra Kazhagam ruling party Tamil Nadu
#BJP → Bharatiya Janata Party Indian national political party
#NTK → Naam Tamilar Katchi Tamil nationalist political party
```

### 🔹 Emoji Feature Extraction

Emojis are used to capture sarcasm and emotional tone.

We extract two features:

- Total emoji count  
- Sarcastic emoji count

Example sarcasm emojis:

```
😂 🤣 😏 🙃 🤡 🤦 🤷 🙄 💀 🔥
```

---

# 🔥 Feature Extraction

## 1️⃣ TF-IDF Features

TF-IDF captures important words and phrases from tweets.

Parameters used:

```
max_features = 8000
ngram_range = (1,2)
min_df = 2
```

## 2️⃣ Dimensionality Reduction

TF-IDF features are reduced using **Truncated SVD**.

```
TF-IDF → 256 dimensional vector
```

---

# ⚙️ Model Architecture

The hybrid architecture combines multiple feature representations.

### Input Features

1. **XLM-RoBERTa CLS embedding** (768)  
2. **Mean pooled token embeddings** (768)  
3. **TF-IDF + SVD features** (256)  
4. **Emoji sarcasm features** (2)

These features are concatenated and passed to a classifier.

### Architecture Flow

```
Tweet Text
   │
XLM-RoBERTa Encoder
   │
 ┌───────────────┐
 CLS Embedding   Mean Pooling
 └───────────────┘
        │
 TF-IDF + SVD Features
        │
 Emoji Sarcasm Features
        │
 Feature Concatenation
        │
 Fully Connected Layer
        │
 Softmax Classification
```

---

# 🧪 Training Strategy

### Optimizer

```
AdamW
Learning Rate = 2e-5
Weight Decay = 0.01
```

### Loss Function

```
Weighted Cross Entropy Loss
Label Smoothing = 0.1
```

### Additional Techniques

- Progressive **layer unfreezing**
- **Gradient clipping**
- **Class weighting** for imbalanced data

---

# 📊 Results

## Model Variants

| Model | Macro-F1 |
|------|------|
| XLM-RoBERTa Only | 0.331 |
| XLM-RoBERTa + TF-IDF | 0.346 |
| **XLM-RoBERTa + TF-IDF + Emoji** | **0.356** |

### Development Set Performance

Macro-F1 Score: **0.397**

| Class | F1 Score |
|------|------|
| Negative | 0.25 |
| Neutral | 0.22 |
| None of the above | 0.95 |
| Opinionated | 0.45 |
| Positive | 0.36 |
| Sarcastic | 0.43 |
| Substantiated | 0.12 |

---

# 📂 Dataset

The dataset contains **Tamil political tweets** annotated into **seven sentiment classes**:

1. Positive  
2. Negative  
3. Sarcastic  
4. Opinionated  
5. Substantiated  
6. Neutral  
7. None of the above  

Dataset files:

```
PS_train.csv
PS_dev.csv
PS_test_without_labels.csv
```

---

# 📦 Installation

Install dependencies:

```bash
pip install -q transformers sentencepiece torch scikit-learn emoji accelerate pandas numpy
```

---

## ▶️ Running the Model (Google Colab)

Open the notebook **Hybrid_XLMR_TFIDF_Emoji.ipynb** in **Google Colab** and install the required libraries by running `!pip install -q transformers sentencepiece torch scikit-learn emoji accelerate pandas numpy`. Since the dataset is stored in Google Drive, mount your drive using `from google.colab import drive` and `drive.mount('/content/drive')`. After mounting the drive, run all the cells in the notebook sequentially to load the dataset, preprocess the tweets, train the **Hybrid XLM-RoBERTa + TF-IDF + Emoji feature fusion model**, evaluate the model on the development dataset, and finally generate predictions for the test dataset. The final predictions will be saved as **submission.csv**.

---

# 🛠 Dependencies

Main libraries used:

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas
- NumPy
- Emoji

---

# 👩‍💻 Authors

**Kanimozhi Selvi C S**  
**Inigashree N S**  
**Moneissh A G**  
**Kavinraj J**

Department of Computer Science and Engineering  
Kongu Engineering College  
Tamil Nadu, India

---

# 📄 Paper

**TamilEcho: Hybrid XLM-RoBERTa with Sarcasm-Aware Feature Fusion for Political Multiclass Sentiment Analysis in Tamil X (Twitter) Comments**

Submitted to:

**DravidianLangTech 2026 Workshop**

---

# 🧠 Mental Health Text Classifier — Team MARS

> \*\*CSS 324 · NLP Project\*\*  
> Fine-tuned MentalBERT for 7-class mental health detection from social media text, deployed as a Gradio web app.

\---



\## Installation



\# CPU (general use)

pip install -r requirements.txt



\# GPU (CUDA 12.4)

pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements-full.txt





## 📋 Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Classes](#classes)
* [Pipeline](#pipeline)
* [Models \& Results](#models--results)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Team](#team)

\---

## Overview

MARS is a mental health text classification system that detects psychological conditions from short social media posts. Given a piece of text, the model returns one of seven mental health categories along with a risk level and per-class confidence scores.

**Key design constraints:**

* Suicidal recall ≥ **0.90** (safety-critical)
* Depression recall ≥ **0.85**
* Maximize macro F1 across all 7 classes

The final deployed model is a fine-tuned **MentalBERT** ([mental-bert-base-uncased](https://huggingface.co/mental/mental-bert-base-uncased)), a domain-specific BERT pre-trained on mental health corpora.

\---

## Dataset

|Property|Value|
|-|-|
|Source|Combined social media posts (Reddit, Twitter)|
|Total samples|\~53,000|
|Classes|7|
|Format|`statement` (text) + `status` (label)|

The dataset is **class-imbalanced** — Normal and Depression dominate. SMOTE oversampling and class-weighted loss are used to compensate.

\---

## Classes

|Label|Risk Level|Colour|
|-|-|-|
|🟢 Normal|Low|Green|
|🟡 Anxiety|Medium|Yellow|
|🟡 Stress|Medium|Yellow|
|🟡 Bipolar|Medium|Yellow|
|🟡 Personality Disorder|Medium|Yellow|
|🔴 Depression|High|Red|
|🔴 Suicidal|High|Red|

\---

## Pipeline

```
Raw Text
   │
   ▼
Text Cleaning (8 steps)
   ├─ Remove Reddit artifacts (r/subreddit, u/user, deleted posts)
   ├─ Strip HTML, URLs, emails
   ├─ Unicode normalization + emoji removal
   ├─ Special character cleaning
   ├─ Duplicate removal
   └─ Short post filtering (< 5 words)
   │
   ▼
Feature Engineering
   ├─ TF-IDF (20k features, unigrams + bigrams)
   ├─ VADER \& TextBlob sentiment scores
   ├─ Psycholinguistic features (first-person pronouns, absolutist words, negations)
   ├─ Suicide-specific keyword flags
   └─ SBERT / MentalBERT embeddings
   │
   ▼
Model Training
   ├─ Baselines: Logistic Regression, SVM (TF-IDF)
   ├─ Embeddings: SBERT + LogReg, MentalBERT embeddings + SVM
   └─ Fine-tuned MentalBERT (HuggingFace Trainer, weighted cross-entropy)
   │
   ▼
Threshold Tuning
   └─ Per-class thresholds tuned to meet recall targets
   │
   ▼
Gradio Deployment
```

\---

## Models \& Results

|Model|Macro F1|Suicidal Recall|Depression Recall|Passes Constraints|
|-|-|-|-|-|
|LogReg (TF-IDF)|0.71|0.82|0.79|❌|
|SVM (TF-IDF)|0.73|0.85|0.81|❌|
|SBERT + LogReg|0.78|0.88|0.84|❌|
|MentalBERT Embeddings|0.80|0.91|0.86|✅|
|**Fine-tuned MentalBERT**|**0.87**|**0.94**|**0.91**|✅|

> ✅ = Suicidal recall ≥ 0.90 \*\*and\*\* Depression recall ≥ 0.85

\---

## Installation

### Requirements

* Python 3.9+
* CUDA GPU recommended for fine-tuning (inference runs on CPU)

```bash
git clone https://github.com/your-org/mental-health-mars.git
cd mental-health-mars

pip install -r requirements.txt
```

**`requirements.txt`**

```
gradio
transformers
torch
sentencepiece
```

### Download the model

The fine-tuned model weights (`mental\_bert\_final/`) must be present in the project root. Either train it yourself using the notebook, or download the checkpoint from the releases page.

\---

## Usage

### Web App (Gradio)

```bash
python app.py
```

Open `http://localhost:7860` in your browser. Type or paste any text and click **Анализировать →**.

The app displays:

* **Top predicted class** with risk level and confidence
* **Progress bars** for all 7 classes

### Inference in Python

```python
from transformers import pipeline

LABEL\_MAP = {
    "LABEL\_0": "Anxiety",
    "LABEL\_1": "Bipolar",
    "LABEL\_2": "Depression",
    "LABEL\_3": "Normal",
    "LABEL\_4": "Personality Disorder",
    "LABEL\_5": "Stress",
    "LABEL\_6": "Suicidal",
}

classifier = pipeline("text-classification", model="mental\_bert\_final", top\_k=None)

text = "I've been feeling really overwhelmed and can't sleep."
results = sorted(classifier(text)\[0], key=lambda x: x\["score"], reverse=True)
top\_label = LABEL\_MAP\[results\[0]\["label"]]
print(f"Prediction: {top\_label} ({results\[0]\['score']:.1%})")
```

### Training (Notebook)

Open `mental\_health\_MARS\_v3.ipynb` and run cells sequentially. The notebook covers:

1. Data loading \& EDA
2. Text cleaning pipeline
3. Feature engineering (TF-IDF, VADER, SBERT, MentalBERT embeddings)
4. Baseline models (LogReg, SVM, Random Forest, LightGBM)
5. Neural models (MLP, CNN, fine-tuned MentalBERT)
6. Threshold tuning for recall-first strategy
7. Binary classifier for Depression vs. Suicidal disambiguation
8. SHAP feature importance
9. Model saving \& leaderboard

\---

## Project Structure

```
mental-health-mars/
├── app.py                        # Gradio web app
├── mental\_health\_MARS\_v3.ipynb   # Full training notebook
├── Combined\_Data.csv             # Raw dataset
├── mental\_bert\_final/            # Fine-tuned model weights
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
├── saved\_models/                 # Sklearn model artifacts
│   ├── tfidf.pkl
│   ├── label\_encoder.pkl
│   └── leaderboard.csv
├── requirements.txt
└── README.md
```

\---

## ⚠️ Disclaimer

This tool is a **research prototype** for an academic course. It is **not** a substitute for professional mental health evaluation. If you or someone you know is in crisis:

* **Russia:** 8-800-2000-122 (free, 24/7)
* **International:** [findahelpline.com](https://findahelpline.com)

\---

## Team

**Team MARS** — CSS 324, 2024–2025

\---

## License

MIT License — see `LICENSE` for details.


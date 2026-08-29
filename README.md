# RoBERTa Sentiment Analysis

A lightweight web application that classifies English text as **Negative**, **Neutral**, or **Positive** using a pretrained Transformer model.

The application runs locally with Streamlit and displays the predicted sentiment, confidence score, and probability for every class.

## Features

- Three-class sentiment classification
- Confidence score for the predicted sentiment
- Probability breakdown for all sentiment classes
- Preprocessing for usernames and URLs
- Automatic CPU or CUDA device selection
- Cached model loading for faster repeated predictions
- Simple Streamlit interface

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- RoBERTa

## Model

This project uses the pretrained [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) model.

The model returns three sentiment classes:

| Class ID | Sentiment |
| --- | --- |
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

This repository does not train or fine-tune the model. It integrates the published model for local inference.

## Application Workflow

```text
User text
   ↓
Text preprocessing
   ↓
RoBERTa tokenizer
   ↓
Pretrained sentiment model
   ↓
Softmax probabilities
   ↓
Sentiment, confidence, and class probabilities
```

## Project Structure

```text
sentiment-analysis-roberta/
├── app.py               # Streamlit user interface
├── sentiment_model.py   # Model loading and prediction logic
├── requirements.txt     # Direct Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Files excluded from Git
```

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-roberta.git
cd sentiment-analysis-roberta
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 4. Install CPU-only PyTorch

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 5. Install project dependencies

```powershell
pip install -r requirements.txt
```

## Run the Application

```powershell
python -m streamlit run app.py
```

Open the displayed local address, normally:

```text
http://localhost:8501
```

The pretrained model is downloaded from Hugging Face during the first run. Later runs use the locally cached files.

## Example

Input:

```text
I absolutely loved this movie!
```

Example output:

```text
Predicted sentiment: Positive
Confidence: 99.16%

Negative: 0.22%
Neutral:  0.61%
Positive: 99.16%
```

## Dataset

No dataset is required to run this application because it uses an already-trained model. Text is entered manually through the Streamlit interface and analyzed in real time.

## Limitations

- The model is designed primarily for English social-media text.
- Sarcasm, mixed emotions, and domain-specific language may be misclassified.
- The first application startup can be slow on a CPU because the model must be downloaded and loaded into memory.
- Predictions reflect patterns learned by the pretrained model and should not be treated as objective judgments.


## Model Credit

The pretrained sentiment model was published by [CardiffNLP](https://huggingface.co/cardiffnlp).

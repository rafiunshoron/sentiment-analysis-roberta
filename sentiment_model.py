import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


class SentimentAnalyzer:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _preprocess(text: str) -> str:
        processed_words = []

        for word in text.split():
            if word.startswith("@") and len(word) > 1:
                word = "@user"
            elif word.startswith("http"):
                word = "http"

            processed_words.append(word)

        return " ".join(processed_words)

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        processed_text = self._preprocess(text)

        encoded_input = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        encoded_input = {
            key: value.to(self.device)
            for key, value in encoded_input.items()
        }

        with torch.inference_mode():
            output = self.model(**encoded_input)
            probabilities = torch.softmax(output.logits, dim=-1).squeeze(0)

        probabilities = probabilities.cpu()
        predicted_index = int(torch.argmax(probabilities).item())

        scores = {
            LABELS[index]: round(float(probability), 4)
            for index, probability in enumerate(probabilities)
        }

        return {
            "label": LABELS[predicted_index],
            "confidence": round(
                float(probabilities[predicted_index]), 4
            ),
            "scores": scores,
            "device": str(self.device),
        }
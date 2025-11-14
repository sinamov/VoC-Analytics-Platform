import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This is a one-time setup script for local development.
# It downloads the large sentiment model so our app doesn't have to.

sentiment_model_path = "models/sentiment_model"
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"

if __name__ == "__main__":
    if os.path.exists(sentiment_model_path):
        print(f"✅ Sentiment model already exists at: {sentiment_model_path}")
    else:
        print(f"Sentiment model not found locally. Downloading from '{sentiment_model_name}'...")
        
        # Download and save the model
        tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
        
        tokenizer.save_pretrained(sentiment_model_path)
        model.save_pretrained(sentiment_model_path)
        
        print(f"✅ Sentiment Model downloaded and saved to: {sentiment_model_path}")
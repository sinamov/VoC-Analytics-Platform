import os
from dotenv import load_dotenv
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import TypedDict, List
import json

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI

# --- 0. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
print("✅ .env file loaded")

# --- 1. SET UP ROBUST PATHS (THE FIX) ---
# Get the absolute path of the directory where this script (agent.py) lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the project root)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Build absolute paths to our models
ner_model_path = os.path.join(PROJECT_ROOT, "models", "ner_model")
sentiment_model_path = os.path.join(PROJECT_ROOT, "models", "sentiment_model")

# --- 2. SET UP THE LLM ---
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini" 
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("API key or endpoint not found. Make sure .env is correct.")

llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY
)
print("✅ LLM Client Initialized")

# --- 3. LOAD OUR LOCAL "TOOLS" (THE MODELS) ---
# (Now we use our robust paths)

# Load Custom NER Model
try:
    nlp_ner = spacy.load(ner_model_path)
    print(f"✅ Custom NER Model Loaded from: {ner_model_path}")
except IOError:
    print(f"❌ ERROR: Could not load NER model from {ner_model_path}.")
    print("➡️ SOLUTION: Manually download/unzip it from Databricks.")
    nlp_ner = None

# Load Sentiment Model
try:
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
    print(f"✅ Sentiment Model Loaded from: {sentiment_model_path}")
except IOError:
    print(f"❌ ERROR: Could not load Sentiment model from {sentiment_model_path}.")
    print("➡️ SOLUTION: Run 'python setup_models.py' in your terminal.")
    sentiment_tokenizer = None
    sentiment_model = None

# Define the sentiment labels
sentiment_labels = ["negative", "neutral", "positive"]

# --- 4. DEFINE THE AGENT'S "STATE" ---
class VoCState(TypedDict):
    raw_text: str
    classification: str
    aspects: List[str]
    sentiment: str
    summary: str
print("✅ Agent State Defined")

# --- 5. CREATE OUR "TOOL" FUNCTIONS ---
def run_aspect_analysis(text: str) -> List[str]:
    """Uses our custom spaCy model to extract aspects."""
    if nlp_ner is None:
        return ["ERROR: NER model not loaded"]
    doc = nlp_ner(text)
    aspects = [ent.text for ent in doc.ents if ent.label_ == "ASPECT"]
    return aspects

def run_sentiment_analysis(text: str) -> str:
    """Uses our pre-trained Hugging Face model to get sentiment."""
    if sentiment_model is None or sentiment_tokenizer is None:
        return "ERROR: Sentiment model not loaded"
    try:
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)
        predicted_index = torch.argmax(probabilities, dim=-1).item()
        return sentiment_labels[predicted_index]
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "neutral"
print("✅ Tool Functions Defined")

# --- NEXT STEPS ---
# (Rest of agent code will go here)
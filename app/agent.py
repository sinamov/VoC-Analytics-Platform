import os
from dotenv import load_dotenv
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import TypedDict, List, Dict, Any
import json
from collections import defaultdict

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI

# --- 0. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
print("✅ .env file loaded")

# --- 1. SET UP ROBUST PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ner_model_path = os.path.join(PROJECT_ROOT, "models", "ner_model")
sentiment_model_path = os.path.join(PROJECT_ROOT, "models", "sentiment_model")

# --- 2. SET UP THE LLM ---
AZURE_OPENAI_DEPLOYMENT = "gpt-35-turbo" # Make sure this matches your deployment
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
    raise ValueError("API key or endpoint not found. Make sure .env is correct.")

llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    temperature=0
)
print("✅ LLM Client Initialized")

# --- 3. LOAD OUR LOCAL "TOOLS" (THE MODELS) ---
try:
    nlp_ner = spacy.load(ner_model_path)
    # We add the 'sentencizer' to the model's pipeline.
    # It will run before the 'ner' component.
    nlp_ner.add_pipe('sentencizer', before='ner')
    print(f"✅ Custom NER Model Loaded and Sentencizer added")

except IOError:
    print(f"❌ ERROR: Could not load NER model from {ner_model_path}.")
    nlp_ner = None

try:
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
    print(f"✅ Sentiment Model Loaded")
except IOError:
    print(f"❌ ERROR: Could not load Sentiment model from {sentiment_model_path}.")
    sentiment_tokenizer = None
    sentiment_model = None

sentiment_labels = ["negative", "neutral", "positive"]

# --- 4. DEFINE THE AGENT'S "STATE" ---
class VoCState(TypedDict):
    raw_text: str
    classification: str
    absa_results: List[Dict[str, str]] # e.g., [{"aspect": "shuffle", "sentiment": "negative"}]
    summary: str
print("✅ Agent State Defined")

# --- 5. CREATE OUR "TOOL" FUNCTIONS ---
# This tool function is now simpler, as the complex logic
# will live inside the 'absa_analyst' node
def run_sentiment_analysis(text: str) -> str:
    """Runs sentiment analysis on a single piece of text."""
    if sentiment_model is None or sentiment_tokenizer is None:
        return "ERROR: Sentiment model not loaded"
    try:
        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)
        predicted_index = torch.argmax(probabilities, dim=-1).item()
        return sentiment_labels[predicted_index]
    except Exception:
        return "neutral"

def aggregate_sentiments(sentiments: List[str]) -> str:
    """
    Aggregates a list of sentiments into one.
    Business Logic: "Negative" wins. "Positive" is second. "Neutral" is last.
    """
    if "negative" in sentiments:
        return "negative"
    if "positive" in sentiments:
        return "positive"
    return "neutral"

print("✅ Tool Functions Defined")


# -----------------------------------------------------------------
# --- 6. DEFINE THE AGENT "NODES" (***FINAL, ROBUST VERSION***) ---
# -----------------------------------------------------------------

def router_agent(state: VoCState) -> dict:
    print("--- ➡️ Executing Router Node ---")
    text = state["raw_text"]
    prompt = f"""
    You are a text classification agent. Classify the following email text
    into one: 'Bug Report', 'Feature Request', 'General Praise', 'Spam/Irrelevant'.
    Email: "{text}"
    Classification:
    """
    response = llm.invoke(prompt)
    classification = response.content.strip("'\"").replace("Classification: ", "")
    return {"classification": classification}


def absa_analyst(state: VoCState) -> dict:
    """
    Node 2: Runs our TRUE ABSA logic, as you described.
    """
    print("--- ➡️ Executing Sentence-Level ABSA Analyst Node ---")
    text = state["raw_text"]
    
    if nlp_ner is None:
        return {"absa_results": [{"aspect": "overall", "sentiment": run_sentiment_analysis(text)}]}

    # 1. Process the text with the NER model
    doc = nlp_ner(text)
    
    # 2. Store all sentiments for each aspect
    # e.g., {"shuffle": ["negative", "neutral"], "query optimizer": ["positive"]}
    aspect_sentiments = defaultdict(list)

    if not doc.ents:
        # No aspects found by the model. Fall back to overall sentiment.
        sentiment = run_sentiment_analysis(text)
        return {"absa_results": [{"aspect": "overall", "sentiment": sentiment}]}
    
    # 3. Loop through every entity (aspect) found
    for ent in doc.ents:
        if ent.label_ == "ASPECT":
            # 4. Find the sentence that contains this aspect
            sentence = ent.sent.text
            
            # 5. Get the sentiment for that specific sentence
            sentiment = run_sentiment_analysis(sentence)
            
            # 6. Store it. Use .lower() to group "Shuffle" and "shuffle"
            aspect_sentiments[ent.text.lower()].append(sentiment)

    # 7. Aggregate the results
    final_results = []
    for aspect, sentiments in aspect_sentiments.items():
        final_sentiment = aggregate_sentiments(sentiments)
        final_results.append({"aspect": aspect, "sentiment": final_sentiment})
        
    return {"absa_results": final_results}


def summarizer_agent(state: VoCState) -> dict:
    """
    Node 3: Writes a final summary (this prompt is now even more powerful)
    """
    print("--- ➡️ Executing Summarizer Node ---")
    
    # Convert the list of dicts into a clean string for the prompt
    absa_summary = "\n".join(
        [f"- {item['aspect']}: {item['sentiment']}" for item in state["absa_results"]]
    )
    
    prompt = f"""
    You are a technical analyst. Write a one-sentence summary
    of the following Voice of the Customer (VoC) analysis.
    
    Original Text: {state['raw_text']}
    Classification: {state['classification']}
    
    Aspect-Based Sentiment Analysis:
    {absa_summary}
    
    One-sentence summary:
    """
    
    response = llm.invoke(prompt)
    summary = response.content
    return {"summary": summary}

print("✅ Agent Nodes Defined (Robust ABSA)")

# -----------------------------------------------------------------
# --- 7. DEFINE THE "ROUTER" (THE CONDITIONAL LOGIC) ---
# -----------------------------------------------------------------

def route_logic(state: VoCState):
    print("--- Routing Logic ---")
    classification = state["classification"]
    
    if classification == 'Spam/Irrelevant':
        return "end_spam"
    else:
        return "continue_analysis"
print("✅ Router Logic Defined")

# -----------------------------------------------------------------
# --- 8. BUILD AND COMPILE THE GRAPH (THE FLOWCHART) ---
# -----------------------------------------------------------------
workflow = StateGraph(VoCState)

# Add nodes
workflow.add_node("router_agent", router_agent)
workflow.add_node("absa_analyst", absa_analyst)
workflow.add_node("summarizer_agent", summarizer_agent)

# Spam exit path
workflow.add_node("end_spam_node", lambda state: {
    "summary": "Email classified as Spam.",
    "absa_results": []
})
workflow.add_edge("end_spam_node", END)

# Define entry and graph structure
workflow.set_entry_point("router_agent")
workflow.add_conditional_edges(
    "router_agent",
    route_logic,
    {"end_spam": "end_spam_node", "continue_analysis": "absa_analyst"}
)
workflow.add_edge("absa_analyst", "summarizer_agent")
workflow.add_edge("summarizer_agent", END)

# Compile
compiled_agent_app = workflow.compile()
print("✅ LangGraph Agent Compiled! (Robust ABSA)")

# -----------------------------------------------------------------
# --- 9. A SIMPLE TEST ---
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- RUNNING AGENT TEST (V3) ---")
    
    # We use the same email, but we know our NER model
    # (trained on 25 examples) will likely fail to find aspects.
    # This is a *model* issue, not a *logic* issue.
    # Our code will now gracefully handle it.
    test_email = """
    Hi team,
    the shuffle in the new feature is not reliable and when i want to log in ti the structured streaming it crashes. But I have to say that the performance of the dynamic allocation although pretty slow is exceptional. They dynamic allocation must be improved through more memory allocation to the users. They dynamic allocation is very very slow and that hurts perforemance.
    
    - A Frustrated User
    """
    
    inputs = {"raw_text": test_email}
    final_state = compiled_agent_app.invoke(inputs)
    
    print("\n--- FINAL AGENT STATE (V3) ---")
    print(json.dumps(final_state, indent=2))
    
    # --- NER MODEL ISOLATION TEST ---
    print("\n--- NER MODEL ISOLATION TEST ---")
    if nlp_ner:
        doc = nlp_ner(test_email)
        if not doc.ents:
            print("NOTE: The NER model found no aspects. This is a model performance issue.")
            print("The agent will correctly fall back to 'overall' sentiment.")
        else:
            print(f"NER model found: {[ent.text for ent in doc.ents]}")
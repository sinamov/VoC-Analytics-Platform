Customer Feedback Analytics (VoC) Platform

This is an end-to-end MLOps project that builds a "Voice of the Customer" (VoC) analytics platform. The application uses a custom-trained AI agent to analyze user feedback for technical aspects, sentiment, and classification, and presents the findings in an interactive web dashboard.

Features

This application is a Streamlit web app with three main features:

Live Sandbox: A real-time interface to analyze a single piece of text (like an email or feedback form) using the custom AI agent.

Main Dashboard: A pre-computed dashboard (powered by Plotly and Altair) that aggregates all analyzed data, showing top-discussed aspects and sentiment breakdowns.

Batch Analysis: A batch uploader that allows a user to upload a CSV of emails, runs the agent over each row, and provides a downloadable results file.

Tech Stack

This project is built using a modern, scalable, cloud-native tech stack, demonstrating skills across the entire MLOps lifecycle.

Cloud & Infrastructure (IaC):

Azure: The cloud platform for all services.

Azure Bicep / ARM: Used for Infrastructure as Code to provision all resources programmatically.

Azure CLI: Used for deploying Bicep templates and managing resources.

Data Engineering (ETL):

Azure Databricks: The Spark platform used for the large-scale data processing job.

PySpark: Used to write the distributed ETL job to parse and clean raw .mbox files.

Azure Blob Storage: Stores the raw, unstructured .mbox file data.

Delta Lake: The "Bronze" table format used to store the cleaned, structured data.

Machine Learning & NLP:

spaCy: Used to train a custom NER model from scratch to identify specific technical ASPECTs (e.g., "query optimizer").

Hugging Face transformers: Used for sentiment analysis (cardiffnlp/twitter-roberta-base-sentiment).

PyTorch: The backend framework for the Hugging Face model.

AI Agent & Application:

LangGraph: Used to build the "brain" of the AI agent, creating a robust, multi-step graph of "workers" (nodes).

LangChain (langchain-openai): Used to connect to the Azure OpenAI LLM for routing and summarization.

Streamlit: The Python framework used to build the interactive web application.

Plotly & Altair: Used to create the beautiful, interactive dashboard visualizations.

DevOps & Deployment:

Git & GitHub: For all version control, including the cloud-to-local sync workflow.

Docker: Used to containerize the final application, including all models and dependencies.

Azure Container Apps (ACA): The final serverless platform for hosting the deployed Docker image.

How to Run Locally

Clone the Repository:

git clone [https://github.com/your-username/VoC-Analytics-Platform.git](https://github.com/your-username/VoC-Analytics-Platform.git)
cd VoC-Analytics-Platform


Create & Activate Virtual Environment:

python3 -m venv venv
source venv/bin/activate


Create .env File:
Create a file named .env in the root of the project. Get your credentials from the Azure Portal (under your OpenAI service's "Keys and Endpoint" page).

AZURE_OPENAI_API_KEY="your-key-here"
AZURE_OPENAI_ENDPOINT="your-endpoint-url-here"


Install Dependencies:

pip install -r requirements.txt


Download Models:
This project uses two models. The custom ner_model must be downloaded from the Databricks "release" (or provided in the repo). The sentiment model is downloaded by our setup script.

Sentiment Model: Run the setup script to download the 500MB sentiment model.

python setup_models.py


NER Model: (Add instructions here if you've hosted your ner_model_zip.zip file, otherwise, manually add the ner_model folder to models/).

Run the App:

python -m streamlit run app/app.py

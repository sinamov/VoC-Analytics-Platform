# Voice of Customer (VoC) Analytics Platform

This is an end-to-end Agentic-AI-NLP web app that builds a "Voice of Customer" (VoC) analytics platform. The application uses a custom-trained AI agent to analyze customer feedback for technical aspects, sentiment, and classification, and presents the findings in an interactive web dashboard.

## Features

This application is a Streamlit web app with three main features:

* **Live Sandbox:** A real-time interface to analyze a single piece of text (like an email or feedback form) using the custom AI agent.
* **Main Dashboard:** A pre-computed dashboard (powered by Plotly and Altair) that aggregates all analyzed data, showing top-discussed aspects and sentiment breakdowns.
* **Batch Analysis:** A batch uploader that allows a user to upload a CSV of emails, runs the agent over each row, and provides a downloadable results file.

## Tech Stack

This project is built using a modern, scalable, cloud-native tech stack, demonstrating skills across the entire MLOps lifecycle.

### Cloud & Infrastructure (IaC)

* **Azure:** The cloud platform for all services.
* **Azure Bicep / ARM:** Used for Infrastructure as Code to provision all resources programmatically.
* **Azure CLI:** Used for deploying Bicep templates and managing resources.

### Data Engineering (ETL)

* **Azure Databricks:** The Spark platform used for the large-scale data processing job.
* **PySpark:** Used to write the distributed ETL job to parse and clean raw .mbox files.
* **Azure Blob Storage:** Stores the raw, unstructured .mbox file data.
* **Delta Lake:** The "Bronze" table format used to store the cleaned, structured data.

### Machine Learning & NLP

* **spaCy:** Used to train a custom NER model from scratch to identify specific technical ASPECTs (e.g., "query optimizer").
* **Hugging Face `transformers`:** Used for sentiment analysis (`cardiffnlp/twitter-roberta-base-sentiment`).
* **PyTorch:** The backend framework for the Hugging Face model.

### AI Agent & Application

* **LangGraph:** Used to build the "brain" of the AI agent, creating a robust, multi-step graph of "workers" (nodes).
* **LangChain (`langchain-openai`):** Used to connect to the Azure OpenAI LLM for routing and summarization.
* **Streamlit:** The Python framework used to build the interactive web application.
* **Plotly & Altair:** Used to create the interactive dashboard visualizations.

### DevOps & Deployment

* **Git & GitHub:** For all version control, including the cloud-to-local sync workflow.
* **Docker:** Used to containerize the final application, including all models and dependencies.
* **Azure Container Apps (ACA):** The final serverless platform for hosting the deployed Docker image.

## How to Run Locally

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/VoC-Analytics-Platform.git](https://github.com/your-username/VoC-Analytics-Platform.git)
    cd VoC-Analytics-Platform
    ```

2.  **Create & Activate Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Create `.env` File:**
    Create a file named `.env` in the root of the project. Get your credentials from the Azure Portal (under your OpenAI service's "Keys and Endpoint" page).
    ```ini
    AZURE_OPENAI_API_KEY="your-key-here"
    AZURE_OPENAI_ENDPOINT="your-endpoint-url-here"
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download Models:**
    This project uses two models. The custom `ner_model` must be downloaded from the repo (or databricks if you want to retrain your NER model). The sentiment model is downloaded by our setup script.
    * **Sentiment Model:** Run the setup script to download the 500MB sentiment model.
        ```bash
        python setup_models.py
        ```
    * **NER Model:** (Manually download `ner_model_zip.zip` from your Databricks workspace, unzip it, and place the `ner_model` folder inside your local `models/` directory).

6.  **Run the App:**
    ```bash
    python -m streamlit run app/app.py
    ```
## Deployment to Azure

This application is designed to be deployed as a containerized app on Azure Container Apps (ACA), which offers a generous free tier and scales to zero.

1.  **(One-Time) Install Docker**
    You must have Docker Desktop installed and running on your local machine to build the image.

2.  **Log in to Azure Container Registry (ACR)**
    In your terminal, log in to the ACR we created with Bicep (`voc...acr`).
    ```bash
    az acr login --name <your-registry-name>
    ```

3.  **Build & Push the Docker Image**
    We use `docker buildx` to build a multi-platform image (specifically `linux/amd64`) that can run on Azure's servers, even if you are on an Apple Silicon Mac.
    This command builds the image, tags it for ACR, and pushes it all in one step.
    ```bash
    docker buildx build --platform linux/amd64 -t <your-registry-name>.azurecr.io/voc-app:latest --push .
    ```

4.  **Deploy to Azure Container Apps**
    We will deploy using the robust, two-step method.

    **First, create the ACA Environment (One-Time Setup):**
    This creates the serverless "sandbox" for our app.
    ```bash
    az containerapp env create \
      --name voc-env \
      --resource-group voc-analytics-rg \
      --location eastus
    ```

    **Second, create and deploy the app:**
    This command pulls your image from ACR, injects your secrets, and makes the app public.
    ```bash
    az containerapp create \
      --name voc-analytics-app \
      --resource-group voc-analytics-rg \
      --environment voc-env \
      --image <your-registry-name>.azurecr.io/voc-app:latest \
      --registry-server <your-registry-name>.azurecr.io \
      --target-port 8501 \
      --ingress external \
      --min-replicas 0 \
      --secrets "openai-key=YOUR_KEY_HERE" "openai-endpoint=YOUR_ENDPOINT_HERE" \
      --env-vars "AZURE_OPENAI_API_KEY=secretref:openai-key" "AZURE_OPENAI_ENDPOINT=secretref:openai-endpoint"
    ```
    
    After this command finishes, the CLI will output the public URL for your application.
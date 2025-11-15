# Use the official Streamlit base image
FROM streamlit/base:latest

# Set the working directory inside the container
WORKDIR /app

# Copy our requirements file and install libraries
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- This is the key "Build Time" step ---
# Copy the model setup script and run it ONCE to download the model
COPY setup_models.py .
RUN python setup_models.py

# Copy the rest of our application code
# This includes the 'app/' folder and the 'models/ner_model/' folder
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# The command to run when the container starts
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
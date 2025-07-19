# Import necessary libraries for file operations and JSON handling
import os
import json
# Import pandas for data manipulation and CSV reading
import pandas as pd
# Import dotenv for loading environment variables from .env files
from dotenv import load_dotenv
# Import Neo4j graph integration from LangChain community package
from langchain_community.graphs import Neo4jGraph
# Import OpenAI chat model wrapper from LangChain
from langchain_community.chat_models import ChatOpenAI
# Import prompt template functionality from LangChain
from langchain.prompts import PromptTemplate
# Import retrieval-based question answering chain from LangChain
from langchain.chains import RetrievalQA
# Import OpenAI embeddings for vector representations
from langchain_community.embeddings import OpenAIEmbeddings
# Import FAISS vector store for similarity search
from langchain_community.vectorstores import FAISS

# Import os again (duplicate import - could be removed)
import os

# Set the OpenAI API key as an environment variable
# WARNING: This is a security risk - API keys should not be hardcoded in source code
os.environ["OPENAI_API_KEY"] = "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"

# Check if the OpenAI API key environment variable is set (returns True/False)
print("OpenAI API Key set:", os.getenv("OPENAI_API_KEY") is not None)

# Define the path to the CSV file containing training data
file_path = 'train.csv'  # Adjust path if needed

# Read the CSV file into a pandas DataFrame and limit to first 10 rows for testing
df = pd.read_csv(file_path).head(10)
# Print the shape (dimensions) of the loaded DataFrame
print("Loaded DataFrame:", df.shape)
# Display the first few rows of the DataFrame
print(df.head())

# Convert the DataFrame to a list of dictionaries (JSON-like format)
# Each row becomes a dictionary with column names as keys
jsonData = df.to_dict(orient='records')

# Print the column names of the DataFrame for debugging
print("DataFrame columns:", df.columns)

# Start testing the Neo4j database connection
print("Testing Neo4j connection...")
try:
    # Create a Neo4j graph connection using Neo4j Aura cloud service
    # WARNING: Database credentials should not be hardcoded in source code
    graph = Neo4jGraph(
        url="neo4j+s://b3eeb433.databases.neo4j.io",  # Neo4j Aura connection URL
        username="neo4j",                              # Database username
        password="nKQXCtdNXFOlDq_6m9f6gyHFG_ecuzegpkq4bIT91Js"  # Database password
    )
    
    # Execute a simple test query to verify connection
    result = graph.query("RETURN 1 AS test")
    # Print success message with the query result
    print("Neo4j connection successful! Result:", result)

    # Loop through each record in the JSON data to create graph nodes
    for i, node in enumerate(jsonData):
        # Extract the product name/ID from the current record
        product_name = node["PRODUCT_ID"]  # or whatever the correct column is
        # Print progress information for current node being processed
        print(f"Processing node {i}: {product_name}")
        # Execute Cypher query to create or merge a Product node in Neo4j
        # MERGE ensures the node is created only if it doesn't already exist
        graph.query(f"MERGE (p:Product {{name: '{product_name}'}})")

# Handle any exceptions that occur during Neo4j operations
except Exception as e:
    # Print error message if connection or operations fail
    print("Neo4j connection failed:", e)

# Define a system prompt for the AI agent that will generate Cypher queries
system_prompt = '''
You are a helpful agent designed to fetch product data from a knowledge graph.
Each product can be linked to the following entity types:
- category
- brand
- color
- measurement
- characteristic

Given a user query, determine the Cypher query needed to retrieve products based on those relationships.
'''

# Define a sample user question for testing the system
user_question = "Show me waterproof items in kitchen category."
# Combine system prompt, user question, and instruction to create full prompt
prompt = system_prompt + "\nUser question: " + user_question + "\nReturn only Cypher query:"

# Import requests library for making HTTP calls to external APIs
import requests

# Define function to interact with Ollama local LLM server
def ollama_chat(prompt, model="mistral"):
    # Make POST request to Ollama API endpoint
    response = requests.post(
        "http://localhost:11434/api/generate",  # Ollama server URL
        json={
            "model": model,      # Specify which model to use (default: mistral)
            "prompt": prompt,    # The prompt to send to the model
            "stream": False      # Disable streaming response
        }
    )
    # Parse the JSON response from Ollama
    data = response.json()
    # Check if response contains the expected "response" field
    if "response" in data:
        # Return the generated text response
        return data["response"]
    else:
        # Print error information if response format is unexpected
        print("Ollama error:", data)
        # Return formatted error message
        return f"Ollama error: {data.get('error', 'Unknown error')}"

# Example usage of the ollama_chat function
# Define a test prompt for the LLM
prompt = "What are some waterproof items for adults?"
# Call the function with the prompt using mistral model
result = ollama_chat(prompt, model="mistral")
# Print the generated response from the LLM
print(result)
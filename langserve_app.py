import os
from uvicorn import run
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from fastapi import FastAPI

# Initialize the FastAPI app
app = FastAPI(title="LangServe LLaMA2 Chatbot", version="1.0")

# Define the LLaMA2 model
llm = OllamaLLM(model="llama2")  # Ensure the model name matches your setup

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    "Answer the following question concisely:\n{question}"
)

# Add LangServe routes
add_routes(app, prompt_template | llm, path="/invoke")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 11435))  # Default to 11435 if PORT is not set
    run(app, host="0.0.0.0", port=port)
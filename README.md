# Promtior Chatbot

**Description:**  
This project was developed as part of the Promtior technical test. The chatbot leverages LangChain, Ollama, and the LLaMA2 model to answer questions based on web documents and PDF files. The solution is fully containerized using Docker, allowing for seamless deployment on local machines or cloud platforms. The architecture focuses on information retrieval and generating precise, context-aware responses.

## Technologies Used
- **LangChain**: Framework for integrating Large Language Models (LLMs).
- **Ollama**: Server for hosting LLaMA2 models.
- **FAISS**: VectorStore for efficient information retrieval.
- **FastAPI**: Framework for exposing the chatbot's REST API.
- **Uvicorn**: ASGI server for running the FastAPI application.
- **GitHub Actions**: Automation tool for workflows (CI/CD).
- **Docker**: Containerization of services for easy deployment.

## Key Features
- Load data from multiple sources: web documents and PDF files.
- Retrieve the most relevant content using FAISS.
- Summarize and generate concise answers focused on user queries.
- Expose an API endpoint for interacting with the chatbot.

## Component Diagram

![Component Diagram](doc/PromtiorChatbotComponentDiagram.png)

---

# Deployment with Docker on Local Machines

## Prerequisites
- Ensure Docker and Docker Compose are installed on your machine.
- Verify you have sufficient memory and CPU resources to run both the chatbot and the LLaMA2 model.

## Steps to Deploy
- **Clone the Repository**: Clone the project repository to your local machine:
  ```sh
  git clone https://github.com/nicofarizano/chatbot-RAG.git
  cd <repository_directory>

- **Start the Services**: Run the following command to start the containers using the provided [`docker-compose.yml`](docker-compose.yml) file:
  ```sh
  docker-compose up

- **Access the Chatbot:** Once the containers are running, access the chatbot service at:
  URL: [http://localhost:11435](http://localhost:11435)

## Test and Verify:
Open a browser and navigate to the chatbot endpoint.
Ask sample questions, such as:
- What services does Promtior offer?
- When was the company founded?

**Shut Down the Services**: When you're done testing, stop and remove the containers using:
  ```sh
  docker-compose down
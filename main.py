import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from loaders import DocumentLoader
from config import get_settings
from model import Query, ChatResponse, AIModel
from contextlib import asynccontextmanager

# Configuración de logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar componentes
settings = get_settings()
os.environ["USER_AGENT"] = settings.user_agent
ai_model = AIModel()
document_loader = DocumentLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        documents = document_loader.load_all_documents()
        ai_model.initialize_vectorstore(documents)
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise
    yield

# Configurar FastAPI
app = FastAPI(
    title="Promtior Chatbot",
    description="RAG chatbot API using Ollaama and LLaMA2",
    version="2.0.0",
    lifespan=lifespan,
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta principal para cargar la interfaz web
@app.get("/")
def read_root():
    """Endpoint to serve the web interface"""
    try:
        return FileResponse("index.html")
    except Exception as e:
        logger.error(f"Error serving index.html file: {e}")
        raise HTTPException(status_code=404, detail="File not found")

# Ruta para manejar preguntas enviadas al chatbot
@app.post("/ask", response_model=ChatResponse)
async def ask_question(query: Query) -> ChatResponse:
    """Endpoint to process user questions"""
    try:
        response = ai_model.generate_response(query.question)
        return ChatResponse(answer=response)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port, reload=True)
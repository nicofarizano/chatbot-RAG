import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM
from loaders import DocumentLoader
from config import get_settings
import logging

# Configuración de logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI y configurar CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar configuración
settings = get_settings()

# Clase para manejar la entrada JSON
class Query(BaseModel):
    question: str

# Cargar documentos y crear VectorStore
logger.info("Cargando documentos...")
loader = DocumentLoader()
all_docs = loader.load_all_documents()
logger.info(f"Total de fragmentos: {len(all_docs)}")

embedding_model = GPT4AllEmbeddings()
vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)
logger.info("VectorStore creado con éxito.")

# Configurar el modelo Ollama con Llama2
llm = OllamaLLM(model=settings.model_name)
logger.info(f"Modelo {settings.model_name} cargado.")

# Función para resumir documentos recuperados basándose en una pregunta
def summarize_documents(llm, docs, question, max_words=settings.max_words_response):
    combined_text = " ".join([doc.page_content for doc in docs])
    max_length = settings.max_context_length
    if len(combined_text) > max_length:
        combined_text = combined_text[:max_length]

    summary_prompt = (
        f"A continuación, tienes un texto. Responde a la siguiente pregunta basándote en el texto. "
        f"Prioriza información relevante y no incluyas contenido que no esté relacionado con la pregunta. "
        f"Limita tu respuesta a no más de {max_words} palabras.\n\n"
        f"Pregunta: {question}\n\n"
        f"Texto:\n{combined_text}"
    )

    try:
        summary = llm.invoke(summary_prompt)
    except Exception as e:
        logger.error(f"Error en LLM: {e}")
        return "Ocurrió un error procesando tu pregunta."

    return summary

# Ruta principal para cargar la interfaz web
@app.get("/")
def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

# Ruta para manejar preguntas enviadas al chatbot
@app.post("/ask")
async def ask_question(query: Query):
    question = query.question
    docs = vectorstore.similarity_search(question, k=5)
    if len(docs) == 0:
        return {"answer": "No se encontraron documentos relevantes para tu pregunta."}
    
    response = summarize_documents(llm, docs, question)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
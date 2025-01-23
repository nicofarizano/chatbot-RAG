import os
os.environ["USER_AGENT"] = "promtior-bot"

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM
import logging

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar middleware CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "llama2"  # Modelo usado en Ollama

# Clase para manejar la entrada JSON
class Query(BaseModel):
    question: str

# Cargar datos desde la web
logger.info("Cargando datos...")
web_loader = WebBaseLoader("https://www.promtior.ai")   
web_data = web_loader.load()    

# Cargar datos desde el archivo PDF proporcionado
pdf_loader = PyPDFLoader("AI Engineer.pdf")
pdf_data = pdf_loader.load()    

# Dividir los datos cargados en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)  
web_splits = text_splitter.split_documents(web_data)
pdf_splits = text_splitter.split_documents(pdf_data)

# Combinar los fragmentos en un solo conjunto
all_splits = web_splits + pdf_splits    
logger.info(f"Total de fragmentos: {len(all_splits)}")

# Crear el VectorStore con FAISS y GPT4AllEmbeddings
embedding_model = GPT4AllEmbeddings()
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_model)
logger.info("VectorStore creado con éxito.")

# Configurar el modelo Llama2
llm = OllamaLLM(model=MODEL)
logger.info(f"Modelo {MODEL} cargado.")

# Función para resumir documentos recuperados basándose en una pregunta
def summarize_documents(llm, docs, question, max_words=200):
    combined_text = " ".join([doc.page_content for doc in docs])
    max_length = 5000
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

# Ruta principal para cargar la interfaz web (archivo HTML)
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
    
    response = summarize_documents(llm, docs, question, max_words=200)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
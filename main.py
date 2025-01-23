import os
os.environ["USER_AGENT"] = "promtior-bot"

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM

MODEL = "llama2"  # Modelo usado en Ollama
app = FastAPI() # Inicializar la aplicación FastAPI

# Clase para manejar la entrada JSON
class Query(BaseModel):
    question: str

# Cargar datos desde la web
print("Cargando datos...")
web_loader = WebBaseLoader("https://www.promtior.ai")   
web_data = web_loader.load()    

# Cargar datos desde el PDF
pdf_loader = PyPDFLoader("AI Engineer.pdf")
pdf_data = pdf_loader.load()    

# Dividir datos en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)  
web_splits = text_splitter.split_documents(web_data)
pdf_splits = text_splitter.split_documents(pdf_data)

# Combinar todos los fragmentos
all_splits = web_splits + pdf_splits    
print(f"Total de fragmentos: {len(all_splits)}")

# Crear el VectorStore con FAISS y GPT4AllEmbeddings
embedding_model = GPT4AllEmbeddings()
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_model)
print("VectorStore creado con éxito.")

# Configurar el modelo Llama2
llm = OllamaLLM(model=MODEL)
print(f"Modelo {MODEL} cargado.")

def summarize_documents(llm, docs, question, max_words=200): # Resume los documentos recuperados utilizando el modelo de lenguaje, enfocado en la pregunta inicial.
    combined_text = " ".join([doc.page_content for doc in docs])
    summary_prompt = (
        f"A continuación, tienes un texto. Responde a la siguiente pregunta basándote en el texto. "
        f"Prioriza información relevante y no incluyas contenido que no esté relacionado con la pregunta. "
        f"Limita tu respuesta a no más de {max_words} palabras.\n\n"
        f"Pregunta: {question}\n\n"
        f"Texto:\n{combined_text}"
    )
    summary = llm.invoke(summary_prompt)
    return summary

@app.post("/ask")
async def ask_question(query: Query):
    """
    Endpoint para manejar preguntas de los usuarios.
    """
    question = query.question
    docs = vectorstore.similarity_search(question, k=5)  # Recuperar k documentos
    if len(docs) == 0:
        return {"answer": "No se encontraron documentos relevantes para tu pregunta."}
    
    # Generar la respuesta
    response = summarize_documents(llm, docs, question, max_words=200)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)